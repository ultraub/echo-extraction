"""Run the three-stage extraction pipeline over a JSONL of reports.

    python scripts/run_extraction.py --input reports.jsonl --output out.jsonl \
        --api-base http://127.0.0.1:30000/v1 --model Qwen/Qwen3.5-27B-FP8

Each report goes through:

    Stage 1  extract   findings the report states outright
    Stage 2  infer     findings that follow from what is stated
             merge + clean   (deterministic, src/merge.py)
    Stage 3  validate  corrections to the merged result, checked against the report
             apply corrections, clean, drop uncited values

Every stage is one chat completion with `response_format=json_schema`
(strict), so outputs conform to src/schemas.py by construction. The server
can be anything OpenAI-compatible that enforces the schema at decode time —
a local SGLang/vLLM server or a cloud endpoint. See README.md.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompts import build_explicit_prompt, build_inference_prompt, build_validation_prompt
from src.schemas import build_explicit_schema, build_inference_schema, build_corrections_schema
from src.merge import merge_results, clean, apply_corrections, remove_empty_evidence

SYSTEM_MSG = "Respond with valid JSON only."

# Output-token budgets per stage (observed p95 in the paper's cohort plus headroom).
MAX_TOKENS = {"explicit": 16384, "inference": 32768, "validation": 8192}


# =============================================================================
# Model calls
# =============================================================================

def make_client(api_base: str, api_key: str):
    import openai
    return openai.OpenAI(base_url=api_base, api_key=api_key or "EMPTY")


def complete(client, model: str, prompt: str, schema: dict, max_tokens: int,
             local: bool, retries: int = 3) -> str:
    """One schema-constrained chat completion. Returns the raw text ('' on failure)."""
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_MSG},
                  {"role": "user", "content": prompt}],
        temperature=1.0,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "extraction", "strict": True, "schema": schema}},
    )
    if local:
        # Sampling used for the self-hosted Qwen3.5 runs in the paper. min_p
        # suppresses the whitespace-loop degeneration this model family shows
        # at long output lengths; thinking is disabled so the whole token
        # budget goes to the JSON.
        kwargs.update(max_tokens=max_tokens, top_p=1.0,
                      extra_body={"min_p": 0.2, "top_k": 40,
                                  "chat_template_kwargs": {"enable_thinking": False}})
    else:
        kwargs.update(max_completion_tokens=max_tokens)

    err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            if text:
                return text
            err = f"empty response (finish_reason={resp.choices[0].finish_reason})"
        except Exception as e:  # network / API errors
            err = e
        time.sleep(2 ** attempt)
    print(f"  [ERROR] request failed after {retries} attempts: {err}", file=sys.stderr)
    return ""


def parse_json(text: str):
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {str(e)[:120]}"


# =============================================================================
# Pipeline for one report
# =============================================================================

def normalize_text(text: str) -> str:
    return (text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
                .replace("’", "'").replace("‘", "'"))


def number_events(result: dict) -> dict:
    for i, ev in enumerate(result.get("events", [])):
        ev["event_id"] = str(ev.get("event_id") or i + 1)
    return result


def process_report(client, cfg, source_text: str):
    """Returns (extraction, meta). meta records per-stage timing and parse errors."""
    text = normalize_text(source_text)
    meta = {}

    def call(stage, prompt, schema):
        t0 = time.time()
        raw = complete(client, cfg.model, prompt, schema, MAX_TOKENS[stage], cfg.local)
        obj, err = parse_json(raw)
        meta[stage] = {"seconds": round(time.time() - t0, 1), "error": err}
        return obj

    # Stage 1
    explicit = call("explicit",
                    cfg.demo_prefix + build_explicit_prompt(text, use_segments=cfg.segments),
                    cfg.explicit_schema)
    if not explicit or not explicit.get("events"):
        return {"events": []}, meta
    explicit = number_events(explicit)

    # Stage 2
    inferred = call("inference",
                    build_inference_prompt(text, json.dumps(explicit, indent=2),
                                           use_segments=cfg.segments),
                    cfg.inference_schema)
    merged = clean(merge_results(explicit, number_events(inferred) if inferred else None))

    # Stage 3
    validation = call("validation",
                      build_validation_prompt(text, json.dumps(merged, indent=2),
                                              use_segments=cfg.segments),
                      cfg.corrections_schema)
    if validation:
        # Keep Stage 3's edits verbatim so every correction is auditable —
        # including corrected_value strings before any normalization.
        meta["validation"]["output"] = validation
        merged = clean(apply_corrections(merged, validation))
    if cfg.require_evidence:
        merged = remove_empty_evidence(merged)
    return merged, meta


# =============================================================================
# Entry point
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="JSONL with row_id and source_text per line")
    p.add_argument("--output", required=True, help="JSONL to write (one record per report)")
    p.add_argument("--model", required=True, help="Model id (or deployment name) sent to the API")
    p.add_argument("--api-base", default="http://127.0.0.1:30000/v1",
                   help="OpenAI-compatible base URL. Default: local SGLang server.")
    p.add_argument("--api-key", default=None,
                   help="API key; falls back to OPENAI_API_KEY / AZURE_OPENAI_API_KEY. Not needed locally.")
    p.add_argument("--workers", type=int, default=8,
                   help="Reports processed concurrently (the server batches the requests).")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N reports.")
    # Prompt-component ablation (defaults reproduce the paper's primary configuration).
    p.add_argument("--demo-shots", type=int, default=0,
                   help="Prepend N worked report->extraction examples to Stage 1 (needs --demo-file).")
    p.add_argument("--demo-file", default=None,
                   help="JSONL of demonstrations, each with source_text and extraction.")
    p.add_argument("--no-reasoning", action="store_true",
                   help="Drop the per-field reasoning property from the schemas.")
    p.add_argument("--no-evidence", action="store_true",
                   help="Make the evidence span optional and keep uncited values.")
    p.add_argument("--segments", action="store_true",
                   help="Also extract 17-segment AHA wall motion (per-segment values). "
                        "Off by default; the paper's reported runs did not enable this "
                        "mode and it is unevaluated.")
    cfg = p.parse_args()

    # Load .env (credentials) if present, then resolve the key.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    cfg.local = "127.0.0.1" in cfg.api_base or "localhost" in cfg.api_base
    cfg.api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    if not cfg.local and not cfg.api_key:
        p.error("--api-key (or OPENAI_API_KEY / AZURE_OPENAI_API_KEY) is required for a remote endpoint")

    cfg.require_evidence = not cfg.no_evidence
    cfg.explicit_schema = build_explicit_schema(include_reasoning=not cfg.no_reasoning,
                                                include_raw_text=cfg.require_evidence,
                                                use_segments=cfg.segments)
    cfg.inference_schema = build_inference_schema(include_reasoning=not cfg.no_reasoning,
                                                  include_raw_text=cfg.require_evidence,
                                                  use_segments=cfg.segments)
    cfg.corrections_schema = build_corrections_schema()

    cfg.demo_prefix = ""
    if cfg.demo_shots > 0:
        if not cfg.demo_file:
            p.error("--demo-shots requires --demo-file")
        from src.fewshot import load_demo_pool, select_demos, build_demo_prefix
        cfg.demo_prefix = build_demo_prefix(select_demos(load_demo_pool(cfg.demo_file), cfg.demo_shots))

    reports = [json.loads(l) for l in open(cfg.input) if l.strip()]
    if cfg.limit:
        reports = reports[:cfg.limit]
    print(f"{len(reports)} reports  model={cfg.model}  api={cfg.api_base}  workers={cfg.workers}")

    client = make_client(cfg.api_base, cfg.api_key)
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(cfg.output, "w") as out, ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(process_report, client, cfg, r["source_text"]): r for r in reports}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                extraction, meta = fut.result()
            except Exception as e:
                extraction, meta = {"events": []}, {"error": repr(e)}
            record = {k: v for k, v in r.items() if k != "source_text"}
            record["extraction"] = extraction
            record["meta"] = meta
            out.write(json.dumps(record) + "\n")
            out.flush()
            done += 1
            errs = [s for s, m in meta.items() if isinstance(m, dict) and m.get("error")]
            print(f"  [{done}/{len(reports)}] {r.get('row_id')}: events={len(extraction.get('events', []))}"
                  + (f"  errors={errs}" if errs else ""))
    print(f"Wrote {cfg.output}")


if __name__ == "__main__":
    main()

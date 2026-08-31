"""Optional few-shot demonstrations for the extraction stage.

The pipeline runs zero-shot by default. Passing `--demo-shots K --demo-file
<file>` prepends K worked report→extraction examples to the extraction
prompt, which raises recall on reports whose phrasing the model would
otherwise only resolve through the inference stage.

The demo file is JSONL, one record per line, each carrying a report and the
extraction it should produce:

    {"row_id": "...", "source_text": "...", "proc_name": "...",
     "extraction": {"events": [...]}}

Draw demonstrations from a different cohort than the one being extracted, so
that examples never leak evaluation data. Use extractions you have verified
— a wrong demonstration teaches the wrong thing.
"""

import json


def load_demo_pool(source_jsonl: str, extraction_jsonl: str = None):
    """Return [(row_id, source_text, extraction, proc_name), ...].

    Pass the same path twice when one file carries both `source_text` and
    `extraction` (the usual case). Pass two paths to join a report file
    against a separate extraction-output file by report id.
    """
    extraction_jsonl = extraction_jsonl or source_jsonl

    srcs = {}
    for line in open(source_jsonl):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        srcs[str(r["row_id"])] = r

    exts = {}
    for line in open(extraction_jsonl):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rid = str(r.get("report_id") or r.get("row_id"))
        exts[rid] = r.get("extraction", r)

    pool = []
    for rid in sorted(srcs.keys() & exts.keys()):
        if not (exts[rid].get("events") or []):
            continue  # an empty extraction teaches nothing
        pool.append((rid, srcs[rid]["source_text"], exts[rid],
                     srcs[rid].get("proc_name", "")))
    return pool


def select_demos(pool, k, prefer_exam_diversity=True):
    """Pick k demonstrations. Deterministic — no RNG — so runs reproduce.

    With prefer_exam_diversity, spread across distinct `proc_name` values
    before repeating one, so the examples show more than a single exam type.
    """
    if k <= 0:
        return []
    if not prefer_exam_diversity:
        return pool[:k]

    chosen, seen = [], set()
    for rec in pool:
        if rec[3] not in seen:
            chosen.append(rec)
            seen.add(rec[3])
        if len(chosen) == k:
            return chosen
    for rec in pool:                      # fill remainder in order
        if rec not in chosen:
            chosen.append(rec)
        if len(chosen) == k:
            break
    return chosen


def _compact(extraction: dict) -> str:
    """Render an extraction as the compact JSON a demonstration should show."""
    events = []
    for ev in extraction.get("events", []):
        slim = {k: ev[k] for k in ("event_type", "event_label") if k in ev}
        for ft in ("severity_fields", "numeric_fields", "categorical_fields"):
            if ev.get(ft):
                slim[ft] = ev[ft]
        events.append(slim)
    return json.dumps({"events": events}, ensure_ascii=False)


def build_demo_prefix(demos) -> str:
    """Format demonstrations into text to prepend to the extraction prompt.

    Returns "" for an empty list, which leaves the prompt untouched (the
    zero-shot default).
    """
    if not demos:
        return ""
    out = [
        "## WORKED EXAMPLES (reference extractions from a different cohort)",
        "Study the following verified report→extraction pairs, then apply the "
        "same rules to the target report. Do not copy values from these "
        "examples; extract only from the target report.",
    ]
    for i, (_rid, src, ext, _exam) in enumerate(demos, 1):
        out.append(f"\n### Example {i} — report:\n{src.strip()}")
        out.append(f"### Example {i} — correct extraction:\n{_compact(ext)}")
    out.append("\n## END EXAMPLES — now extract from the target report below.\n")
    return "\n".join(out)

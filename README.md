# Structured extraction of echocardiography reports

Turns free-text echocardiography reports into structured data across 48
clinical fields using a general-purpose language model — no fine-tuning, no
task-specific training. The model is constrained by a JSON schema at decode
time, so every output conforms to the schema by construction.

This is the extraction pipeline from the accompanying paper, reduced to what
the paper describes. It is about 1,500 lines of Python, most of it prompt
text and schema.

---

## How it works

Each report goes through three model calls (Figure 1 in the paper):

**1. Extract** — pull values the report states outright.
`"EF 50%"` → `ejection_fraction = 50`

**2. Infer** — derive findings that follow from what is stated.
`"RVSP 45 mmHg"` → `pulmonary_hypertension`
Inference is deliberately conservative: the prompt forbids reading "normal"
into a silent report. Absence of a finding is not evidence of normality.

**3. Validate** — re-read the source and return corrections and removals,
which are applied to the merged Stage 1+2 result.

Between and after the calls, `src/merge.py` does the deterministic work:
join inferred values onto the extracted ones, drop `not_mentioned`
placeholders, collapse duplicates, apply the Stage 3 edits, and finally
drop any value that cites no supporting text.

Every stage sends `response_format={"type":"json_schema","strict":true}`, so
each field's enum is enforced during generation rather than validated
afterwards. **This is the load-bearing detail**: backends that accept a
schema-shaped request but only apply best-effort JSON formatting will fail on
a schema this deeply nested. Test yours before trusting it — see *Checking
your backend* below.

---

## Layout

```
src/
  config.py      The 48 fields and every enumerated value they accept
  schemas.py     Builds the JSON schemas used to constrain decoding
  prompts.py     The three prompts, verbatim from the paper's supplement
  merge.py       Merge, clean, apply corrections, evidence filter
  fewshot.py     Optional worked examples for Stage 1 (ablation)
scripts/
  run_extraction.py   Runs the pipeline over a JSONL of reports
examples/
  example_input.jsonl  3 synthetic reports showing the input format
  demos.jsonl          3 synthetic report→extraction pairs for few-shot prompting
```

Start with `src/config.py` (what gets extracted) and `src/prompts.py` (how
it is asked for).

---

## Running it

```bash
pip install -r requirements.txt
```

### Against a local open-weights model

This is the paper's primary configuration (Qwen3.5-27B-FP8 on two H100s)
and keeps protected health information inside your institution. Start any
OpenAI-compatible server with schema-constrained decoding; we used SGLang:

```bash
pip install "sglang[all]==0.5.10.post1"
python -m sglang.launch_server --model-path Qwen/Qwen3.5-27B-FP8 \
    --tp 2 --port 30000 --grammar-backend xgrammar

python scripts/run_extraction.py \
    --input examples/example_input.jsonl --output out/results.jsonl \
    --model Qwen/Qwen3.5-27B-FP8
```

The runner's default `--api-base` is `http://127.0.0.1:30000/v1`. When it
points at localhost the runner sends the sampling settings used in the
paper (`temperature=1, top_p=1, top_k=40, min_p=0.2`, thinking disabled).

### Against a cloud endpoint

```bash
cp .env.template .env      # add your key
python scripts/run_extraction.py \
    --input examples/example_input.jsonl --output out/results.jsonl \
    --api-base "https://<resource>.openai.azure.com/openai/v1" \
    --model gpt-5.4-mini
```

`--api-base` takes any OpenAI-compatible base URL. Remote calls send
`temperature=1` and `max_completion_tokens`; provider-specific sampling
parameters are not sent.

---

## Input and output

Input is JSONL, one report per line. Only `row_id` and `source_text` are
required; any other keys ride along into the output.

```json
{"row_id": "EX0001", "source_text": "PATIENT/TEST INFORMATION: ...", "proc_name": "TTE (Complete)"}
```

Output is JSONL, one record per report, with the input's metadata, an
`extraction`, and per-stage `meta` (timing, parse errors, and the Stage 3
corrections object verbatim). Any value Stage 3 edited is additionally
stamped `corrected_by_validation: true` and keeps its pre-correction value,
so validation's effect on each field stays auditable.

The extraction is a list of **events**. A report usually describes one
study state and yields one event, but some describe several — rest and
stress in a stress echo, pre- and post-bypass in an intraoperative TEE —
and each becomes its own event with its own fields. Within an event, each
field holds a list of values; each value carries the verbatim span it came
from, whether that span was narrative or structured text, and whether it
was stated or inferred:

```json
{"severity": "moderate",
 "raw_text": "Moderate (2+) mitral regurgitation.",
 "source_type": "narrative",
 "source": "explicit",
 "extraction_reasoning": "..."}
```

Keeping the span and the explicit/inferred flag on every value is what makes
the output auditable, and what makes the paper's *explicit-only* analysis
possible: filter to `source == "explicit"` and you have the higher-precision
configuration without re-running anything.

---

## Ablation flags

The paper's prompt-component ablation (described in the supplementary
material) is exposed as three flags. Defaults reproduce the primary
configuration.

```bash
--demo-shots 1 --demo-file examples/demos.jsonl   # in-context demonstrations
--no-reasoning                                    # drop per-field reasoning
--no-evidence                                     # evidence span optional; keep uncited values
```

In the paper, requiring per-field reasoning slightly *hurt* accuracy, one
worked example helped while three did not, and requiring evidence spans made
no difference to accuracy (its value is auditability).

The demo file is JSONL with `source_text` and a verified `extraction` per
line. The bundled `examples/demos.jsonl` is synthetic — replace it with
demonstrations from your own data, drawn from a **different cohort** than
the one you are extracting.

---

## Wall-motion segments (optional)

`--segments` additionally extracts regional wall motion on the 17-segment
AHA/ASE model, one entry per segment (`{segment, motion, raw_text, ...}`),
with segment-aware instructions injected into all three prompts. **This
mode is off by default and was not enabled in the runs reported in the
paper** — the paper's wall-motion results come from the
`global_lv_wall_motion`, `rv_wall_motion`, and
`interventricular_septal_motion` fields — so segment-level accuracy is
unevaluated. It is included because the extraction schema and annotation
instrument define the 17-segment model; validate it on your own data
before relying on it.

---

## Checking your backend

Schema enforcement is the one thing that must work. This request distinguishes
real token-level constraint from best-effort JSON — the schema permits only a
string no model would produce on its own:

```python
schema = {"type": "object",
          "properties": {"animal": {"type": "string", "enum": ["xyzqwerty"]}},
          "required": ["animal"], "additionalProperties": False}
# prompt: "What is the fastest land animal? Reply with one word."
# response_format: {"type":"json_schema",
#                   "json_schema":{"name":"r","strict":True,"schema":schema}}
```

A constrained backend must return `{"animal": "xyzqwerty"}`. If it returns
`"cheetah"`, the schema is not being enforced and this pipeline will produce
parse failures on the real schema.

Reasoning models routed through a plain OpenAI-compatible API can also spend
their whole token budget thinking and return empty content; disable thinking
at the server or model level if you see `finish_reason=length` with no text.

---

## The 48 fields

Defined in `src/config.py`: 19 severity/graded fields (chamber sizes, valve
regurgitation and stenosis, ventricular function, wall thickness, pulmonary
hypertension, pericardial effusion), 10 categorical fields (diastolic grade,
wall-motion patterns, valve morphology, pericardial physiology), and 19
numeric measurements (ejection fraction, RVSP, TAPSE, valve gradients and
areas, diastolic indices).

Enumerated values are clinically granular rather than binary — mitral valve
morphology, for instance, takes any of nine values. Two sentinels apply to
every field: `not_mentioned` when the report is silent, and `conflict` when
the report contradicts itself.

---

## Citing this work

If you use this pipeline in published work, please cite the accompanying
paper — see `CITATION.cff`.

## License

Apache License 2.0 — see `LICENSE`. The license governs the code. It does not
cover the MIMIC-III data used in the paper, which remains subject to the
PhysioNet credentialed access agreement and is not distributed here.

## Scope

This repository is the extraction pipeline only. The evaluation and
adjudication tooling used to produce the paper's accuracy figures is not
included. Both files in `examples/` are synthetic and contain no patient
data.

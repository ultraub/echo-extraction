"""Prompt text for the three pipeline stages (extract, infer, validate).

These are the prompts used in the paper, reproduced verbatim in the
supplementary material. Each builder fills the report text (and, for later stages, the
JSON produced so far) into the template.
"""

import json
from typing import List, Optional

from .config import (
    CATEGORICAL_ENUM_VALUES,
    FUNCTION_FIELDS,
    REGURGITATION_FIELDS,
    STENOSIS_FIELDS,
    SEGMENT_MOTION_VALUES,
    WALL_SEGMENTS,
)

# =============================================================================
# Prompt templates
# =============================================================================

EXPLICIT_PROMPT = """You are an expert echocardiography report analyzer.

## TASK
Extract structured cardiac findings from this echocardiogram report.

## CORE PRINCIPLES
1. **Verbatim raw_text**: Copy text EXACTLY as it appears in the report
2. **Evidence required**: Every extracted value must have supporting raw_text
3. **Explicit only**: Extract only what is directly stated - do not infer
4. **Not mentioned — two equivalent representations**: If a finding is not mentioned in the report, use EITHER (A) an empty array, OR (B) a single entry with `value: "not_mentioned"`. Both are accepted by the grammar and produce the same downstream result. NEVER emit `value: "normal"` (or any other substantive value) when there is no supporting text — that is option (C), the forbidden one.
   - The output schema requires every severity/categorical field *name* to appear as a key in your output. It does NOT require the value to be a non-empty array. An empty array satisfies the grammar and means "this field has no extraction."
   - **(A) — EMPTY ARRAY (preferred when there is nothing to record):**
     `tv_morphology: []`  ✓
   - **(B) — `not_mentioned` ENTRY (acceptable when you want an audit trail):**
     `tv_morphology: [{{"value": "not_mentioned", "raw_text": "TV morphology not addressed in report.", "source_type": "narrative", "extraction_reasoning": "No specific morphological description of the tricuspid valve is provided."}}]`  ✓
     `value` must be the enum string `not_mentioned`. Never put the literal string `"not_mentioned"` in `raw_text`.
   - **(C) — FORBIDDEN (value contradicts reasoning):**
     `tv_morphology: [{{"value": "normal", "raw_text": "not_mentioned", "extraction_reasoning": "No specific morphological description of the tricuspid valve is provided. OMITTING per strict rule."}}]`  ✗ NEVER
   - If you find yourself writing "OMITTING", "omit per rule", "no mention of X", or any similar phrase in `extraction_reasoning`, STOP — that field belongs in option (A) or (B), not as a populated `normal` entry. Pick (A) by default; pick (B) when you want to record that you considered the field and judged it absent.
5. **Conflict**: Use "conflict" as the value when EITHER (a) the report contains contradictory information about the same finding, OR (b) the appropriate description is not available in the field's enum options (e.g., "repair" not listed for av_morphology). The raw_text should explain what was found.

## TERMINOLOGY MAPPING

**Severity terms** (map to standard values):
- "trivial", "physiologic", "trace" → trace
- "mild-to-moderate", "mild to moderate" → mild_moderate
- "1+" → mild | "2+" → moderate | "3+" → moderate_severe | "4+" → severe
- "significant" (unquantified) → abnormal_ungraded
- "borderline" → use the milder adjacent grade

**Synonyms** (extract to standard field):
- "insufficiency" = "regurgitation" (use regurgitation field)
- "MR/AR/TR/PR" = mitral/aortic/tricuspid/pulmonic regurgitation
- "AS/MS" = aortic/mitral stenosis
- "LVEF", "EF" = ejection_fraction
- "PHTN", "PAH" = pulm_htn

**Diastolic grade terminology**:
- "impaired relaxation", "delayed relaxation" → grade_I
- "pseudonormal" → grade_II
- "restrictive filling", "restrictive pattern" → grade_III

**Interventricular septal motion terminology**:
- "paradoxical motion", "paradoxical septal motion", "septal bounce" → dyskinesis
- "D-shaped", "septal flattening", "RV pressure overload" → systolic_flattening
- "RV volume overload" → diastolic_flattening

## MULTIPLE EVENTS
Reports may contain multiple distinct time points:
- Pre-bypass vs post-bypass (intraoperative)
- Rest vs stress vs recovery phases

Create separate events only when distinct findings exist. Do NOT create empty events for "unchanged" or "no change" comparisons.
DO NOT make new events for comparison studies that occurred on other dates. Events should reflect findings that are part of the same study, not comparisons to other studies.

## FIELD TYPES

**ALL severity and categorical fields below ALSO accept two shared values, in addition to the per-field list shown:**
- `not_mentioned` — finding is not addressed by the report. Equivalent to emitting an empty array `[]` for this field; choose whichever you find clearer (see CORE PRINCIPLE 4 for the full rule). Never substitute `normal`.
- `conflict` — when the report contradicts itself for this finding, or when no listed enum value fits.

These two are valid in EVERY severity/categorical enum (they are omitted from the per-field listings only to avoid repetition). Reminder: an empty array `[]` is ALSO a valid output for any field with no extraction — do NOT emit a `normal` entry merely because of empty values.

**Graded Fields - Chamber Size** (normal, mild, moderate, severe, abnormal; lv/rv also: small):
lv_cavity_size, rv_cavity_size, la_cavity_size, ra_cavity_size, aortic_root, ascending_aorta
**Note**: aortic_root and ascending_aorta are for DIAMETER only.
Do not use these fields for other aortic findings (atheroma, calcification, dissection, aneurysm morphology).

**Graded Fields - Function** (normal, abnormal_ungraded, mild, mild_moderate, moderate, moderate_severe, severe, hyperdynamic):
LV_systolic, RV_systolic

**Graded Fields - Diastolic** (normal, grade_I, grade_II, grade_III, indeterminate):
diastolic_grade

**Graded Fields - Regurgitation** (normal, trace, mild, mild_moderate, moderate, moderate_severe, severe, abnormal_ungraded):
MV_regurgitation, AV_regurgitation, TV_regurgitation, PV_regurgitation

**Graded Fields - Stenosis** (normal, mild, moderate, severe, abnormal_ungraded):
AV_stenosis, MV_stenosis, TV_stenosis, PV_stenosis

**Graded Fields - Other** (normal, mild, moderate, severe OR specific values):
lv_wall_thickness (normal, concentric_mild, concentric_moderate, concentric_severe, asymmetric_septal_hypertrophy)
pulm_htn (normal, mild, moderate, severe)
pericardial_effusion (normal, trace, small, moderate, large)

(Reminder: every graded field above also accepts `not_mentioned` and `conflict`. Use `not_mentioned` when the report does not address the finding.)

**Numeric Fields** (include qualifier: exact, range, approximate, gt, lt, gte, lte):
ejection_fraction, rvsp, tapse, e_e_prime_avg, la_volume_index, aortic_valve_peak_velocity, av_mean_gradient, av_area, mv_area, mv_mean_gradient, mv_e_a_ratio, ra_pressure_numeric, lv_posterior_wall_diameter, iv_septal_diameter, rv_fractional_area_change, e_prime_septal, e_prime_lateral, e_e_prime_septal, e_e_prime_lateral

**Categorical Fields** (each also accepts `not_mentioned` and `conflict` as described above):
- global_lv_wall_motion: normal | hypokinesis | akinesis | dyskinesis | aneurysm
- rv_wall_thickness: normal | abnormal
- rv_wall_motion: normal | abnormal
- interventricular_septal_motion: normal | diastolic_flattening | systolic_flattening | dyskinesis
- av_morphology: normal | sclerotic | thickened | bicuspid | prosthetic | endocarditis
- mv_morphology: normal | thickened | calcified | prolapse | repair | prosthetic | rheumatic | myxomatous | endocarditis
- tv_morphology: normal | thickened | prosthetic | repair | endocarditis
- pv_morphology: normal | prosthetic | endocarditis
- pericardial_physiology: normal | tamponade | constriction

## WALL MOTION EXTRACTION RULES

Use **global_lv_wall_motion** when ALL walls have the SAME motion pattern:
- "Normal wall motion" → global_lv_wall_motion: normal AND IVS motion: normal
- "Global hypokinesis" or "diffusely hypokinetic" → global_lv_wall_motion: hypokinesis
- "Diffuse akinesis" → global_lv_wall_motion: akinesis
- "No regional wall motion abnormalities" → global_lv_wall_motion: normal

## RV FUNCTION EXTRACTION RULES

RV wall motion findings should populate BOTH fields when applicable:
- **rv_wall_motion** (categorical): normal | abnormal
- **RV_systolic** (severity): normal | abnormal_ungraded | mild | mild_moderate | moderate | moderate_severe | severe | hyperdynamic

Examples:
- "Normal RV free wall motion" → rv_wall_motion: normal AND RV_systolic: normal
- "Normal RV chamber size and free wall motion" → rv_wall_motion: normal AND RV_systolic: normal
- "RV hypokinesis" or "reduced RV function" → rv_wall_motion: abnormal AND RV_systolic: [grade if specified, else abnormal_ungraded]
- "Mildly reduced RV systolic function" → rv_wall_motion: abnormal AND RV_systolic: mild

## LV FUNCTION EXTRACTION RULES

LV systolic function maps to the same severity scale as RV_systolic. Note that "hyperdynamic" is a valid value (not "normal").

## NUMERIC FIELDS — STRICT RULE
**Only populate a numeric field if an actual measured number appears in the report text.**
- Qualitative descriptions ("severe PAH", "mild dysfunction") give NO numeric measurement → output `[]`
- Do NOT estimate, infer, or substitute typical population values
- Do NOT use qualitative text as evidence for numeric fields (e.g. "normal systolic function" is NOT evidence for EF = 55%)
- Most reports will have empty arrays `[]` for most numeric fields — this is correct and expected

## OUTPUT
For each field, include:
- value (severity/numeric/categorical)
- raw_text (verbatim from report)
- source_type ("structured" or "narrative")
- extraction_reasoning (brief explanation of why this text supports this field)

## Report Text
{report_text}
"""

INFERENCE_PROMPT = """
You are an expert echocardiography report analyzer.

## TASK
Add inferences ONLY when the report provides positive, direct support.
When in doubt, OMIT the field. Missing a real finding is acceptable;
fabricating a finding is not. Prefer fewer high-quality inferences over
many speculative ones. Then detect any genuine conflicts in the data.

## Explicit Findings Already Extracted
{explicit_json}

## HARD RULES (NEVER VIOLATE)

**A. DO NOT infer numeric fields. These should only be explictly derived.**
Calculations may be made from existing numeric fields, but do NOT populate a numeric 
field unless a specific number is cited in the report text, corresponding to the field. 
Qualitative descriptions are NOT sufficient evidence for numeric values.

**B. SEVERITY / CATEGORICAL fields require positive supporting text.**
"Not mentioned" is NEVER evidence. Inference is allowed only when the report
contains explicit text that directly supports the inferred value.
Absence of an opposite finding is NOT sufficient to infer normality.
Always look for positive evidence supporting the specific severity or category.
Inferred fields must be logically consistent, and incontestable when supported
by the evidence. There must be no room for reasonable disagreement.

If no positive evidence exists for an inference, emit an EMPTY ARRAY `[]` for that field —
that is the correct way to say "no inference here." The output schema requires every field
name to appear as a key, but it does NOT require the value to be a non-empty array; `[]`
satisfies the grammar. Equivalently, you may emit a single `value: "not_mentioned"` entry if
you want an audit trail (every enum accepts it — see ## Graded Field Values and
## Categorical Values below). Do NOT emit `value: "normal"` (or any other substantive value)
with hedging reasoning like "OMITTING" or "no mention" — that is a self-contradiction and
will be discarded.

The evidence quote must reference the SAME anatomical structure as
the field. Do NOT cite a different structure that happens to sound
similar (e.g. "intact interatrial septum" is NOT evidence for
interventricular_septal_motion — different structures).

**C. Per-field DO NOT INFER list (high error risk, never infer):**
- GLS values
- lv_diastolic_grade from a single parameter alone (needs E/e' + LA size + multiple)
- lv_wall_thickness from low LVEF (dysfunction != hypertrophy)

## INFERENCE REQUIREMENTS (when an inference IS valid)
- Every inference needs an "evidence" array with 1-2 supporting quotes from the report.
- Each evidence item has: raw_text (verbatim from report), source_type ("structured" or "narrative").
- Use 2 evidence items when multiple report passages support the inference.
- Include inference_reasoning explaining how the evidence supports the inferred value.
- Do NOT repeat values already in explicit findings. 
- raw_text must be the actual report text supporting your inference, not generated text.

Example evidence array format:
  "evidence": [ item1, item2 ]
  where each item has "raw_text" and "source_type" fields

## CONFLICT DETECTION
Flag genuine contradictions within the same event:
- Numeric values contradicting qualitative descriptions (e.g., EF 55% but "severely depressed")
- Conflicting severity grades for the same finding
- Physiologically implausible combinations
- Conflicts can be detected within explicit findings or between explicit findings. 
- Always evaluate conflicts across all data in the event.

NOT conflicts: changes between events (pre/post), expected disease patterns

## Shared values (apply to EVERY graded and categorical field below)
- `not_mentioned` — finding is not addressed by the report. Equivalent to emitting an empty array `[]` for that field (see HARD RULE B above for the full rule). Never substitute `normal`.
- `conflict` — for contradictory information, or when no listed enum value fits.

Reminder: an empty array `[]` is also a valid output for any field with no inference. Do NOT emit a `normal` entry merely because the grammar requires the field name to be present.

## Graded Field Values (each ALSO accepts `not_mentioned` and `conflict`)
- Chamber Size: normal, mild, moderate, severe, abnormal (+ small for lv/rv_cavity_size only)
- Function (LV/RV_systolic): normal, abnormal_ungraded, mild, mild_moderate, moderate, moderate_severe, severe, hyperdynamic
- Diastolic (diastolic_grade): normal, grade_I, grade_II, grade_III, indeterminate
- Regurgitation: normal, trace, mild, mild_moderate, moderate, moderate_severe, severe, abnormal_ungraded
- Stenosis: normal, mild, moderate, severe, abnormal_ungraded

## Categorical Values (each ALSO accepts `not_mentioned` and `conflict`)
- global_lv_wall_motion: normal | hypokinesis | akinesis | dyskinesis | aneurysm
- lv_wall_thickness: normal | concentric_mild | concentric_moderate | concentric_severe | asymmetric_septal_hypertrophy
- pericardial_effusion: normal | trace | small | moderate | large
- pulm_htn: normal | mild | moderate | severe
- rv_wall_thickness: normal | abnormal
- rv_wall_motion: normal | abnormal
- interventricular_septal_motion: normal | diastolic_flattening | systolic_flattening | dyskinesis
- av_morphology: normal | sclerotic | thickened | bicuspid | prosthetic | endocarditis
- mv_morphology: normal | thickened | calcified | prolapse | repair | prosthetic | rheumatic | myxomatous | endocarditis
- tv_morphology: normal | thickened | prosthetic | repair | endocarditis
- pv_morphology: normal | prosthetic | endocarditis
- pericardial_physiology: normal | tamponade | constriction

## Report Text
{report_text}
"""

VALIDATION_PROMPT = """
You are validating echocardiography extraction results and outputting ONLY the corrections needed.

## TASK
Review the extracted findings against the original report. Output ONLY what needs to be corrected or removed - do NOT re-extract everything.

## Original Report
{report_text}

## Extracted Findings to Validate
{formatted_extraction}

## VALIDATION CHECKS

**IMPORTANT: For EXPLICIT findings, prefer KEEPING them — only REMOVE if completely fabricated or generated 
from findings not mentioned in the report (e.g. inferring normal from absence of abnormal).

 For INFERRED findings, apply strict checks below; speculative inferences should be REMOVED.**

1. **Explicit-finding Evidence Check** (lenient — applies to source="explicit" only):
   - If raw_text is paraphrased but the finding IS supported by the report → KEEP (optionally correct the quote)
   - If raw_text references text that exists in some form in the report → KEEP
   - Only REMOVE if the finding is not supported in the report or drawn from lack of mention (e.g. "not mentioned therefore normal")

2. **Value Schema Compliance**:
   - For fields with [valid: ...] shown, the extracted value MUST be one of the valid options
   - If value is not in the valid list, add a CORRECTION with the closest valid value
   - Map synonyms to valid values (e.g., "trivial" → "trace", "none" → "normal")

3. **Value Appropriateness**:
   - Verify the extracted value is reasonably supported by the report
   - If value is close but not exact, add a CORRECTION
   - Only REMOVE if value is completely wrong or fabricated

4. **Inference Validity Check (STRICT — applies to source="inferred" only)**:
   - REMOVE any inference whose inference_reasoning contains qualifiers semantically similar to:
     "cannot infer", "not quantified", "no data", "without specific data",
     "this is an assumption", "not explicitly", "inferred as normal".
   - REMOVE any NUMERIC inference whose evidence cites only qualitative
     descriptions (e.g. "normal function", "structurally normal valve",
     "chamber size normal") rather than an actual number, range, or
     measurement from the report. Numeric inferences are valid only when
     the report contains a real measurement supporting THIS specific field.
   - REMOVE any SEVERITY or CATEGORICAL inference whose only evidence is
     the absence of an opposite finding (e.g. "not mentioned therefore
     normal", or chamber size cited to infer wall thickness/motion).
   - REMOVE inferences whose evidence quote references a DIFFERENT anatomical
     structure than the field name. Common error: "intact interatrial septum"
     cited for interventricular_septal_motion (different structures).
   - For GLS%: always REMOVE if inferred.

5. **Anatomical Alignment**:
   - Aortic valve fields (av_*) must reference aortic/AV content
   - Mitral valve fields (mv_*) must reference mitral/MV content
   - If misaligned, add a REMOVAL (the field is in wrong location)

6. **Conflict Detection**: Flag genuine contradictions
   - KEEP conflicts identifying numeric values contradicting qualitative descriptions
   - KEEP conflicts identifying significant conflicting severity grades for the same finding
   - KEEP conflicts identifying physiologically implausible combinations
   - NOT conflicts: changes between events, expected disease patterns
   - Conflicts should not be used to identify fields flagged for correction/removal

## OUTPUT FORMAT

Output corrections and removals for fields that need attention. Do NOT list fields that are correct.

- **corrections**: Fields needing fix (provide corrected_raw_text and/or corrected_value)
- **removals**: Fields to remove entirely (no valid evidence)
- **event_removals**: Entire events to remove (empty or invalid)
- **conflicts**: Detected inconsistencies (including EF-severity mismatches)
- **validation_passed**: true if no corrections/removals needed, false otherwise

If extraction is perfect, output empty arrays and validation_passed=true.
"""


# =============================================================================
# Optional 17-segment wall-motion instructions (--segments).
# NOT used in the paper's runs; included because the extraction schema and
# annotation instrument define the 17-segment model. Unevaluated.
# =============================================================================

WALL_MOTION_SEGMENT_INSTRUCTIONS = """
## WALL MOTION SEGMENTS (17-Segment Model)

Extract segment-level wall motion abnormalities using the 17-segment model.

**Valid Segments** (use ONLY these exact names):
basal_anterior, mid_anterior, apical_anterior,
basal_anteroseptal, mid_anteroseptal, apical_septal,
basal_inferoseptal, mid_inferoseptal,
basal_inferior, mid_inferior, apical_inferior,
basal_inferolateral, mid_inferolateral, apical_lateral,
basal_anterolateral, mid_anterolateral,
apex

**Valid Motion Values** (use ONLY these exact values):
- normal
- hypokinesis (includes: hypo, hypokinetic, reduced)
- akinesis (includes: akinetic, absent motion)
- dyskinesis (includes: dyskinetic, paradoxical)
- aneurysm (includes: aneurysmal)

CRITICAL: The "motion" field must be one of: normal, hypokinesis, akinesis, dyskinesis, aneurysm
Do NOT put segment names in the motion field.

**Segment Name Mapping** (IMPORTANT - use exact segment names):

| Report Terms | Segment Name |
|--------------|--------------|
| "anterior apex", "apical anterior" | apical_anterior |
| "lateral apex", "apical lateral", "apical_lateral" | apical_lateral |
| "inferior apex", "apical inferior" | apical_inferior |
| "septal apex", "apical septal", "anteroseptal apex" | apical_septal |
| "apex" (alone, true apex tip) | apex |
| "posterior" segments | inferolateral (e.g., "basal posterior" → basal_inferolateral) |

CRITICAL: "apical_lateral" and "apex" are DIFFERENT segments. Do not conflate them.
- "lateral apex" or "apical lateral" → apical_lateral (NOT apex)
- "apex" alone (the true apex tip) → apex

**Parsing Rules for Implicit Segment Mentions**:

1. **"Remaining segments" phrases**: When report states "remaining segments are [abnormality]" or "other segments are [abnormality]", apply that abnormality to ALL 17 segments not explicitly listed elsewhere.

2. **Generic "apex" expansion**: When "the apex" is mentioned ALONE (without individual apical segments listed), expand to: apical_anterior, apical_septal, apical_inferior, apical_lateral. If "apex" appears ALONGSIDE other apical segments (e.g., "apical inferior - akinetic; apex - dyskinetic"), keep apex as distinct segment 17.

3. **Territory/wall name expansion**:
   - "lateral wall" → basal_anterolateral, mid_anterolateral, basal_inferolateral, mid_inferolateral, apical_lateral
   - "anterior wall" → basal_anterior, mid_anterior, apical_anterior
   - "inferior wall" → basal_inferior, mid_inferior, apical_inferior
   - "septal wall" → basal_anteroseptal, mid_anteroseptal, basal_inferoseptal, mid_inferoseptal, apical_septal

**Extraction Rules**:
- Extract each segment mentioned with its motion abnormality
- Apply parsing rules above to expand implicit segment references
- Use verbatim raw_text for each segment extraction

**Output Format** (in wall_motion_segments array):
- segment: one of the valid segment names listed above
- motion: one of: normal, hypokinesis, akinesis, dyskinesis, aneurysm
- raw_text: verbatim text mentioning this segment
- source_type: "structured" or "narrative"
"""

WALL_MOTION_SEGMENT_INFERENCE_INSTRUCTIONS = """
## WALL MOTION SEGMENT INFERENCES

You may infer segment motion from:
- Regional descriptions mentioning coronary territories (e.g., "LAD territory akinesis")
- Anatomical groupings (e.g., "inferior wall akinetic")
- "Remaining segments" phrases (e.g., "the remaining segments are hypokinetic")
- Generic apex descriptions (e.g., "extensive akinesis of the apex")

**Expansion Rules** (apply when inferring from grouped descriptions):

1. **"Remaining segments are [X]"**: Infer abnormality X for ALL 17-model segments not explicitly mentioned with a different abnormality.

2. **Generic "apex" (alone)**: When "the apex" is described without individual apical segments listed nearby, expand to apical_anterior, apical_septal, apical_inferior, apical_lateral.

3. **Wall/territory names**: Expand to constituent segments:
   - "lateral wall" → anterolateral + inferolateral segments + apical_lateral
   - "anterior wall" → anterior segments + apical_anterior
   - "inferior wall" → inferior segments + apical_inferior
   - "septal wall" → anteroseptal + inferoseptal segments + apical_septal

**Coronary Territory Mapping**:
- LAD: basal/mid anterior, basal/mid anteroseptal, apical_anterior, apical_septal, apex
- RCA: basal/mid inferior, basal/mid inferoseptal, apical_inferior
- LCx: basal/mid anterolateral, basal/mid inferolateral, apical_lateral

**DO NOT infer** segments without clear anatomical, territorial, or "remaining" reference.

Each inferred segment needs an evidence array with 1-2 supporting quotes and inference_reasoning.
"""


# =============================================================================
# Validation Helpers
# =============================================================================

# Valid values by field type


# =============================================================================
# Helpers for rendering the extraction inside the validation prompt
# =============================================================================

REGURGITATION_VALUES = ["normal", "trace", "mild", "mild_moderate", "moderate", "moderate_severe", "severe", "abnormal_ungraded"]
STENOSIS_VALUES = ["normal", "mild", "moderate", "severe", "abnormal_ungraded"]
LV_RV_CHAMBER_VALUES = ["normal", "mild", "moderate", "severe", "small", "abnormal"]
LA_RA_CHAMBER_VALUES = ["normal", "mild", "moderate", "severe", "abnormal"]
FUNCTION_VALUES = ["normal", "abnormal_ungraded", "mild", "mild_moderate", "moderate", "moderate_severe", "severe", "hyperdynamic"]
LV_WALL_THICKNESS_VALUES = ["normal", "concentric_mild", "concentric_moderate", "concentric_severe", "asymmetric_septal_hypertrophy"]
PULM_HTN_VALUES = ["normal", "mild", "moderate", "severe"]
PERICARDIAL_EFFUSION_VALUES = ["normal", "trace", "small", "moderate", "large"]
AORTIC_SIZE_VALUES = ["normal", "mild", "moderate", "severe", "abnormal"]


def get_valid_values_for_field(field_name: str) -> Optional[List[str]]:
    """Get valid values for a severity/categorical field.

    Returns None for numeric fields or unknown fields.
    """
    # Categorical fields from config
    if field_name in CATEGORICAL_ENUM_VALUES:
        return CATEGORICAL_ENUM_VALUES[field_name]

    # Regurgitation fields
    if field_name in REGURGITATION_FIELDS or field_name.lower().replace("_", "").endswith("regurgitation"):
        return REGURGITATION_VALUES

    # Stenosis fields
    if field_name in STENOSIS_FIELDS or field_name.lower().replace("_", "").endswith("stenosis"):
        return STENOSIS_VALUES

    # Function fields
    if field_name in FUNCTION_FIELDS:
        return FUNCTION_VALUES

    # Chamber size fields
    if field_name in ["lv_cavity_size", "rv_cavity_size"]:
        return LV_RV_CHAMBER_VALUES
    if field_name in ["la_cavity_size", "ra_cavity_size"]:
        return LA_RA_CHAMBER_VALUES

    # Aortic size fields
    if field_name in ["aortic_root", "ascending_aorta"]:
        return AORTIC_SIZE_VALUES

    # Other graded fields
    if field_name == "lv_wall_thickness":
        return LV_WALL_THICKNESS_VALUES
    if field_name == "pulm_htn":
        return PULM_HTN_VALUES
    if field_name == "pericardial_effusion":
        return PERICARDIAL_EFFUSION_VALUES

    # Numeric fields return None
    return None


def format_extraction_with_valid_values(extraction_json: str) -> str:
    """Format extraction JSON with inline valid values for severity/categorical fields.

    Args:
        extraction_json: JSON string of extraction results

    Returns:
        Formatted string with valid values shown inline for each field
    """
    try:
        data = json.loads(extraction_json)
    except json.JSONDecodeError:
        return extraction_json  # Return as-is if not valid JSON

    lines = []

    # Process events
    events = data.get("events", [data]) if "events" in data else [data]

    for event_idx, event in enumerate(events):
        if len(events) > 1:
            event_id = event.get("event_id", f"event_{event_idx + 1}")
            lines.append(f"\n### Event: {event_id}")

        # Process each field in the event
        for field_name, field_data in event.items():
            if field_name in ["event_id", "event_type", "event_date", "event_context", "wall_motion_segments"]:
                continue

            if not isinstance(field_data, dict):
                continue

            # Get valid values for this field
            valid_values = get_valid_values_for_field(field_name)

            # Format field with valid values inline
            if valid_values:
                valid_str = ", ".join(valid_values)
                lines.append(f"\n**{field_name}** [valid: {valid_str}]")
            else:
                lines.append(f"\n**{field_name}**")

            # Show the extracted value and raw_text
            value = field_data.get("value")
            raw_text = field_data.get("raw_text", "")
            source = field_data.get("source", "explicit")

            lines.append(f"  value: {json.dumps(value)}")
            if raw_text:
                lines.append(f"  raw_text: {json.dumps(raw_text)}")
            if source == "inferred":
                lines.append(f"  source: inferred")

        # Handle wall_motion_segments separately (don't repeat valid values for each)
        segments = event.get("wall_motion_segments", [])
        if segments:
            lines.append(f"\n**wall_motion_segments** ({len(segments)} segments)")
            for seg in segments:
                seg_name = seg.get("segment", "unknown")
                motion = seg.get("motion", "unknown")
                raw_text = seg.get("raw_text", "")
                lines.append(f"  - {seg_name}: {motion}")
                if raw_text:
                    lines.append(f"    raw_text: {json.dumps(raw_text)}")

    return "\n".join(lines)


# =============================================================================
# Builders
# =============================================================================

def build_explicit_prompt(report_text: str, use_segments: bool = False) -> str:
    """Stage 1: extract findings the report states outright."""
    prompt = EXPLICIT_PROMPT
    if use_segments:
        prompt = prompt.replace(
            "## OUTPUT",
            WALL_MOTION_SEGMENT_INSTRUCTIONS + "\n## OUTPUT",
        )
    return prompt.format(report_text=report_text)


def build_inference_prompt(report_text: str, explicit_json: str,
                           use_segments: bool = False) -> str:
    """Stage 2: infer findings from the report plus the Stage 1 output."""
    prompt = INFERENCE_PROMPT
    if use_segments:
        prompt = prompt.replace(
            "## CONFLICT DETECTION",
            WALL_MOTION_SEGMENT_INFERENCE_INSTRUCTIONS + "\n## CONFLICT DETECTION",
        )
    return prompt.format(
        report_text=report_text,
        explicit_json=explicit_json,
    )


def build_validation_prompt(report_text: str, extraction_json: str,
                            use_segments: bool = False) -> str:
    """Stage 3: check the merged Stage 1+2 output against the report.

    The extraction is rendered with each field's permitted values inline so
    the model can correct a value without consulting the schema.
    """
    prompt = VALIDATION_PROMPT
    formatted_extraction = format_extraction_with_valid_values(extraction_json)

    if use_segments:
        segments_str = ", ".join(WALL_SEGMENTS)
        motion_str = ", ".join(SEGMENT_MOTION_VALUES)
        wall_motion_reference = f"""
## Wall Motion Reference
**Valid segments**: {segments_str}
**Valid motion values**: {motion_str}
"""
        prompt = prompt.replace(
            "## VALIDATION CHECKS",
            wall_motion_reference + "\n## VALIDATION CHECKS",
        )
        segment_validation = """
7. **Global vs Segment Wall Motion (STRICT)**:
   - global_lv_wall_motion and wall_motion_segments are MUTUALLY EXCLUSIVE
   - If BOTH are present, add a REMOVAL for the less appropriate one:
     - If report describes UNIFORM wall motion (all walls same) → REMOVE wall_motion_segments, KEEP global_lv_wall_motion
     - If report describes REGIONAL abnormalities (specific walls/segments) → REMOVE global_lv_wall_motion, KEEP wall_motion_segments
   - Never keep both - one MUST be removed

8. **Wall Motion Segment Validation**:
   - Each segment must be one of the valid segments listed above
   - Motion values must be one of: normal, hypokinesis, akinesis, dyskinesis, aneurysm
   - raw_text must verbatim support the segment and motion extracted
   - For segment corrections/removals, use field_type="wall_motion_segments" and field_name=segment_name
   - If segment name doesn't match standard nomenclature, add a CORRECTION with corrected_value
   - If no evidence supports a segment, add a REMOVAL
"""
        prompt = prompt.replace(
            "## OUTPUT FORMAT",
            segment_validation + "\n## OUTPUT FORMAT",
        )

    return prompt.format(
        report_text=report_text,
        formatted_extraction=formatted_extraction,
    )

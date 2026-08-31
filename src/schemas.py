"""JSON schemas that constrain decoding at each stage.

Every stage's request carries one of these as
`response_format={"type": "json_schema", "strict": true, ...}`, so the
model can only emit values the schema permits. Per-field enums come from
src/config.py.

Output shape (Stage 1, Stage 3 input, and the final result):

    {"events": [
        {"event_id": "1", "event_type": "...", "event_label": "...",
         "date": "...", "is_baseline": true, "temporal_sequence": 1,
         "severity_fields":    {"<field>": [ {severity, raw_text, source_type, extraction_reasoning} ]},
         "numeric_fields":     {"<field>": [ {value, value_min, value_max, qualifier, unit, raw_text, ...} ]},
         "categorical_fields": {"<field>": [ {value, raw_text, source_type, ...} ]}}
    ]}

A report is a list of *events* because one report can describe more than
one study state — rest and stress in a stress echo, pre- and post-bypass in
an intraoperative TEE. Most reports produce a single event.

Two flags exist for the prompt-component ablation in the paper:
  include_reasoning=False drops the per-field `*_reasoning` property.
  include_raw_text=False  makes the evidence span optional instead of required.

use_segments=True adds 17-segment wall-motion arrays (--segments). Off by
default; the paper's runs did not enable it.
"""

from .config import (
    ESSENTIAL_SEVERITY_FIELDS,
    ESSENTIAL_NUMERIC_FIELDS,
    ESSENTIAL_CATEGORICAL_FIELDS,
    SEVERITY_ENUM_VALUES,
    SOURCE_TYPE_VALUES,
    CATEGORICAL_ENUM_VALUES,
    SEGMENT_MOTION_VALUES,
    WALL_SEGMENTS,
)

_REASONING = {"type": "string", "maxLength": 300}
_NUMERIC_VALUE_PROPS = {
    "value": {"type": "number"},
    "value_min": {"type": ["number", "null"]},
    "value_max": {"type": ["number", "null"]},
    "qualifier": {
        "type": "string",
        "enum": ["exact", "range", "gt", "lt", "gte", "lte", "approximate", "conflict"],
    },
    "unit": {"type": "string"},
}
_NUMERIC_VALUE_REQUIRED = ["value", "value_min", "value_max", "qualifier", "unit"]


def _obj(props, required):
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def _strip_required(schema, key):
    """Remove `key` from every 'required' list in a schema (in place)."""
    if isinstance(schema, dict):
        if isinstance(schema.get("required"), list):
            schema["required"] = [k for k in schema["required"] if k != key]
        for v in schema.values():
            _strip_required(v, key)
    elif isinstance(schema, list):
        for item in schema:
            _strip_required(item, key)
    return schema


# =============================================================================
# Stage 1 — explicit extraction
# =============================================================================

def build_explicit_schema(include_reasoning: bool = True,
                          include_raw_text: bool = True,
                          use_segments: bool = False) -> dict:
    """Each extracted value cites a verbatim span (`raw_text`), says whether
    that span was structured or narrative text, and (by default) carries a
    short reasoning string. Property order is deliberate: the model writes
    the span and reasoning before it commits to the value."""

    def severity_item(field):
        props = {
            "raw_text": {"type": "string"},
            "source_type": {"type": "string", "enum": SOURCE_TYPE_VALUES},
            "severity": {"type": "string", "enum": SEVERITY_ENUM_VALUES[field]},
        }
        required = ["severity", "raw_text", "source_type"]
        if include_reasoning:
            props["extraction_reasoning"] = _REASONING
            required.append("extraction_reasoning")
        return _obj(props, required)

    numeric_props = {
        "raw_text": {"type": "string"},
        "source_type": {"type": "string", "enum": SOURCE_TYPE_VALUES},
        **_NUMERIC_VALUE_PROPS,
    }
    numeric_required = _NUMERIC_VALUE_REQUIRED + ["raw_text", "source_type"]
    if include_reasoning:
        numeric_props["extraction_reasoning"] = _REASONING
        numeric_required.append("extraction_reasoning")
    numeric_item = _obj(numeric_props, numeric_required)

    def categorical_item(field):
        props = {
            "raw_text": {"type": "string"},
            "source_type": {"type": "string", "enum": SOURCE_TYPE_VALUES},
            "value": {"type": "string", "enum": CATEGORICAL_ENUM_VALUES[field]},
        }
        required = ["value", "raw_text", "source_type"]
        if include_reasoning:
            props["extraction_reasoning"] = _REASONING
            required.append("extraction_reasoning")
        return _obj(props, required)

    event_props = {
        "event_id": {"type": "string"},
        "event_type": {"type": "string"},
        "event_label": {"type": "string"},
        "date": {"type": "string"},
        "is_baseline": {"type": "boolean"},
        "temporal_sequence": {"type": "integer"},
        "severity_fields": _obj(
            {f: {"type": "array", "items": severity_item(f)} for f in ESSENTIAL_SEVERITY_FIELDS},
            list(ESSENTIAL_SEVERITY_FIELDS)),
        "numeric_fields": _obj(
            {f: {"type": "array", "items": numeric_item} for f in ESSENTIAL_NUMERIC_FIELDS},
            list(ESSENTIAL_NUMERIC_FIELDS)),
        "categorical_fields": _obj(
            {f: {"type": "array", "items": categorical_item(f)} for f in ESSENTIAL_CATEGORICAL_FIELDS},
            list(ESSENTIAL_CATEGORICAL_FIELDS)),
    }
    event_required = ["event_id", "event_type", "event_label", "date", "is_baseline",
                      "temporal_sequence", "severity_fields", "numeric_fields", "categorical_fields"]
    if use_segments:
        segment_item = _obj(
            {
                "segment": {"type": "string", "enum": WALL_SEGMENTS},
                "motion": {"type": "string", "enum": SEGMENT_MOTION_VALUES},
                "raw_text": {"type": "string"},
                "source_type": {"type": "string", "enum": SOURCE_TYPE_VALUES},
            },
            ["segment", "motion", "raw_text", "source_type"],
        )
        event_props["wall_motion_segments"] = {"type": "array", "items": segment_item}
        event_required.append("wall_motion_segments")

    event = _obj(event_props, event_required)
    schema = _obj({"events": {"type": "array", "items": event}}, ["events"])
    if not include_raw_text:
        _strip_required(schema, "raw_text")
    return schema


# =============================================================================
# Stage 2 — inference
# =============================================================================

def build_inference_schema(include_reasoning: bool = True,
                           include_raw_text: bool = True,
                           use_segments: bool = False) -> dict:
    """Inferred values cite one or two supporting spans in an `evidence`
    array instead of a single `raw_text`. Events are keyed by the
    `event_id` Stage 1 assigned; the merge step joins on it."""

    evidence_item = _obj(
        {"raw_text": {"type": "string"},
         "source_type": {"type": "string", "enum": SOURCE_TYPE_VALUES}},
        ["raw_text", "source_type"],
    )
    evidence = {"type": "array", "items": evidence_item, "minItems": 1, "maxItems": 2}

    def severity_item(field):
        props = {"evidence": evidence,
                 "severity": {"type": "string", "enum": SEVERITY_ENUM_VALUES[field]}}
        required = ["severity", "evidence"]
        if include_reasoning:
            props["inference_reasoning"] = _REASONING
            required.append("inference_reasoning")
        return _obj(props, required)

    numeric_props = {"evidence": evidence, **_NUMERIC_VALUE_PROPS}
    numeric_required = _NUMERIC_VALUE_REQUIRED + ["evidence"]
    if include_reasoning:
        numeric_props["inference_reasoning"] = _REASONING
        numeric_required.append("inference_reasoning")
    numeric_item = _obj(numeric_props, numeric_required)

    def categorical_item(field):
        props = {"evidence": evidence,
                 "value": {"type": "string", "enum": CATEGORICAL_ENUM_VALUES[field]}}
        required = ["value", "evidence"]
        if include_reasoning:
            props["inference_reasoning"] = _REASONING
            required.append("inference_reasoning")
        return _obj(props, required)

    conflict = _obj(
        {"event_id": {"type": "string"}, "field": {"type": "string"},
         "conflict_type": {"type": "string"}, "description": {"type": "string"}},
        ["event_id", "field", "conflict_type", "description"],
    )

    event_props = {
        "event_id": {"type": "string"},
        "inferred_severity": _obj(
            {f: {"type": "array", "items": severity_item(f)} for f in ESSENTIAL_SEVERITY_FIELDS},
            list(ESSENTIAL_SEVERITY_FIELDS)),
        "inferred_numeric": _obj(
            {f: {"type": "array", "items": numeric_item} for f in ESSENTIAL_NUMERIC_FIELDS},
            list(ESSENTIAL_NUMERIC_FIELDS)),
        "inferred_categorical": _obj(
            {f: {"type": "array", "items": categorical_item(f)} for f in ESSENTIAL_CATEGORICAL_FIELDS},
            list(ESSENTIAL_CATEGORICAL_FIELDS)),
    }
    event_required = ["event_id", "inferred_severity", "inferred_numeric", "inferred_categorical"]
    if use_segments:
        seg_props = {
            "segment": {"type": "string", "enum": WALL_SEGMENTS},
            "motion": {"type": "string", "enum": SEGMENT_MOTION_VALUES},
            "evidence": evidence,
        }
        seg_required = ["segment", "motion", "evidence"]
        if include_reasoning:
            seg_props["inference_reasoning"] = _REASONING
            seg_required.append("inference_reasoning")
        event_props["inferred_wall_motion_segments"] = {
            "type": "array", "items": _obj(seg_props, seg_required)}
        event_required.append("inferred_wall_motion_segments")

    event = _obj(event_props, event_required)
    schema = _obj(
        {"events": {"type": "array", "items": event},
         "conflicts": {"type": "array", "items": conflict}},
        ["events", "conflicts"],
    )
    if not include_raw_text:
        _strip_required(schema, "raw_text")
    return schema


# =============================================================================
# Stage 3 — validation
# =============================================================================

def build_corrections_schema() -> dict:
    """Stage 3 does not re-emit the extraction; it returns edits to apply to
    it. `corrected_value` is a free string (a correction may target a
    severity, categorical, or numeric field), which is why src/merge.py
    applies a correction only when its value is legal for the target field."""
    correction = _obj(
        {
            "event_id": {"type": "string"},
            "field_type": {"type": "string"},
            "field_name": {"type": "string"},
            "correction_type": {"type": "string",
                                "enum": ["fix_raw_text", "fix_value", "fix_both"]},
            "corrected_raw_text": {"type": "string"},
            "corrected_value": {"type": "string"},
            "reason": {"type": "string"},
        },
        ["event_id", "field_type", "field_name", "correction_type",
         "corrected_raw_text", "corrected_value", "reason"],
    )
    removal = _obj(
        {"event_id": {"type": "string"}, "field_type": {"type": "string"},
         "field_name": {"type": "string"}, "reason": {"type": "string"}},
        ["event_id", "field_type", "field_name", "reason"],
    )
    event_removal = _obj(
        {"event_id": {"type": "string"}, "reason": {"type": "string"}},
        ["event_id", "reason"],
    )
    conflict = _obj(
        {"event_id": {"type": "string"}, "field": {"type": "string"},
         "conflict_type": {"type": "string"}, "description": {"type": "string"}},
        ["event_id", "field", "conflict_type", "description"],
    )
    return _obj(
        {
            "corrections": {"type": "array", "items": correction},
            "removals": {"type": "array", "items": removal},
            "event_removals": {"type": "array", "items": event_removal},
            "conflicts": {"type": "array", "items": conflict},
            "validation_passed": {"type": "boolean"},
        },
        ["corrections", "removals", "event_removals", "conflicts", "validation_passed"],
    )

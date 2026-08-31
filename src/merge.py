"""Deterministic (non-LLM) steps between and after the model calls.

  merge_results(explicit, inferred)
      Joins Stage 2's inferred values onto Stage 1's events by event_id and
      tags every value with source="explicit" or "inferred".

  clean(result)
      Applied after the merge and again after Stage 3 corrections:
        - merges events that share (date, event_type)
        - drops "not_mentioned" placeholders
        - keeps at most one explicit value per field
        - validates values and drops duplicates
        - validates and dedupes wall-motion segments (--segments mode only)

  apply_corrections(result, corrections)
      Applies the edits Stage 3 returned. `corrected_value` is a free string
      (the corrections schema serves all three field types), so a correction
      is applied only if its value is already legal for the target field —
      otherwise it is rejected and the extracted value stands; the attempt
      remains visible in the run metadata. Every item that is changed is
      stamped with corrected_by_validation=true and keeps its pre-correction
      value (value_before_correction / raw_text_before_correction), so
      corrected values stay distinguishable from values the model extracted
      directly.

  remove_empty_evidence(result)
      Drops any value that cites no span. Skipped in the evidence ablation.
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Optional

from .config import (SEVERITY_ENUM_VALUES, CATEGORICAL_ENUM_VALUES,
                     SEGMENT_MOTION_VALUES, WALL_SEGMENTS)

FIELD_TYPES = ("severity_fields", "numeric_fields", "categorical_fields")


# =============================================================================
# Merge
# =============================================================================

def merge_results(explicit: dict, inferred: Optional[dict]) -> dict:
    """Combine Stage 1 and Stage 2 output into one event list."""
    if not explicit:
        return {"events": [], "conflicts": []}

    result = deepcopy(explicit)
    for event in result.get("events", []):
        for ft in FIELD_TYPES:
            for items in event.get(ft, {}).values():
                for occ in items if isinstance(items, list) else []:
                    if isinstance(occ, dict):
                        occ["source"] = "explicit"
        for seg in event.get("wall_motion_segments", []) or []:
            if isinstance(seg, dict):
                seg["source"] = "explicit"

    if inferred:
        by_id = {e["event_id"]: e for e in result.get("events", [])}
        for inf_event in inferred.get("events", []):
            target = by_id.get(inf_event.get("event_id"))
            if target is None:
                continue
            for inf_ft, ft in (("inferred_severity", "severity_fields"),
                               ("inferred_numeric", "numeric_fields"),
                               ("inferred_categorical", "categorical_fields")):
                for field, items in inf_event.get(inf_ft, {}).items():
                    if not isinstance(items, list):
                        continue
                    for occ in items:
                        if isinstance(occ, dict):
                            occ["source"] = "inferred"
                    target.setdefault(ft, {}).setdefault(field, []).extend(items)
            for seg in inf_event.get("inferred_wall_motion_segments", []) or []:
                if isinstance(seg, dict):
                    seg["source"] = "inferred"
                target.setdefault("wall_motion_segments", []).append(seg)
        result["conflicts"] = inferred.get("conflicts", [])
    else:
        result["conflicts"] = []
    return result


# =============================================================================
# Clean
# =============================================================================

def _span(occ: dict) -> str:
    if "raw_text" in occ:
        return (occ.get("raw_text") or "").strip()[:100]
    ev = occ.get("evidence")
    if isinstance(ev, list) and ev:
        return (ev[0].get("raw_text") or "").strip()[:100]
    return ""


def _dedupe(items: list, field_type: str) -> list:
    """Collapse values with the same (value, span); prefer explicit over inferred."""
    seen = OrderedDict()
    for occ in items:
        if not isinstance(occ, dict):
            continue
        if field_type == "severity_fields":
            key = (str(occ.get("severity", "")), _span(occ))
        elif field_type == "numeric_fields":
            key = (str(occ.get("value")), str(occ.get("value_min")), str(occ.get("value_max")),
                   occ.get("qualifier", "exact"), _span(occ))
        else:
            key = (str(occ.get("value", "")), _span(occ))
        if key not in seen or (occ.get("source") == "explicit"
                               and seen[key].get("source") == "inferred"):
            seen[key] = occ
    return list(seen.values())


def _valid(occ: dict, field_type: str, field: str) -> bool:
    if field_type == "severity_fields":
        allowed = SEVERITY_ENUM_VALUES.get(field)
        return allowed is None or occ.get("severity") in allowed
    if field_type == "numeric_fields":
        return occ.get("value") not in (None, 0)
    allowed = CATEGORICAL_ENUM_VALUES.get(field)
    return allowed is None or occ.get("value") in allowed


def _merge_duplicate_events(events: list) -> list:
    """Events with the same (date, event_type) are the same study described
    in different paragraphs; union their fields."""
    groups, keyless = OrderedDict(), []
    for ev in events:
        key = (ev.get("date", ""), ev.get("event_type", ""))
        (groups.setdefault(key, []) if any(key) else keyless).append(ev)
    out = []
    for group in groups.values():
        merged = deepcopy(group[0])
        for dup in group[1:]:
            for ft in FIELD_TYPES:
                for field, items in dup.get(ft, {}).items():
                    if isinstance(items, list) and items:
                        merged.setdefault(ft, {}).setdefault(field, []).extend(items)
        out.append(merged)
    return out + keyless


def clean(result: dict) -> dict:
    result = deepcopy(result)
    if result.get("events"):
        result["events"] = _merge_duplicate_events(result["events"])
    for event in result.get("events", []):
        for ft in FIELD_TYPES:
            cleaned = {}
            for field, items in (event.get(ft) or {}).items():
                if not isinstance(items, list):
                    items = [items] if items else []
                items = [o for o in items if isinstance(o, dict)
                         and o.get("severity", o.get("value")) != "not_mentioned"]
                explicit = [o for o in items if o.get("source") == "explicit"][:1]
                inferred = [o for o in items if o.get("source") != "explicit"]
                items = [o for o in explicit + inferred if _valid(o, ft, field)]
                items = _dedupe(items, ft)
                if items:
                    cleaned[field] = items
            if ft in event:
                event[ft] = cleaned
        if "wall_motion_segments" in event:
            event["wall_motion_segments"] = _clean_segments(event["wall_motion_segments"])
    return result


def _clean_segments(segments: list) -> list:
    """Drop segments with invalid names/motions; collapse duplicates by
    (segment, motion), preferring explicit over inferred. Only relevant when
    the optional --segments mode is on."""
    if not isinstance(segments, list):
        return segments
    seen = OrderedDict()
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if seg.get("segment") not in WALL_SEGMENTS or seg.get("motion") not in SEGMENT_MOTION_VALUES:
            continue
        key = (seg["segment"], seg["motion"])
        if key not in seen or (seg.get("source") == "explicit"
                               and seen[key].get("source") == "inferred"):
            seen[key] = seg
    return list(seen.values())


# =============================================================================
# Stage 3 corrections and the evidence filter
# =============================================================================

def _validated_correction_value(field_type: str, field: str, val: str):
    """Return (value, True) if `val` is legal for the target field, else
    (None, False). Enum matching is case-insensitive but resolves to the
    canonical enum spelling; numeric values must parse as a number."""
    if field_type == "numeric_fields":
        try:
            return float(val), True
        except ValueError:
            return None, False
    allowed = (SEVERITY_ENUM_VALUES if field_type == "severity_fields"
               else CATEGORICAL_ENUM_VALUES).get(field)
    if allowed is None:
        return None, False
    wanted = val.strip().lower()
    for a in allowed:
        if a.lower() == wanted:
            return a, True
    return None, False


def apply_corrections(result: dict, validation: dict) -> dict:
    result = deepcopy(result)
    by_id = {e.get("event_id"): e for e in result.get("events", [])}

    for rm in validation.get("removals", []):
        ev = by_id.get(rm.get("event_id"))
        ft, fn = rm.get("field_type"), rm.get("field_name")
        if ev is None:
            continue
        if ft == "wall_motion_segments":
            ev["wall_motion_segments"] = [s for s in ev.get("wall_motion_segments", [])
                                          if s.get("segment") != fn]
        elif ft in ev and fn in ev[ft]:
            ev[ft][fn] = []

    for corr in validation.get("corrections", []):
        ev = by_id.get(corr.get("event_id"))
        ft, fn = corr.get("field_type"), corr.get("field_name")
        if ev is None or ft not in ev or fn not in ev[ft]:
            continue
        ct = corr.get("correction_type")
        raw, val = corr.get("corrected_raw_text", ""), corr.get("corrected_value", "")
        for item in ev[ft][fn]:
            if ct in ("fix_raw_text", "fix_both") and raw:
                if "raw_text" in item:
                    item.setdefault("raw_text_before_correction", item["raw_text"])
                    item["raw_text"] = raw
                    item["corrected_by_validation"] = True
                elif item.get("evidence"):
                    item["evidence"][0].setdefault("raw_text_before_correction",
                                                   item["evidence"][0].get("raw_text", ""))
                    item["evidence"][0]["raw_text"] = raw
                    item["corrected_by_validation"] = True
            if ct in ("fix_value", "fix_both") and val:
                new_val, ok = _validated_correction_value(ft, fn, val)
                if not ok:
                    continue  # reject: the extracted value stands
                value_key = "severity" if ft == "severity_fields" else "value"
                item.setdefault("value_before_correction", item.get(value_key))
                item["corrected_by_validation"] = True
                item[value_key] = new_val

    dropped = {r.get("event_id") for r in validation.get("event_removals", [])}
    result["events"] = [e for e in result.get("events", []) if e.get("event_id") not in dropped]
    result["conflicts"] = validation.get("conflicts", [])
    result["validation_passed"] = validation.get("validation_passed", False)
    result["corrections_applied"] = len(validation.get("corrections", []))
    result["removals_applied"] = len(validation.get("removals", []))
    return result


def remove_empty_evidence(result: dict) -> dict:
    def cited(item) -> bool:
        if not isinstance(item, dict):
            return False
        if "raw_text" in item:
            return bool((item.get("raw_text") or "").strip())
        ev = item.get("evidence")
        return isinstance(ev, list) and any(
            isinstance(e, dict) and (e.get("raw_text") or "").strip() for e in ev)

    for event in result.get("events", []):
        for ft in FIELD_TYPES:
            for field, items in (event.get(ft) or {}).items():
                if isinstance(items, list):
                    event[ft][field] = [i for i in items if cited(i)]
    return result

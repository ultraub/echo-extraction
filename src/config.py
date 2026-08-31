"""The extraction schema: the 48 fields and the values each accepts.

Editing this file changes what the pipeline extracts — the JSON schemas
(src/schemas.py) are derived from these definitions. The prompt text
(src/prompts.py) names the fields and values in prose and must be kept in
step by hand.
"""

# =============================================================================
# Severity / graded fields (19)
# =============================================================================

# Chamber sizes. aortic_root and ascending_aorta refer to diameter only.
CHAMBER_SIZE_FIELDS = [
    "lv_cavity_size",
    "rv_cavity_size",
    "la_cavity_size",
    "ra_cavity_size",
    "aortic_root",
    "ascending_aorta",
]

FUNCTION_FIELDS = [
    "LV_systolic",
    "RV_systolic",
]

REGURGITATION_FIELDS = [
    "MV_regurgitation",
    "AV_regurgitation",
    "TV_regurgitation",
    "PV_regurgitation",
]

STENOSIS_FIELDS = [
    "AV_stenosis",
    "MV_stenosis",
    "TV_stenosis",
    "PV_stenosis",
]

OTHER_GRADED_FIELDS = [
    "lv_wall_thickness",
    "pulm_htn",
    "pericardial_effusion",
]

ESSENTIAL_SEVERITY_FIELDS = (
    CHAMBER_SIZE_FIELDS
    + FUNCTION_FIELDS
    + REGURGITATION_FIELDS
    + STENOSIS_FIELDS
    + OTHER_GRADED_FIELDS
)

# Every severity field accepts these two sentinels plus its own grade list.
# not_mentioned: the report is silent on this finding (dropped after extraction).
# conflict:      the report contradicts itself, or no listed value fits.
SEVERITY_ENUM_VALUES = {
    "lv_cavity_size":   ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "small", "abnormal"],
    "rv_cavity_size":   ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "small", "abnormal"],
    "la_cavity_size":   ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "abnormal"],
    "ra_cavity_size":   ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "abnormal"],
    "aortic_root":      ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "abnormal"],
    "ascending_aorta":  ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "abnormal"],
    "LV_systolic": ["not_mentioned", "conflict", "normal", "abnormal_ungraded", "mild", "mild_moderate", "moderate", "moderate_severe", "severe", "hyperdynamic"],
    "RV_systolic": ["not_mentioned", "conflict", "normal", "abnormal_ungraded", "mild", "mild_moderate", "moderate", "moderate_severe", "severe", "hyperdynamic"],
    "MV_regurgitation": ["not_mentioned", "conflict", "normal", "trace", "mild", "mild_moderate", "moderate", "moderate_severe", "severe", "abnormal_ungraded"],
    "AV_regurgitation": ["not_mentioned", "conflict", "normal", "trace", "mild", "mild_moderate", "moderate", "moderate_severe", "severe", "abnormal_ungraded"],
    "TV_regurgitation": ["not_mentioned", "conflict", "normal", "trace", "mild", "mild_moderate", "moderate", "moderate_severe", "severe", "abnormal_ungraded"],
    "PV_regurgitation": ["not_mentioned", "conflict", "normal", "trace", "mild", "mild_moderate", "moderate", "moderate_severe", "severe", "abnormal_ungraded"],
    "AV_stenosis": ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "abnormal_ungraded"],
    "MV_stenosis": ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "abnormal_ungraded"],
    "TV_stenosis": ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "abnormal_ungraded"],
    "PV_stenosis": ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe", "abnormal_ungraded"],
    "lv_wall_thickness":    ["not_mentioned", "conflict", "normal", "concentric_mild", "concentric_moderate", "concentric_severe", "asymmetric_septal_hypertrophy"],
    "pulm_htn":             ["not_mentioned", "conflict", "normal", "mild", "moderate", "severe"],
    "pericardial_effusion": ["not_mentioned", "conflict", "normal", "trace", "small", "moderate", "large"],
}

# =============================================================================
# Categorical fields (10)
# =============================================================================

CATEGORICAL_ENUM_VALUES = {
    "diastolic_grade": ["not_mentioned", "conflict", "normal", "grade_I", "grade_II", "grade_III", "indeterminate"],
    "global_lv_wall_motion": ["not_mentioned", "conflict", "normal", "hypokinesis", "akinesis", "dyskinesis"],
    "rv_wall_thickness": ["not_mentioned", "conflict", "normal", "abnormal"],
    "rv_wall_motion": ["not_mentioned", "conflict", "normal", "abnormal"],
    "interventricular_septal_motion": ["not_mentioned", "conflict", "normal", "diastolic_flattening", "systolic_flattening", "dyskinesis"],
    "av_morphology": ["not_mentioned", "conflict", "normal", "sclerotic", "thickened", "bicuspid", "prosthetic", "endocarditis"],
    "mv_morphology": ["not_mentioned", "conflict", "normal", "thickened", "calcified", "prolapse", "repair", "prosthetic", "rheumatic", "myxomatous", "endocarditis"],
    "tv_morphology": ["not_mentioned", "conflict", "normal", "thickened", "prosthetic", "repair", "endocarditis"],
    "pv_morphology": ["not_mentioned", "conflict", "normal", "prosthetic", "endocarditis"],
    "pericardial_physiology": ["not_mentioned", "conflict", "normal", "tamponade", "constriction"],
}

ESSENTIAL_CATEGORICAL_FIELDS = list(CATEGORICAL_ENUM_VALUES)

# =============================================================================
# Numeric fields (19)
# =============================================================================

ESSENTIAL_NUMERIC_FIELDS = [
    "ejection_fraction",
    "rvsp",
    "tapse",
    "e_e_prime_avg",
    "la_volume_index",
    "aortic_valve_peak_velocity",
    "av_mean_gradient",
    "av_area",
    "mv_area",
    "mv_mean_gradient",
    "mv_e_a_ratio",
    "ra_pressure_numeric",
    "lv_posterior_wall_diameter",
    "iv_septal_diameter",
    "rv_fractional_area_change",
    "e_prime_septal",
    "e_prime_lateral",
    "e_e_prime_septal",
    "e_e_prime_lateral",
]

# =============================================================================
# Wall-motion segments (optional; enabled with --segments)
# =============================================================================
# The 17-segment AHA/ASE model. Off by default: the runs reported in the
# paper did not enable segment-level extraction (wall motion was captured
# via the global_lv_wall_motion / rv_wall_motion / interventricular_
# septal_motion fields above), and this mode is therefore unevaluated.

WALL_SEGMENTS = [
    "basal_anterior",
    "mid_anterior",
    "apical_anterior",
    "basal_anteroseptal",
    "mid_anteroseptal",
    "apical_septal",
    "basal_inferoseptal",
    "mid_inferoseptal",
    "basal_inferior",
    "mid_inferior",
    "apical_inferior",
    "basal_inferolateral",
    "mid_inferolateral",
    "apical_lateral",
    "basal_anterolateral",
    "mid_anterolateral",
    "apex",
]

SEGMENT_MOTION_VALUES = [
    "normal",
    "hypokinesis",
    "akinesis",
    "dyskinesis",
    "aneurysm",
    "conflict",
]

# =============================================================================
# Provenance tags attached to every extracted value
# =============================================================================

SOURCE_VALUES = ["explicit", "inferred"]          # stated outright vs derived
SOURCE_TYPE_VALUES = ["structured", "narrative"]  # where in the report the span sits

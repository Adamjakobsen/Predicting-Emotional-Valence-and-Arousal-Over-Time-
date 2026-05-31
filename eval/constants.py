# Central constants used by scripts in this repository
from typing import Dict, List, Set

TEMPLATES_DEFAULT: Dict[str, str] = {
    "subtask1": "subtask1-template.csv",
    "subtask2a": "subtask2a-template.csv",
    "subtask2b": "subtask2b-template.csv",
}

REQUIRED_COLUMNS_DEFAULT: Dict[str, List[str]] = {
    # keep user_id in key columns but remove from prediction-only list if desired
    "subtask1": ["user_id", "text_id", "pred_valence", "pred_arousal"],
    "subtask2a": ["user_id", "pred_state_change_valence", "pred_state_change_arousal"],
    "subtask2b": ["user_id", "pred_dispo_change_valence", "pred_dispo_change_arousal"],
}

# Key columns (always required for grouping / matching template rows)
KEY_COLUMNS_DEFAULT: Dict[str, List[str]] = {
    "subtask1": ["user_id", "text_id"],
    "subtask2a": ["user_id"],
    "subtask2b": ["user_id"],
}

NULL_VALUES_DEFAULT: Set[str] = {"", "NULL", "null", "None", "none"}

DEFAULT_ASSETS_DIR: str = "assets"

# prediction column names by task and dimension
PRED_COLS_DEFAULT = {
    "subtask1": {"valence": "pred_valence", "arousal": "pred_arousal"},
    "subtask2a": {"valence": "pred_state_change_valence", "arousal": "pred_state_change_arousal"},
    "subtask2b": {"valence": "pred_dispo_change_valence", "arousal": "pred_dispo_change_arousal"},
}

# label column names by task and dimension
LABEL_COLS_DEFAULT = {
    "subtask1": {"valence": "valence", "arousal": "arousal"},
    "subtask2a": {"valence": "state_change_valence", "arousal": "state_change_arousal"},
    "subtask2b": {"valence": "disp_change_valence", "arousal": "disp_change_arousal"},
}

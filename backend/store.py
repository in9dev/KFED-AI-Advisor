"""
store.py — tiny JSON-file "database" standing in for KFED's real, scattered
systems (CRM, LMS/training records, coaching notes). In production this
would be replaced by API calls into Qudorat / the actual KFED data stores;
the ProfilerAgent doesn't care where unify() gets its data from.
"""

import json
import os
import threading

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_SEED_PATH = os.path.join(_DATA_DIR, "beneficiaries_seed.json")
_STATE_PATH = os.path.join(_DATA_DIR, "state.json")

_lock = threading.Lock()


def _load_seed():
    with open(_SEED_PATH, encoding="utf-8") as f:
        return json.load(f)


def _default_state():
    return {"profiles": {}, "pathways": {}, "escalations": [], "quiz_progress": {}, "next_beneficiary_seq": 3000}


def load_state():
    with _lock:
        if not os.path.exists(_STATE_PATH):
            state = _default_state()
            _save_state(state)
            return state
        with open(_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)


def _save_state(state):
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def save_state(state):
    with _lock:
        _save_state(state)


def reset_state():
    with _lock:
        state = _default_state()
        _save_state(state)
        return state


SEED = _load_seed()


def lookup_scattered_records(beneficiary_id):
    """Pulls together the three scattered KFED-style sources for one
    beneficiary_id, simulating 'profiles/training history/coaching notes
    live in different systems' from the challenge brief."""
    crm = SEED.get("crm_profiles", {}).get(beneficiary_id)
    training = SEED.get("training_history", {}).get(beneficiary_id, [])
    coaching = SEED.get("coaching_notes", {}).get(beneficiary_id, [])
    if not crm and not training and not coaching:
        return None
    return {"crm": crm, "training_history": training, "coaching_notes": coaching}

from __future__ import annotations

import hashlib


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def derive_seed_bundle(*, global_seed: int, family: str, family_trial_index: int) -> dict[str, int]:
    """Derive deterministic seeds for all trial-level randomness targets."""

    trial_seed = _stable_seed(global_seed, family, family_trial_index, "trial")
    return {
        "trial_seed": trial_seed,
        "input_signal_seed": _stable_seed(trial_seed, "input_signal"),
        "mask_seed": _stable_seed(trial_seed, "mask"),
        "morphology_seed": _stable_seed(trial_seed, "morphology"),
    }

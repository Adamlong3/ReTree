from .recovery import RecoveryMemory, target_logit_consistency_gate
from .utils import (
    collect_prompt_hashes,
    extract_context_feature,
    load_and_process_dataset,
    load_calibration_dataset,
    prompt_hashes,
    sample,
)


__all__ = [
    "DFlashDraftModel",
    "RecoveryMemory",
    "collect_prompt_hashes",
    "extract_context_feature",
    "load_and_process_dataset",
    "load_calibration_dataset",
    "prompt_hashes",
    "sample",
    "target_logit_consistency_gate",
]


def __getattr__(name: str):
    if name == "DFlashDraftModel":
        from .dflash import DFlashDraftModel

        return DFlashDraftModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

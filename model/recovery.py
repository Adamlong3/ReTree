from collections import defaultdict
import math
import torch
import json
import os


class RecoveryMemory:
    def __init__(self, freq_threshold: int = 6):
        self.freq_threshold = freq_threshold
        self._table: dict[tuple[int, int], int] = defaultdict(int)

    def update(self, draft_token: int, target_token: int):
        self._table[(int(draft_token), int(target_token))] += 1

    def is_frequent(self, draft_token: int, target_token: int) -> bool:
        return (
            self._table.get((int(draft_token), int(target_token)), 0)
            >= self.freq_threshold
        )

    def get_frequency(self, draft_token: int, target_token: int) -> int:
        return self._table.get((int(draft_token), int(target_token)), 0)

    def top_k_pairs(self, k: int = 20) -> list[tuple[tuple[int, int], int]]:
        sorted_items = sorted(self._table.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def total_pairs(self) -> int:
        return len(self._table)

    def total_rejections(self) -> int:
        return sum(self._table.values())

    def save(self, path: str):
        serializable = {f"{k[0]},{k[1]}": v for k, v in self._table.items()}
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        with open(path, "w") as f:
            json.dump(serializable, f)

    def load(self, path: str):
        with open(path, "r") as f:
            serializable = json.load(f)
        self._table = defaultdict(int)
        for k_str, v in serializable.items():
            parts = k_str.split(",")
            self._table[(int(parts[0]), int(parts[1]))] = int(v)

    @staticmethod
    def from_file(path: str, freq_threshold: int = 6) -> "RecoveryMemory":
        memory = RecoveryMemory(freq_threshold=freq_threshold)
        memory.load(path)
        return memory


def token_pair_has_stop(
    draft_token: int,
    target_token: int,
    stop_token_ids: set[int] | None,
) -> bool:
    """
    Stop-token-safe helper.

    Return True if either side of a draft->target pair is a stop token.

    We use this to block both directions:
        X -> EOS
        EOS -> X
    """
    if stop_token_ids is None:
        return False

    draft_token = int(draft_token)
    target_token = int(target_token)

    return draft_token in stop_token_ids or target_token in stop_token_ids


def target_logit_consistency_gate(
    target_logits: torch.Tensor,
    draft_token_id: int,
    target_top_token_id: int,
    position: int,
    threshold: float = 0.01,
) -> bool:
    """
    Target-logit consistency gate:
        exp(logit_draft - logit_target_top) >= threshold

    Equivalent:
        logit_draft - logit_target_top >= log(threshold)

    target_logits: [seq_len, vocab]
    """
    if int(draft_token_id) == int(target_top_token_id):
        return True

    if threshold <= 0:
        return True

    logit_draft = target_logits[position, int(draft_token_id)]
    logit_top = target_logits[position, int(target_top_token_id)]

    return bool((logit_draft - logit_top >= math.log(threshold)).item())

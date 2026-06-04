from collections import defaultdict
from typing import Optional
import math
import torch
import json
import os


class OnlineCorrectionMemory:
    def __init__(self, freq_threshold: int = 6):
        self.freq_threshold = freq_threshold
        self._table: dict[tuple[int, int], int] = defaultdict(int)

    def update(self, draft_token: int, target_token: int):
        self._table[(int(draft_token), int(target_token))] += 1

    def is_frequent(self, draft_token: int, target_token: int) -> bool:
        return self._table.get((int(draft_token), int(target_token)), 0) >= self.freq_threshold

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
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
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
    def from_file(path: str, freq_threshold: int = 6) -> "OnlineCorrectionMemory":
        memory = OnlineCorrectionMemory(freq_threshold=freq_threshold)
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
    """
    if stop_token_ids is None:
        return False

    draft_token = int(draft_token)
    target_token = int(target_token)

    return draft_token in stop_token_ids or target_token in stop_token_ids


def semantic_consistency_gate(
    target_logits: torch.Tensor,
    draft_token_id: int,
    target_top_token_id: int,
    position: int,
    threshold: float = 0.01,
) -> bool:
    if int(draft_token_id) == int(target_top_token_id):
        return True

    if threshold <= 0:
        return True

    logit_draft = target_logits[position, int(draft_token_id)]
    logit_top = target_logits[position, int(target_top_token_id)]

    return bool((logit_draft - logit_top >= math.log(threshold)).item())


def correction_verify_block(
    block_output_ids: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
    memory: Optional[OnlineCorrectionMemory],
    consistency_threshold: float = 0.01,
    stop_token_ids: set[int] | None = None,
) -> tuple[int, torch.Tensor, list[bool]]:
    block_size = block_output_ids.shape[1] - 1

    if temperature < 1e-5:
        posterior = torch.argmax(target_logits, dim=-1)
    else:
        bsz, seq_len, vocab_size = target_logits.shape
        logits_flat = target_logits.view(-1, vocab_size) / temperature
        probs = torch.softmax(logits_flat, dim=-1)
        posterior = torch.multinomial(probs, num_samples=1).view(bsz, seq_len)

    draft_tokens = block_output_ids[0, 1:]
    target_tokens = posterior[0, :-1]

    accepted_count = 0
    recovered_flags: list[bool] = []

    for i in range(block_size):
        draft_tok = int(draft_tokens[i].item())
        target_tok = int(target_tokens[i].item())

        if draft_tok == target_tok:
            accepted_count += 1
            recovered_flags.append(False)
            continue

        if token_pair_has_stop(draft_tok, target_tok, stop_token_ids):
            break

        is_freq = memory.is_frequent(draft_tok, target_tok) if memory is not None else False

        is_safe = semantic_consistency_gate(
            target_logits[0],
            draft_token_id=draft_tok,
            target_top_token_id=target_tok,
            position=i,
            threshold=consistency_threshold,
        )

        if memory is not None:
            memory.update(draft_tok, target_tok)

        if is_freq and is_safe:
            accepted_count += 1
            recovered_flags.append(True)
        else:
            break

    return accepted_count, posterior, recovered_flags

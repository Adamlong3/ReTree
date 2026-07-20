from collections import defaultdict
from collections.abc import Iterable, Mapping
import json
import math
import os

import torch


TokenPair = tuple[int, int]


def _serialize_counts(counts: Mapping[TokenPair, int]) -> dict[str, int]:
    return {
        f"{draft_token},{target_token}": int(count)
        for (draft_token, target_token), count in counts.items()
        if int(count) > 0
    }


def _deserialize_counts(serialized: Mapping[str, int]) -> dict[TokenPair, int]:
    counts: dict[TokenPair, int] = {}
    for key, value in serialized.items():
        parts = str(key).split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid recovery-memory key: {key!r}")
        count = int(value)
        if count < 0:
            raise ValueError(f"Recovery-memory counts must be non-negative: {key!r}")
        if count > 0:
            counts[(int(parts[0]), int(parts[1]))] = count
    return counts


class RecoveryMemory:
    """Two-stage recovery memory with a fixed prior and causal online counts."""

    def __init__(self, freq_threshold: int = 6):
        if freq_threshold < 0:
            raise ValueError("freq_threshold must be non-negative")
        self.freq_threshold = int(freq_threshold)
        self._prior_counts: dict[TokenPair, int] = defaultdict(int)
        self._online_counts: dict[TokenPair, int] = defaultdict(int)

    @staticmethod
    def _pair(draft_token: int, target_token: int) -> TokenPair:
        return int(draft_token), int(target_token)

    @staticmethod
    def _increment(
        table: dict[TokenPair, int],
        draft_token: int,
        target_token: int,
        count: int = 1,
    ) -> None:
        count = int(count)
        if count < 0:
            raise ValueError("Recovery-memory updates must be non-negative")
        if count > 0:
            table[RecoveryMemory._pair(draft_token, target_token)] += count

    def record_prior(self, draft_token: int, target_token: int, count: int = 1) -> None:
        """Record one offline calibration event."""
        self._increment(
            self._prior_counts, draft_token, target_token, count=count
        )

    def record_online(self, draft_token: int, target_token: int, count: int = 1) -> None:
        """Record one causally observed inference event."""
        self._increment(
            self._online_counts, draft_token, target_token, count=count
        )

    def _record_ranked_divergences(
        self,
        ranked_child_tokens: Iterable[int],
        target_token: int,
        top_k: int,
        stop_token_ids: set[int] | None,
        online: bool,
    ) -> int:
        if top_k <= 0:
            return 0

        target_token = int(target_token)
        recorded = 0
        for child_token in ranked_child_tokens:
            child_token = int(child_token)
            if child_token == target_token:
                continue
            if token_pair_has_stop(child_token, target_token, stop_token_ids):
                continue
            if online:
                self.record_online(child_token, target_token)
            else:
                self.record_prior(child_token, target_token)
            recorded += 1
            if recorded >= top_k:
                break
        return recorded

    def record_prior_divergences(
        self,
        ranked_child_tokens: Iterable[int],
        target_token: int,
        top_k: int,
        stop_token_ids: set[int] | None,
    ) -> int:
        return self._record_ranked_divergences(
            ranked_child_tokens,
            target_token,
            top_k,
            stop_token_ids,
            online=False,
        )

    def record_online_divergences(
        self,
        ranked_child_tokens: Iterable[int],
        target_token: int,
        top_k: int,
        stop_token_ids: set[int] | None,
    ) -> int:
        return self._record_ranked_divergences(
            ranked_child_tokens,
            target_token,
            top_k,
            stop_token_ids,
            online=True,
        )

    def get_prior_frequency(self, draft_token: int, target_token: int) -> int:
        return self._prior_counts.get(self._pair(draft_token, target_token), 0)

    def get_online_frequency(self, draft_token: int, target_token: int) -> int:
        return self._online_counts.get(self._pair(draft_token, target_token), 0)

    def get_frequency(self, draft_token: int, target_token: int) -> int:
        pair = self._pair(draft_token, target_token)
        return self._prior_counts.get(pair, 0) + self._online_counts.get(pair, 0)

    def snapshot_frequencies(
        self, pairs: Iterable[TokenPair]
    ) -> dict[TokenPair, int]:
        """Snapshot combined frequencies before recording the current mismatch."""
        return {
            self._pair(*pair): self.get_frequency(*pair)
            for pair in pairs
        }

    def is_frequent(self, draft_token: int, target_token: int) -> bool:
        return self.get_frequency(draft_token, target_token) >= self.freq_threshold

    def prior_counts(self) -> dict[TokenPair, int]:
        return dict(self._prior_counts)

    def online_counts(self) -> dict[TokenPair, int]:
        return dict(self._online_counts)

    def online_delta_since(
        self, snapshot: Mapping[TokenPair, int]
    ) -> dict[TokenPair, int]:
        delta: dict[TokenPair, int] = {}
        for pair, current in self._online_counts.items():
            change = int(current) - int(snapshot.get(pair, 0))
            if change < 0:
                raise ValueError("Online recovery-memory counts cannot decrease")
            if change > 0:
                delta[pair] = change
        return delta

    def apply_online_delta(self, delta: Mapping[TokenPair, int]) -> None:
        for (draft_token, target_token), count in delta.items():
            self.record_online(draft_token, target_token, count=int(count))

    def clear_online(self) -> None:
        self._online_counts = defaultdict(int)

    def _combined_counts(self) -> dict[TokenPair, int]:
        combined = dict(self._prior_counts)
        for pair, count in self._online_counts.items():
            combined[pair] = combined.get(pair, 0) + count
        return combined

    def top_k_pairs(self, k: int = 20) -> list[tuple[TokenPair, int]]:
        sorted_items = sorted(
            self._combined_counts().items(), key=lambda item: item[1], reverse=True
        )
        return sorted_items[:k]

    def prior_total_pairs(self) -> int:
        return len(self._prior_counts)

    def online_total_pairs(self) -> int:
        return len(self._online_counts)

    def total_pairs(self) -> int:
        return len(self._combined_counts())

    def prior_total_events(self) -> int:
        return sum(self._prior_counts.values())

    def online_total_events(self) -> int:
        return sum(self._online_counts.values())

    def total_rejections(self) -> int:
        return self.prior_total_events() + self.online_total_events()

    def save_prior(self, path: str) -> None:
        """Save only the immutable offline prior in the legacy flat JSON format."""
        self._write_json(path, _serialize_counts(self._prior_counts))

    def save_runtime_state(self, path: str) -> None:
        """Save the fixed prior and this run's online delta without conflating them."""
        payload = {
            "format": "retree-recovery-state-v1",
            "freq_threshold": self.freq_threshold,
            "prior_counts": _serialize_counts(self._prior_counts),
            "online_counts": _serialize_counts(self._online_counts),
        }
        self._write_json(path, payload)

    @staticmethod
    def _write_json(path: str, payload: object) -> None:
        directory = os.path.dirname(path) if os.path.dirname(path) else "."
        os.makedirs(directory, exist_ok=True)
        temporary_path = f"{path}.tmp.{os.getpid()}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def load(self, path: str, load_online: bool = False) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("Recovery-memory files must contain a JSON object")

        if "prior_counts" in payload:
            prior_payload = payload["prior_counts"]
            online_payload = payload.get("online_counts", {}) if load_online else {}
        elif "rescue_pairs" in payload:
            prior_payload = payload["rescue_pairs"]
            online_payload = {}
        else:
            prior_payload = payload
            online_payload = {}

        if not isinstance(prior_payload, dict) or not isinstance(online_payload, dict):
            raise TypeError("Recovery-memory count tables must be JSON objects")

        self._prior_counts = defaultdict(int, _deserialize_counts(prior_payload))
        self._online_counts = defaultdict(int, _deserialize_counts(online_payload))

    @staticmethod
    def from_file(
        path: str,
        freq_threshold: int = 6,
        load_online: bool = False,
    ) -> "RecoveryMemory":
        memory = RecoveryMemory(freq_threshold=freq_threshold)
        memory.load(path, load_online=load_online)
        return memory


def token_pair_has_stop(
    draft_token: int,
    target_token: int,
    stop_token_ids: set[int] | None,
) -> bool:
    """Return whether either side of a recovery pair is a stop token."""
    if stop_token_ids is None:
        return False
    return (
        int(draft_token) in stop_token_ids
        or int(target_token) in stop_token_ids
    )


def target_logit_consistency_gate(
    target_logits: torch.Tensor,
    draft_token_id: int,
    target_top_token_id: int,
    position: int,
    threshold: float = 0.01,
) -> bool:
    """Check exp(logit_draft - logit_target) >= threshold."""
    if int(draft_token_id) == int(target_top_token_id):
        return True
    if threshold <= 0:
        return True

    logit_draft = target_logits[position, int(draft_token_id)]
    logit_top = target_logits[position, int(target_top_token_id)]
    return bool((logit_draft - logit_top >= math.log(threshold)).item())

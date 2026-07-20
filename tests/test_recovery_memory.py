import json
import tempfile
import unittest
from pathlib import Path

from model.recovery import RecoveryMemory
from model.utils import prompt_hashes


class RecoveryMemoryTest(unittest.TestCase):
    def test_prior_and_online_counts_are_separate(self):
        memory = RecoveryMemory(freq_threshold=3)
        memory.record_prior(10, 20, count=2)
        memory.record_online(10, 20)

        self.assertEqual(memory.get_prior_frequency(10, 20), 2)
        self.assertEqual(memory.get_online_frequency(10, 20), 1)
        self.assertEqual(memory.get_frequency(10, 20), 3)
        self.assertTrue(memory.is_frequent(10, 20))

    def test_ranked_recording_skips_exact_and_stop_tokens(self):
        memory = RecoveryMemory()
        recorded = memory.record_prior_divergences(
            [20, 99, 10, 11, 12],
            target_token=20,
            top_k=2,
            stop_token_ids={99},
        )

        self.assertEqual(recorded, 2)
        self.assertEqual(memory.get_prior_frequency(10, 20), 1)
        self.assertEqual(memory.get_prior_frequency(11, 20), 1)
        self.assertEqual(memory.get_prior_frequency(12, 20), 0)

    def test_prior_file_does_not_persist_online_delta(self):
        memory = RecoveryMemory(freq_threshold=4)
        memory.record_prior(10, 20, count=3)
        memory.record_online(10, 20, count=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            prior_path = Path(tmpdir) / "prior.json"
            runtime_path = Path(tmpdir) / "runtime.json"
            memory.save_prior(str(prior_path))
            memory.save_runtime_state(str(runtime_path))

            reloaded_prior = RecoveryMemory.from_file(str(prior_path))
            self.assertEqual(reloaded_prior.get_frequency(10, 20), 3)
            self.assertEqual(reloaded_prior.online_total_events(), 0)

            runtime_as_fresh_prior = RecoveryMemory.from_file(str(runtime_path))
            self.assertEqual(runtime_as_fresh_prior.get_frequency(10, 20), 3)
            self.assertEqual(runtime_as_fresh_prior.online_total_events(), 0)

            reloaded_runtime = RecoveryMemory.from_file(
                str(runtime_path), load_online=True
            )
            self.assertEqual(reloaded_runtime.get_prior_frequency(10, 20), 3)
            self.assertEqual(reloaded_runtime.get_online_frequency(10, 20), 2)

            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertIn("prior_counts", payload)
            self.assertIn("online_counts", payload)

    def test_online_delta_can_be_synchronized_without_copying_prior(self):
        first = RecoveryMemory()
        second = RecoveryMemory()
        first.record_prior(10, 20, count=5)
        second.record_prior(10, 20, count=5)

        before = first.online_counts()
        first.record_online(10, 20, count=2)
        second.apply_online_delta(first.online_delta_since(before))

        self.assertEqual(second.get_prior_frequency(10, 20), 5)
        self.assertEqual(second.get_online_frequency(10, 20), 2)

    def test_pre_update_snapshot_prevents_self_recovery(self):
        memory = RecoveryMemory(freq_threshold=2)
        memory.record_prior(10, 20)

        snapshot = memory.snapshot_frequencies([(10, 20)])
        memory.record_online(10, 20)

        self.assertLess(snapshot[(10, 20)], memory.freq_threshold)
        self.assertEqual(memory.get_frequency(10, 20), 2)
        next_snapshot = memory.snapshot_frequencies([(10, 20)])
        self.assertGreaterEqual(next_snapshot[(10, 20)], memory.freq_threshold)


class PromptHashTest(unittest.TestCase):
    def test_prompt_hash_normalizes_unicode_and_whitespace(self):
        self.assertEqual(
            prompt_hashes(["Solve   x + 1\nnow"]),
            prompt_hashes(["Solve x + 1 now"]),
        )

    def test_prompt_hash_includes_individual_turns(self):
        first_turn_hashes = prompt_hashes(["first"])
        multi_turn_hashes = prompt_hashes(["first", "second"])
        self.assertTrue(first_turn_hashes.intersection(multi_turn_hashes))


if __name__ == "__main__":
    unittest.main()

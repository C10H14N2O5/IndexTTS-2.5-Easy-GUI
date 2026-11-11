"""Pure step-scheduling rules shared by the real and the fake LoRA trainer."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from indextts_training.schedule import (
    ADAPTER_STATE_FILE, checkpoint_due, normalize_dense, read_adapter_state, validation_due, write_adapter_state,
)


def due_steps(end_step, **kwargs):
    return [step for step in range(1, end_step + 1) if validation_due(step, end_step=end_step, **kwargs)]


class CheckpointDueTests(unittest.TestCase):
    def test_every_save_interval_and_end_step(self):
        self.assertEqual([s for s in range(1, 11) if checkpoint_due(s, 3, 10)], [3, 6, 9, 10])
        self.assertEqual([s for s in range(1, 8) if checkpoint_due(s, 3, 7)], [3, 6, 7])

    def test_end_step_is_always_saved_even_when_early(self):
        # An early stop between two save points still produces a checkpoint.
        self.assertTrue(checkpoint_due(5, 100, 5))
        self.assertFalse(checkpoint_due(4, 100, 5))

    def test_non_positive_save_every_saves_every_step(self):
        self.assertTrue(all(checkpoint_due(s, 0, 9) for s in range(1, 10)))
        self.assertTrue(all(checkpoint_due(s, -4, 9) for s in range(1, 10)))


class ValidationDueTests(unittest.TestCase):
    def test_interval_none_or_zero_follows_save_every(self):
        for interval in (None, 0):
            with self.subTest(interval=interval):
                self.assertEqual(due_steps(10, save_every=4, interval=interval), [4, 8, 10])

    def test_custom_interval_is_independent_of_checkpoints(self):
        self.assertEqual(due_steps(10, save_every=4, interval=3), [3, 6, 9, 10])

    def test_end_step_always_validated(self):
        self.assertEqual(due_steps(7, save_every=100, interval=50), [7])
        self.assertTrue(validation_due(1, end_step=1, save_every=10))

    def test_dense_range_inclusive_and_counted_from_start(self):
        dense = {'start_step': 5, 'end_step': 11, 'interval_steps': 3}
        # Global interval 10, dense every 3 from 5: 5, 8, 11 (inclusive end), plus 10 and end 20.
        self.assertEqual(due_steps(20, save_every=10, dense=dense), [5, 8, 10, 11, 20])

    def test_dense_end_not_on_grid_is_not_forced(self):
        dense = {'start_step': 2, 'end_step': 9, 'interval_steps': 4}
        self.assertEqual(due_steps(30, save_every=30, dense=dense), [2, 6, 30])

    def test_dense_single_step_range(self):
        dense = {'start_step': 7, 'end_step': 7, 'interval_steps': 5}
        self.assertEqual(due_steps(20, save_every=20, dense=dense), [7, 20])

    def test_dense_beyond_end_step_is_harmless(self):
        dense = {'start_step': 4, 'end_step': 100, 'interval_steps': 2}
        self.assertEqual(due_steps(9, save_every=50, dense=dense), [4, 6, 8, 9])

    def test_dense_values_may_be_numeric_strings(self):
        dense = {'start_step': '3', 'end_step': '6', 'interval_steps': '3'}
        self.assertEqual(due_steps(10, save_every=10, dense=dense), [3, 6, 10])

    def test_invalid_dense_is_ignored(self):
        invalid = [
            None, {}, {'start_step': 5}, {'start_step': 0, 'end_step': 5, 'interval_steps': 1},
            {'start_step': 6, 'end_step': 5, 'interval_steps': 1}, {'start_step': 1, 'end_step': 5, 'interval_steps': 0},
            {'start_step': 1, 'end_step': 5, 'interval_steps': -2}, {'start_step': 'x', 'end_step': 5, 'interval_steps': 1},
            {'start_step': None, 'end_step': 5, 'interval_steps': 1},
        ]
        for dense in invalid:
            with self.subTest(dense=dense):
                self.assertIsNone(normalize_dense(dense))
                self.assertEqual(due_steps(10, save_every=5, dense=dense), [5, 10])

    def test_normalize_dense_valid(self):
        self.assertEqual(normalize_dense({'start_step': 1, 'end_step': 1, 'interval_steps': 1}), (1, 1, 1))
        self.assertEqual(normalize_dense({'start_step': 2, 'end_step': 9, 'interval_steps': 3, 'extra': 1}), (2, 9, 3))


class AdapterStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_roundtrip_early_stop(self):
        write_adapter_state(self.directory, 30, 100, 30, False)
        self.assertEqual(read_adapter_state(self.directory), {
            'step': 30, 'schedule_total': 100, 'end_step': 30, 'stopped': False, 'early_stopped': True})

    def test_flags_for_normal_intermediate_and_stopped_runs(self):
        cases = [
            ((100, 100, 100, False), False),   # planned finish
            ((20, 100, 30, False), False),     # intermediate checkpoint of an early-stop run
            ((12, 100, 30, True), False),      # user stop
            ((30, 100, 30, True), False),      # stop flag wins even at the end step
        ]
        for args, early in cases:
            with self.subTest(args=args):
                write_adapter_state(self.directory, *args)
                state = read_adapter_state(self.directory)
                self.assertEqual(state['early_stopped'], early)
                self.assertEqual(state['stopped'], args[3])
                self.assertEqual((state['step'], state['schedule_total'], state['end_step']), args[:3])

    def test_values_are_coerced_to_int(self):
        write_adapter_state(str(self.directory), '7', 10.0, '7', 0)
        state = json.loads((self.directory / ADAPTER_STATE_FILE).read_text(encoding='utf-8'))
        self.assertEqual(state, {'step': 7, 'schedule_total': 10, 'end_step': 7, 'stopped': False,
                                 'early_stopped': True})

    def test_missing_or_malformed_sidecar(self):
        self.assertIsNone(read_adapter_state(self.directory))
        self.assertIsNone(read_adapter_state(self.directory / 'missing'))
        path = self.directory / ADAPTER_STATE_FILE
        for content in ['{not json', '[]', '"text"', '{}', '{"step": "5"}', '{"step": null}', '']:
            with self.subTest(content=content):
                path.write_text(content, encoding='utf-8')
                self.assertIsNone(read_adapter_state(self.directory))
        path.write_bytes(b'\xff\xfe\x00garbage')
        self.assertIsNone(read_adapter_state(self.directory))


if __name__ == '__main__':
    unittest.main()

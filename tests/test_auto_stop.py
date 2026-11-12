"""Validation-loss auto stop state machine (pure Python, no torch)."""
from __future__ import annotations

import json
import math
import unittest

from indextts_training.auto_stop import AutoStopMonitor, monitor_from_config

FINGERPRINT = 'f' * 64
NO_EVENT = {'new_best': False, 'entered_watch': False, 'recovered': False, 'stop': False}


def event(**flags):
    return {**NO_EVENT, **flags}


def monitor(patience=10, min_delta=0.1, interval=2, fingerprint=FINGERPRINT):
    return AutoStopMonitor(patience, min_delta, interval, fingerprint)


class NormalPhaseTests(unittest.TestCase):
    def test_first_observation_sets_best_and_valley(self):
        m = monitor()
        self.assertEqual(m.observe(5, 2.0), event(new_best=True))
        self.assertEqual((m.phase, m.best_step, m.best_loss, m.valley_step, m.valley_loss), ('normal', 5, 2.0, 5, 2.0))

    def test_monotone_significant_decrease_stays_normal_and_moves_valley(self):
        m = monitor(min_delta=0.1)
        for step, loss in [(10, 3.0), (20, 2.8), (30, 2.5), (40, 2.3)]:
            self.assertEqual(m.observe(step, loss), event(new_best=True))
            self.assertEqual(m.phase, 'normal')
            self.assertEqual((m.valley_step, m.valley_loss), (step, loss))
            self.assertEqual((m.best_step, m.best_loss), (step, loss))
            self.assertFalse(m.extra_validation_due(step + 1))
        self.assertIsNone(m.deadline_step)
        self.assertIsNone(m.watch_start_step)

    def test_plateau_tiny_gain_and_rise_enter_watching(self):
        for label, loss, best in [('plateau', 2.0, False), ('tiny gain', 1.95, True), ('rise', 2.4, False)]:
            with self.subTest(label):
                m = monitor(patience=10, min_delta=0.1)
                m.observe(10, 2.0)
                self.assertEqual(m.observe(20, loss), event(new_best=best, entered_watch=True))
                self.assertEqual(m.phase, 'watching')
                self.assertEqual((m.watch_start_step, m.deadline_step), (20, 30))
                # The valley stays at the last significant point.
                self.assertEqual((m.valley_step, m.valley_loss), (10, 2.0))
                self.assertEqual(m.best_step, 20 if best else 10)

    def test_gain_exactly_min_delta_is_significant(self):
        m = monitor(min_delta=0.5)
        m.observe(1, 2.0)
        self.assertEqual(m.observe(2, 1.5), event(new_best=True))
        self.assertEqual((m.phase, m.valley_step), ('normal', 2))


class WatchingPhaseTests(unittest.TestCase):
    def watching(self, patience=10, min_delta=0.1, interval=2):
        m = monitor(patience, min_delta, interval)
        m.observe(10, 2.0)
        m.observe(20, 2.2)  # watch starts at 20, deadline 30
        self.assertEqual(m.phase, 'watching')
        return m

    def test_small_gain_updates_best_but_not_valley_and_does_not_recover(self):
        m = self.watching()
        self.assertEqual(m.observe(22, 1.95), event(new_best=True))
        self.assertEqual(m.phase, 'watching')
        self.assertEqual((m.best_step, m.best_loss), (22, 1.95))
        self.assertEqual((m.valley_step, m.valley_loss), (10, 2.0))
        self.assertEqual(m.watch_start_step, 20)
        self.assertAlmostEqual(m.improvement, 0.05)
        # A worse point inside the watch keeps the best watch improvement.
        self.assertEqual(m.observe(24, 2.3), NO_EVENT)
        self.assertAlmostEqual(m.improvement, 0.05)

    def test_significant_gain_recovers_and_moves_valley(self):
        m = self.watching()
        self.assertEqual(m.observe(24, 1.9), event(new_best=True, recovered=True))
        self.assertEqual(m.phase, 'normal')
        self.assertEqual((m.valley_step, m.valley_loss), (24, 1.9))
        self.assertIsNone(m.watch_start_step)
        self.assertIsNone(m.watch_best_loss)
        self.assertIsNone(m.deadline_step)
        self.assertEqual(m.improvement, 0.0)
        self.assertFalse(m.extra_validation_due(26))
        # A later failure starts a fresh watch from the new step.
        self.assertEqual(m.observe(30, 1.9)['entered_watch'], True)
        self.assertEqual((m.watch_start_step, m.deadline_step), (30, 40))

    def test_recovery_needs_valley_gain_not_watch_gain(self):
        m = self.watching(min_delta=0.1)
        m.observe(22, 2.5)
        # 2.5 -> 1.95 is a large drop inside the watch but only 0.05 below the valley.
        self.assertEqual(m.observe(24, 1.95), event(new_best=True))
        self.assertEqual(m.phase, 'watching')

    def test_stop_at_exact_deadline_step(self):
        m = self.watching(patience=10)
        self.assertEqual(m.observe(28, 2.1), NO_EVENT)
        self.assertEqual(m.observe(29, 2.1), NO_EVENT)
        self.assertEqual(m.phase, 'watching')
        self.assertEqual(m.observe(30, 2.1), event(stop=True))
        self.assertEqual(m.phase, 'stopped')

    def test_late_observation_past_deadline_stops(self):
        m = self.watching(patience=10)
        self.assertEqual(m.observe(35, 2.1), event(stop=True))

    def test_improvement_at_deadline_recovers_instead_of_stopping(self):
        m = self.watching(patience=10)
        self.assertEqual(m.observe(30, 1.5), event(new_best=True, recovered=True))
        self.assertEqual(m.phase, 'normal')

    def test_small_new_best_at_deadline_still_stops(self):
        m = self.watching(patience=10)
        self.assertEqual(m.observe(30, 1.99), event(new_best=True, stop=True))
        self.assertEqual((m.phase, m.best_step), ('stopped', 30))

    def test_observe_after_stop_is_a_no_op(self):
        m = self.watching(patience=10)
        m.observe(30, 2.1)
        before = m.to_dict()
        self.assertEqual(m.observe(40, 0.1), NO_EVENT)
        self.assertEqual(m.observe(50, 5.0), NO_EVENT)
        self.assertEqual(m.to_dict(), before)
        self.assertEqual(m.phase, 'stopped')
        self.assertFalse(m.extra_validation_due(32))


class ExtraValidationTests(unittest.TestCase):
    def test_normal_phase_never_requests_extra_validation(self):
        m = monitor(patience=10, interval=2)
        self.assertFalse(any(m.extra_validation_due(step) for step in range(0, 50)))
        m.observe(10, 2.0)
        self.assertFalse(any(m.extra_validation_due(step) for step in range(0, 50)))

    def test_schedule_counts_from_watch_start_and_includes_deadline(self):
        m = monitor(patience=7, interval=3)
        m.observe(10, 2.0)
        m.observe(20, 2.0)  # watch 20, deadline 27
        due = [step for step in range(15, 40) if m.extra_validation_due(step)]
        # Offsets 3 and 6 plus the deadline 27; not the watch start itself (offset 0) and
        # also later interval multiples (the trainer has stopped by then).
        self.assertEqual(due[:3], [23, 26, 27])
        self.assertNotIn(20, due)
        self.assertFalse(m.extra_validation_due(19))
        self.assertFalse(m.extra_validation_due(21))

    def test_deadline_on_interval_grid_is_listed_once(self):
        m = monitor(patience=6, interval=2)
        m.observe(1, 2.0)
        m.observe(4, 2.0)
        self.assertEqual([step for step in range(1, 11) if m.extra_validation_due(step)], [6, 8, 10])

    def test_interval_equal_to_patience_only_validates_deadline(self):
        m = monitor(patience=5, interval=5)
        m.observe(1, 2.0)
        m.observe(2, 3.0)
        self.assertEqual([step for step in range(1, 8) if m.extra_validation_due(step)], [7])


class MinDeltaTests(unittest.TestCase):
    def test_zero_min_delta_any_decrease_is_significant_equal_is_not(self):
        m = monitor(min_delta=0)
        m.observe(1, 2.0)
        self.assertEqual(m.observe(2, 2.0 - 1e-9), event(new_best=True))
        self.assertEqual(m.phase, 'normal')
        self.assertEqual(m.valley_step, 2)
        self.assertEqual(m.observe(3, 2.0 - 1e-9), event(entered_watch=True))
        self.assertEqual(m.phase, 'watching')
        self.assertEqual(m.observe(4, 2.0 - 2e-9), event(new_best=True, recovered=True))

    def test_negative_or_tiny_settings_are_clamped(self):
        m = AutoStopMonitor(0, -1, 0)
        self.assertEqual((m.patience_steps, m.min_delta, m.watch_interval_steps), (1, 0.0, 1))

    def test_non_finite_loss_raises_without_changing_state(self):
        m = monitor()
        m.observe(1, 2.0)
        before = m.to_dict()
        for value in (math.nan, math.inf, -math.inf, 'nan'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Non-finite validation loss'):
                m.observe(2, value)
        self.assertEqual(m.to_dict(), before)


class PersistenceTests(unittest.TestCase):
    def test_snapshot_fields(self):
        m = monitor(patience=10, min_delta=0.1, interval=2)
        self.assertEqual(m.snapshot(), {
            'phase': 'normal', 'best_loss': None, 'best_step': None, 'valley_loss': None, 'valley_step': None,
            'watch_start_step': None, 'deadline_step': None, 'improvement': 0.0, 'min_delta': 0.1,
            'patience_steps': 10, 'watch_interval_steps': 2, 'grace_steps': 0})
        m.observe(10, 2.0)
        m.observe(20, 2.2)
        m.observe(22, 1.96)
        snapshot = m.snapshot()
        self.assertEqual({key: snapshot[key] for key in ('phase', 'best_step', 'valley_step', 'watch_start_step',
                                                         'deadline_step')},
                         {'phase': 'watching', 'best_step': 22, 'valley_step': 10, 'watch_start_step': 20,
                          'deadline_step': 30})
        self.assertAlmostEqual(snapshot['improvement'], 0.04)
        self.assertEqual((snapshot['best_loss'], snapshot['valley_loss']), (1.96, 2.0))
        m.observe(30, 2.5)
        stopped = m.snapshot()
        self.assertEqual((stopped['phase'], stopped['watch_start_step'], stopped['deadline_step']), ('stopped', 20, 30))
        json.dumps(stopped)  # JSON-safe for the GUI snapshot

    def test_to_dict_restore_roundtrip_in_watching(self):
        m = monitor(patience=10, min_delta=0.1, interval=3)
        m.observe(10, 2.0)
        m.observe(20, 2.2)
        m.observe(23, 1.97)
        data = json.loads(json.dumps(m.to_dict()))
        self.assertEqual(data['version'], 1)
        self.assertEqual(data['fingerprint'], FINGERPRINT)
        restored = monitor(patience=10, min_delta=0.1, interval=3)
        self.assertTrue(restored.restore(data))
        self.assertEqual(restored.to_dict(), m.to_dict())
        self.assertEqual(restored.snapshot(), m.snapshot())
        self.assertEqual([s for s in range(20, 31) if restored.extra_validation_due(s)], [23, 26, 29, 30])
        # The restored monitor continues identically.
        self.assertEqual(restored.observe(30, 2.1), m.observe(30, 2.1))
        self.assertEqual(restored.phase, 'stopped')

    def test_roundtrip_in_normal_phase_and_empty_state(self):
        fresh = monitor()
        other = monitor()
        self.assertTrue(other.restore(fresh.to_dict()))
        self.assertEqual(other.to_dict(), fresh.to_dict())
        fresh.observe(5, 1.0)
        self.assertTrue(other.restore(fresh.to_dict()))
        self.assertEqual((other.phase, other.valley_step, other.best_loss), ('normal', 5, 1.0))

    def test_restore_rejects_fingerprint_mismatch_and_bad_version(self):
        source = monitor()
        source.observe(5, 1.0)
        data = source.to_dict()
        for label, bad in [('fingerprint', {**data, 'fingerprint': 'e' * 64}), ('version', {**data, 'version': 2}),
                           ('no version', {key: value for key, value in data.items() if key != 'version'}),
                           ('not a dict', [data]), ('none', None)]:
            with self.subTest(label):
                target = monitor()
                self.assertFalse(target.restore(bad))
                self.assertEqual(target.to_dict(), monitor().to_dict())

    def test_monitor_without_fingerprint_accepts_any_saved_fingerprint(self):
        source = monitor()
        source.observe(5, 1.0)
        target = monitor(fingerprint=None)
        self.assertTrue(target.restore(source.to_dict()))
        self.assertEqual(target.valley_step, 5)

    def test_stopped_phase_restores_as_normal_from_its_valley(self):
        m = monitor(patience=4, min_delta=0.1, interval=2)
        m.observe(10, 2.0)
        m.observe(12, 2.05)
        m.observe(16, 2.1)
        self.assertEqual(m.phase, 'stopped')
        restored = monitor(patience=4, min_delta=0.1, interval=2)
        self.assertTrue(restored.restore(m.to_dict()))
        self.assertEqual(restored.phase, 'normal')
        self.assertIsNone(restored.watch_start_step)
        self.assertIsNone(restored.watch_best_loss)
        self.assertEqual((restored.valley_step, restored.valley_loss, restored.best_step), (10, 2.0, 10))
        self.assertFalse(restored.extra_validation_due(18))
        # Training continues: a significant improvement keeps it in normal.
        self.assertEqual(restored.observe(20, 1.8), event(new_best=True))

    def test_inconsistent_watching_state_restores_as_normal(self):
        m = monitor()
        data = {**m.to_dict(), 'phase': 'watching', 'valley_loss': 2.0, 'valley_step': 3, 'watch_start_step': None}
        self.assertTrue(m.restore(data))
        self.assertEqual(m.phase, 'normal')


class ConfigTests(unittest.TestCase):
    def test_monitor_from_config(self):
        self.assertIsNone(monitor_from_config({}))
        self.assertIsNone(monitor_from_config({'auto_stop': None}))
        self.assertIsNone(monitor_from_config({'auto_stop': {}}))
        self.assertIsNone(monitor_from_config(None))
        m = monitor_from_config({'auto_stop': {'patience_steps': 200, 'min_delta': 0.005, 'watch_interval_steps': 25}},
                                FINGERPRINT)
        self.assertIsInstance(m, AutoStopMonitor)
        self.assertEqual((m.patience_steps, m.min_delta, m.watch_interval_steps, m.fingerprint, m.phase),
                         (200, 0.005, 25, FINGERPRINT, 'normal'))
        self.assertEqual(m.grace_steps, 0)  # configs saved before the grace period existed
        m = monitor_from_config({'auto_stop': {'patience_steps': 200, 'min_delta': 0.005, 'watch_interval_steps': 25,
                                               'grace_steps': 150}})
        self.assertEqual((m.grace_steps, m.snapshot()['grace_steps']), (150, 150))


class GracePeriodTests(unittest.TestCase):
    def grace(self, grace=40, patience=20, min_delta=0.1, interval=10):
        return AutoStopMonitor(patience, min_delta, interval, FINGERPRINT, grace_steps=grace)

    def test_plateau_and_rise_inside_grace_never_watch_but_track_best_and_valley(self):
        m = self.grace(grace=40)
        self.assertEqual(m.observe(10, 6.04), event(new_best=True))
        self.assertEqual(m.observe(20, 6.04), NO_EVENT)          # warmup plateau
        self.assertEqual(m.observe(30, 6.10), NO_EVENT)          # a rise
        self.assertEqual(m.observe(40, 6.03), event(new_best=True))  # boundary step is still exempt
        self.assertEqual(m.phase, 'normal')
        self.assertEqual((m.best_step, m.valley_step, m.valley_loss), (40, 10, 6.04))
        self.assertFalse(any(m.extra_validation_due(step) for step in range(1, 60)))
        self.assertTrue(m.in_grace(40))
        self.assertFalse(m.in_grace(41))

    def test_significant_gain_inside_grace_moves_valley(self):
        m = self.grace(grace=40)
        m.observe(10, 6.0)
        m.observe(20, 5.5)
        self.assertEqual((m.valley_step, m.valley_loss), (20, 5.5))

    def test_first_validation_after_grace_follows_the_normal_rules(self):
        m = self.grace(grace=40, patience=20)
        m.observe(10, 6.0)
        m.observe(40, 6.0)
        self.assertEqual(m.observe(50, 6.0), event(entered_watch=True))
        self.assertEqual((m.watch_start_step, m.deadline_step), (50, 70))
        m2 = self.grace(grace=40)
        m2.observe(10, 6.0)
        self.assertEqual(m2.observe(50, 5.0), event(new_best=True))
        self.assertEqual((m2.phase, m2.valley_step), ('normal', 50))

    def test_restored_watch_inside_grace_is_cancelled(self):
        source = AutoStopMonitor(20, 0.1, 10, FINGERPRINT)
        source.observe(10, 6.0)
        source.observe(20, 6.0)
        self.assertEqual(source.phase, 'watching')
        target = self.grace(grace=100)
        self.assertTrue(target.restore(source.to_dict()))
        self.assertEqual(target.observe(30, 6.0), NO_EVENT)
        self.assertEqual((target.phase, target.watch_start_step, target.deadline_step), ('normal', None, None))
        self.assertEqual(target.observe(40, 6.0), NO_EVENT)  # still exempt, no stop at the old deadline

    def test_zero_grace_matches_the_original_behaviour(self):
        plain, zero = AutoStopMonitor(20, 0.1, 10, FINGERPRINT), self.grace(grace=0)
        for step, loss in [(10, 6.0), (20, 6.0), (30, 5.95), (40, 6.1)]:
            self.assertEqual(plain.observe(step, loss), zero.observe(step, loss))
        self.assertEqual(plain.to_dict(), zero.to_dict())


class ManagerMessageTests(unittest.TestCase):
    def run_manager(self, result):
        from indextts_training.manager import TrainingManager
        manager = TrainingManager()
        manager.train_fn = lambda config, callback, stop_event: result
        manager._start_locked({})
        manager.thread.join(10)
        self.assertFalse(manager.thread.is_alive())
        return manager.snapshot()

    def test_auto_stop_finish_message_and_state(self):
        snapshot = {'phase': 'stopped', 'best_step': 300, 'best_loss': 1.25}
        state = self.run_manager({'step': 450, 'total': 1000, 'stopped': False, 'early_stopped': False,
                                  'auto_stopped': True, 'min_delta': 0.005, 'patience_steps': 150,
                                  'best_loss': 1.25, 'best_step': 300, 'auto_stop': snapshot, 'output': 'x'})
        self.assertFalse(state['running'] or state['failed'])
        self.assertTrue(state['auto_stopped'])
        self.assertEqual(state['auto_stop'], snapshot)
        self.assertEqual(state['message'],
                         'Training stopped automatically at step 450: no validation improvement of at least 0.005 '
                         'within 150 steps. Best validation loss 1.250000 at step 300.')

    def test_regular_finish_resets_auto_stop_flags(self):
        state = self.run_manager({'step': 10, 'total': 10, 'stopped': False, 'early_stopped': False})
        self.assertEqual(state['message'], 'Training complete.')
        self.assertFalse(state['auto_stopped'])
        self.assertIsNone(state['auto_stop'])


if __name__ == '__main__':
    unittest.main()

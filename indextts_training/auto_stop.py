"""Validation-loss based automatic stop for LoRA training (pure Python).

State machine
-------------
``normal``
    Every validation is compared with the confirmed valley (the bottom of the
    last "U"). A *significant improvement* (``valley - loss >= min_delta`` and
    ``> 0``) moves the valley to the new point. Anything else - a rise, a
    plateau or a tiny improvement - starts ``watching``.
``watching``
    The valley stays fixed. Extra validations run every
    ``watch_interval_steps`` counted from the step the watch started, plus one
    at the patience deadline. A significant improvement over the valley moves
    the valley, ends the watch and restores normal validation. Reaching
    ``watch_start_step + patience_steps`` while still watching stops training.

``best_loss`` is tracked independently and updates on any improvement; it is
used to keep the best weights, while ``valley_loss`` is the patience reference.
"""
from __future__ import annotations

import math

NORMAL = 'normal'
WATCHING = 'watching'
STOPPED = 'stopped'


class AutoStopMonitor:
    def __init__(self, patience_steps, min_delta, watch_interval_steps, fingerprint=None):
        self.patience_steps = max(1, int(patience_steps))
        self.min_delta = max(0.0, float(min_delta))
        self.watch_interval_steps = max(1, int(watch_interval_steps))
        self.fingerprint = fingerprint
        self.phase = NORMAL
        self.best_loss = None
        self.best_step = None
        self.valley_loss = None
        self.valley_step = None
        self.watch_start_step = None
        self.watch_best_loss = None

    # ------------------------------------------------------------------ queries
    @property
    def deadline_step(self):
        if self.phase != WATCHING or self.watch_start_step is None:
            return None
        return self.watch_start_step + self.patience_steps

    @property
    def improvement(self):
        """Improvement over the valley reached during the current watch (never negative)."""
        if self.phase != WATCHING or self.valley_loss is None or self.watch_best_loss is None:
            return 0.0
        return max(0.0, self.valley_loss - self.watch_best_loss)

    def _significant(self, loss):
        gain = self.valley_loss - loss
        return gain > 0 and gain >= self.min_delta

    def extra_validation_due(self, step):
        """Additional validation steps while watching (union with the normal schedule)."""
        if self.phase != WATCHING or self.watch_start_step is None:
            return False
        offset = step - self.watch_start_step
        return offset > 0 and (offset % self.watch_interval_steps == 0 or step == self.deadline_step)

    # ------------------------------------------------------------------ update
    def observe(self, step, loss):
        step = int(step)
        loss = float(loss)
        if not math.isfinite(loss):
            raise ValueError(f'Non-finite validation loss: {loss}')
        event = {'new_best': False, 'entered_watch': False, 'recovered': False, 'stop': False}
        if self.phase == STOPPED:
            return event
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss, self.best_step = loss, step
            event['new_best'] = True
        if self.valley_loss is None:
            self.valley_loss, self.valley_step = loss, step
            return event
        if self.phase == NORMAL:
            if self._significant(loss):
                self.valley_loss, self.valley_step = loss, step
            else:
                self.phase = WATCHING
                self.watch_start_step = step
                self.watch_best_loss = loss
                event['entered_watch'] = True
            return event
        # watching
        if self.watch_best_loss is None or loss < self.watch_best_loss:
            self.watch_best_loss = loss
        if self._significant(loss):
            self.valley_loss, self.valley_step = loss, step
            self.phase = NORMAL
            self.watch_start_step = None
            self.watch_best_loss = None
            event['recovered'] = True
        elif step >= self.deadline_step:
            self.phase = STOPPED
            event['stop'] = True
        return event

    # ------------------------------------------------------------- persistence
    def snapshot(self):
        return {
            'phase': self.phase, 'best_loss': self.best_loss, 'best_step': self.best_step,
            'valley_loss': self.valley_loss, 'valley_step': self.valley_step,
            'watch_start_step': self.watch_start_step if self.phase != NORMAL else None,
            'deadline_step': self.watch_start_step + self.patience_steps
            if self.phase != NORMAL and self.watch_start_step is not None else None,
            'improvement': self.improvement, 'min_delta': self.min_delta,
            'patience_steps': self.patience_steps, 'watch_interval_steps': self.watch_interval_steps,
        }

    def to_dict(self):
        return {
            'version': 1, 'fingerprint': self.fingerprint, 'phase': self.phase,
            'best_loss': self.best_loss, 'best_step': self.best_step,
            'valley_loss': self.valley_loss, 'valley_step': self.valley_step,
            'watch_start_step': self.watch_start_step, 'watch_best_loss': self.watch_best_loss,
        }

    def restore(self, data):
        """Restore saved state; returns False (and keeps a fresh state) when it does not apply."""
        if not isinstance(data, dict) or data.get('version') != 1:
            return False
        if self.fingerprint is not None and data.get('fingerprint') != self.fingerprint:
            return False
        phase = data.get('phase')
        if phase not in (NORMAL, WATCHING):
            # A stopped run resumes as a watch that already passed its deadline would
            # stop again immediately; start the comparison afresh from its valley instead.
            phase = NORMAL
        self.phase = phase
        for key in ('best_loss', 'valley_loss', 'watch_best_loss'):
            value = data.get(key)
            setattr(self, key, float(value) if value is not None else None)
        for key in ('best_step', 'valley_step', 'watch_start_step'):
            value = data.get(key)
            setattr(self, key, int(value) if value is not None else None)
        if self.phase == NORMAL:
            self.watch_start_step = None
            self.watch_best_loss = None
        elif self.watch_start_step is None or self.valley_loss is None:
            self.phase = NORMAL
        return True


def monitor_from_config(config, fingerprint=None):
    """Build a monitor from ``config['auto_stop']`` or return None when disabled."""
    settings = config.get('auto_stop') if isinstance(config, dict) else None
    if not settings:
        return None
    return AutoStopMonitor(settings['patience_steps'], settings['min_delta'],
                           settings['watch_interval_steps'], fingerprint)

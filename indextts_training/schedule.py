"""Pure step-scheduling rules shared by the real and the fake LoRA trainer.

``end_step`` is the step at which the run finishes. It equals the planned
learning-rate schedule length unless early stop is enabled, in which case the
schedule keeps its planned horizon and only the loop ends sooner.
"""
from __future__ import annotations

import json
from pathlib import Path

ADAPTER_STATE_FILE = 'adapter_state.json'


def write_adapter_state(directory, step, schedule_total, end_step, stopped):
    """Small sidecar so listings learn the step without loading trainer_state.pt."""
    step, schedule_total, end_step = int(step), int(schedule_total), int(end_step)
    data = {'step': step, 'schedule_total': schedule_total, 'end_step': end_step, 'stopped': bool(stopped),
            'early_stopped': bool(not stopped and end_step < schedule_total and step == end_step)}
    Path(directory, ADAPTER_STATE_FILE).write_text(json.dumps(data, indent=2), encoding='utf-8')


def read_adapter_state(directory):
    try:
        data = json.loads(Path(directory, ADAPTER_STATE_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get('step'), int) else None


def normalize_dense(dense):
    """Return ``(start, end, interval)`` for an enabled dense range, else None."""
    if not dense:
        return None
    try:
        start = int(dense['start_step'])
        end = int(dense['end_step'])
        interval = int(dense['interval_steps'])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 1 or end < start or interval < 1:
        return None
    return start, end, interval


def checkpoint_due(step, save_every, end_step):
    return step % max(1, int(save_every)) == 0 or step == end_step


def validation_due(step, *, end_step, save_every, interval=None, dense=None):
    """Validate at the end step, every global interval and inside the dense range.

    ``interval`` of ``None``/``0`` follows the checkpoint frequency. Dense steps
    are counted from the range start, so the start step itself is validated.
    """
    if step == end_step:
        return True
    if step % max(1, int(interval or save_every)) == 0:
        return True
    window = normalize_dense(dense)
    if window is not None:
        start, end, every = window
        if start <= step <= end and (step - start) % every == 0:
            return True
    return False

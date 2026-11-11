"""Post-hoc checkpoint validation without CUDA, PEFT or IndexTTS weights."""

from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import indextts_training
from indextts_training import posthoc


class Wrapped:
    """Minimal PeftModel stand-in that records adapter management calls."""

    def __init__(self, base, path, adapter_name, events):
        self.base = base
        self.events = events
        self.adapters = {adapter_name: path}
        self.active = adapter_name
        self.disabled = False

    def load_adapter(self, path, adapter_name, is_trainable):
        self.events.append(('load_adapter', path, adapter_name, is_trainable))
        self.adapters[adapter_name] = path

    def set_adapter(self, name):
        self.events.append(('set_adapter', name))
        assert name in self.adapters
        self.active = name

    def delete_adapter(self, name):
        self.events.append(('delete_adapter', name))
        assert name != self.active
        del self.adapters[name]

    @contextmanager
    def disable_adapter(self):
        self.events.append(('disable_adapter',))
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


@pytest.fixture
def env(tmp_path, monkeypatch):
    events = []
    base = SimpleNamespace(name='base')
    tts = SimpleNamespace(gpt=SimpleNamespace(gpt=base))
    tts_factory = Mock(return_value=tts)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=Mock(return_value=True), manual_seed_all=Mock(), empty_cache=Mock()),
        manual_seed=Mock(),
    )

    def from_pretrained(model, path, adapter_name, is_trainable):
        assert model is base
        events.append(('from_pretrained', path, adapter_name, is_trainable))
        return Wrapped(model, path, adapter_name, events)

    losses = {}
    failures = set()
    stop_during = set()
    cancel_state = {'set': False}

    def validation_loss(tts_arg, rows, device, step, stop, on_sample=None):
        assert tts_arg is tts and device == 'cuda:0'
        model = tts.gpt.gpt
        if model is base:
            label = 'base'
        elif model.disabled:
            label = 'disabled'
        else:
            label = model.adapters[model.active]
        events.append(('validate', label, step))
        for index, _ in enumerate(rows):
            if label in stop_during and index == 1:
                cancel_state['set'] = True  # the user cancels mid-target
            if stop.is_set():
                return None
            if on_sample is not None:
                on_sample(index, len(rows))
        if label in failures:
            raise RuntimeError(f'Validation failed at step {step} for clip.wav: boom')
        return losses.get(label, 1.0)

    fake_trainer = ModuleType('indextts_training.trainer')
    fake_trainer._freeze_index_tts_modules = Mock(return_value=[])
    fake_trainer._validation_loss = validation_loss
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=from_pretrained)))
    monkeypatch.setitem(sys.modules, 'indextts.infer_v2_5', SimpleNamespace(IndexTTS2=tts_factory))
    monkeypatch.setitem(sys.modules, 'indextts_training.trainer', fake_trainer)
    monkeypatch.setattr(indextts_training, 'trainer', fake_trainer, raising=False)
    (tmp_path / 'config.yaml').touch()
    rows = [{'audio': 'a.wav'}, {'audio': 'b.wav'}]
    results = []
    progress = []

    def run(targets, cancelled=lambda: False, **kwargs):
        return posthoc.evaluate_checkpoints(
            tmp_path, targets, rows, on_progress=lambda *args: progress.append(args),
            on_result=lambda target, loss: results.append((target['name'], loss)), cancelled=cancelled, **kwargs)

    return SimpleNamespace(events=events, base=base, tts=tts, tts_factory=tts_factory, torch=torch, losses=losses,
                           failures=failures, stop_during=stop_during, cancel_state=cancel_state, trainer=fake_trainer, rows=rows,
                           results=results, progress=progress, run=run, model_dir=tmp_path)


def target(name, path, step):
    return {'name': name, 'path': path, 'step': step}


BASE = target('base_model', None, 0)
FIRST = target('checkpoint-step-000100', 'ckpt100', 100)
SECOND = target('checkpoint-step-000200', 'ckpt200', 200)
FINAL = target('final_adapter', 'final', 300)


def test_baseline_then_adapters_swap_and_delete_previous(env):
    env.losses.update(base=2.5, ckpt100=1.5, ckpt200=1.25, final=1.0)
    outcome = env.run([BASE, FIRST, SECOND, FINAL])
    assert outcome == {'completed': ['base_model', 'checkpoint-step-000100', 'checkpoint-step-000200',
                                     'final_adapter'], 'stopped': False}
    assert env.events == [
        ('validate', 'base', 0),
        ('from_pretrained', 'ckpt100', 'posthoc_1', False),
        ('validate', 'ckpt100', 100),
        ('load_adapter', 'ckpt200', 'posthoc_2', False),
        ('set_adapter', 'posthoc_2'),
        ('delete_adapter', 'posthoc_1'),
        ('validate', 'ckpt200', 200),
        ('load_adapter', 'final', 'posthoc_3', False),
        ('set_adapter', 'posthoc_3'),
        ('delete_adapter', 'posthoc_2'),
        ('validate', 'final', 300),
    ]
    assert env.results == [('base_model', 2.5), ('checkpoint-step-000100', 1.5),
                           ('checkpoint-step-000200', 1.25), ('final_adapter', 1.0)]
    assert all(isinstance(loss, float) for _, loss in env.results)
    assert list(env.tts.gpt.gpt.adapters) == ['posthoc_3']
    env.trainer._freeze_index_tts_modules.assert_called_once_with(env.tts)
    kwargs = env.tts_factory.call_args.kwargs
    assert kwargs['device'] == 'cuda:0' and kwargs['use_bf16'] is True
    assert kwargs['cfg_path'] == str(env.model_dir / 'config.yaml') and kwargs['model_dir'] == str(env.model_dir)
    assert not any(kwargs[key] for key in ('use_cuda_kernel', 'use_gpt_latent', 'use_torch_compile',
                                           'use_deepspeed', 'use_accel', 'use_qwen_emo'))
    env.torch.manual_seed.assert_called_once_with(1234)
    env.torch.cuda.manual_seed_all.assert_called_once_with(1234)
    env.torch.cuda.empty_cache.assert_called()


def test_baseline_after_adapter_disables_it_temporarily(env):
    env.losses.update(disabled=2.75, ckpt100=1.5, ckpt200=1.25)
    outcome = env.run([FIRST, BASE, SECOND], seed=7)
    assert outcome['completed'] == ['checkpoint-step-000100', 'base_model', 'checkpoint-step-000200']
    assert env.events == [
        ('from_pretrained', 'ckpt100', 'posthoc_0', False),
        ('validate', 'ckpt100', 100),
        ('disable_adapter',),
        ('validate', 'disabled', 0),
        ('load_adapter', 'ckpt200', 'posthoc_2', False),
        ('set_adapter', 'posthoc_2'),
        ('delete_adapter', 'posthoc_0'),
        ('validate', 'ckpt200', 200),
    ]
    assert env.results == [('checkpoint-step-000100', 1.5), ('base_model', 2.75), ('checkpoint-step-000200', 1.25)]
    assert env.tts.gpt.gpt.disabled is False
    env.torch.manual_seed.assert_called_once_with(7)


def test_progress_reports_samples_and_target_completion(env):
    env.run([BASE, FIRST])
    assert [(done, total, item['name'], sample, count) for done, total, item, sample, count in env.progress] == [
        (0, 4, 'base_model', 0, 2), (1, 4, 'base_model', 1, 2), (2, 4, 'base_model', 2, 2),
        (2, 4, 'checkpoint-step-000100', 0, 2), (3, 4, 'checkpoint-step-000100', 1, 2),
        (4, 4, 'checkpoint-step-000100', 2, 2),
    ]


def test_cancel_between_targets_keeps_completed(env):
    outcome = env.run([FIRST, SECOND, FINAL], cancelled=lambda: len(env.results) >= 1)
    assert outcome == {'completed': ['checkpoint-step-000100'], 'stopped': True}
    assert env.results == [('checkpoint-step-000100', 1.0)]
    assert [event for event in env.events if event[0] == 'validate'] == [('validate', 'ckpt100', 100)]
    env.torch.cuda.empty_cache.assert_called()


def test_cancel_during_validation_discards_partial_target(env):
    env.stop_during.add('ckpt200')
    outcome = env.run([FIRST, SECOND, FINAL], cancelled=lambda: env.cancel_state['set'])
    assert outcome == {'completed': ['checkpoint-step-000100'], 'stopped': True}
    assert env.results == [('checkpoint-step-000100', 1.0)]
    assert ('validate', 'final', 300) not in env.events


def test_cancel_before_start_runs_nothing(env):
    outcome = env.run([BASE, FIRST], cancelled=lambda: True)
    assert outcome == {'completed': [], 'stopped': True}
    assert env.events == [] and env.results == []


def test_error_propagates_after_earlier_results(env):
    env.failures.add('ckpt200')
    with pytest.raises(RuntimeError, match='Validation failed at step 200'):
        env.run([BASE, FIRST, SECOND, FINAL])
    assert env.results == [('base_model', 1.0), ('checkpoint-step-000100', 1.0)]
    assert ('validate', 'final', 300) not in env.events
    env.torch.cuda.empty_cache.assert_called()


def test_load_error_propagates(env):
    def broken(*_args, **_kwargs):
        raise OSError('adapter_model.safetensors is corrupt')
    sys.modules['peft'].PeftModel.from_pretrained = broken
    with pytest.raises(OSError, match='corrupt'):
        env.run([BASE, FIRST])
    assert env.results == [('base_model', 1.0)]


def test_empty_rows_error(env):
    with pytest.raises(RuntimeError, match='Validation manifest is empty'):
        posthoc.evaluate_checkpoints(env.model_dir, [FIRST], [])
    env.tts_factory.assert_not_called()


def test_cuda_and_model_config_are_required(env, tmp_path):
    env.torch.cuda.is_available.return_value = False
    with pytest.raises(RuntimeError, match='requires an NVIDIA CUDA device'):
        env.run([FIRST])
    env.torch.cuda.is_available.return_value = True
    with pytest.raises(RuntimeError, match='model config not found'):
        posthoc.evaluate_checkpoints(tmp_path / 'missing', [FIRST], env.rows)
    env.tts_factory.assert_not_called()


def test_none_step_is_passed_as_zero(env):
    env.run([target('final_adapter', 'final', None)])
    assert env.events[-1] == ('validate', 'final', 0)

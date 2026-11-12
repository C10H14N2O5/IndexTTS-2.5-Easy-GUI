"""Exercise the optimizer loop and TensorBoard contract without CUDA or models."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from threading import Event
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest


TRAINER = Path(__file__).resolve().parents[1] / 'indextts_training/trainer.py'


class Loss(float):
    def __new__(cls, value, backward):
        loss = super().__new__(cls, value)
        loss.backward = backward
        return loss

    def __truediv__(self, divisor):
        return Loss(float(self) / divisor, self.backward)

    def detach(self):
        return self

    def float(self):
        return self

    def item(self):
        return float(self)


@pytest.fixture
def training(tmp_path, monkeypatch):
    flags = SimpleNamespace(grad=True, autocast=False)
    stop = Event()
    parameter = SimpleNamespace(requires_grad=True, numel=lambda: 1)
    adapter = SimpleNamespace(training=False, parameters=lambda: [parameter], save_pretrained=Mock())
    adapter.train = lambda mode=True: setattr(adapter, 'training', mode)
    adapter.eval = lambda: adapter.train(False)
    adapter.get_base_model = lambda: adapter
    tts = SimpleNamespace(gpt=SimpleNamespace(gpt=adapter))
    optimizer = Mock(param_groups=[{'lr': 0.001}])
    optimizer.state_dict.side_effect = lambda: {'steps': optimizer.step.call_count}
    scheduler = Mock()
    scheduler.state_dict.side_effect = lambda: {'steps': scheduler.step.call_count}
    writer = MagicMock()
    writer.__enter__.return_value = writer
    writer_factory = Mock(return_value=writer)
    bfloat16 = object()
    lr_lambdas = []

    def lambda_lr(_optimizer, lr_lambda):
        lr_lambdas.append(lr_lambda)
        return scheduler

    @contextmanager
    def no_grad():
        previous = flags.grad
        flags.grad = False
        try:
            yield
        finally:
            flags.grad = previous

    @contextmanager
    def autocast(device, dtype):
        assert device == 'cuda' and dtype is bfloat16
        flags.autocast = True
        try:
            yield
        finally:
            flags.autocast = False

    nn = ModuleType('torch.nn')
    nn.functional = ModuleType('torch.nn.functional')
    nn.utils = SimpleNamespace(clip_grad_norm_=lambda *_: 1.0)
    torch = SimpleNamespace(
        nn=nn, no_grad=no_grad, autocast=autocast, bfloat16=bfloat16,
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=Mock(), empty_cache=Mock()),
        manual_seed=Mock(), isfinite=math.isfinite,
        optim=SimpleNamespace(AdamW=lambda *_, **__: optimizer,
                              lr_scheduler=SimpleNamespace(LambdaLR=lambda_lr)),
        save=lambda state, path: path.write_text(json.dumps(state), encoding='utf-8'),
        load=lambda path, map_location: json.loads(path.read_text(encoding='utf-8')),
    )
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.nn', nn)
    monkeypatch.setitem(sys.modules, 'torch.nn.functional', nn.functional)
    monkeypatch.setitem(sys.modules, 'torchaudio', ModuleType('torchaudio'))
    monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', SimpleNamespace(SummaryWriter=writer_factory))
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(
        LoraConfig=Mock(), get_peft_model=lambda model, _: model,
        PeftModel=SimpleNamespace(from_pretrained=lambda *_, **__: adapter),
    ))
    monkeypatch.setitem(sys.modules, 'indextts.infer_v2_5', SimpleNamespace(IndexTTS2=lambda **_: tts))
    spec = importlib.util.spec_from_file_location('trainer_under_test', TRAINER)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    train_row = {'audio': 'training.wav', 'loss': 1.0}
    val_rows = [{'audio': 'first.wav', 'loss': 2.0}, {'audio': 'second.wav', 'loss': 4.0}]
    train_manifest = tmp_path / 'manifest.jsonl'
    train_manifest.write_text(json.dumps(train_row), encoding='utf-8')
    (tmp_path / 'config.yaml').touch()
    out = tmp_path / 'output'
    config = {
        'manifest': str(train_manifest), 'model_dir': str(tmp_path), 'output_dir': str(out),
        'batch': 1, 'grad_accum': 1, 'grad_clip': 1.0, 'save_every': 2,
        'mode': 'Steps', 'steps': 4, 'rank': 1, 'alpha': 1, 'dropout': 0.1, 'lr': 0.001,
        'validation': {'rows': val_rows, 'interval_steps': None, 'fingerprint': 'a' * 64,
                       'manifest': str(tmp_path / 'validation.jsonl')},
    }
    samples = []
    backward = Mock()
    # Optional per-step validation loss: every validation row returns this value.
    validation_losses = {}
    validation_step = {}

    def prepare(_tts, item, device, include_auxiliary_conditioning):
        assert device == 'cuda:0'
        assert not include_auxiliary_conditioning
        assert adapter.training == flags.grad == (item['audio'] == 'training.wav')
        if item.get('broken'):
            raise OSError('cannot decode clip')
        return (item,) + (None,) * 10

    def loss(_tts, item, *_):
        assert flags.autocast
        assert adapter.training == flags.grad == (item['audio'] == 'training.wav')
        samples.append((item['audio'], optimizer.step.call_count))
        if item.get('stop'):
            stop.set()
        value = item['loss']
        if item['audio'] != 'training.wav':
            value = validation_losses.get(validation_step.get('step'), value)
        value = Loss(value, backward)
        return value, value, None, None, None

    trainer._freeze_index_tts_modules = Mock(return_value=[])
    trainer._prepare_sample = prepare
    trainer._loss = loss
    trainer._generate_eval = Mock(return_value=None)
    validation_loss = trainer._validation_loss

    def tracked_validation_loss(_tts, _rows, _device, step, *args, **kwargs):
        validation_step['step'] = step  # global step, also after a resume
        return validation_loss(_tts, _rows, _device, step, *args, **kwargs)

    trainer._validation_loss = tracked_validation_loss
    callback = Mock()
    return SimpleNamespace(
        trainer=trainer, config=config, out=out, writer=writer, writer_factory=writer_factory,
        optimizer=optimizer, scheduler=scheduler, adapter=adapter, flags=flags, samples=samples,
        backward=backward, stop=stop, tts=tts, callback=callback, lr_lambdas=lr_lambdas,
        validation_losses=validation_losses,
        run=lambda: trainer.train_lora(config, callback, stop),
    )


def validation_points(writer):
    return [(step, value) for tag, value, step in
            (record.args for record in writer.add_scalar.call_args_list) if tag == 'validation/loss']


def test_checkpoints_and_final_validation_restore_training_mode(training):
    result = training.run()
    assert result['step'] == 4
    assert validation_points(training.writer) == [(2, 3.0), (4, 3.0)]
    assert training.samples == [
        ('training.wav', 0), ('training.wav', 1), ('first.wav', 2), ('second.wav', 2),
        ('training.wav', 2), ('training.wav', 3), ('first.wav', 4), ('second.wav', 4),
    ]
    assert training.backward.call_count == training.optimizer.step.call_count == training.scheduler.step.call_count == 4
    assert training.adapter.training and training.flags.grad
    assert [entry.args[2] for entry in training.trainer._generate_eval.call_args_list] == [2, 4]
    state = json.loads((training.out / 'checkpoint-step-000004/trainer_state.pt').read_text(encoding='utf-8'))
    assert state['config']['validation'] == {
        key: value for key, value in training.config['validation'].items() if key != 'rows'
    }


def test_custom_interval_is_independent_of_checkpoint_frequency(training):
    training.config.update(save_every=3, steps=5)
    training.config['validation']['interval_steps'] = 2
    training.run()
    assert validation_points(training.writer) == [(2, 3.0), (4, 3.0), (5, 3.0)]
    assert [entry.args[2] for entry in training.trainer._generate_eval.call_args_list] == [3, 5]
    assert not (training.out / 'checkpoint-step-000002').exists()


def test_resume_uses_global_steps_and_purges_replayed_events(training, tmp_path):
    resume = tmp_path / 'checkpoint-step-000003'
    resume.mkdir()
    state = {'step': 3, 'optimizer': {'steps': 3}, 'scheduler': {'steps': 3}}
    (resume / 'trainer_state.pt').write_text(json.dumps(state), encoding='utf-8')
    training.config.update(resume=str(resume), steps=6)
    training.config['validation']['interval_steps'] = 2
    training.run()
    assert validation_points(training.writer) == [(4, 3.0), (6, 3.0)]
    training.writer_factory.assert_called_once_with(str(training.out / 'tensorboard'), purge_step=4)
    assert training.optimizer.load_state_dict.call_args == call(state['optimizer'])
    assert training.scheduler.load_state_dict.call_args == call(state['scheduler'])


def test_disabled_validation_preserves_training(training):
    training.config.pop('validation')
    assert training.run()['step'] == 4
    assert validation_points(training.writer) == []
    training.writer.add_custom_scalars.assert_not_called()
    training.writer.add_text.assert_not_called()


def test_training_manifest_must_match_preflight_snapshot(training):
    manifest = Path(training.config['manifest'])
    training.config['validation']['training_fingerprint'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.write_text('{}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Training manifest changed after validation preflight'):
        training.run()
    training.optimizer.step.assert_not_called()
    training.writer_factory.assert_not_called()


@pytest.mark.parametrize(('interval', 'row_change', 'message'), [
    (1, {'broken': True}, 'cannot decode clip'),
    (None, {'loss': float('nan')}, 'Non-finite validation loss'),
])
def test_failed_validation_preserves_resumable_checkpoint(training, interval, row_change, message):
    training.config['validation']['interval_steps'] = interval
    training.config['validation']['rows'][1].update(row_change)
    failed_step = interval or training.config['save_every']
    with pytest.raises(RuntimeError, match=rf'Validation failed at step {failed_step} for second\.wav: {message}'):
        training.run()
    path = training.out / f'checkpoint-step-{failed_step:06d}'
    state = json.loads((path / 'trainer_state.pt').read_text(encoding='utf-8'))
    assert state['step'] == state['optimizer']['steps'] == state['scheduler']['steps'] == failed_step
    assert validation_points(training.writer) == []
    assert training.adapter.training and training.flags.grad
    training.adapter.save_pretrained.assert_called_once_with(str(path), safe_serialization=True)
    training.writer.__exit__.assert_called_once()


def test_stop_during_validation_discards_partial_round(training):
    training.config['validation']['interval_steps'] = 1
    training.config['validation']['rows'][0]['stop'] = True
    result = training.run()
    assert result['stopped'] and result['step'] == 1
    assert (training.out / 'final_adapter/trainer_state.pt').is_file()
    assert validation_points(training.writer) == []
    assert training.samples == [('training.wav', 0), ('first.wav', 1)]
    assert training.adapter.training and training.flags.grad


def test_tensorboard_event_log_contains_validation_and_overlay(training, tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
    from tensorboard.plugins.custom_scalar import layout_pb2, metadata
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    logdir = str(tmp_path / 'events')
    events = EventFileWriter(logdir)

    def record(value, step):
        events.add_event(event_pb2.Event(
            step=step, wall_time=time.time(), summary=summary_pb2.Summary(value=[value]),
        ))

    def record_layout(config):
        layout = layout_pb2.Layout()
        for title, charts in config.items():
            category = layout.category.add(title=title)
            for label, (kind, tags) in charts.items():
                assert kind == 'Multiline'
                category.chart.add(title=label).multiline.tag.extend(tags)
        record(summary_pb2.Summary.Value(
            tag=metadata.CONFIG_SUMMARY_TAG, metadata=metadata.create_summary_metadata(),
            tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,
                                          string_val=[layout.SerializeToString()]),
        ), 0)

    # Use TensorBoard's lightweight event writer behind the mocked torch API.
    training.writer.add_custom_scalars.side_effect = record_layout
    training.writer.add_scalar.side_effect = lambda tag, value, step: record(
        summary_pb2.Summary.Value(tag=tag, simple_value=value), step,
    )
    try:
        training.run()
    finally:
        events.close()

    log = EventAccumulator(logdir).Reload()
    assert [(point.step, point.value) for point in log.Scalars('validation/loss')] == [(2, 3.0), (4, 3.0)]
    assert [point.step for point in log.Scalars('train/loss_ema')] == [1, 2, 3, 4]
    layout = layout_pb2.Layout.FromString(log.Tensors(metadata.CONFIG_SUMMARY_TAG)[0].tensor_proto.string_val[0])
    assert list(layout.category[0].chart[0].multiline.tag) == ['train/loss_ema', 'validation/loss']
    tag, dataset, step = training.writer.add_text.call_args.args
    assert tag == 'validation/dataset' and step == 0
    assert training.config['validation']['fingerprint'] in dataset and 'Samples: 2' in dataset


def adapter_state(path):
    return json.loads((path / 'adapter_state.json').read_text(encoding='utf-8'))


def callback_progress(training):
    return [entry.kwargs for entry in training.callback.call_args_list if 'step' in entry.kwargs]


def test_planned_run_reports_schedule_and_writes_sidecars(training):
    result = training.run()
    assert (result['step'], result['total'], result['schedule_total'], result['end_step']) == (4, 4, 4, 4)
    assert not result['early_stopped'] and not result['stopped']
    for name, step in [('checkpoint-step-000002', 2), ('checkpoint-step-000004', 4), ('final_adapter', 4)]:
        assert adapter_state(training.out / name) == {
            'step': step, 'schedule_total': 4, 'end_step': 4, 'stopped': False, 'early_stopped': False}
    progress = callback_progress(training)
    assert [(item['step'], item['total'], item['schedule_total']) for item in progress] == [
        (1, 4, 4), (2, 4, 4), (3, 4, 4), (4, 4, 4)]
    assert all('LR schedule' not in item['message'] for item in progress)


def test_early_stop_ends_loop_but_keeps_planned_lr_schedule(training):
    training.config.update(steps=400, end_step=5)
    result = training.run()
    assert result['step'] == 5 and result['end_step'] == 5
    assert result['total'] == result['schedule_total'] == 400
    assert result['early_stopped'] and not result['stopped']
    assert training.optimizer.step.call_count == training.scheduler.step.call_count == 5
    # Checkpoint, validation and evaluation are forced at the off-grid end step.
    assert validation_points(training.writer) == [(2, 3.0), (4, 3.0), (5, 3.0)]
    assert [entry.args[2] for entry in training.trainer._generate_eval.call_args_list] == [2, 4, 5]
    assert sorted(path.name for path in training.out.glob('checkpoint-step-*')) == [
        'checkpoint-step-000002', 'checkpoint-step-000004', 'checkpoint-step-000005']
    assert not (training.out / 'checkpoint-step-000006').exists()
    state = json.loads((training.out / 'checkpoint-step-000005/trainer_state.pt').read_text(encoding='utf-8'))
    assert state['step'] == state['optimizer']['steps'] == state['scheduler']['steps'] == 5
    assert state['config']['end_step'] == 5
    assert (training.out / 'final_adapter/trainer_state.pt').is_file()
    assert adapter_state(training.out / 'checkpoint-step-000004') == {
        'step': 4, 'schedule_total': 400, 'end_step': 5, 'stopped': False, 'early_stopped': False}
    for name in ['checkpoint-step-000005', 'final_adapter']:
        assert adapter_state(training.out / name) == {
            'step': 5, 'schedule_total': 400, 'end_step': 5, 'stopped': False, 'early_stopped': True}
    progress = callback_progress(training)
    assert [(item['step'], item['total'], item['schedule_total']) for item in progress] == [
        (step, 5, 400) for step in range(1, 6)]
    assert progress[-1]['message'].startswith('Training step 5/5 (LR schedule 400)')
    # The LR lambda keeps the planned 400-step horizon: not decayed at the stop step.
    [lr_lambda] = training.lr_lambdas
    warm = 20
    assert lr_lambda(0) == pytest.approx(1 / warm)
    assert lr_lambda(4) == pytest.approx(5 / warm)
    assert lr_lambda(warm) == pytest.approx(1.0)
    assert lr_lambda(100) > 0.9
    assert lr_lambda(210) == pytest.approx(0.55, abs=0.01)
    assert lr_lambda(400) == pytest.approx(0.1)
    assert lr_lambda(10_000) == pytest.approx(0.1)


def test_early_stop_at_planned_total_is_a_normal_finish(training):
    training.config['end_step'] = 4
    result = training.run()
    assert result['step'] == 4 and not result['early_stopped']
    assert adapter_state(training.out / 'final_adapter')['early_stopped'] is False


def test_end_step_zero_or_none_means_planned_total(training):
    training.config['end_step'] = 0
    result = training.run()
    assert (result['step'], result['end_step'], result['early_stopped']) == (4, 4, False)


@pytest.mark.parametrize('end_step', [5, -1, 1000])
def test_invalid_end_step_raises_before_training(training, end_step):
    training.config['end_step'] = end_step
    with pytest.raises(RuntimeError, match=rf'Early stop step {end_step} must be between 1 and the planned 4 steps'):
        training.run()
    training.optimizer.step.assert_not_called()
    training.writer_factory.assert_not_called()


def make_resume(tmp_path, step):
    resume = tmp_path / f'checkpoint-step-{step:06d}'
    resume.mkdir()
    state = {'step': step, 'optimizer': {'steps': step}, 'scheduler': {'steps': step}}
    (resume / 'trainer_state.pt').write_text(json.dumps(state), encoding='utf-8')
    return resume


@pytest.mark.parametrize('end_step', [2, 3])
def test_resume_at_or_beyond_end_step_raises(training, tmp_path, end_step):
    training.config.update(resume=str(make_resume(tmp_path, 3)), steps=6, end_step=end_step)
    with pytest.raises(RuntimeError, match=rf'already at step 3, at or beyond the end step {end_step}'):
        training.run()
    training.optimizer.step.assert_not_called()
    training.writer_factory.assert_not_called()
    assert not (training.out / 'final_adapter').exists()


def test_resume_at_planned_total_raises(training, tmp_path):
    training.config.update(resume=str(make_resume(tmp_path, 4)), steps=4)
    with pytest.raises(RuntimeError, match='already at step 4, at or beyond the end step 4'):
        training.run()


def test_resume_then_early_stop(training, tmp_path):
    training.config.update(resume=str(make_resume(tmp_path, 3)), steps=10, end_step=5)
    result = training.run()
    assert result['step'] == 5 and result['early_stopped']
    assert training.optimizer.step.call_count == 2
    assert validation_points(training.writer) == [(4, 3.0), (5, 3.0)]
    progress = callback_progress(training)
    assert [(item['step'], item['total'], item['schedule_total']) for item in progress] == [(4, 5, 10), (5, 5, 10)]


def test_dense_range_adds_validation_points_without_checkpoints(training):
    training.config.update(steps=12, save_every=6)
    training.config['validation']['dense'] = {'start_step': 2, 'end_step': 5, 'interval_steps': 2}
    training.run()
    assert [step for step, _ in validation_points(training.writer)] == [2, 4, 6, 12]
    assert [entry.args[2] for entry in training.trainer._generate_eval.call_args_list] == [6, 12]
    assert sorted(path.name for path in training.out.glob('checkpoint-step-*')) == [
        'checkpoint-step-000006', 'checkpoint-step-000012']
    state = json.loads((training.out / 'checkpoint-step-000006/trainer_state.pt').read_text(encoding='utf-8'))
    assert state['config']['validation']['dense'] == training.config['validation']['dense']


def test_dense_range_is_inclusive_and_combines_with_interval(training):
    training.config.update(steps=10, save_every=10)
    training.config['validation'].update(interval_steps=4,
                                         dense={'start_step': 3, 'end_step': 9, 'interval_steps': 3})
    training.run()
    assert [step for step, _ in validation_points(training.writer)] == [3, 4, 6, 8, 9, 10]


def test_dense_range_ignored_without_validation_rows(training):
    training.config.pop('validation')
    training.config['steps'] = 6
    assert training.run()['step'] == 6
    assert validation_points(training.writer) == []


def test_stopped_run_writes_stopped_sidecar(training):
    training.config['validation']['interval_steps'] = 1
    training.config['validation']['rows'][0]['stop'] = True
    training.config.update(steps=10, end_step=6)
    result = training.run()
    assert result['stopped'] and not result['early_stopped']
    assert adapter_state(training.out / 'final_adapter') == {
        'step': 1, 'schedule_total': 10, 'end_step': 6, 'stopped': True, 'early_stopped': False}


def test_validation_loss_reports_each_sample(training):
    seen = []
    loss = training.trainer._validation_loss(
        training.tts, training.config['validation']['rows'], 'cuda:0', 7, Event(),
        on_sample=lambda index, count: seen.append((index, count)))
    assert loss == 3.0
    assert seen == [(0, 2), (1, 2)]
    assert training.adapter.training is False  # restored to the pre-call mode


# ---------------------------------------------------------------- auto stop

U_SHAPE = {4: 3.0, 8: 2.5, 12: 2.45, 14: 2.6, 15: 2.7}


def enable_auto_stop(training, patience=3, min_delta=0.1, interval=2, losses=None, grace=None, **config):
    training.config.update(steps=20, save_every=4, **config)
    training.config['auto_stop'] = {'patience_steps': patience, 'min_delta': min_delta,
                                    'watch_interval_steps': interval}
    if grace is not None:
        training.config['auto_stop']['grace_steps'] = grace
    training.validation_losses.update(U_SHAPE if losses is None else losses)


def saved_paths(training):
    return [Path(entry.args[0]).name for entry in training.adapter.save_pretrained.call_args_list]


def trainer_state(path):
    return json.loads((path / 'trainer_state.pt').read_text(encoding='utf-8'))


def scalar_points(writer, name):
    return [(step, value) for tag, value, step in
            (record.args for record in writer.add_scalar.call_args_list) if tag == name]


def messages(training):
    return [entry.kwargs['message'] for entry in training.callback.call_args_list if 'message' in entry.kwargs]


def auto_stop_snapshots(training):
    return [entry.kwargs['auto_stop'] for entry in training.callback.call_args_list if 'auto_stop' in entry.kwargs]


def test_auto_stop_ends_u_shaped_run_and_keeps_best_adapter(training):
    enable_auto_stop(training)
    result = training.run()
    # Normal validations at 4/8/12; the watch starting at 12 adds 14 and the deadline 15.
    assert validation_points(training.writer) == [(4, 3.0), (8, 2.5), (12, 2.45), (14, 2.6), (15, 2.7)]
    assert training.optimizer.step.call_count == 15
    assert scalar_points(training.writer, 'auto_stop/watching') == [(4, 0.0), (8, 0.0), (12, 1.0), (14, 1.0), (15, 1.0)]
    assert scalar_points(training.writer, 'auto_stop/best_loss') == [(4, 3.0), (8, 2.5), (12, 2.45), (14, 2.45), (15, 2.45)]
    assert scalar_points(training.writer, 'auto_stop/valley_loss') == [(4, 3.0), (8, 2.5), (12, 2.5), (14, 2.5), (15, 2.5)]
    # The best adapter is saved at every new best, before that step's checkpoint.
    assert saved_paths(training) == [
        'best_adapter', 'checkpoint-step-000004', 'best_adapter', 'checkpoint-step-000008',
        'best_adapter', 'checkpoint-step-000012', 'checkpoint-step-000015', 'final_adapter']
    best = training.out / 'best_adapter'
    assert adapter_state(best) == {'step': 12, 'schedule_total': 20, 'end_step': 20, 'stopped': False,
                                   'early_stopped': False, 'kind': 'best', 'validation_loss': 2.45}
    assert trainer_state(best)['step'] == 12
    # Forced off-grid checkpoint and evaluation at the stop step.
    assert sorted(path.name for path in training.out.glob('checkpoint-step-*')) == [
        'checkpoint-step-000004', 'checkpoint-step-000008', 'checkpoint-step-000012', 'checkpoint-step-000015']
    assert [entry.args[2] for entry in training.trainer._generate_eval.call_args_list] == [4, 8, 12, 15]
    for name in ['checkpoint-step-000015', 'final_adapter']:
        assert adapter_state(training.out / name) == {'step': 15, 'schedule_total': 20, 'end_step': 20,
                                                      'stopped': False, 'early_stopped': False, 'auto_stopped': True}
        assert trainer_state(training.out / name)['auto_stop']['phase'] == 'stopped'
    assert 'auto_stopped' not in adapter_state(training.out / 'checkpoint-step-000012')
    assert result['step'] == 15 and result['total'] == result['schedule_total'] == result['end_step'] == 20
    assert result['auto_stopped'] and not result['early_stopped'] and not result['stopped']
    assert (result['best_step'], result['best_loss'], result['valley_step'], result['watch_start_step']) == (12, 2.45, 8, 12)
    assert (result['min_delta'], result['patience_steps']) == (0.1, 3)
    assert result['auto_stop']['phase'] == 'stopped' and result['auto_stop']['deadline_step'] == 15
    assert result['output'] == str(training.out / 'final_adapter')
    text = messages(training)
    assert 'Saved best adapter at step 12 (validation loss 2.450000).' in text
    assert 'Auto stop: no significant improvement at step 12; validating every 2 steps until step 15.' in text
    snapshots = auto_stop_snapshots(training)
    assert [item['phase'] for item in snapshots] == ['normal', 'normal', 'watching', 'watching', 'stopped']
    assert snapshots[2]['deadline_step'] == 15
    assert training.adapter.training and training.flags.grad


def test_auto_stop_state_is_saved_after_the_steps_validation(training):
    enable_auto_stop(training)
    training.run()
    state = trainer_state(training.out / 'checkpoint-step-000012')['auto_stop']
    assert state == {'version': 1, 'fingerprint': 'a' * 64, 'phase': 'watching', 'best_loss': 2.45, 'best_step': 12,
                     'valley_loss': 2.5, 'valley_step': 8, 'watch_start_step': 12, 'watch_best_loss': 2.45}
    assert trainer_state(training.out / 'checkpoint-step-000004')['auto_stop']['valley_step'] == 4


def test_auto_stop_recovery_continues_to_planned_end(training):
    enable_auto_stop(training, patience=6, losses={4: 3.0, 8: 2.95, 10: 2.99, 12: 2.8, 16: 2.6, 20: 2.4})
    result = training.run()
    # Watch 8..14 adds step 10; the recovery at 12 cancels the extra validation at 14.
    assert [step for step, _ in validation_points(training.writer)] == [4, 8, 10, 12, 16, 20]
    assert sorted(path.name for path in training.out.glob('checkpoint-step-*')) == [
        f'checkpoint-step-{step:06d}' for step in (4, 8, 12, 16, 20)]
    assert result['step'] == 20 and not result['auto_stopped'] and not result['early_stopped']
    assert (result['best_step'], result['valley_step'], result['watch_start_step']) == (20, 20, None)
    assert result['auto_stop']['phase'] == 'normal'
    assert 'Auto stop: validation loss improved to 2.800000 at step 12; resuming normal validation.' in messages(training)
    assert 'auto_stopped' not in adapter_state(training.out / 'final_adapter')
    assert adapter_state(training.out / 'best_adapter')['step'] == 20


WARMUP_PLATEAU = {step: 6.0 if step <= 8 else 6.0 - 0.3 * (step - 8) for step in range(1, 21)}


@pytest.mark.parametrize('grace', [None, 0, 8])
def test_grace_period_ignores_the_warmup_plateau(training, grace):
    enable_auto_stop(training, patience=3, interval=1, losses=WARMUP_PLATEAU, grace=grace)
    training.config['validation']['interval_steps'] = 2
    result = training.run()
    steps = [step for step, _ in validation_points(training.writer)]
    if grace:
        # Validations up to step 8 are exempt; the loss then falls and training runs to the end.
        assert steps == list(range(2, 21, 2))
        assert result['step'] == 20 and not result['auto_stopped']
        assert result['auto_stop']['phase'] == 'normal' and result['auto_stop']['grace_steps'] == 8
        assert not any('no significant improvement' in text for text in messages(training))
    else:
        # Without a grace period the plateau starts a watch at step 4 and stops at 7.
        assert steps == [2, 4, 5, 6, 7]
        assert result['step'] == 7 and result['auto_stopped']
        assert result['auto_stop']['grace_steps'] == 0


def test_auto_stop_requires_validation_rows(training):
    training.config.pop('validation')
    training.config['auto_stop'] = {'patience_steps': 3, 'min_delta': 0.1, 'watch_interval_steps': 2}
    with pytest.raises(RuntimeError, match='Auto stop needs validation loss'):
        training.run()
    training.optimizer.step.assert_not_called()
    training.writer_factory.assert_not_called()


def test_disabled_auto_stop_keeps_result_shape(training):
    result = training.run()
    assert result['auto_stopped'] is False
    assert 'best_step' not in result and 'auto_stop' not in result
    assert trainer_state(training.out / 'final_adapter')['auto_stop'] is None
    assert not (training.out / 'best_adapter').exists()
    assert auto_stop_snapshots(training) == []
    assert scalar_points(training.writer, 'auto_stop/watching') == []


def resume_with_auto_stop(tmp_path, step, monitor_state):
    resume = make_resume(tmp_path, step)
    state = json.loads((resume / 'trainer_state.pt').read_text(encoding='utf-8'))
    state['auto_stop'] = monitor_state
    (resume / 'trainer_state.pt').write_text(json.dumps(state), encoding='utf-8')
    return resume


def watching_state(fingerprint):
    from indextts_training.auto_stop import AutoStopMonitor
    monitor = AutoStopMonitor(3, 0.1, 2, fingerprint)
    monitor.observe(4, 3.0)
    monitor.observe(8, 2.5)
    monitor.observe(12, 2.45)
    assert monitor.phase == 'watching'
    return monitor.to_dict()


def test_resume_restores_auto_stop_watch(training, tmp_path):
    enable_auto_stop(training, resume=str(resume_with_auto_stop(tmp_path, 12, watching_state('a' * 64))))
    result = training.run()
    assert 'Auto stop state restored from step 12.' in messages(training)
    # The restored watch still validates at 14 and stops at its deadline 15.
    assert validation_points(training.writer) == [(14, 2.6), (15, 2.7)]
    assert result['auto_stopped'] and result['step'] == 15
    assert (result['best_step'], result['valley_step'], result['watch_start_step']) == (12, 8, 12)
    assert training.optimizer.step.call_count == 3
    assert not (training.out / 'best_adapter').exists()  # no new best in this run


def test_resume_with_changed_validation_set_starts_fresh(training, tmp_path):
    enable_auto_stop(training, losses={16: 2.6, 20: 2.7},
                     resume=str(resume_with_auto_stop(tmp_path, 12, watching_state('b' * 64))))
    result = training.run()
    assert ('Auto stop state was not restored: the validation set changed. Starting a new comparison.'
            in messages(training))
    assert validation_points(training.writer) == [(16, 2.6), (20, 2.7)]
    assert result['step'] == 20 and not result['auto_stopped']
    assert (result['best_step'], result['valley_step'], result['watch_start_step']) == (16, 16, 20)


def test_resume_from_checkpoint_without_auto_stop_state(training, tmp_path):
    enable_auto_stop(training, losses={16: 2.6, 20: 2.5}, resume=str(make_resume(tmp_path, 12)))
    result = training.run()
    assert not any('Auto stop state' in text for text in messages(training))
    assert result['best_step'] == 20 and not result['auto_stopped']


def test_resume_from_auto_stopped_checkpoint_continues(training, tmp_path):
    from indextts_training.auto_stop import AutoStopMonitor
    monitor = AutoStopMonitor(3, 0.1, 2, 'a' * 64)
    monitor.restore(watching_state('a' * 64))
    monitor.observe(15, 2.7)
    assert monitor.phase == 'stopped'
    enable_auto_stop(training, losses={16: 2.3, 20: 2.1},
                     resume=str(resume_with_auto_stop(tmp_path, 15, monitor.to_dict())))
    result = training.run()
    assert 'Auto stop state restored from step 15.' in messages(training)
    assert [step for step, _ in validation_points(training.writer)] == [16, 20]
    assert result['step'] == 20 and not result['auto_stopped']
    assert (result['best_step'], result['valley_step']) == (20, 20)


@pytest.mark.parametrize(('end_step', 'final_step', 'auto', 'early'), [
    (13, 13, False, True),   # fixed early stop first (still watching at 13)
    (18, 15, True, False),   # auto stop first
    (15, 15, True, False),   # both at once: reported as an auto stop
])
def test_auto_stop_combined_with_fixed_early_stop(training, end_step, final_step, auto, early):
    enable_auto_stop(training, end_step=end_step)
    result = training.run()
    assert (result['step'], result['auto_stopped'], result['early_stopped']) == (final_step, auto, early)
    assert result['schedule_total'] == 20 and result['end_step'] == end_step
    final = adapter_state(training.out / 'final_adapter')
    assert final['step'] == final_step
    assert final.get('auto_stopped', False) is auto
    assert final['early_stopped'] is early
    assert adapter_state(training.out / f'checkpoint-step-{final_step:06d}')['early_stopped'] is early
    assert result['auto_stop']['phase'] == ('stopped' if auto else 'watching')


def test_auto_stop_validation_failure_saves_checkpoint_without_observation(training):
    enable_auto_stop(training, losses={})
    training.config['validation']['rows'][1]['broken'] = True
    with pytest.raises(RuntimeError, match='Validation failed at step 4'):
        training.run()
    state = trainer_state(training.out / 'checkpoint-step-000004')
    assert state['step'] == 4 and state['auto_stop']['best_step'] is None
    assert not (training.out / 'best_adapter').exists()
    training.trainer._generate_eval.assert_not_called()


def test_stop_during_validation_does_not_observe(training):
    enable_auto_stop(training)
    training.config['validation']['interval_steps'] = 1
    training.config['validation']['rows'][0]['stop'] = True
    result = training.run()
    assert result['stopped'] and not result['auto_stopped'] and result['step'] == 1
    assert result['auto_stop']['best_step'] is None
    assert not (training.out / 'best_adapter').exists()
    assert auto_stop_snapshots(training) == []

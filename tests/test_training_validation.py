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
                              lr_scheduler=SimpleNamespace(LambdaLR=lambda *_: scheduler)),
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
        value = Loss(item['loss'], backward)
        return value, value, None, None, None

    trainer._freeze_index_tts_modules = Mock(return_value=[])
    trainer._prepare_sample = prepare
    trainer._loss = loss
    trainer._generate_eval = Mock(return_value=None)
    callback = Mock()
    return SimpleNamespace(
        trainer=trainer, config=config, out=out, writer=writer, writer_factory=writer_factory,
        optimizer=optimizer, scheduler=scheduler, adapter=adapter, flags=flags, samples=samples,
        backward=backward, stop=stop,
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

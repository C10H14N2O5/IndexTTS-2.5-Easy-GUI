"""Seed synthetic, explicitly isolated demo data; never use the user's live root.

    python -m tools.gui_server.fake.seed_demo --data-dir .runtime/webui-demo
    python -m tools.gui_server.launcher --fake --data-dir .runtime/webui-demo

All audio is an audible test tone, not a cloned or prerecorded human voice.
Re-running is additive: existing audio, transcripts and project edits are kept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import paths

MARKER = '.indextts-fake-demo.json'


def seed() -> dict:
    """Populate the already redirected root; caller must validate/redirect first."""
    from . import write_tone, _ensure_fake_model, _write_adapter
    from indextts_training import dataset
    from ..services.history import remember

    paths.ensure_dirs()
    _ensure_fake_model()
    voices = [
        ('Demo_Warm', '欢迎使用 IndexTTS。这是一段用于界面验收的合成测试音。', 210),
        ('Demo_Bright', '你好，世界。试听音频与真实模型效果不同。', 310),
        ('Demo_Calm', 'A quiet moment. This demo plays synthetic test tones.', 160),
    ]
    for name, text, pitch in voices:
        wav = paths.VOICES_DIR / f'{name}.wav'
        if not wav.exists():
            write_tone(wav, 3.2, pitch)
        transcript = wav.with_suffix('.txt')
        if not transcript.exists():
            transcript.write_text(text + '\n', encoding='utf-8')
    for name, text in [('Warm_narrator', '温暖、自然、舒缓的讲述，停顿清晰。'),
                       ('Bright_conversation', '轻快而充满好奇，像与朋友面对面聊天。')]:
        path = paths.EMOTION_DESCRIPTIONS_DIR / f'{name}.txt'
        if not path.exists():
            path.write_text(text + '\n', encoding='utf-8')

    corpus = paths.ROOT / 'demo_corpus'
    corpus.mkdir(parents=True, exist_ok=True)
    sentences = ['清晨的阳光落在窗台上。', '今天我们一起探索声音的可能。', '每一段文字，都可以拥有自己的表达。']
    for index, text in enumerate(sentences, 1):
        wav = corpus / f'clip_{index:03d}.wav'
        if not wav.exists():
            write_tone(wav, 3.0 + index * .2, 180 + index * 35)
        sidecar = wav.with_suffix('.txt')
        if not sidecar.exists():
            sidecar.write_text(text + '\n', encoding='utf-8')
    project_name = 'Demo_Narrator'
    project_file = paths.TRAINING_PROJECTS_DIR / project_name / 'project.json'
    if not project_file.exists():
        meta = dataset.prepare_dataset(str(corpus), project_name, language='ZH', root=paths.TRAINING_DIR)
        meta['training'] = {'adapter_name': project_name, 'mode': 'Steps', 'training_steps': 24,
                            'save_steps': 6, 'seed': 1234, 'eval_enabled': True,
                            'eval_text': '欢迎试听训练中的模型。', 'eval_reference': str(paths.VOICES_DIR / 'Demo_Warm.wav'),
                            'eval_reference_transcript': voices[0][1], 'eval_language': 'Chinese'}
        dataset._atomic_write(project_file, json.dumps(meta, ensure_ascii=False, indent=2).encode('utf-8'))

    adapter = paths.TRAINING_OUTPUTS_DIR / project_name / 'final_adapter'
    if not adapter.exists():
        _write_adapter(adapter, {'rank': 16, 'alpha': 32, 'dropout': .05})
    output = paths.OUTPUTS_DIR / 'indextts_2_5_demo.wav'
    if not output.exists():
        write_tone(output, 4.5, 245)
        from ..services.synthesis import SynthesisRequest
        remember(output, 'synthesis', SynthesisRequest(voice_path=str(paths.VOICES_DIR / 'Demo_Warm.wav'),
                 text='清晨的阳光落在窗台上，新的一天开始了。').model_dump())
    return {'root': str(paths.ROOT), 'voices': [name for name, _, _ in voices],
            'project': project_name, 'corpus': str(corpus)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='Empty/new directory reserved only for this fake demo')
    args = parser.parse_args(argv)
    directory = Path(args.data_dir).expanduser().resolve()
    code_root = Path(__file__).resolve().parents[3]
    protected = [code_root, code_root / 'voices', code_root / 'outputs', code_root / 'training',
                 code_root / 'models', code_root / 'assets', code_root / 'emotion_descriptions']
    if (any(directory == root or root.is_relative_to(directory) for root in protected)
            or any(directory.is_relative_to(root) for root in protected[1:])):
        parser.error('--data-dir must not be the repository, a live data folder, or an ancestor of either')
    marker = directory / MARKER
    if directory.exists() and any(directory.iterdir()) and not marker.is_file():
        parser.error('--data-dir must be empty or contain an existing fake-demo marker; existing data was not changed')
    if marker.is_file():
        try:
            valid_marker = json.loads(marker.read_text(encoding='utf-8')).get('kind') == 'indextts-fake-demo'
        except (OSError, ValueError, AttributeError):
            valid_marker = False
        if not valid_marker:
            parser.error('Invalid fake-demo marker; existing data was not changed')
    directory.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text('{"kind":"indextts-fake-demo","version":1}\n', encoding='utf-8')
    paths.use_data_root(directory)
    print(json.dumps(seed(), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

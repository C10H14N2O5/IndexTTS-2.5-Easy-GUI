"""Fake generation end-to-end: forwarding parity, joins, busy and cancellation."""
from __future__ import annotations

import json
import threading
import wave
from pathlib import Path
from unittest.mock import patch

from gui_api_support import GuiApiTestCase
from tools.gui_server import paths
from tools.gui_server.fake import FakeTTS
from tools.gui_server.services import engine, synthesis
from tools.gui_server.services.catalog import EMOTION_PRESETS
from indextts_training.activity import acquire, release
from indextts.utils.text_chunking import CHUNK_CHOICES


class SynthesisApiTests(GuiApiTestCase):
    def body(self, **overrides):
        return {'voice_path': str(self.audio()), 'text': '你好世界。', **overrides}

    def test_saved_synthesis_sidecar_history_and_media(self):
        job = self.submit('/api/synthesis', self.body(reference_transcript='原文', language='Chinese'))
        done = self.wait_job(job['id'])
        self.assertEqual(done['state'], 'succeeded', done)
        result = done['result']
        self.assertTrue(result['saved'])
        self.assertGreater(result['duration'], 0)
        output = Path(result['path'])
        self.assertEqual(output.parent, paths.OUTPUTS_DIR)
        sidecar = json.loads(Path(str(output) + '.json').read_text(encoding='utf-8'))
        self.assertEqual(sidecar['params']['reference_transcript'], '原文')
        self.assertEqual(sidecar['params']['language'], 'ZH')
        history = self.client.get('/api/history').json()
        self.assertEqual(history[0]['path'], str(output))
        self.assertEqual(history[0]['params'], result['params'])
        self.assertEqual(self.client.get('/api/media', params={'path': str(output)}).content, output.read_bytes())
        self.assertEqual(self.client.delete('/api/history/' + output.name).status_code, 200)
        self.assertFalse(output.exists())
        self.assertFalse(Path(str(output) + '.json').exists())

    def test_preview_is_not_history_and_all_decoding_params_forwarded(self):
        calls = []
        original = FakeTTS.infer
        def record(model, *args, **kwargs):
            calls.append(kwargs)
            return original(model, *args, **kwargs)
        body = self.body(save_outputs=False, language='JA', duration_factor=1.25, interval_silence=350,
                         text_normalization=False, max_text_tokens=180, do_sample=False, top_p=.7, top_k=60,
                         temperature=1.1, num_beams=4, repetition_penalty=12, length_penalty=-1,
                         max_mel_tokens=2000, reference_transcript='manual reference',
                         emotion_mode='Emotion vector', emotion_vector=[.1,.2,.3,.4,.5,.6,.7,.8],
                         emotion_alpha=.8, random_emotion=True, use_gpt_latent=True)
        with patch.object(FakeTTS, 'infer', record):
            done = self.wait_job(self.submit('/api/synthesis', body)['id'])
        self.assertEqual(done['state'], 'succeeded', done)
        self.assertEqual(self.client.get('/api/history').json(), [])
        self.assertEqual(Path(done['result']['path']), paths.TEMP_DIR / 'preview.wav')
        kwargs = calls[0]
        for key in ['duration_factor', 'interval_silence', 'text_normalization', 'do_sample', 'top_p', 'top_k',
                    'temperature', 'num_beams', 'repetition_penalty', 'length_penalty', 'max_mel_tokens']:
            self.assertEqual(kwargs[key], body[key])
        self.assertEqual(kwargs['reference_text'], 'manual reference')
        self.assertEqual(kwargs['max_text_tokens_per_segment'], 180)
        self.assertEqual(kwargs['emo_vector'], body['emotion_vector'])
        self.assertTrue(kwargs['use_random'])
        self.assertTrue(engine.loaded_info()['options']['use_gpt_latent'])

    def test_quick_sample_authoritative_and_missing_reference_rejected(self):
        voice = self.audio()
        body = {'quick_sample': str(voice), 'voice_path': str(self.root / 'missing.wav'), 'text': 'Saved voice wins.'}
        done = self.wait_job(self.submit('/api/synthesis', body)['id'])
        self.assertEqual(done['state'], 'succeeded', done)
        for invalid in [self.body(voice_path=''), self.body(text=' '), self.body(emotion_mode='Emotion reference audio'),
                        self.body(language='FR'), self.body(chunk_mode='unknown'), self.body(emotion_vector=[2]*8)]:
            response = self.client.post('/api/synthesis', json=invalid)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn('error', response.json())

    def test_audio_and_description_emotion_conditioning(self):
        calls = []
        original = FakeTTS.infer
        def record(model, *args, **kwargs):
            calls.append(kwargs)
            return original(model, *args, **kwargs)
        emotion_audio = self.audio('.runtime/temp/emotion.wav')
        description = self.client.post('/api/emotion-descriptions', json={'name': 'warm', 'text': 'Gentle narration.'}).json()
        with patch.object(FakeTTS, 'infer', record):
            for body in [self.body(emotion_mode='Emotion reference audio', emotion_audio=str(emotion_audio)),
                         self.body(emotion_mode='Emotion description', emotion_description=description['path']),
                         self.body(emotion_mode='Emotion description', text='Fallback emotion text')]:
                done = self.wait_job(self.submit('/api/synthesis', body)['id'])
                self.assertEqual(done['state'], 'succeeded', done)
        self.assertEqual(calls[0]['emo_audio_prompt'], str(emotion_audio))
        self.assertEqual(calls[1]['emo_text'], 'Gentle narration.')
        self.assertEqual(calls[2]['emo_text'], 'Fallback emotion text')
        self.assertTrue(calls[1]['use_emo_text'])
        self.assertTrue(engine.loaded_info()['options']['use_qwen_emo'])

    def test_chunk_join_inserts_exact_silence(self):
        calls = []
        original = FakeTTS.infer
        def record(model, *args, **kwargs):
            calls.append(kwargs)
            return original(model, *args, **kwargs)
        body = self.body(text='第一行。\n第二行。', chunk_mode='Lines', chunk_silence=.2)
        self.assertIn('Lines', CHUNK_CHOICES)
        with patch.object(FakeTTS, 'infer', record):
            done = self.wait_job(self.submit('/api/synthesis', body)['id'])
        self.assertEqual(done['state'], 'succeeded', done)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call['interval_silence'] == 0 for call in calls))
        self.assertAlmostEqual(done['result']['duration'], 1.4, places=3)
        self.assertEqual(list(paths.TEMP_DIR.glob('inference_*')), [])

    def test_failure_cleans_staging_and_releases_gpu(self):
        done = self.wait_job(self.submit('/api/synthesis', self.body(text='[fail]'))['id'])
        self.assertEqual(done['state'], 'failed', done)
        self.assertIn('Fake engine failure', done['error']['message'])
        self.assertEqual([p for p in paths.OUTPUTS_DIR.iterdir() if p.is_file()], [])
        self.assertEqual(list(paths.TEMP_DIR.glob('inference_*')), [])
        success = self.wait_job(self.submit('/api/synthesis', self.body(text='Recovered'))['id'])
        self.assertEqual(success['state'], 'succeeded', success)

    def test_busy_conflict_cancel_between_chunks_and_cleanup(self):
        started = threading.Event()
        release = threading.Event()
        calls = []
        original = FakeTTS.infer
        def slow(model, *args, **kwargs):
            calls.append(kwargs)
            started.set()
            if not release.wait(8):
                raise RuntimeError('Test barrier timeout')
            return original(model, *args, **kwargs)
        with patch.object(FakeTTS, 'infer', slow):
            job = self.submit('/api/synthesis', self.body(text='One.\nTwo.', chunk_mode='Lines'))
            try:
                self.assertTrue(started.wait(8))
                busy = self.client.post('/api/synthesis', json=self.body())
                self.assertEqual(busy.status_code, 409, busy.text)
                self.assertEqual(busy.json()['error']['busy_job']['id'], job['id'])
                self.assertEqual(self.client.post('/api/model/unload').status_code, 409)
                self.assertEqual(self.client.delete('/api/voices').status_code, 409)
                cancelled = self.client.post('/api/jobs/' + job['id'] + '/cancel')
                self.assertEqual(cancelled.status_code, 200, cancelled.text)
            finally:
                release.set()
            done = self.wait_job(job['id'])
        self.assertEqual(done['state'], 'cancelled', done)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.client.get('/api/history').json(), [])
        self.assertEqual(list(paths.TEMP_DIR.glob('inference_*')), [])

    def test_sidecar_write_failure_does_not_publish_audio(self):
        with patch('tools.gui_server.services.history.remember', side_effect=OSError('Simulated disk full')):
            done = self.wait_job(self.submit('/api/synthesis', self.body())['id'])
        self.assertEqual(done['state'], 'failed', done)
        self.assertEqual(self.client.get('/api/history').json(), [])
        self.assertEqual([p for p in paths.OUTPUTS_DIR.iterdir() if p.is_file()], [])

    def test_inference_blocked_by_training_activity(self):
        token = acquire('training')
        try:
            response = self.client.post('/api/synthesis', json=self.body())
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()['error']['code'], 'busy')
        finally:
            release(token)

    def test_asr_reference_success_failure_cancel_and_auto_transcript(self):
        voice = self.audio()
        done = self.wait_job(self.submit('/api/asr/transcribe', {'path': str(voice)})['id'])
        self.assertEqual(done['state'], 'succeeded', done)
        self.assertIn('reference', done['result']['text'])
        failed = self.wait_job(self.submit('/api/asr/transcribe', {'path': str(self.audio('voices/fail.wav'))})['id'])
        self.assertEqual(failed['state'], 'failed', failed)
        job = self.submit('/api/asr/transcribe', {'path': str(voice)})
        self.client.post('/api/jobs/' + job['id'] + '/cancel')
        self.assertEqual(self.wait_job(job['id'])['state'], 'cancelled')
        auto = self.wait_job(self.submit('/api/synthesis', self.body(reference_auto_transcribe=True))['id'])
        self.assertEqual(auto['state'], 'succeeded', auto)
        self.assertIn('reference', auto['result']['params']['reference_transcript'])

    def test_auto_asr_cancellation_does_not_report_failure(self):
        job = self.submit('/api/synthesis', self.body(reference_auto_transcribe=True))
        self.client.post('/api/jobs/' + job['id'] + '/cancel')
        self.assertEqual(self.wait_job(job['id'])['state'], 'cancelled')

    def test_dialogue_joins_mixed_emotions_and_skips_blank_turns(self):
        voice = self.audio()
        description = self.client.post('/api/emotion-descriptions', json={'name': 'warm', 'text': 'Soft delivery'}).json()
        calls = []
        original = FakeTTS.infer
        def record(model, *args, **kwargs):
            calls.append(kwargs)
            return original(model, *args, **kwargs)
        body = {'turns': [{'voice_path': str(voice), 'text': 'Hello', 'preset': 'Joyful'},
                          {'voice_path': '', 'text': ' '},
                          {'voice_path': str(voice), 'text': 'World', 'emotion_source': 'Emotion Description',
                           'description_path': description['path']}], 'dialogue_silence': .3}
        with patch.object(FakeTTS, 'infer', record):
            done = self.wait_job(self.submit('/api/dialogue', body)['id'])
        self.assertEqual(done['state'], 'succeeded', done)
        self.assertEqual(done['result']['turns'], 2)
        self.assertAlmostEqual(done['result']['duration'], 1.5, places=3)
        self.assertEqual(calls[0]['emo_vector'], EMOTION_PRESETS['Joyful'])
        self.assertEqual(calls[1]['emo_text'], 'Soft delivery')
        self.assertEqual(calls[0]['emo_alpha'], .65)
        self.assertEqual(self.client.get('/api/history').json()[0]['kind'], 'dialogue')

    def test_dialogue_validation_is_synchronous(self):
        voice = self.audio()
        for turns in [[], [{'voice_path': str(voice), 'text': ''}],
                      [{'voice_path': '', 'text': 'Missing'}],
                      [{'voice_path': str(voice), 'text': 'Description', 'emotion_source': 'Emotion Description'}],
                      [{'voice_path': str(voice), 'text': 'Too many'}] * 13]:
            response = self.client.post('/api/dialogue', json={'turns': turns})
            self.assertEqual(response.status_code, 400, response.text)
        self.assertIsNone(self.state.jobs.running_gpu_job())

    def test_history_accepts_legacy_outputs_and_cleanup_preserves_subdirectories(self):
        legacy = self.audio('outputs/legacy.wav')
        abx = self.audio('outputs/abx/session/keep.wav')
        exports = paths.OUTPUTS_DIR / 'exports/keep.pth'
        exports.parent.mkdir(parents=True, exist_ok=True)
        exports.write_bytes(b'keep')
        history = self.client.get('/api/history').json()
        self.assertEqual(history[0]['name'], 'legacy.wav')
        self.assertIsNone(history[0]['params'])
        self.assertEqual(self.client.delete('/api/outputs').json(), {'deleted': 1})
        self.assertFalse(legacy.exists())
        self.assertTrue(abx.exists())
        self.assertTrue(exports.exists())

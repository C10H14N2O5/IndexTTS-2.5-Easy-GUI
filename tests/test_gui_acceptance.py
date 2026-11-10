"""Bounded fake cross-area acceptance; no real GPU/model-quality claims.

Tests use real HTTP routes, project/proofreader persistence and job/lease managers.
Only model/trainer/merge execution is fake. Deterministic barriers hold accepted
work while competing API requests race, rather than relying on timing sleeps.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from gui_api_support import GuiApiTestCase
from tools.gui_server import paths
from tools.gui_server.fake import FakeTTS, _write_adapter


class GuiAcceptanceTests(GuiApiTestCase):
    def prepare(self):
        uploaded = self.upload_audio('listener.wav')
        voice = self.client.post('/api/voices', json={'name': 'listener', 'source_path': uploaded['path'],
                                                     'transcript': 'Frozen reference transcript.'})
        self.assertEqual(voice.status_code, 201, voice.text)
        self.voice = voice.json()['path']
        corpus = self.audio('corpus/train.wav', .7)
        corpus.with_suffix('.txt').write_text('Training corpus transcript.', encoding='utf-8')
        response = self.client.post('/api/projects', json={'name': 'acceptance', 'source_folder': str(corpus.parent), 'language': 'EN'})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(self.client.put('/api/projects/acceptance/dataset', json={'prepare': True}).status_code, 200)
        saved = self.client.put('/api/projects/acceptance/training', json={'training_steps': 2, 'save_steps': 1, 'eval_enabled': False})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.adapter = paths.TRAINING_OUTPUTS_DIR / 'comparison' / 'final_adapter'
        _write_adapter(self.adapter, {'rank': 16, 'alpha': 32})
        self.session_id = self.create_comparison()['id']
        return saved.json()

    def create_comparison(self, texts='First test sentence.', repetitions=1):
        response = self.client.post('/api/abx/sessions', json={'name': 'Acceptance comparison', 'adapter_a': '',
            'adapter_b': str(self.adapter), 'voice': self.voice, 'language': 'EN', 'texts': texts,
            'repetitions': repetitions, 'settings': {'temperature': .45, 'num_beams': 2}})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def finish_training(self):
        self.wait_until(lambda: not self.state.training.snapshot()['running'], timeout=25)
        self.state.training.thread.join(10)
        result = self.client.get('/api/training/status').json()
        self.assertEqual(result['status'], 'succeeded', result)
        return result

    def finish_export(self, expected='succeeded'):
        self.wait_until(lambda: self.state.export.snapshot()['status'] not in ('running', 'cancelling'), timeout=25)
        self.state.export.thread.join(10)
        result = self.client.get('/api/export/status').json()
        self.assertEqual(result['status'], expected, result)
        return result

    def request_work(self, kind, suffix=''):
        if kind == 'synthesis':
            return self.client.post('/api/synthesis', json={'voice_path': self.voice, 'text': 'Lease acceptance.', 'language': 'EN'})
        if kind == 'abx':
            return self.client.post(f'/api/abx/sessions/{self.session_id}/generate')
        if kind == 'training':
            return self.client.post('/api/training/start', json={'project': 'acceptance'})
        return self.client.post('/api/export/start', json={'adapter': str(self.adapter), 'destination': f'lease{suffix}.pth'})

    def finish_work(self, kind, response):
        if kind in ('synthesis', 'abx'):
            result = self.wait_job(response.json()['job']['id'], timeout=25)
            self.assertEqual(result['state'], 'succeeded', result)
        elif kind == 'training':
            self.finish_training()
        else:
            self.finish_export()
        from indextts_training.activity import acquire, release
        lease = acquire('acceptance recovery probe')
        release(lease)

    @contextlib.contextmanager
    def held_workers(self):
        release = threading.Event()
        entered = threading.Event()
        original_infer = FakeTTS.infer
        original_train = self.state.training.train_fn
        original_export = self.state.export._worker
        def gate():
            entered.set()
            if not release.wait(12):
                raise RuntimeError('Acceptance worker barrier timed out')
        def infer(model, *args, **kwargs):
            gate()
            return original_infer(model, *args, **kwargs)
        def train(*args, **kwargs):
            gate()
            return original_train(*args, **kwargs)
        def export(*args, **kwargs):
            gate()
            return original_export(*args, **kwargs)
        with patch.object(FakeTTS, 'infer', infer), patch.object(self.state.training, 'train_fn', train), patch.object(self.state.export, '_worker', export):
            try:
                yield entered, release
            finally:
                release.set()

    def test_seeded_corpus_edit_reprepare_train_and_synthesize(self):
        from tools.gui_server.fake.seed_demo import seed
        seeded = seed()
        name = seeded['project']
        original = self.client.get(f'/api/projects/{name}').json()
        defaults = self.client.get('/api/training/defaults').json()
        self.assertEqual((defaults['seed'], defaults['learning_rate'], defaults['grad_clip']), (1234, 4e-5, .5))
        self.assertEqual(original['dataset_status']['status'], 'ready')
        self.assertEqual(original['training']['seed'], defaults['seed'])
        manifest = Path(original['manifest'])
        before = manifest.read_bytes()
        proofread = self.client.post('/proofread/api/session').json()
        headers = {'X-Session-Token': proofread['token']}
        selected = self.client.post('/proofread/api/select', headers=headers, json={
            'directory': seeded['corpus'], 'project': name, 'language': 'ZH', 'role': 'training'})
        self.assertEqual(selected.status_code, 200, selected.text)
        row = selected.json()['items'][0]
        item = self.client.get('/proofread/api/item', headers=headers, params={'id': row['id']}).json()
        changed = self.client.post('/proofread/api/save', headers=headers, json={
            'id': item['id'], 'version': item['version'], 'text': '校对后的新文本。', 'reviewed': True})
        self.assertEqual(changed.status_code, 200, changed.text)
        stale = self.client.get(f'/api/projects/{name}').json()
        self.assertTrue(stale['dataset_status']['stale'])
        self.assertEqual(manifest.read_bytes(), before)
        rejected = self.client.post('/api/training/start', json={'project': name})
        self.assertEqual(rejected.status_code, 400, rejected.text)
        rebuilt = self.client.post('/proofread/api/manifest/update', headers=headers, json={'role': 'training'})
        self.assertEqual(rebuilt.status_code, 200, rebuilt.text)
        self.assertFalse(rebuilt.json()['status']['stale'])
        self.assertNotEqual(manifest.read_bytes(), before)
        reference = str(paths.VOICES_DIR / 'Demo_Warm.wav')
        saved = self.client.put(f'/api/projects/{name}/training', json={
            'training_steps': 3, 'save_steps': 1, 'seed': 2468, 'eval_reference': reference,
            'eval_enabled': True, 'eval_text': '训练中试听。', 'eval_language': 'Chinese'})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(self.client.post('/api/training/start', json={'project': name}).status_code, 202)
        done = self.finish_training()
        self.assertEqual(done['step'], 3)
        run = json.loads((paths.TRAINING_OUTPUTS_DIR / name / 'training_config.json').read_text(encoding='utf-8'))
        self.assertEqual((run['seed'], run['lr'], run['grad_clip']), (2468, defaults['learning_rate'], defaults['grad_clip']))
        self.assertEqual(run['eval_language'], 'ZH')
        self.assertFalse(run['differentiable_asr_enabled'])
        checkpoints = self.client.get(f'/api/projects/{name}/checkpoints').json()
        self.assertEqual(len(checkpoints), 4)
        evaluations = [entry['evaluation'] for entry in checkpoints if entry['evaluation']]
        self.assertEqual(len(evaluations), 3)
        for audio in evaluations:
            self.assertEqual(self.client.get('/api/media', params={'path': audio}).status_code, 200)
        metrics = self.client.get(f'/api/projects/{name}/metrics', params={'max_points': 2}).json()
        self.assertTrue(metrics['available'])
        self.assertTrue(all(len(series['points']) <= 2 for series in metrics['series']))
        final = next(entry['path'] for entry in checkpoints if entry['kind'] == 'final')
        job = self.submit('/api/synthesis', {'voice_path': reference, 'adapter_path': final, 'text': '训练之后的试听。'})
        result = self.wait_job(job['id'])
        self.assertEqual(result['state'], 'succeeded', result)
        audio = result['result']['path']
        self.assertEqual(self.client.get('/api/media', params={'path': audio}, headers={'Range': 'bytes=0-43'}).status_code, 206)
        history = next(row for row in self.client.get('/api/history').json() if row['path'] == audio)
        self.assertEqual(history['params']['adapter_path'], final)
        self.assertEqual(history['params']['text'], '训练之后的试听。')

    def test_explicit_external_manifest_stays_usable_without_disabling_managed_freshness(self):
        project = self.prepare()
        owned = Path(project['manifest'])
        external = self.root / 'manually-maintained.jsonl'
        external.write_bytes(owned.read_bytes())
        captures = []
        def tiny_trainer(config, callback, stop_event):
            captures.append(config)
            return {'output': '', 'stopped': False}
        self.state.training.train_fn = tiny_trainer
        selected = self.client.put('/api/projects/acceptance/dataset', json={'manifest': str(external), 'source_folder': ''})
        self.assertEqual(selected.status_code, 200, selected.text)
        status = selected.json()['dataset_status']
        self.assertFalse(status['managed'])
        self.assertFalse(status['stale'])
        self.assertEqual(status['status'], 'ready')
        self.assertEqual(selected.json()['manifest'], str(external))
        # Source metadata left over from the formerly managed corpus must not
        # claim ownership of this explicit, user-maintained manifest.
        (self.root / 'corpus/train.txt').write_text('Edited after the external snapshot.', encoding='utf-8')
        self.assertFalse(self.client.get('/api/projects/acceptance').json()['dataset_status']['stale'])
        self.assertEqual(self.client.post('/api/training/start', json={'project': 'acceptance'}).status_code, 202)
        self.finish_training()
        self.assertEqual(captures[0]['manifest'], str(external))
        self.assertEqual(len(captures), 1)
        for invalid in ['', '{not-json}\n']:
            with self.subTest(invalid=invalid):
                external.write_text(invalid, encoding='utf-8')
                status = self.client.get('/api/projects/acceptance').json()['dataset_status']
                self.assertFalse(status['managed'])
                self.assertTrue(status['stale'])
                self.assertEqual(self.client.post('/api/training/start', json={'project': 'acceptance'}).status_code, 400)
                self.assertEqual(len(captures), 1)
        restored = self.client.put('/api/projects/acceptance/dataset', json={
            'manifest': str(owned), 'source_folder': str(self.root / 'corpus')})
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertTrue(restored.json()['dataset_status']['managed'])
        self.assertTrue(restored.json()['dataset_status']['stale'])
        self.assertEqual(self.client.post('/api/training/start', json={'project': 'acceptance'}).status_code, 400)
        prepared = self.client.put('/api/projects/acceptance/dataset', json={'prepare': True})
        self.assertEqual(prepared.status_code, 200, prepared.text)
        self.assertFalse(prepared.json()['dataset_status']['stale'])
        self.assertEqual(self.client.post('/api/training/start', json={'project': 'acceptance'}).status_code, 202)
        self.finish_training()
        self.assertEqual(len(captures), 2)
        self.assertEqual(captures[1]['manifest'], str(owned))

    def test_saved_evaluation_reference_remains_playable_without_exposing_project_metadata(self):
        self.prepare()
        upload = self.upload_audio('eval.wav')
        saved = self.client.put('/api/projects/acceptance/training', json={'eval_reference': upload['path']})
        self.assertEqual(saved.status_code, 200, saved.text)
        reference = saved.json()['training']['eval_reference']
        self.assertTrue(Path(reference).is_relative_to(paths.TRAINING_PROJECTS_DIR))
        response = self.client.get('/api/media', params={'path': reference})
        self.assertEqual(response.status_code, 200, response.text[:200])
        self.assertEqual(response.content[:4], b'RIFF')
        denied = self.client.get('/api/media', params={'path': str(paths.TRAINING_PROJECTS_DIR / 'acceptance/project.json')})
        self.assertEqual(denied.status_code, 403)
        unrelated = self.audio('training/projects/acceptance/private.wav')
        self.assertEqual(self.client.get('/api/media', params={'path': str(unrelated)}).status_code, 403)
        nested = self.audio('training/projects/acceptance/nested/eval_reference.wav')
        self.assertEqual(self.client.get('/api/media', params={'path': str(nested)}).status_code, 403)

    def test_project_evaluation_media_exception_rejects_escaping_symlinks(self):
        self.prepare()
        outside = self.audio('private.wav')
        link = paths.TRAINING_PROJECTS_DIR / 'acceptance/eval_reference.wav'
        try:
            link.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f'Host cannot create an unprivileged test symlink: {exc}')
        self.assertEqual(self.client.get('/api/media', params={'path': str(outside)}).status_code, 403)
        self.assertEqual(self.client.get('/api/media', params={'path': str(link)}).status_code, 403)

    def test_blind_trials_freeze_reference_settings_and_expose_identity_only_on_reveal(self):
        self.prepare()
        session = self.create_comparison('First sentence.\nSecond sentence.', 1)
        sid = session['id']
        disk_path = paths.ABX_DIR / sid / 'session.json'
        frozen = json.loads(disk_path.read_text(encoding='utf-8'))
        self.assertFalse(frozen['settings']['use_gpt_latent'])
        self.assertEqual(self.client.put('/api/settings', json={'use_gpt_latent': True}).status_code, 200)
        # The comparison owns a reference snapshot; library changes cannot
        # silently alter the stimulus or break generation after setup.
        removed = self.client.delete('/api/voices/listener.wav')
        self.assertEqual(removed.status_code, 200, removed.text)
        observed = []
        original_infer = FakeTTS.infer
        def capture_infer(model, *args, **kwargs):
            import torch
            observed.append((model.options, torch.initial_seed(), kwargs.get('reference_text')))
            return original_infer(model, *args, **kwargs)
        with patch.object(FakeTTS, 'infer', capture_infer):
            self.assertEqual(self.wait_job(self.submit(f'/api/abx/sessions/{sid}/generate', {})['id'])['state'], 'succeeded')
        self.assertEqual(len(observed), 4)
        self.assertTrue(all(options[1] is False for options, _, _ in observed))
        self.assertTrue(all(transcript == 'Frozen reference transcript.' for _, _, transcript in observed))
        self.assertEqual([seed for _, seed, _ in observed[:2]], [seed for _, seed, _ in observed[2:]])
        exposed = []
        for index in range(2):
            detail = self.client.get(f'/api/abx/sessions/{sid}', params={'trial': index}).json()
            self.assertIsNone(detail['results'])
            self.assertNotIn('clips', detail)
            self.assertNotIn('settings', detail)
            trial = detail['trial']
            self.assertEqual(set(trial), {'index', 'total', 'text', 'rep', 'reference', 'sample_1', 'sample_2', 'score'})
            for field in ('sample_1', 'sample_2'):
                clip = Path(trial[field])
                self.assertRegex(clip.name, r'^[0-9a-f]{8}\.wav$')
                self.assertNotIn(str(self.adapter), str(clip))
                self.assertEqual(self.client.get('/api/media', params={'path': str(clip)}).status_code, 200)
            exposed.append(trial)
            rated = self.client.put(f'/api/abx/sessions/{sid}/rate', json={'index': index, 'score': index + 1})
            self.assertEqual(rated.status_code, 200, rated.text)
            self.assertIsNone(rated.json()['results'])
        for index, trial in enumerate(exposed):
            reloaded = self.client.get(f'/api/abx/sessions/{sid}', params={'trial': index}).json()['trial']
            self.assertEqual((reloaded['sample_1'], reloaded['sample_2']), (trial['sample_1'], trial['sample_2']))
        self.assertEqual(self.client.get('/api/media', params={'path': str(disk_path)}).status_code, 403)
        persisted = json.loads(disk_path.read_text(encoding='utf-8'))
        self.assertEqual(persisted['settings'], frozen['settings'])
        self.assertEqual(persisted['reference']['transcript'], 'Frozen reference transcript.')
        for text_index in range(2):
            pair = [clip for clip in persisted['clips'] if clip['text_index'] == text_index]
            self.assertEqual(pair[0]['seed'], pair[1]['seed'])
        revealed = self.client.post(f'/api/abx/sessions/{sid}/reveal').json()
        self.assertTrue(revealed['revealed'])
        self.assertEqual(revealed['results']['rated'], 2)
        for row in revealed['results']['rows']:
            self.assertEqual(row['sample_1'].lower(), persisted['trials'][row['index']]['first'])
        self.assertEqual(self.client.put(f'/api/abx/sessions/{sid}/rate', json={'index': 0, 'score': 0}).status_code, 400)

    def test_each_compute_owner_rejects_peers_and_recovers_the_shared_lease(self):
        self.prepare()
        kinds = ('synthesis', 'abx', 'training', 'export')
        for owner in kinds:
            with self.subTest(owner=owner), self.held_workers() as (entered, release):
                accepted = self.request_work(owner, owner)
                self.assertEqual(accepted.status_code, 202, accepted.text)
                self.assertTrue(entered.wait(6), owner)
                for contender in kinds:
                    if contender != owner:
                        response = self.request_work(contender, owner)
                        self.assertEqual(response.status_code, 409, (owner, contender, response.text))
                self.assertEqual(self.client.post('/api/model/unload').status_code, 409)
                self.assertEqual(self.client.delete('/api/voices/listener.wav').status_code, 409)
                # Reference ASR is CPU work, not a compute-lease contender.
                asr = self.submit('/api/asr/reference', {'path': self.voice})
                self.assertEqual(self.wait_job(asr['id'])['state'], 'succeeded')
                release.set()
                self.finish_work(owner, accepted)
                self.assertEqual(self.client.post('/api/model/unload').status_code, 200)
        self.assertTrue(Path(self.voice).is_file())

    def test_simultaneous_compute_starts_have_exactly_one_winner(self):
        self.prepare()
        kinds = ('synthesis', 'abx', 'training', 'export')
        barrier = threading.Barrier(len(kinds))
        def start(kind):
            barrier.wait(timeout=6)
            return kind, self.request_work(kind, 'race')
        with self.held_workers() as (entered, release), ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(start, kinds))
            self.assertEqual(sorted(response.status_code for _, response in results), [202, 409, 409, 409],
                             [(kind, response.status_code, response.text) for kind, response in results])
            self.assertTrue(entered.wait(6))
            winner, response = next((kind, response) for kind, response in results if response.status_code == 202)
            release.set()
            self.finish_work(winner, response)
        self.assertFalse(any(job['state'] == 'running' for job in self.client.get('/api/jobs').json()))

    def test_export_download_tokens_are_task_scoped_and_reports_match_outputs(self):
        self.prepare()
        completed = []
        for name in ('first.pth', 'second.pth'):
            start = self.client.post('/api/export/start', json={'adapter': str(self.adapter), 'destination': name})
            self.assertEqual(start.status_code, 202, start.text)
            result = self.finish_export()
            completed.append(result)
            model = self.client.get(result['downloads']['model'])
            report_response = self.client.get(result['downloads']['report'])
            self.assertEqual((model.status_code, report_response.status_code), (200, 200))
            report = report_response.json()
            self.assertIs(report['source_files_unchanged'], True)
            self.assertIs(report['round_trip_tensors_identical'], True)
            self.assertEqual(report['dtype'], 'float32')
            self.assertEqual(report['output_sha256'], hashlib.sha256(model.content).hexdigest())
            self.assertEqual(report['output'], result['output'])
            self.assertEqual(self.client.get('/api/media', params={'path': result['output']}).status_code, 403)
            self.assertEqual(self.client.get(result['downloads']['model'].split('?')[0]).status_code, 404)
        for index, result in enumerate(completed):
            other = completed[1 - index]
            for kind in ('model', 'report'):
                bare = result['downloads'][kind].split('?')[0]
                self.assertEqual(self.client.get(bare, params={'token': other['token']}).status_code, 404)
                self.assertEqual(self.client.get(result['downloads'][kind]).status_code, 200)
            self.assertEqual(self.client.get(f"/exports/{result['task_id']}/unknown", params={'token': result['token']}).status_code, 404)

    def test_export_never_publishes_false_missing_or_nonboolean_verification(self):
        self.prepare()
        script = self.root / 'fake-invalid-export.py'
        for index, evidence in enumerate([False, None, 'false', 1]):
            with self.subTest(evidence=evidence):
                report = {} if evidence is None else {'source_files_unchanged': evidence, 'round_trip_tensors_identical': evidence}
                script.write_text('import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser()\n'
                    'p.add_argument("--output")\na,rest=p.parse_known_args()\nf=Path(a.output)\n'
                    'f.write_bytes(b"not-a-verified-checkpoint")\n'
                    f'f.with_suffix(".merge.json").write_text({json.dumps(json.dumps(report))})\n', encoding='utf-8')
                with patch.object(self.state.export, 'script', script):
                    started = self.client.post('/api/export/start', json={'adapter': str(self.adapter), 'destination': f'invalid-{index}.pth'})
                    self.assertEqual(started.status_code, 202, started.text)
                    done = self.finish_export('failed')
                self.assertFalse(done['downloads'])
                self.assertIn('validation report is incomplete', done['error'])
                self.assertFalse(Path(done['output']).exists())
                self.assertFalse(Path(done['output']).with_suffix('.merge.json').exists())
                self.assertFalse(any(Path(done['output']).parent.glob('.gpt-export-*')))
        # Failure cleanup releases the same lease used by inference.
        completed = self.wait_job(self.submit('/api/synthesis', {'voice_path': self.voice, 'text': 'Recovered.'})['id'])
        self.assertEqual(completed['state'], 'succeeded', completed)

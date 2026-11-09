"""Manifest correctness checks use only temporary audio and sidecars."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

from indextts_training import dataset as ds


def write_audio(path, sample=b'\0\0'):
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(sample * 16000)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base/'中文语料'
        self.source.mkdir()
        self.training = self.base/'training'
        self.audio = self.source/'sample.wav'
        write_audio(self.audio)
        self.text = self.audio.with_suffix('.txt')
        self.text.write_text('原话。', encoding='utf-8-sig')

    def prepare(self):
        return ds.prepare_dataset(str(self.source), 'voice', 'ZH', root=self.training)

    def test_saved_text_updates_both_fields_and_preserves_project(self):
        meta = self.prepare()
        project = self.training/'projects/voice/project.json'
        meta.update(training={'rank': 16}, custom={'keep': True})
        project.write_text(json.dumps(meta), encoding='utf-8')
        self.assertFalse(ds.dataset_manifest_status('voice', self.training)['stale'])
        self.text.write_text('校对后的原话。', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Update the training manifest'):
            ds.validate_project_manifest('voice', meta['manifest'], self.training)
        result = ds.refresh_project_manifest('voice', self.training)
        row = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
        self.assertEqual(row['text'], '校对后的原话。')
        self.assertEqual(row['reference_text'], row['text'])
        self.assertEqual(result['training'], {'rank': 16})
        self.assertEqual(result['custom'], {'keep': True})
        self.assertFalse(ds.dataset_manifest_status('voice', self.training)['stale'])

    def test_legacy_project_is_checked_without_migration(self):
        meta = self.prepare()
        meta.pop('source_signature'); meta.pop('manifest_sha256')
        project = self.training/'projects/voice/project.json'
        project.write_text(json.dumps(meta), encoding='utf-8')
        before = project.read_bytes()
        self.assertFalse(ds.dataset_manifest_status('voice', self.training)['stale'])
        self.assertEqual(project.read_bytes(), before)
        self.text.write_text('外部修改', encoding='utf-8')
        self.assertTrue(ds.dataset_manifest_status('voice', self.training)['stale'])

    def test_external_manifest_is_not_blocked(self):
        self.prepare()
        self.text.write_text('新文本', encoding='utf-8')
        ds.validate_project_manifest('voice', self.base/'external.jsonl', self.training)

    def test_training_blocks_updates(self):
        meta = self.prepare()
        before = Path(meta['manifest']).read_bytes()
        with patch('indextts_training.manager.TRAINING_MANAGER.snapshot', return_value={'running': True}):
            with self.assertRaisesRegex(ValueError, 'Training is running'):
                ds.refresh_project_manifest('voice', self.training)
        self.assertEqual(Path(meta['manifest']).read_bytes(), before)

    def test_failed_metadata_write_rolls_back_manifest(self):
        meta = self.prepare()
        before = Path(meta['manifest']).read_bytes()
        project = self.training/'projects/voice/project.json'
        old_project = project.read_bytes()
        self.text.write_text('新文本', encoding='utf-8')
        write = ds._atomic_write
        def fail_project(path, data):
            if Path(path) == project:
                raise OSError('disk full')
            return write(path, data)
        with patch.object(ds, '_atomic_write', side_effect=fail_project):
            with self.assertRaises(OSError):
                ds.refresh_project_manifest('voice', self.training)
        self.assertEqual(Path(meta['manifest']).read_bytes(), before)
        self.assertEqual(project.read_bytes(), old_project)
        self.assertTrue(ds.dataset_manifest_status('voice', self.training)['stale'])

    def test_empty_text_does_not_erase_old_manifest(self):
        meta = self.prepare()
        before = Path(meta['manifest']).read_bytes()
        self.text.write_text('', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'No valid'):
            ds.refresh_project_manifest('voice', self.training)
        self.assertEqual(Path(meta['manifest']).read_bytes(), before)

    def test_language_and_new_files_invalidate_manifest(self):
        meta = self.prepare()
        project = self.training/'projects/voice/project.json'
        meta['language'] = 'JA'
        project.write_text(json.dumps(meta), encoding='utf-8')
        self.assertTrue(ds.dataset_manifest_status('voice', self.training)['stale'])
        ds.refresh_project_manifest('voice', self.training)
        (self.source/'second.wav').write_bytes(self.audio.read_bytes())
        self.assertTrue(ds.dataset_manifest_status('voice', self.training)['stale'])

    def test_identical_copy_of_source_still_requires_rebuild(self):
        meta = self.prepare()
        other = self.base/'other'
        shutil.copytree(self.source, other, copy_function=shutil.copy2)
        meta['source_folder'] = str(other)
        project = self.training/'projects/voice/project.json'
        project.write_text(json.dumps(meta), encoding='utf-8')
        self.assertTrue(ds.dataset_manifest_status('voice', self.training)['stale'])

    def test_recursive_inventory_excludes_metadata_and_accepts_bom(self):
        sub = self.source/'sub'; sub.mkdir()
        (sub/'sample.wav').write_bytes(self.audio.read_bytes())
        (sub/'sample.txt').write_text('子目录', encoding='utf-8')
        hidden = self.source/'.proofread'; hidden.mkdir()
        (hidden/'skip.wav').write_bytes(self.audio.read_bytes())
        rows, stats = ds.scan_dataset(str(self.source))
        self.assertEqual(stats['samples'], 2)
        self.assertEqual({r['text'] for r in rows}, {'原话。', '子目录'})

    def prepare_validation(self):
        training_meta = self.prepare()
        folder = self.base/'验证集'
        folder.mkdir()
        audio = folder/'validation.wav'
        write_audio(audio, b'\x01\0')
        audio.with_suffix('.txt').write_text('验证语句。', encoding='utf-8')
        write_audio(folder/'unlabeled.wav')
        linked = ds.configure_validation('voice', source_folder=folder, root=self.training)
        self.assertTrue(linked['validation']['enabled'])
        ds.refresh_project_manifest('voice', self.training, role='validation')
        return training_meta['manifest'], audio

    def test_validation_manifests_and_settings_are_independent(self):
        training_manifest, audio = self.prepare_validation()
        status = ds.dataset_manifest_status('voice', self.training, role='validation')
        validation_manifest = Path(status['manifest'])
        self.assertEqual(validation_manifest, Path(training_manifest).with_name('validation.jsonl'))
        self.assertEqual(status['status'], 'ready')
        self.assertEqual((status['stats']['samples'], status['stats']['valid']), (2, 1))
        before = Path(training_manifest).read_bytes()
        audio.with_suffix('.txt').write_text('验证校对完成。', encoding='utf-8')
        self.assertFalse(ds.dataset_manifest_status('voice', self.training)['stale'])
        self.assertTrue(ds.dataset_manifest_status('voice', self.training, role='validation')['stale'])
        with self.assertRaisesRegex(ValueError, 'Validation manifest needs updating'):
            ds.validate_validation_manifest('voice', training_manifest, self.training)
        ds.refresh_project_manifest('voice', self.training, role='validation')
        self.assertEqual(Path(training_manifest).read_bytes(), before)
        self.assertEqual(json.loads(validation_manifest.read_bytes())['language'], 'ZH')
        self.assertEqual(json.loads(validation_manifest.read_bytes())['text'], '验证校对完成。')
        ds.configure_validation('voice', enabled=False, interval_steps=8, root=self.training)
        validation = ds.validation_settings(ds.load_project('voice', self.training))
        self.assertEqual((validation['source_folder'], validation['manifest'], validation['interval_steps']),
                         (str(audio.parent.resolve()), str(validation_manifest), 8))
        self.assertIsNone(ds.validate_validation_manifest('voice', training_manifest, self.training))
        active = ds.validate_validation_manifest('voice', training_manifest, self.training,
                                                 enabled=True, interval_steps=None)
        self.assertIsNone(active['interval_steps'])
        self.assertEqual(len(active['rows']), 1)
        self.assertEqual(active['fingerprint'], validation['manifest_sha256'])
        self.assertEqual(active['training_fingerprint'], hashlib.sha256(before).hexdigest())
        self.assertFalse(ds.load_project('voice', self.training)['validation']['enabled'])

    def test_unconfigured_validation_only_blocks_when_enabled(self):
        meta = self.prepare()
        saved = ds.configure_validation('voice', enabled=False, interval_steps=None, root=self.training)
        self.assertNotIn('validation', saved)
        self.assertIsNone(ds.validate_validation_manifest('voice', meta['manifest'], self.training))
        with self.assertRaisesRegex(ValueError, 'save a validation source folder'):
            ds.validate_validation_manifest(None, meta['manifest'], self.training, enabled=True)
        with self.assertRaisesRegex(ValueError, 'save a validation source folder'):
            ds.validate_validation_manifest('voice', meta['manifest'], self.training, enabled=True)
        ds.configure_validation('voice', source_folder=self.base/'not-prepared', root=self.training)
        with self.assertRaisesRegex(ValueError, 'Validation manifest has not been prepared'):
            ds.validate_validation_manifest('voice', meta['manifest'], self.training)
        self.assertIsNone(ds.validate_validation_manifest('voice', meta['manifest'], self.training, enabled=False))

    def test_validation_preflight_checks_actual_training_manifest(self):
        training_manifest, audio = self.prepare_validation()
        ds.validate_validation_manifest('voice', training_manifest, self.training)
        external = self.base/'external.jsonl'
        duplicate = self.base/'other-filename.wav'
        shutil.copyfile(audio, duplicate)
        for sample in (audio, duplicate):
            with self.subTest(sample=sample):
                external.write_text(json.dumps({'audio': str(sample), 'text': 'Training text'}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Training and validation contain the same audio'):
                    ds.validate_validation_manifest('voice', external, self.training)

    def test_empty_validation_and_invalid_intervals_are_rejected(self):
        training_manifest, _ = self.prepare_validation()
        for invalid in (False, 0, -1, 2.5, float('inf')):
            with self.subTest(interval=invalid), self.assertRaisesRegex(ValueError, 'positive whole number'):
                ds.validate_validation_manifest('voice', training_manifest, self.training, interval_steps=invalid)
        self.assertEqual(ds.normalize_validation_interval(4.0), 4)
        validation = ds.load_project('voice', self.training)['validation']
        Path(validation['manifest']).write_text('', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Validation manifest is empty'):
            ds.validate_validation_manifest('voice', training_manifest, self.training)

    def test_clone_preserves_validation_source_but_prepares_its_own_manifest(self):
        _, audio = self.prepare_validation()
        original = ds.load_project('voice', self.training)['validation']
        clone = ds.clone_project('voice', self.training)
        validation = clone['validation']
        self.assertEqual(validation['source_folder'], original['source_folder'])
        self.assertNotEqual(validation['manifest'], original['manifest'])
        self.assertEqual(validation['stats']['prepared_manifest'], validation['manifest'])
        self.assertFalse(ds.dataset_manifest_status(clone['project_name'], self.training, role='validation')['stale'])
        original_bytes = Path(original['manifest']).read_bytes()
        audio.with_suffix('.txt').write_text('在副本中更新。', encoding='utf-8')
        ds.refresh_project_manifest(clone['project_name'], self.training, role='validation')
        self.assertEqual(Path(original['manifest']).read_bytes(), original_bytes)
        self.assertNotEqual(Path(validation['manifest']).read_bytes(), original_bytes)


if __name__ == '__main__':
    unittest.main()

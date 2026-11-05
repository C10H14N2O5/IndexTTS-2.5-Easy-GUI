"""Manifest correctness checks use only temporary audio and sidecars."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from indextts_training import dataset as ds


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base/'中文语料'
        self.source.mkdir()
        self.training = self.base/'training'
        self.audio = self.source/'sample.wav'
        with wave.open(str(self.audio), 'wb') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\0\0'*16000)
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
        import shutil
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


if __name__ == '__main__':
    unittest.main()

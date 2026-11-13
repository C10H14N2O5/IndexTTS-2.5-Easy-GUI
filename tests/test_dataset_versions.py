"""Frozen dataset versions: snapshots, selection, integrity and repair (temporary data only)."""
import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest
import wave
from pathlib import Path

from indextts_training import dataset as ds
from indextts_training import dataset_versions as dv


def write_audio(path, sample=b'\0\0', seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(sample * int(16000 * seconds))


class DatasetVersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / '语料'
        self.training = self.base / 'training'
        self.clips = []
        for index in range(3):
            self.add_clip(f'clip{index}', bytes([index + 1, 0]))
        ds.prepare_dataset(str(self.source), 'voice', 'ZH', root=self.training)

    def add_clip(self, name, sample=b'\x05\0', text=None):
        audio = self.source / f'{name}.wav'
        write_audio(audio, sample)
        audio.with_suffix('.txt').write_text(text or f'{name} 文本', encoding='utf-8')
        self.clips.append(audio)
        return audio

    def rebuild(self, role='training'):
        return ds.refresh_project_manifest('voice', self.training, role=role)

    def freeze(self, **kwargs):
        return dv.freeze('voice', 'training', root=self.training, **kwargs)

    def test_freeze_is_byte_identical_and_survives_working_changes(self):
        working = Path(ds.load_project('voice', self.training)['manifest'])
        info = self.freeze(note='first batch')
        self.assertEqual((info['id'], info['name'], info['revision']), ('train-001', 'v1', 1))
        folder = dv.versions_root('voice', 'training', self.training) / 'train-001'
        self.assertEqual((folder / 'manifest.jsonl').read_bytes(), working.read_bytes())
        self.assertEqual(info['manifest_sha256'], hashlib.sha256(working.read_bytes()).hexdigest())
        self.assertEqual(info['stats']['valid'], 3)
        audio = [json.loads(line) for line in (folder / 'audio.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(audio[0]['sha256'], hashlib.sha256(Path(audio[0]['audio']).read_bytes()).hexdigest())
        # The working corpus keeps growing; the version keeps its rows and text.
        self.clips[0].with_suffix('.txt').write_text('改过的文本', encoding='utf-8')
        self.add_clip('clip9')
        self.rebuild()
        report = dv.verify('voice', 'training', 'train-001', self.training)
        self.assertEqual(report['status'], 'ready')
        self.assertEqual(len(report['rows']), 3)
        self.assertEqual(dv.diff(report['rows'], dv.working_rows('voice', 'training', self.training)),
                         dict(added=1, removed=0, text_changed=1))

    def test_freeze_requires_up_to_date_managed_manifest_and_rejects_duplicates(self):
        self.freeze()
        with self.assertRaises(dv.VersionError) as caught:
            self.freeze()
        self.assertEqual(caught.exception.kind, 'conflict')
        self.add_clip('new')
        with self.assertRaisesRegex(dv.VersionError, 'up-to-date training manifest'):
            self.freeze()
        self.rebuild()
        with self.assertRaises(dv.VersionError) as caught:
            self.freeze(name='V1')
        self.assertEqual(caught.exception.kind, 'conflict')  # names are unique, case-insensitive
        second = self.freeze(name='  扩充  后 ')
        self.assertEqual((second['id'], second['name']), ('train-002', '扩充 后'))
        meta = ds.load_project('voice', self.training)
        meta['manifest'] = str(self.base / 'external.jsonl')
        Path(meta['manifest']).write_text(json.dumps({'audio': str(self.clips[0]), 'text': 'x'}), encoding='utf-8')
        (self.training / 'projects/voice/project.json').write_text(json.dumps(meta), encoding='utf-8')
        with self.assertRaisesRegex(dv.VersionError, 'external training manifest'):
            self.freeze()

    def test_ids_are_never_reused_and_selection_follows_deletion(self):
        first = self.freeze(activate=True)
        self.assertEqual(ds.load_project('voice', self.training)['dataset_version'], first['id'])
        self.add_clip('more')
        self.rebuild()
        second = self.freeze()
        dv.delete('voice', 'training', second['id'], self.training)
        self.add_clip('again')
        self.rebuild()
        self.assertEqual(self.freeze()['id'], 'train-003')
        result = dv.delete('voice', 'training', first['id'], self.training)
        self.assertTrue(result['was_active'])
        self.assertIsNone(ds.load_project('voice', self.training)['dataset_version'])
        with self.assertRaises(dv.VersionError) as caught:
            dv.load_version('voice', 'training', first['id'], self.training)
        self.assertEqual(caught.exception.kind, 'not_found')
        with self.assertRaises(dv.VersionError):
            dv.load_version('voice', 'training', '../outside', self.training)
        with self.assertRaises(dv.VersionError):
            dv.load_version('voice', 'validation', 'train-003', self.training)

    def test_rename_and_select(self):
        info = self.freeze()
        renamed = dv.update('voice', 'training', info['id'], name='基线', note='说明', root=self.training)
        self.assertEqual((renamed['name'], renamed['note']), ('基线', '说明'))
        with self.assertRaisesRegex(dv.VersionError, 'Enter a version name'):
            dv.update('voice', 'training', info['id'], name='  ', root=self.training)
        dv.select('voice', 'training', info['id'], self.training)
        active = dv.resolve_active('voice', 'training', self.training)
        self.assertEqual((active['version_id'], active['status'], active['stale']), (info['id'], 'ready', False))
        self.assertEqual(active['stats']['valid'], 3)
        dv.select('voice', 'training', None, self.training)
        self.assertIsNone(dv.resolve_active('voice', 'training', self.training)['version_id'])
        with self.assertRaises(dv.VersionError):
            dv.select('voice', 'training', 'train-099', self.training)

    def test_integrity_detects_missing_changed_and_ignores_touch(self):
        info = self.freeze()
        stamp = time.time() + 10
        os.utime(self.clips[1], (stamp, stamp))  # same bytes, new mtime: still intact
        self.assertEqual(dv.verify('voice', 'training', info['id'], self.training)['status'], 'ready')
        write_audio(self.clips[1], b'\x09\0')  # same size, different content
        self.clips[2].unlink()
        report = dv.verify('voice', 'training', info['id'], self.training)
        self.assertEqual(report['status'], 'damaged')
        self.assertEqual((len(report['missing']), len(report['changed'])), (1, 1))
        with self.assertRaises(dv.VersionError):
            dv.usable_rows('voice', 'training', info['id'], self.training)
        folder = dv.versions_root('voice', 'training', self.training) / info['id']
        (folder / 'manifest.jsonl').write_text('{}\n', encoding='utf-8')
        self.assertEqual(dv.verify('voice', 'training', info['id'], self.training)['status'], 'invalid')

    def test_repair_relocates_drops_and_accepts_with_a_new_revision(self):
        info = self.freeze()
        moved = self.source / 'session_02' / 'clip0.wav'
        moved.parent.mkdir()
        shutil.move(self.clips[0], moved)
        self.clips[1].unlink()
        write_audio(self.clips[2], b'\x07\0', seconds=1.5)
        report = dv.integrity_report('voice', 'training', info['id'], self.training)
        self.assertEqual((report['missing_count'], report['changed_count'], report['relocatable']), (2, 1, 1))
        self.assertEqual({item['relocated_to'] for item in report['missing']}, {str(moved.resolve()), None})
        [repaired] = dv.repair('voice', 'training', info['id'], changed='accept', root=self.training)
        self.assertEqual((repaired['id'], repaired['revision']), (info['id'], 2))
        self.assertNotEqual(repaired['manifest_sha256'], info['manifest_sha256'])
        history = repaired['history'][-1]
        self.assertEqual((len(history['relocated']), len(history['dropped']), len(history['accepted'])), (1, 1, 1))
        self.assertEqual(history['previous_manifest_sha256'], info['manifest_sha256'])
        result = dv.verify('voice', 'training', info['id'], self.training)
        self.assertEqual(result['status'], 'ready')
        rows = {Path(row['audio']).name: row for row in result['rows']}
        self.assertEqual(set(rows), {'clip0.wav', 'clip2.wav'})
        self.assertAlmostEqual(rows['clip2.wav']['duration'], 1.5, places=2)
        self.assertEqual(dv.repair('voice', 'training', info['id'], root=self.training), [])

    def test_repair_affected_versions_and_refuse_to_empty(self):
        first = self.freeze()
        self.add_clip('extra')
        self.rebuild()
        second = self.freeze()
        self.clips[0].unlink()
        report = dv.integrity_report('voice', 'training', first['id'], self.training)
        self.assertEqual([item['id'] for item in report['affected']], [second['id']])
        repaired = dv.repair('voice', 'training', first['id'], changed='drop', scope='affected', root=self.training)
        self.assertEqual({item['id'] for item in repaired}, {first['id'], second['id']})
        for path in self.clips[1:]:
            path.unlink(missing_ok=True)
        with self.assertRaisesRegex(dv.VersionError, 'would remove every clip'):
            dv.repair('voice', 'training', first['id'], root=self.training)

    def test_validation_versions_feed_the_training_preflight(self):
        folder = self.base / '验证'
        write_audio(folder / 'v.wav', b'\x11\0')
        (folder / 'v.txt').write_text('验证', encoding='utf-8')
        ds.configure_validation('voice', source_folder=folder, root=self.training)
        self.rebuild('validation')
        info = dv.freeze('voice', 'validation', activate=True, root=self.training)
        self.assertEqual(info['id'], 'val-001')
        self.assertEqual(ds.validation_settings(ds.load_project('voice', self.training))['version'], 'val-001')
        # The working validation set can go stale without affecting the frozen one.
        (folder / 'v.txt').write_text('改了', encoding='utf-8')
        manifest = ds.load_project('voice', self.training)['manifest']
        snapshot = ds.validate_validation_manifest('voice', manifest, self.training)
        self.assertEqual((snapshot['version_id'], snapshot['revision']), ('val-001', 1))
        self.assertEqual(snapshot['fingerprint'], info['manifest_sha256'])
        self.assertEqual(snapshot['rows'][0]['text'], '验证')
        self.assertEqual(dv.active_fingerprint('voice', 'validation', self.training),
                         (info['manifest_sha256'], dict(id='val-001', name='v1', revision=1)))
        (folder / 'v.wav').unlink()
        with self.assertRaises(dv.VersionError):
            ds.validate_validation_manifest('voice', manifest, self.training)

    def test_clone_copies_versions_and_used_by_reads_training_configs(self):
        info = self.freeze(activate=True)
        clone = ds.clone_project('voice', self.training)
        self.assertEqual(clone['dataset_version'], info['id'])
        self.assertEqual(dv.verify(clone['project_name'], 'training', info['id'], self.training)['status'], 'ready')
        output = self.training / 'outputs/adapter_a'
        output.mkdir(parents=True)
        (output / 'training_config.json').write_text(json.dumps(
            {'project': 'voice', 'dataset': {'version_id': info['id']}}), encoding='utf-8')
        self.assertEqual(dv.used_by('voice', 'training', info['id'], self.training), ['adapter_a'])
        self.assertEqual(dv.used_by(clone['project_name'], 'training', info['id'], self.training), [])


if __name__ == '__main__':
    unittest.main()

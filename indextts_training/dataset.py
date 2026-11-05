from __future__ import annotations
import functools, hashlib, json, os, re, secrets, shutil, statistics, threading, wave
from pathlib import Path

AUDIO_EXTS={'.wav','.flac','.mp3','.m4a','.ogg','.aac'}
DATASET_LOCK = threading.RLock()


def project_transaction(function):
    @functools.wraps(function)
    def locked(*args, **kwargs):
        with DATASET_LOCK:
            return function(*args, **kwargs)
    return locked


def audio_files(folder):
    """The same bounded recursive inventory is used by training and proofreading."""
    root = Path(folder).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Source Audio Folder does not exist.')
    found = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != '.proofread' and not (Path(current)/d).is_symlink()
                         and not (hasattr(Path(current)/d, 'is_junction') and (Path(current)/d).is_junction()))
        for name in files:
            p = Path(current)/name
            if p.suffix.lower() in AUDIO_EXTS and p.is_file() and not p.is_symlink():
                if p.resolve().is_relative_to(root):
                    found.append(p)
    return sorted(found)


def _read_text(audio):
    target = audio.with_suffix('.txt')
    if target.is_symlink():
        raise ValueError(f'Transcript links are not supported: {target}')
    return target.read_text(encoding='utf-8-sig').strip() if target.is_file() else ''


def _source_signature(folder):
    root = Path(folder).expanduser().resolve(strict=True)
    digest = hashlib.sha256(os.path.normcase(str(root)).encode("utf-8"))
    for p in audio_files(root):
        st = p.stat()
        txt = p.with_suffix('.txt')
        if txt.is_symlink():
            raise ValueError(f'Transcript links are not supported: {txt}')
        record = [p.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns,
                  hashlib.sha256(txt.read_bytes()).hexdigest() if txt.is_file() else None]
        digest.update(json.dumps(record, ensure_ascii=False).encode('utf-8'))
    return digest.hexdigest()


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + secrets.token_hex(8) + '.tmp')
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _project_file(name, root):
    name = str(name or '')
    if not name or name in ('.', '..') or '/' in name or '\\' in name or ':' in name:
        raise ValueError('Select a valid dataset project.')
    base = Path(root).resolve()/'projects'
    result = base/name/'project.json'
    if not result.resolve().is_relative_to(base):
        raise ValueError('Invalid project path.')
    return result

def _duration(path: Path) -> float:
    try:
        import soundfile as sf
        info=sf.info(str(path)); return float(info.frames)/float(info.samplerate)
    except Exception:
        try:
            with wave.open(str(path),'rb') as w: return w.getnframes()/float(w.getframerate())
        except Exception: return 0.0

def _auto_clip_bounds(rows):
    durations=sorted(r['duration'] for r in rows if r['duration'] > 0)
    if not durations: return .5, 30.0
    lo=durations[max(0, int(len(durations)*.05))]
    hi=durations[min(len(durations)-1, max(0, int(len(durations)*.95)))]
    return max(.5, min(lo, 10.0)), min(30.0, max(hi, max(.5, lo)+1.0))

def scan_dataset(folder: str, min_seconds=None, max_seconds=None):
    root=Path(folder).expanduser()
    if not root.is_dir(): raise ValueError('Source Audio Folder does not exist.')
    rows=[]
    for p in audio_files(root):
        dur=_duration(p); text=_read_text(p)
        rows.append({'audio':str(p.resolve()),'text':text,'duration':dur,'valid': bool(text)})
    auto_min, auto_max = _auto_clip_bounds(rows)
    min_seconds = auto_min if min_seconds is None else float(min_seconds)
    max_seconds = auto_max if max_seconds is None else float(max_seconds)
    for r in rows: r['valid'] = bool(r['text']) and min_seconds <= r['duration'] <= max_seconds
    durations=[r['duration'] for r in rows]
    valid=[r for r in rows if r['valid']]
    stats={
      'samples':len(rows),'valid':len(valid),'missing_text':sum(not bool(r['text']) for r in rows),
      'total_seconds':sum(durations),'median_seconds':statistics.median(durations) if durations else 0.0,
      'p90_seconds':sorted(durations)[max(0,int(len(durations)*.9)-1)] if durations else 0.0,
      'auto_min_seconds':min_seconds, 'auto_max_seconds':max_seconds,
    }
    return rows, stats

def _manifest_rows(rows, language):
    return [dict(audio=r['audio'], text=r['text'], reference_text=r['text'],
                 duration=r['duration'], language=language) for r in rows if r['valid']]


def _require_idle_training():
    from .manager import TRAINING_MANAGER
    if TRAINING_MANAGER.snapshot()['running']:
        raise ValueError('Training is running. Update the manifest after training stops.')


def prepare_dataset(source_folder, project_name, language='EN', min_seconds=None, max_seconds=None,
                    transcribe_missing=False, whisper_model=None, whisper_language=None, root='training'):
    """Build only from saved sidecars. Legacy ASR arguments remain readable, not executable."""
    if transcribe_missing:
        raise ValueError('Use Recognize missing text in the proofreading workbench before updating the manifest.')
    from .proofread import directory_lock, directory_asr_running
    with DATASET_LOCK, directory_lock(source_folder):
        _require_idle_training()
        if directory_asr_running(source_folder):
            raise ValueError('Recognition is running for this directory. Finish recognition before updating the manifest.')
        safe=''.join(c if c.isalnum() or c in '-_' else '_' for c in project_name.strip()).strip('_') or 'dataset'
        root=Path(root)
        project_file=_project_file(safe, root)
        manifest=root/'datasets'/safe/'manifest.jsonl'
        if not manifest.resolve().is_relative_to((root/'datasets').resolve()):
            raise ValueError('Invalid manifest destination.')
        before=_source_signature(source_folder)
        rows,stats=scan_dataset(source_folder,min_seconds,max_seconds)
        valid=_manifest_rows(rows,language)
        if not valid:
            raise ValueError('No valid audio+transcript pairs were found after filtering.')
        if before != _source_signature(source_folder):
            raise ValueError('Source files changed while scanning. Retry updating the manifest.')
        previous=load_project(safe, root)
        meta=dict(previous)
        stats['prepared_manifest']=str(manifest.resolve())
        meta.update(project_name=safe, source_folder=str(Path(source_folder).resolve()),
                    manifest=str(manifest.resolve()), language=language, stats=stats,
                    source_signature=before, manifest_bounds={'min':min_seconds,'max':max_seconds})
        manifest_bytes=('\n'.join(json.dumps(r,ensure_ascii=False) for r in valid)+'\n').encode('utf-8')
        meta['manifest_sha256']=hashlib.sha256(manifest_bytes).hexdigest()
        old_manifest=manifest.read_bytes() if manifest.is_file() else None
        try:
            _atomic_write(manifest,manifest_bytes)
            _atomic_write(project_file,json.dumps(meta,indent=2,ensure_ascii=False).encode('utf-8'))
        except Exception:
            if old_manifest is None:
                manifest.unlink(missing_ok=True)
            else:
                _atomic_write(manifest,old_manifest)
            raise
        return meta


def dataset_manifest_status(project, root='training'):
    """Read-only freshness check, including projects prepared before signatures existed."""
    if not project:
        return dict(stale=True,managed=False,manifest='',message='Select a project to update its training manifest.',stats={})
    with DATASET_LOCK:
        meta=load_project(project,root)
        manifest=str(meta.get('manifest') or '')
        result=dict(stale=True,managed=True,manifest=manifest,stats=meta.get('stats',{}))
        if not manifest or not Path(manifest).is_file():
            return dict(result,message='Training manifest has not been prepared.')
        if not meta.get('source_folder'):
            return dict(result,message='Save the project source folder first.')
        try:
            signature=_source_signature(meta['source_folder'])
            raw=Path(manifest).read_bytes()
            if meta.get('source_signature'):
                stale=(signature != meta['source_signature'] or
                       hashlib.sha256(raw).hexdigest() != meta.get('manifest_sha256'))
                # Saving a changed project language must also invalidate its previous manifest.
                entries=[json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
                stale=stale or any(r.get('language') != meta.get('language','EN') for r in entries)
            else:
                bounds=meta.get('manifest_bounds',{})
                rows,_=scan_dataset(meta['source_folder'],bounds.get('min'),bounds.get('max'))
                expected=_manifest_rows(rows,meta.get('language','EN'))
                actual=[json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
                def canonical(items):
                    return sorted((str(Path(r['audio']).resolve()),r['text'],r.get('reference_text',r['text']),r.get('language')) for r in items)
                stale=canonical(actual)!=canonical(expected)
            return dict(result,stale=stale,message='Training manifest needs updating.' if stale else 'Training manifest is up to date.')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            return dict(result,message=f'Unable to verify training manifest: {exc}')


def refresh_project_manifest(project, root='training'):
    with DATASET_LOCK:
        meta=load_project(project,root)
        if not meta or not meta.get('source_folder'):
            raise ValueError('Save the project with a source folder first.')
        bounds=meta.get('manifest_bounds',{})
        return prepare_dataset(meta['source_folder'],project,meta.get('language','EN'),
                               bounds.get('min'),bounds.get('max'),root=root)


def validate_project_manifest(project, manifest, root='training'):
    if not project:
        return
    meta=load_project(project,root)
    # An explicitly selected external manifest is not owned by this GUI project.
    if not meta.get('manifest') or Path(manifest).resolve()!=Path(meta['manifest']).resolve():
        return
    status=dataset_manifest_status(project,root)
    if status['stale']:
        raise ValueError(status['message']+' Update the training manifest in the proofreading workbench before starting training.')


def list_projects(root='training'):
    p=Path(root)/'projects'; p.mkdir(parents=True,exist_ok=True)
    return sorted(x.parent.name for x in p.glob('*/project.json'))

def load_project(name, root='training'):
    if not name: return {}
    p=_project_file(name,root)
    if not p.is_file(): return {}
    return json.loads(p.read_text(encoding='utf-8'))

def _next_clone_name(name, root):
    """Return the next free incremented project identifier."""
    safe=''.join(c if c.isalnum() or c in '-_ ' else '_' for c in str(name or '').strip()).strip().replace(' ','_') or 'project'
    match=re.match(r'^(.*?)(?:_(\d+)|(\d+))?$', safe)
    base=(match.group(1) if match else safe).rstrip('_') or 'project'
    separator='_' if match and match.group(2) else ''
    suffix=(match.group(2) or match.group(3)) if match else None
    number=int(suffix)+1 if suffix else 2
    root=Path(root)
    while (root/'projects'/f'{base}{separator}{number}').exists() or (root/'datasets'/f'{base}{separator}{number}').exists():
        number+=1
    return f'{base}{separator}{number}'

def _rebase_project_path(value, old_root, new_root):
    """Rebase copied project-local absolute paths while preserving others."""
    if not value:
        return value
    try:
        relative=Path(str(value)).resolve().relative_to(Path(old_root).resolve())
    except (OSError, ValueError):
        return value
    return str((Path(new_root).resolve()/relative))

def clone_project(name, root='training'):
    """Clone project metadata and prepared data into the next numbered name.

    Training outputs are deliberately not copied. Project-local references
    and the prepared manifest are rebased to the clone so the two projects can
    be edited independently without changing the source project.
    """
    root=Path(root)
    source_name=str(name or '').strip()
    source_project=root/'projects'/source_name
    source_file=source_project/'project.json'
    if not source_file.is_file():
        raise ValueError(f'Project not found: {source_name or "(empty)"}')
    try:
        payload=json.loads(source_file.read_text(encoding='utf-8'))
    except Exception as exc:
        raise ValueError(f'Could not read project: {source_name}') from exc

    new_name=_next_clone_name(source_name,root)
    target_project=root/'projects'/new_name
    source_dataset=root/'datasets'/source_name
    target_dataset=root/'datasets'/new_name
    target_project.parent.mkdir(parents=True,exist_ok=True)
    target_dataset.parent.mkdir(parents=True,exist_ok=True)
    try:
        shutil.copytree(source_project,target_project)
        if source_dataset.is_dir():
            shutil.copytree(source_dataset,target_dataset)

        cloned=json.loads((target_project/'project.json').read_text(encoding='utf-8'))
        cloned['project_name']=new_name
        cloned['manifest']=_rebase_project_path(cloned.get('manifest',''),source_dataset,target_dataset)
        stats=cloned.get('stats')
        if isinstance(stats,dict) and stats.get('prepared_manifest'):
            stats['prepared_manifest']=_rebase_project_path(stats['prepared_manifest'],source_dataset,target_dataset)
        training=cloned.get('training')
        if isinstance(training,dict):
            if training.get('eval_reference'):
                training['eval_reference']=_rebase_project_path(training['eval_reference'],source_project,target_project)
            # A default adapter follows the project identity; custom adapter
            # names remain unchanged so every saved field is otherwise cloned.
            if not training.get('adapter_name') or training.get('adapter_name')==source_name:
                training['adapter_name']=new_name
        (target_project/'project.json').write_text(json.dumps(cloned,indent=2,ensure_ascii=False),encoding='utf-8')
    except Exception:
        if target_project.is_dir(): shutil.rmtree(target_project)
        if target_dataset.is_dir(): shutil.rmtree(target_dataset)
        raise
    return cloned

def dataset_summary(stats):
    if not stats: return 'No dataset scanned.'
    return (f"Samples: {stats.get('samples',0)} | Valid: {stats.get('valid',0)} | Missing text: {stats.get('missing_text',0)} | "
            f"Duration: {stats.get('total_seconds',0)/60:.1f} min | Median clip: {stats.get('median_seconds',0):.2f}s | "
            f"P90: {stats.get('p90_seconds',0):.2f}s | Auto range: {stats.get('auto_min_seconds',0):.2f}–{stats.get('auto_max_seconds',0):.2f}s")

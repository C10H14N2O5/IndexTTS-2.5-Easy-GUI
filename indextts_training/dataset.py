from __future__ import annotations
import functools, hashlib, json, os, re, secrets, shutil, statistics, threading, wave
from pathlib import Path

AUDIO_EXTS={'.wav','.flac','.mp3','.m4a','.ogg','.aac'}
DATASET_LOCK = threading.RLock()
_UNSET = object()


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


def validation_settings(project):
    saved = project.get('validation')
    defaults = dict(source_folder='', manifest='', stats={}, enabled=False, interval_steps=None)
    return {**defaults, **saved} if isinstance(saved, dict) else defaults


def normalize_validation_interval(value):
    if value is None:
        return None
    message = 'Validation interval must be a non-negative whole number of steps.'
    try:
        steps = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if isinstance(value, bool) or steps < 0 or steps != value:
        raise ValueError(message)
    return steps or None


@project_transaction
def configure_validation(project, *, source_folder=_UNSET, enabled=_UNSET,
                         interval_steps=_UNSET, root='training'):
    """Save validation options without rewriting either manifest."""
    meta = load_project(project, root)
    if not meta:
        raise ValueError('Select and save a project first.')
    previous = validation_settings(meta)
    settings = dict(previous)
    if source_folder is not _UNSET:
        folder = str(source_folder or '').strip()
        settings['source_folder'] = str(Path(folder).expanduser().resolve()) if folder else ''
        if not previous['source_folder'] or not folder:
            settings['enabled'] = bool(folder)
    if enabled is not _UNSET:
        settings['enabled'] = bool(enabled)
    if interval_steps is not _UNSET:
        settings['interval_steps'] = normalize_validation_interval(interval_steps)
    if settings != previous:
        meta['validation'] = settings
        _atomic_write(_project_file(project, root), json.dumps(meta, indent=2, ensure_ascii=False).encode('utf-8'))
    return meta


def _dataset_settings(meta, role):
    if role == 'training':
        return meta
    if role == 'validation':
        return validation_settings(meta)
    raise ValueError('Unknown dataset role.')


def _manifest_path(root, project, role):
    datasets = Path(root) / 'datasets'
    filename = 'validation.jsonl' if role == 'validation' else 'manifest.jsonl'
    path = datasets / project / filename
    resolved = path.resolve()
    if not resolved.is_relative_to(datasets.resolve()):
        raise ValueError('Invalid manifest destination.')
    if role == 'validation' and resolved != datasets.resolve() / project / filename:
        raise ValueError('Invalid manifest destination.')
    return path


def _decode_manifest(raw):
    rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Manifest rows must be JSON objects.')
    return rows


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
                    transcribe_missing=False, whisper_model=None, whisper_language=None, root='training',
                    *, role='training'):
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
        meta=load_project(safe, root)
        settings=_dataset_settings(meta, role)
        if role == 'validation':
            folder = settings['source_folder']
            if not folder or Path(folder).resolve() != Path(source_folder).resolve():
                raise ValueError('Save the project validation folder first.')
            language = meta.get('language', 'EN')
            meta['validation'] = settings
        manifest=_manifest_path(root, safe, role)
        before=_source_signature(source_folder)
        rows,stats=scan_dataset(source_folder,min_seconds,max_seconds)
        valid=_manifest_rows(rows,language)
        if not valid:
            raise ValueError('No valid audio+transcript pairs were found after filtering.')
        if before != _source_signature(source_folder):
            raise ValueError('Source files changed while scanning. Retry updating the manifest.')
        stats['prepared_manifest']=str(manifest.resolve())
        meta.update(project_name=safe, language=language)
        settings.update(source_folder=str(Path(source_folder).resolve()),
                        manifest=str(manifest.resolve()), stats=stats, source_signature=before,
                        manifest_bounds={'min':min_seconds,'max':max_seconds})
        manifest_bytes=('\n'.join(json.dumps(r,ensure_ascii=False) for r in valid)+'\n').encode('utf-8')
        settings['manifest_sha256']=hashlib.sha256(manifest_bytes).hexdigest()
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


def dataset_manifest_status(project, root='training', *, role='training'):
    """Read-only freshness check, including projects prepared before signatures existed."""
    if not project:
        return dict(stale=True,managed=False,manifest='',status='missing',
                    message=f'Select a project to update its {role} manifest.',stats={})
    with DATASET_LOCK:
        meta=load_project(project,root)
        settings=_dataset_settings(meta, role)
        manifest=str(settings.get('manifest') or '')
        result=dict(stale=True,managed=True,manifest=manifest,stats=settings.get('stats',{}))
        label=role.capitalize()
        if not manifest or not Path(manifest).is_file():
            return dict(result,status='missing',message=f'{label} manifest has not been prepared.')
        if not settings.get('source_folder'):
            source = 'validation folder' if role == 'validation' else 'source folder'
            return dict(result,status='missing',message=f'Save the project {source} first.')
        try:
            if role == 'validation' and Path(manifest).resolve() != _manifest_path(root, project, role).resolve():
                raise ValueError('Invalid manifest destination.')
            signature=_source_signature(settings['source_folder'])
            raw=Path(manifest).read_bytes()
            actual=_decode_manifest(raw)
            if role == 'validation' and not actual:
                return dict(result,status='invalid',message='Validation manifest is empty.')
            if settings.get('source_signature'):
                stale=(signature != settings['source_signature'] or
                       hashlib.sha256(raw).hexdigest() != settings.get('manifest_sha256'))
                # Saving a changed project language must also invalidate its previous manifest.
                stale=stale or any(r.get('language') != meta.get('language','EN') for r in actual)
            else:
                bounds=settings.get('manifest_bounds',{})
                rows,_=scan_dataset(settings['source_folder'],bounds.get('min'),bounds.get('max'))
                expected=_manifest_rows(rows,meta.get('language','EN'))
                def canonical(items):
                    return sorted((str(Path(r['audio']).resolve()),r['text'],r.get('reference_text',r['text']),r.get('language')) for r in items)
                stale=canonical(actual)!=canonical(expected)
            return dict(result,stale=stale,status='stale' if stale else 'ready',
                        message=f'{label} manifest needs updating.' if stale else f'{label} manifest is up to date.')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            return dict(result,status='invalid',message=f'Unable to verify {role} manifest: {exc}')


def refresh_project_manifest(project, root='training', *, role='training'):
    with DATASET_LOCK:
        meta=load_project(project,root)
        settings=_dataset_settings(meta, role)
        if not settings.get('source_folder'):
            if role == 'validation':
                raise ValueError('Save the project validation folder first.')
            raise ValueError('Save the project with a source folder first.')
        bounds=settings.get('manifest_bounds',{})
        return prepare_dataset(settings['source_folder'],project,meta.get('language','EN'),
                               bounds.get('min'),bounds.get('max'),root=root,role=role)


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


def _read_manifest_data(manifest, role):
    try:
        raw = Path(manifest).read_bytes()
        rows = _decode_manifest(raw)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f'Unable to read {role} manifest: {exc}') from exc
    if not rows:
        raise ValueError(f'{role.capitalize()} manifest is empty.')
    return raw, rows


def _manifest_audio_info(row, role):
    try:
        path = Path(row['audio']).resolve(strict=True)
        if not path.is_file():
            raise OSError('Expected a regular audio file.')
        return path, path.stat().st_size
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Unable to read audio listed in the {role} manifest: {row.get("audio", "")}') from exc


def _audio_digest(path, role):
    try:
        with path.open('rb') as audio:
            return hashlib.file_digest(audio, 'sha256').digest()
    except OSError as exc:
        raise ValueError(f'Unable to read audio listed in the {role} manifest: {path}') from exc


def _check_validation_overlap(training_rows, validation_rows):
    validation_paths = {}
    for row in validation_rows:
        path, size = _manifest_audio_info(row, 'validation')
        validation_paths[path] = size

    # Identical files have the same size. Only hash audio in the size groups
    # shared by the two manifests, and reuse each validation digest.
    validation_by_size = {}
    for path, size in validation_paths.items():
        validation_by_size.setdefault(size, []).append(path)

    validation_hashes = {}
    for row in training_rows:
        path, size = _manifest_audio_info(row, 'training')
        if path in validation_paths:
            duplicate = path
        elif size in validation_by_size:
            if size not in validation_hashes:
                validation_hashes[size] = {_audio_digest(candidate, 'validation'): candidate
                                           for candidate in validation_by_size[size]}
            duplicate = validation_hashes[size].get(_audio_digest(path, 'training'))
        else:
            duplicate = None
        if duplicate is not None:
            raise ValueError(f'Training and validation contain the same audio: {path} and {duplicate}')


def validate_validation_manifest(project, training_manifest, root='training', *,
                                 enabled=_UNSET, interval_steps=_UNSET):
    """Preflight enabled validation and snapshot its rows for the training worker."""
    if enabled is not _UNSET and not enabled:
        return None
    with DATASET_LOCK:
        settings = validation_settings(load_project(project, root))
        if enabled is _UNSET and not settings['enabled']:
            return None
        if not settings['source_folder']:
            raise ValueError('Select and save a validation source folder before starting training.')
        interval = normalize_validation_interval(settings['interval_steps'] if interval_steps is _UNSET else interval_steps)
        from .proofread import directory_asr_running, directory_lock
        with directory_lock(settings['source_folder']):
            if directory_asr_running(settings['source_folder']):
                raise ValueError('Finish recognition in the validation folder before starting training.')
            status = dataset_manifest_status(project, root, role='validation')
            if status['stale']:
                raise ValueError(status['message'])
            raw, rows = _read_manifest_data(status['manifest'], 'validation')
            fingerprint = hashlib.sha256(raw).hexdigest()
            if fingerprint != settings.get('manifest_sha256'):
                raise ValueError('Validation manifest needs updating.')
            training_raw, training_rows = _read_manifest_data(training_manifest, 'training')
            _check_validation_overlap(training_rows, rows)
            return dict(rows=rows, fingerprint=fingerprint, interval_steps=interval, manifest=status['manifest'],
                        training_fingerprint=hashlib.sha256(training_raw).hexdigest())


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
        validation=cloned.get('validation')
        if isinstance(validation,dict) and validation.get('manifest'):
            validation['manifest']=_rebase_project_path(validation['manifest'],source_dataset,target_dataset)
            stats=validation.get('stats')
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

from __future__ import annotations
import json, re, shutil, statistics, wave
from pathlib import Path

AUDIO_EXTS={'.wav','.flac','.mp3','.m4a','.ogg','.aac'}

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
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in AUDIO_EXTS or not p.is_file(): continue
        dur=_duration(p); txt=p.with_suffix('.txt'); text=txt.read_text(encoding='utf-8', errors='ignore').strip() if txt.is_file() else ''
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

def _transcribe_missing(rows, model_size='small', language=None, device='cuda'):
    missing=[r for r in rows if not r['text']]
    if not missing: return rows
    from faster_whisper import WhisperModel
    compute='float16' if device=='cuda' else 'int8'
    model=WhisperModel(model_size, device=device, compute_type=compute)
    for i,r in enumerate(missing,1):
        segs,_=model.transcribe(r['audio'], language=None if not language or language=='Auto-detect' else language, vad_filter=True)
        text=' '.join(s.text.strip() for s in segs).strip(); r['text']=text
        if text: Path(r['audio']).with_suffix('.txt').write_text(text+'\n', encoding='utf-8')
        print(f'[dataset] transcribed {i}/{len(missing)}: {Path(r["audio"]).name}', flush=True)
    del model
    return rows

def prepare_dataset(source_folder, project_name, language='EN', min_seconds=None, max_seconds=None,
                    transcribe_missing=False, whisper_model='small', whisper_language='Auto-detect', root='training'):
    rows,stats=scan_dataset(source_folder,min_seconds,max_seconds)
    min_seconds=float(stats.get('auto_min_seconds', .5) if min_seconds is None else min_seconds)
    max_seconds=float(stats.get('auto_max_seconds', 30.0) if max_seconds is None else max_seconds)
    if transcribe_missing:
        rows=_transcribe_missing(rows, whisper_model, whisper_language)
        for r in rows: r['valid']=bool(r['text']) and min_seconds <= r['duration'] <= max_seconds
        stats['missing_text']=sum(not bool(r['text']) for r in rows)
        stats['total_seconds']=sum(r['duration'] for r in rows)
        durations=[r['duration'] for r in rows]
        stats['median_seconds']=statistics.median(durations) if durations else 0.0
        stats['p90_seconds']=sorted(durations)[max(0,int(len(durations)*.9)-1)] if durations else 0.0
    valid=[r for r in rows if r['valid']]
    if not valid: raise ValueError('No valid audio+transcript pairs were found after filtering.')
    safe=''.join(c if c.isalnum() or c in '-_' else '_' for c in project_name.strip()).strip('_') or 'dataset'
    root=Path(root); project_dir=root/'projects'/safe; prepared_dir=root/'datasets'/safe
    project_dir.mkdir(parents=True, exist_ok=True); prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest=prepared_dir/'manifest.jsonl'
    with manifest.open('w',encoding='utf-8') as f:
        for r in valid:
            # The prepared dataset keeps the transcript explicit for both
            # target-text supervision and optional reference-text conditioning.
            # Existing manifests without this field remain supported by the
            # trainer, which falls back to ``text``.
            item={'audio':r['audio'],'text':r['text'],'reference_text':r['text'],
                  'duration':r['duration'],'language':language}
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
    stats['valid']=len(valid); stats['prepared_manifest']=str(manifest.resolve())
    project_file=project_dir/'project.json'
    previous={}
    if project_file.is_file():
        try: previous=json.loads(project_file.read_text(encoding='utf-8'))
        except Exception: previous={}
    meta={'project_name':safe,'source_folder':str(Path(source_folder).resolve()),'manifest':str(manifest.resolve()),'language':language,'stats':stats}
    if previous.get('training'):
        meta['training']=previous['training']
    project_file.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    return meta

def list_projects(root='training'):
    p=Path(root)/'projects'; p.mkdir(parents=True,exist_ok=True)
    return sorted(x.parent.name for x in p.glob('*/project.json'))

def load_project(name, root='training'):
    p=Path(root)/'projects'/str(name)/'project.json'
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

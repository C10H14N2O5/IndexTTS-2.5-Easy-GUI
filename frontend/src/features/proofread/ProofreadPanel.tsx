import { useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type Ref } from 'react';
import { useBlocker, type BlockerFunction } from 'react-router';
import { ArrowLeft, ArrowRight, AudioLines, Check, Circle, FileCheck2, FileText, RefreshCw, Save, Search, Square } from 'lucide-react';
import { Badge, Button, Dialog, EmptyState, Field, IconButton, InlineAlert, ProgressBar, Select, Spinner, TextArea, TextInput } from '@/ui';
import { useT } from '@/i18n';
import { ProofreadApi, type DatasetRole, type ProofreadContext } from './api';
import { ProofreadController, type UnsavedChoice } from './controller';
import { ProofreadAudio } from './ProofreadAudio';
import styles from './Proofread.module.css';

export interface ProofreadPanelHandle {
  /** Run a binding mutation only after resolving edits; keep the panel locked through its promise. */
  prepareContextChange(commit: () => Promise<unknown>): Promise<boolean>;
}
export interface ProofreadPanelProps { project: string; directory: string; role: DatasetRole; language?: string; onUpdated?: () => void; onContextLockedChange?: (locked: boolean) => void; ref?: Ref<ProofreadPanelHandle> }
export function ProofreadPanel({ project, directory, role, language = 'EN', onUpdated, onContextLockedChange, ref }: ProofreadPanelProps) {
  const t = useT();
  const [unsaved, setUnsaved] = useState(false);
  const choiceResolver = useRef<((choice: UnsavedChoice) => void) | null>(null);
  const [controller] = useState(() => new ProofreadController(new ProofreadApi(), () => new Promise<UnsavedChoice>(resolve => { choiceResolver.current = resolve; setUnsaved(true); })));
  const state = useSyncExternalStore(controller.subscribe, controller.getSnapshot, controller.getSnapshot);
  const requested = useMemo<ProofreadContext>(() => ({ project, directory, role, language }), [project, directory, role, language]);
  const initialContext = useRef(requested);
  const audio = useRef<HTMLAudioElement>(null);
  const editor = useRef<HTMLTextAreaElement>(null);
  const [speed, setSpeed] = useState('1');
  const dirty = controller.dirty;
  const busy = Boolean(state.busy);
  const running = state.asr.job.running;
  const visible = controller.visible;
  const index = visible.findIndex(item => item.id === state.current?.id);
  const done = state.items.filter(item => item.reviewed).length;
  const blocker = useBlocker(useCallback<BlockerFunction>(({ currentLocation, nextLocation }) => currentLocation.pathname !== nextLocation.pathname && (controller.dirty || Boolean(controller.getSnapshot().busy) || controller.getSnapshot().asr.job.running), [controller]));
  const choosing = useRef(false);
  const locked = controller.contextLocked;
  const lockCallback = useRef(onContextLockedChange);
  useImperativeHandle(ref, () => ({ prepareContextChange: controller.prepareContextChange }), [controller]);
  useEffect(() => { lockCallback.current = onContextLockedChange; onContextLockedChange?.(locked); }, [locked, onContextLockedChange]);
  useEffect(() => () => lockCallback.current?.(false), []);

  useEffect(() => { controller.setOnUpdated(onUpdated ?? (() => {})); }, [controller, onUpdated]);
  useEffect(() => { void controller.start(initialContext.current); const timer = window.setInterval(() => void controller.poll(), 1500); return () => { window.clearInterval(timer); choiceResolver.current?.('cancel'); choiceResolver.current = null; controller.dispose(); }; }, [controller]);
  useLayoutEffect(() => { controller.setContext(requested); }, [controller, requested]);
  useEffect(() => {
    if (blocker.state !== 'blocked' || choosing.current) return;
    choosing.current = true;
    void controller.allowNavigation().then(allowed => { choosing.current = false; if (allowed) blocker.proceed(); else blocker.reset(); });
  }, [blocker, controller]);
  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => { if (controller.dirty || controller.getSnapshot().busy || controller.getSnapshot().asr.job.running) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', beforeUnload);
    return () => window.removeEventListener('beforeunload', beforeUnload);
  }, [controller]);
  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      if (unsaved || event.isComposing) return;
      const current = controller.getSnapshot();
      if ((event.ctrlKey || event.metaKey) && (event.key.toLowerCase() === 's' || event.key === 'Enter')) {
        event.preventDefault();
        if (current.current && !current.busy) void (event.key === 'Enter' ? controller.saveNext() : controller.save());
      } else if (event.altKey && event.code === 'Space' && current.current) {
        event.preventDefault();
        const player = audio.current;
        if (!player) return;
        if (player.paused) void player.play().catch(error => controller.setError(error instanceof Error ? error.message : String(error)));
        else player.pause();
      }
    };
    document.addEventListener('keydown', shortcut);
    return () => document.removeEventListener('keydown', shortcut);
  }, [controller, unsaved]);

  function choose(choice: UnsavedChoice) { const resolve = choiceResolver.current; choiceResolver.current = null; setUnsaved(false); resolve?.(choice); }
  const asrReason = !state.context.directory ? t('Save a corpus folder first.') : !state.asr.available ? (state.asr.message || t('Speech recognition is unavailable.')) : state.asr.directory_running && !running ? t('This folder already has a transcription task in another session.') : undefined;
  const manifestReason = !state.context.project || !state.context.directory ? t('Save a project and its corpus folder first.') : state.manifest?.training_running ? t('Stop training before updating this manifest.') : running || state.manifest?.asr_running ? t('Finish transcription before updating this manifest.') : state.manifest?.available === false ? state.manifest.message || t('Save the selected corpus folder in project settings first.') : undefined;
  const stats = state.manifest?.stats;
  const currentReviewed = state.current?.reviewed && !dirty;
  const job = state.asr.job;
  const asrSummary = running ? t('{0} / {1} · {2} succeeded · {3} skipped · {4} failed', job.completed, job.total, job.succeeded, job.skipped, job.failed) : job.cancelled ? t('Transcription stopped. Completed results were kept.') : job.total > 0 ? t('Transcription finished: {0} succeeded, {1} skipped, {2} failed.', job.succeeded, job.skipped, job.failed) : t('Only missing or blank TXT files are filled. Existing text is never replaced.');

  return <section className={styles.panel} aria-label={t('Audio proofreading')} aria-busy={busy}>
    <header className={styles.header}><div><h2>{t('Audio proofreading')}</h2><p>{t('Listen, refine the transcript, and mark each clip reviewed.')}</p></div><div className={styles.headerActions}><Badge tone="neutral">{t(state.ready && state.context.role === 'validation' || !state.ready && role === 'validation' ? 'Validation set' : 'Training set')}</Badge><IconButton icon={RefreshCw} label={t('Refresh proofreading files')} pending={state.busy === 'refresh'} disabled={busy || !state.ready} onClick={() => void controller.refresh()} /></div></header>
    {state.retainedContext && <InlineAlert tone="warning" title={t('Previous dataset retained')} action={<Button size="sm" variant="outlined" disabled={busy || running} onClick={() => controller.setContext(requested)}>{t('Switch to saved dataset')}</Button>}>{t('The context change was cancelled. You are still editing {0} · {1}.', state.context.project, state.context.role === 'validation' ? t('Validation set') : t('Training set'))}<div className={styles.path}>{state.context.directory}</div></InlineAlert>}
    {state.error && <InlineAlert tone="error" title={state.errorStatus === 409 ? t('The transcript changed on disk') : state.errorStatus === 423 ? t('This dataset is locked') : t('Proofreading could not complete the operation')} action={state.current ? <Button variant="text" size="sm" onClick={() => { editor.current?.focus(); editor.current?.select(); }}>{t('Select transcript to copy')}</Button> : undefined}>{state.error}{state.errorStatus === 409 && <p>{t('Your edits are preserved. Copy them before reloading the saved text.')}</p>}</InlineAlert>}
    {state.notice && !state.error && <div className={styles.notice} role="status">{state.notice}</div>}
    {!state.ready ? <div className={styles.loading}>{busy ? <><Spinner /><span>{t('Opening proofreading session…')}</span></> : <Button variant="outlined" onClick={() => void controller.start(requested)}>{t('Retry')}</Button>}</div> : <>
      <div className={styles.contextPath} title={state.context.directory}><FileText size={14}/><span>{state.context.directory || t('No corpus folder selected')}</span></div>
      <div className={styles.pipeline}>
        <div><div className={styles.pipelineHeader}><strong>SenseVoiceSmall Q8 <span>· CPU</span></strong>{running ? <Button icon={Square} size="sm" variant="outlined" pending={state.busy === 'batch-stop'} disabled={busy} onClick={() => void controller.stopBatch()}>{t('Stop transcription')}</Button> : <Button size="sm" variant="tonal" pending={state.busy === 'batch-start'} disabled={busy} disabledReason={asrReason} onClick={() => void controller.startBatch()}>{t('Transcribe missing text')}</Button>}</div><p>{asrSummary}</p>{running && <><ProgressBar value={job.total ? job.completed / job.total : null} label={t('Transcription progress')}/><p className={styles.currentFile}>{job.current}</p></>}{job.error && <InlineAlert tone="error">{job.error}</InlineAlert>}<small>{t('Search and filters do not limit batch transcription; subfolders are included.')}</small></div>
        <div><div className={styles.pipelineHeader}><Badge tone={!state.manifest || state.manifest.available === false ? 'neutral' : state.manifest?.stale ? 'warning' : 'success'}>{!state.manifest || state.manifest.available === false ? t('Project setup needed') : state.manifest?.stale ? t('Update needed') : t('Up to date')}</Badge><Button icon={FileCheck2} size="sm" variant="outlined" pending={state.busy === 'manifest'} disabled={busy} disabledReason={manifestReason} onClick={() => void controller.updateManifest()}>{state.context.role === 'validation' ? t('Update validation manifest') : t('Update training manifest')}</Button></div><p>{state.manifest?.status === 'missing' ? t('The manifest has not been prepared.') : state.manifest?.stale ? t('The dataset changed. Update the manifest before training.') : state.manifest?.status === 'ready' ? t('The manifest matches the saved dataset.') : state.manifest?.message || manifestReason}</p>{stats && Number.isFinite(stats.samples) && Number.isFinite(stats.valid) && <p>{t('{0} clips · {1} valid · {2} filtered · {3} missing text', stats.samples, stats.valid, stats.samples - stats.valid, stats.missing_text ?? 0)}</p>}{state.manifest?.manifest && <small className={styles.path}>{state.manifest.manifest}</small>}</div>
      </div>
      <div className={styles.workbench}>
        <aside className={styles.fileSidebar}><div className={styles.progressLabel}><strong>{t('Review progress')}</strong><span>{done} / {state.items.length}</span></div><ProgressBar value={state.items.length ? done / state.items.length : 0} label={t('Review progress')}/><div className={styles.search}><Search size={15}/><TextInput type="search" aria-label={t('Search relative file paths')} placeholder={t('Search files…')} value={state.query} onChange={event => controller.setQuery(event.target.value)} disabled={busy}/></div><Select value={state.filter} onChange={value => controller.setFilter(value)} aria-label={t('Review status filter')} disabled={busy} options={[{ value: 'all', label: t('All clips') }, { value: 'pending', label: t('Pending review') }, { value: 'reviewed', label: t('Reviewed') }, { value: 'missing', label: t('Missing text') }]}/><nav className={styles.fileList} aria-label={t('Audio clips')}>{visible.length ? visible.map(item => <button type="button" key={item.id} title={item.name} aria-current={state.current?.id === item.id ? 'true' : undefined} className={styles.file} disabled={busy} onClick={() => void controller.select(item.id)}><span className={styles.reviewMarker} data-reviewed={item.reviewed || undefined}>{item.reviewed ? <Check size={13}/> : <Circle size={11}/>}</span><span>{item.name}</span>{item.missing_text && <span className={styles.missing}>{t('No text')}</span>}{item.error && <span className={styles.missing} title={item.error}>!</span>}</button>) : <p className={styles.listEmpty}>{state.items.length ? t('No files match this filter.') : t('No supported audio files in this folder.')}</p>}</nav></aside>
        <div className={styles.editorPane}>{state.current ? <>
          <div className={styles.editorHeader}><div><h3>{state.current.name}</h3><p>{index >= 0 ? t('Clip {0} of {1} in this list', index + 1, visible.length) : t('This clip is outside the active filter.')}</p></div><Badge tone={currentReviewed ? 'success' : 'neutral'}>{currentReviewed ? t('Reviewed') : t('Pending review')}</Badge></div>
          <ProofreadAudio src={state.audioSrc} audioRef={audio} speed={speed} onSpeedChange={setSpeed} onError={message => controller.setError(message)}/>
          <Field label={<span className={styles.transcriptLabel}>{t('Transcript')}<span data-dirty={dirty || undefined}>{dirty ? t('Unsaved changes') : t('Saved')}</span></span>} hint={t('Keep the actual words, repetitions and interjections. Do not add words absent from the recording.')}><TextArea ref={editor} aria-label={t('Transcript')} spellCheck={false} minRows={6} maxRows={14} showCount value={state.draft} disabled={busy} placeholder={t('Listen and enter the transcript…')} onChange={event => controller.setDraft(event.target.value)}/></Field>
          <div className={styles.editorActions}><div><Button variant="outlined" icon={ArrowLeft} disabled={busy || index <= 0} onClick={() => void controller.move(-1)}>{t('Previous clip')}</Button><Button variant="outlined" icon={ArrowRight} disabled={busy || index < 0 || index >= visible.length - 1} onClick={() => void controller.move(1)}>{t('Next clip')}</Button></div><div><Button variant="tonal" icon={Save} pending={state.busy === 'save'} disabled={busy} onClick={() => void controller.save()}>{t('Save transcript')}</Button><Button icon={ArrowRight} pending={state.busy === 'save-next'} disabled={busy} onClick={() => void controller.saveNext()}>{t('Save and next')}</Button></div></div>
          <div className={styles.secondaryActions}><Button variant="text" size="sm" disabled={busy || !state.draft.trim()} onClick={() => void controller.review()}>{currentReviewed ? t('Mark as pending') : t('Mark as reviewed')}</Button><Button variant="text" size="sm" pending={state.busy === 'transcribe'} disabled={busy || running} disabledReason={!state.asr.available ? state.asr.message || t('Speech recognition is unavailable.') : undefined} onClick={() => void controller.transcribe()}>{t('Transcribe this clip again')}</Button><Button variant="text" size="sm" pending={state.busy === 'reload'} disabled={busy} onClick={() => void controller.reload()}>{t('Reload saved text')}</Button></div>
        </> : <EmptyState icon={AudioLines} title={t('Start with a clip')} description={t('Save your corpus folder above, then select an audio file. Missing transcripts can be generated in one batch.')}/>}</div>
      </div>
      <footer className={styles.shortcuts}><span><kbd>Ctrl S</kbd>{t('Save transcript')}</span><span><kbd>Ctrl Enter</kbd>{t('Save and next')}</span><span><kbd>Alt Space</kbd>{t('Play / pause')}</span><span>{t('Audio never autoplays when a clip is selected.')}</span></footer>
    </>}
    <Dialog open={unsaved} onOpenChange={open => { if (!open) choose('cancel'); }} title={t('Unsaved transcript')} description={t('Save your edits, discard them, or stay on this clip.')} size="sm" footer={<><Button variant="text" onClick={() => choose('cancel')}>{t('Cancel')}</Button><Button variant="outlined" onClick={() => choose('discard')}>{t('Discard edits')}</Button><Button onClick={() => choose('save')}>{t('Save and continue')}</Button></>}><p>{state.current?.name}</p></Dialog>
  </section>;
}
export default ProofreadPanel;

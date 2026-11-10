import { useEffect, useRef, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router';
import { AudioLines, Play, RotateCcw, Save } from 'lucide-react';
import { AudioPlayer, Badge, Button, ConfirmDialog, Disclosure, EmptyState, Field, InlineAlert, PageHeader, ParamField, Section, SegmentedControl, Select, Switch, TextArea, toast } from '@/ui';
import { formatDuration } from '@/ui/format';
import { errorMessage, isApiError, mediaUrl } from '@/api/client';
import { useGpuJob, useLive } from '@/api/live';
import { useCatalog } from '@/app/bootstrap';
import { useT } from '@/i18n';
import { VoiceEditorDialog } from '@/features/library/VoiceEditorDialog';
import { TranscribeButton } from '@/features/library/TranscribeButton';
import { defaultForm, newTurn, PARAM_LABELS, restoreForm, validateForm, type FormErrors, type NumericParam, type SynthesisForm } from './model';
import { generateSpeech, historyKey, isSynthesisResult, useTrackedJob, type HistoryItem } from './api';
import { usePersistedForm } from './usePersistedForm';
import { VoicePicker } from './VoicePicker';
import { GenerationOptions } from './GenerationOptions';
import { EmotionControls } from './EmotionControls';
import { DialogueEditor } from './DialogueEditor';
import { HistoryPanel } from './HistoryPanel';
import { JobCard } from './JobCard';
import styles from './Synthesis.module.css';

export default function SynthesisPage() {
  const t = useT(); const { mode } = useParams(); const dialogue = mode === 'dialogue';
  const navigate = useNavigate(); const [search, setSearch] = useSearchParams(); const catalog = useCatalog();
  const [form, setForm] = usePersistedForm(); const [errors, setErrors] = useState<FormErrors>({});
  const [resetOpen, setResetOpen] = useState(false); const [saveVoice, setSaveVoice] = useState(false);
  const tracked = useTrackedJob(); const gpu = useGpuJob(); const training = useLive((state) => state.training); const connection = useLive((state) => state.connection);
  const client = useQueryClient(); const submitRef = useRef<HTMLDivElement>(null);
  const mutation = useMutation({ mutationFn: () => generateSpeech(form, dialogue), onSuccess: ({ job }) => tracked.track(job) });
  useEffect(() => {
    const adapter = search.get('adapter');
    if (adapter) { setForm((old) => ({ ...old, adapter_path: adapter })); setSearch({}, { replace: true }); }
  }, [search, setSearch, setForm]);
  useEffect(() => { if (tracked.job?.state === 'succeeded') void client.invalidateQueries({ queryKey: historyKey }); }, [tracked.job?.state, tracked.job?.id, client]);
  function patch(values: Partial<SynthesisForm>) { setForm((old) => ({ ...old, ...values })); setErrors({}); }
  function submit() {
    const next = validateForm(form, dialogue); setErrors(next);
    if (Object.keys(next).length) { submitRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); return; }
    mutation.mutate();
  }
  function reuse(item: HistoryItem) {
    if (!item.params) return;
    const params = item.params;
    const turns = Array.isArray(params.turns) ? params.turns.map((turn) => newTurn(turn as Partial<SynthesisForm['turns'][number]>)) : form.turns;
    setForm(restoreForm({ ...form, ...params, turns }));
    navigate(item.kind === 'dialogue' ? '/synthesis/dialogue' : '/synthesis');
    toast.info(t('Settings restored. Review the voice and text before generating.'));
  }
  const result = isSynthesisResult(tracked.job?.result) ? tracked.job.result : null;
  const busyJob = isApiError(mutation.error) ? mutation.error.busyJob : undefined;
  const busyReason = training?.running ? t('Training is using the GPU. Stop training before synthesizing.') : gpu && gpu.id !== tracked.job?.id ? t('The GPU is busy: {0}', gpu.title) : connection === 'reconnecting' ? t('Waiting for the server connection…') : undefined;
  const running = mutation.isPending || tracked.job?.state === 'running';
  return <div className={styles.page}>
    <PageHeader title={t('Synthesis')} description={t('Give your words a voice. Start with a reference clip and the text you want to hear.')} actions={<Button variant="text" icon={RotateCcw} onClick={() => setResetOpen(true)}>{t('Reset form')}</Button>} />
    <div className={styles.toolbar}><SegmentedControl value={dialogue ? 'dialogue' : 'single'} onChange={(value) => navigate(value === 'dialogue' ? '/synthesis/dialogue' : '/synthesis')} options={[{ value: 'single', label: t('Single voice') }, { value: 'dialogue', label: t('Dialogue') }]} /><span className={styles.muted}>{t('Your draft is saved in this browser.')}</span></div>
    <div className={styles.layout}>
      <div className={styles.stack} ref={submitRef}>
        {Object.keys(errors).length > 0 && <InlineAlert tone="error" title={t('Review the form before generating.')}><ul>{Object.keys(errors).filter((key) => key in PARAM_LABELS).map((key) => <li key={key}>{t(PARAM_LABELS[key as NumericParam])}: {t('Check this value.')}</li>)}</ul></InlineAlert>}
        <Section title={dialogue ? t('Build a conversation') : t('What would you like to say?')} description={dialogue ? t('Each turn has its own voice and delivery. Empty turns are skipped.') : undefined}>
          {dialogue ? <DialogueEditor turns={form.turns} onChange={(turns) => patch({ turns })} errors={errors} /> : <div className={styles.fields}>
            <Field label={t('Target text')} error={errors.text ? t(errors.text) : undefined} required><TextArea aria-label={t('Target text')} placeholder={t('Write a sentence, a story, or a whole new idea…')} value={form.text} onChange={(event) => patch({ text: event.target.value })} minRows={8} maxRows={18} showCount /></Field>
            <div className={styles.two}><Field label={t('Text chunking')} hint={t('Split longer passages into manageable pieces.')}><Select aria-label={t('Text chunking')} value={form.chunk_mode} onChange={(value) => patch({ chunk_mode: value })} options={catalog.chunk_modes.map((value) => ({ value: value as SynthesisForm['chunk_mode'], label: t(value) }))} /></Field>
              {form.chunk_mode !== 'None' && <ParamField label={t('Pause between chunks')} value={form.chunk_silence} defaultValue={0.5} onChange={(value) => patch({ chunk_silence: value })} min={0} max={5} step={0.1} unit={t('s')} />}</div>
          </div>}
        </Section>
        {!dialogue && <Section><EmotionControls form={form} onChange={patch} error={errors.emotion_audio ? t(errors.emotion_audio) : undefined} /></Section>}
        <Section title={t('Generation settings')}><GenerationOptions form={form} onChange={patch} />{dialogue && <ParamField label={t('Pause between speakers')} value={form.dialogue_silence} defaultValue={0.5} onChange={(value) => patch({ dialogue_silence: value })} min={0} max={5} step={0.1} unit={t('s')} />}</Section>
        {(busyReason || busyJob) && <InlineAlert tone="warning" title={busyReason ?? t('The GPU is busy: {0}', busyJob!.title)} action={<Link to={training?.running ? '/projects' : '/synthesis'}>{training?.running ? t('Open training projects') : t('View active task in Activity')}</Link>}>{t('Tasks are not queued. Try again once the current task has finished.')}{busyJob && <JobCard job={busyJob} />}</InlineAlert>}
        {mutation.error && <InlineAlert tone="error" title={t('Could not start synthesis')}>{errorMessage(mutation.error)}</InlineAlert>}
        <div className={styles.actionbar}><Switch checked={form.save_outputs} onCheckedChange={(value) => patch({ save_outputs: value })} label={t('Save output')} description={form.save_outputs ? t('Keep the audio and settings in recent results.') : t('Preview only; temporary audio may be removed.')} /><Button icon={Play} pending={running} pendingLabel={t('Generating…')} disabledReason={busyReason} onClick={submit}>{dialogue ? t('Generate dialogue') : t('Generate speech')}</Button></div>
      </div>
      <aside className={styles.stack}>
        {!dialogue && <Section title={t('Choose your voice')} description={t('The reference defines the speaker, not the words.')}>
          <div className={styles.fields}><VoicePicker path={form.voice_path} name={form.voice_name} onChange={(voice_path, voice_name, transcript) => patch({ voice_path, voice_name, reference_transcript: transcript ?? '' })} error={errors.voice_path ? t(errors.voice_path) : undefined} />
            <Button icon={Save} variant="text" size="sm" disabledReason={!form.voice_path ? t('Add a reference clip first.') : undefined} onClick={() => setSaveVoice(true)}>{t('Save voice to library')}</Button>
            <Disclosure title={t('Reference transcript')} summary={t('Optional pronunciation guidance')} modifiedCount={Number(Boolean(form.reference_transcript)) + Number(form.reference_auto_transcribe)}>
              <div className={styles.fields}><TextArea aria-label={t('Reference transcript')} value={form.reference_transcript} onChange={(event) => patch({ reference_transcript: event.target.value })} placeholder={t('The exact words in your reference clip…')} minRows={3} /><TranscribeButton path={form.voice_path} onTranscript={(reference_transcript) => patch({ reference_transcript })} /><Switch checked={form.reference_auto_transcribe} onCheckedChange={(value) => patch({ reference_auto_transcribe: value })} label={t('Transcribe automatically when empty')} /></div>
            </Disclosure>
          </div>
        </Section>}
        <Section title={t('Your result')}>
          <div className={styles.result}>
            {tracked.job ? <JobCard job={tracked.job} /> : <EmptyState icon={AudioLines} title={t('Ready when you are')} description={t('Your audio will appear here, with live progress along the way.')} />}
            {tracked.error && <InlineAlert tone="error">{errorMessage(tracked.error)}</InlineAlert>}
            {result && <><AudioPlayer src={mediaUrl(result.path, tracked.job?.id)} title={result.name} downloadName={result.name} /><div className={styles.resultMeta}><span>{formatDuration(result.duration ?? 0)}</span><Badge tone={result.saved ? 'success' : 'neutral'}>{result.saved ? t('Saved locally') : t('Temporary preview')}</Badge></div></>}
          </div>
        </Section>
        <HistoryPanel onReuse={reuse} />
      </aside>
    </div>
    <ConfirmDialog open={resetOpen} onOpenChange={setResetOpen} title={t('Reset this draft?')} description={t('This clears your text and turns and restores all synthesis defaults. Saved recordings are kept.')} confirmLabel={t('Reset form')} onConfirm={() => { setForm(defaultForm()); setErrors({}); setResetOpen(false); }} />
    <VoiceEditorDialog open={saveVoice} onOpenChange={setSaveVoice} initial={form.voice_path ? { path: form.voice_path, name: form.voice_name, transcript: form.reference_transcript } : undefined} onSaved={(voice_path, voice_name) => patch({ voice_path, voice_name })} />
  </div>;
}

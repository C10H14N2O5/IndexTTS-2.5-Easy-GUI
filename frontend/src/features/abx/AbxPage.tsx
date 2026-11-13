import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router';
import { ArrowLeft, ArrowRight, FlaskConical, Play, Plus, RefreshCw, Trash2, Unlock } from 'lucide-react';
import { api, errorMessage, mediaUrl, type Schemas } from '@/api/client';
import { useGpuJob, useJobs, useLive, type Job } from '@/api/live';
import { useCatalog, useSettings } from '@/app/bootstrap';
import { useT } from '@/i18n';
import { AudioPlayer, Badge, Button, ConfirmDialog, Disclosure, EmptyState, Field, InlineAlert, KeyValueList, NumberInput, PageHeader, ProgressBar, Section, Select, Skeleton, Stat, TextArea, TextInput, toast } from '@/ui';
import { useVoices } from '@/features/library/api';
import { useAdapters, useTrackedJob } from '@/features/synthesis/api';
import { JobCard } from '@/features/synthesis/JobCard';
import { readSavedForm } from '@/features/synthesis/model';
import { COMPARISON_DRAFT_KEY, comparisonPlan, decodingSettings, readComparisonDraft, SCORE_LABELS, VERDICTS, type ComparisonResults } from './model';
import s from '@/features/projects/Workbench.module.css';

type Session = Schemas['AbxSession'];
const sessionUrl = (id: string) => `/api/abx/sessions/${encodeURIComponent(id)}`;

export default function AbxPage() {
  const t = useT(); const [params, setParams] = useSearchParams(); const client = useQueryClient();
  const sessions = useQuery({ queryKey: ['abx-sessions'], queryFn: () => api.get<Schemas['AbxSummary'][]>('/api/abx/sessions') });
  const [selected, setSelected] = useState(params.get('session') ?? ''); const [index, setIndex] = useState(0); const [creating, setCreating] = useState(!params.get('session')); const [revealOpen, setRevealOpen] = useState(false); const [deleteOpen, setDeleteOpen] = useState(false);
  const sid = selected || sessions.data?.[0]?.id || ''; const tracked = useTrackedJob(); const jobs = useJobs(); const job = tracked.job?.meta?.session_id === sid ? tracked.job : jobs.find(item => item.kind === 'abx_generate' && item.meta?.session_id === sid); const gpu = useGpuJob();
  const training = useLive(state => state.training?.running); const exporting = useLive(state => state.export?.status === 'running' || state.export?.status === 'cancelling');
  const detail = useQuery({ queryKey: ['abx-session', sid, index], queryFn: () => api.get<Session>(sessionUrl(sid), { trial: index }), enabled: Boolean(sid), refetchInterval: job?.state === 'running' ? 1500 : false });
  useEffect(() => { if (job && job.state !== 'running') { void client.invalidateQueries({ queryKey: ['abx-sessions'] }); void client.invalidateQueries({ queryKey: ['abx-session', String(job.meta?.session_id)] }); } }, [client, job]);
  function select(id: string) { setSelected(id); setIndex(0); setCreating(false); void setParams({ session: id }); }
  const generation = useMutation({ mutationFn: (id: string) => api.post<{ job: Job }>(`${sessionUrl(id)}/generate`), onSuccess: ({ job: value }) => tracked.track(value) });
  const rate = useMutation({ mutationFn: (score: number | null) => api.put<Session>(`${sessionUrl(sid)}/rate`, { index, score }), onSuccess: data => { client.setQueryData(['abx-session', sid, index], data); void client.invalidateQueries({ queryKey: ['abx-sessions'] }); } });
  const reveal = useMutation({ mutationFn: () => api.post<Session>(`${sessionUrl(sid)}/reveal`, {}, { trial: index }), onSuccess: data => { client.setQueryData(['abx-session', sid, index], data); void client.invalidateQueries({ queryKey: ['abx-sessions'] }); toast.success(t('Results revealed. Ratings are now locked.')); } });
  const remove = useMutation({ mutationFn: (id: string) => api.delete<Schemas['Ok']>(sessionUrl(id)), onSuccess: (result, id) => {
    // Drop the deleted entry right away so the page never refetches a session that no longer exists.
    const list = sessions.data ?? []; const position = Math.max(0, list.findIndex(item => item.id === id)); const rest = list.filter(item => item.id !== id); const next = rest[Math.min(position, rest.length - 1)];
    client.setQueryData(['abx-sessions'], rest); client.removeQueries({ queryKey: ['abx-session', id] }); void client.invalidateQueries({ queryKey: ['abx-sessions'] });
    setSelected(next?.id ?? ''); setIndex(0); setCreating(open => open || !next); setRevealOpen(false); setDeleteOpen(false);
    void setParams(previous => { const value = new URLSearchParams(previous); value.delete('session'); return value; }, { replace: true });
    toast.success(result?.message || t('Blind comparison deleted.'));
  } });
  const summary = sessions.data?.find(item => item.id === sid); const generating = job?.state === 'running';
  const busy = training ? t('Wait for training to finish.') : exporting ? t('Wait for checkpoint export to finish.') : gpu ? t('The GPU is busy: {0}', gpu.title) : undefined;
  const session = detail.data; const trial = session?.trial;
  const revealedFirst = session?.revealed && session.results ? (session.results as unknown as ComparisonResults).rows.find(row => row.index === index)?.sample_1 : undefined;
  return <div className={s.stack}>
    <PageHeader title={t('Blind A/B')} description={t('Listen without labels. Compare paired samples before revealing which adapter made them.')} actions={<Button variant="outlined" icon={Plus} onClick={() => setCreating(true)}>{t('New comparison')}</Button>}/>
    {creating && <CreateComparison adapterA={params.get('a')} adapterB={params.get('b')} busyReason={busy} onCreated={result => { select(result.id); generation.mutate(result.id); }}/>}
    <Section title={t('Saved comparisons')} actions={<Button variant="text" size="sm" icon={RefreshCw} pending={sessions.isFetching} onClick={() => void sessions.refetch()}>{t('Refresh')}</Button>}>
      {sessions.error && <InlineAlert tone="error">{errorMessage(sessions.error)}</InlineAlert>}
      {sessions.data?.length ? <div className={s.toolbar} style={{ alignItems: 'flex-end' }}><div style={{ flex: 1, minWidth: 240 }}><Field label={t('Comparison')}><Select value={sid} onChange={select} options={sessions.data.map(item => ({ value: item.id, label: `${item.name || item.id} · ${t('{0}/{1} rated', item.rated, item.trials)}${item.revealed ? ` · ${t('Revealed')}` : ''}` }))}/></Field></div><Button variant="text" icon={Trash2} disabled={!summary} disabledReason={generating ? t('Stop generating before deleting this comparison.') : undefined} onClick={() => setDeleteOpen(true)}>{t('Delete comparison')}</Button></div> : <EmptyState icon={FlaskConical} title={t('No blind comparisons yet')} description={t('Choose two different adapters above, including the original model if you wish.')}/>}
    </Section>
    {[detail.error, generation.error, rate.error, reveal.error, tracked.error].filter(Boolean).map((error, n) => <InlineAlert key={n} tone="error">{errorMessage(error)}</InlineAlert>)}
    {job && <JobCard job={job}/>}
    {sid && detail.isPending && <Skeleton height={240}/>}
    {session && !session.generation_complete && <Section title={t('Prepare the listening samples')}><div className={s.fields}><ProgressBar value={session.clips_done / Math.max(1, session.clips_total)} label={t('Sample generation')}/><div className={s.toolbar}><span>{t('{0} of {1} clips ready', session.clips_done, session.clips_total)}</span><Button icon={Play} pending={generation.isPending} disabledReason={busy} onClick={() => generation.mutate(sid)}>{session.clips_done ? t('Continue generating') : t('Generate samples')}</Button></div><p className={s.muted}>{t('Completed clips are retained after cancellation or failure. Continue to generate only the missing clips.')}</p></div></Section>}
    {session && trial && <Section title={t('Trial {0} of {1}', index + 1, session.trials)} actions={<Badge tone={session.revealed ? 'neutral' : 'accent'}>{session.revealed ? t('Revealed') : t('Blind listening')}</Badge>}>
      <div className={s.fields}><ProgressBar value={session.rated / Math.max(1, session.trials)} label={t('Rating progress')}/><div className={s.toolbar}><span className={s.muted}>{t('{0}/{1} rated', session.rated, session.trials)}</span><span className={s.muted}>{t('Repetition {0}', trial.rep + 1)}</span></div>
        <div className={s.sentence}>{trial.text}</div><AudioPlayer src={mediaUrl(trial.reference)} title={t('Reference voice')} compact/>
        <div className={s.two}><AudioPlayer src={mediaUrl(trial.sample_1)} title={revealedFirst ? t('Sample 1 · Adapter {0}', revealedFirst) : t('Sample 1')}/><AudioPlayer src={mediaUrl(trial.sample_2)} title={revealedFirst ? t('Sample 2 · Adapter {0}', revealedFirst === 'A' ? 'B' : 'A') : t('Sample 2')}/></div>
        <div className={s.radioGrid} role="group" aria-label={t('Which sample sounds better?')}>{SCORE_LABELS.map((label, i) => <button key={label} type="button" className={s.rating} aria-pressed={trial.score === i - 2} disabled={session.revealed || rate.isPending} onClick={() => rate.mutate(i - 2)}>{t(label)}</button>)}</div>
        <div className={s.toolbar}><div className={s.actions}><Button variant="outlined" icon={ArrowLeft} disabled={rate.isPending || index === 0} onClick={() => setIndex(index - 1)}>{t('Previous')}</Button><Button variant="outlined" icon={ArrowRight} disabled={rate.isPending || index >= session.trials - 1} onClick={() => setIndex(index + 1)}>{t('Next')}</Button><Button variant="text" disabled={session.next_unrated === null || rate.isPending} onClick={() => { if (session.next_unrated !== null) setIndex(session.next_unrated); }}>{t('Next unrated')}</Button>{!session.revealed && <Button variant="text" disabled={trial.score === null || rate.isPending} onClick={() => rate.mutate(null)}>{t('Clear rating')}</Button>}</div>{!session.revealed && <Button icon={Unlock} disabledReason={!session.rated ? t('Rate at least one trial first.') : undefined} onClick={() => setRevealOpen(true)}>{t('Reveal results')}</Button>}</div>
        <p className={s.muted}>{session.revealed ? t('Ratings cannot be changed after reveal.') : t('Scores are saved immediately. You can change them until you reveal the results.')}</p>
      </div>
    </Section>}
    {session?.revealed && session.results && <ComparisonReport session={session} results={session.results as unknown as ComparisonResults}/>}
    <ConfirmDialog open={revealOpen} onOpenChange={setRevealOpen} title={t('Reveal the adapter identities?')} description={t('This locks all ratings. {0} trial(s) are still unrated.', (session?.trials ?? 0) - (session?.rated ?? 0))} confirmLabel={t('Reveal and lock ratings')} tone="default" onConfirm={() => reveal.mutateAsync()}/>
    <ConfirmDialog open={deleteOpen && Boolean(summary)} onOpenChange={setDeleteOpen} title={t('Delete this blind comparison?')} confirmLabel={t('Delete permanently')} onConfirm={() => remove.mutateAsync(sid)}
      description={summary && [t('This permanently deletes “{0}”: the session record, the reference voice copy and {1} generated clip(s) in outputs/abx/{2}/.', summary.name || summary.id, summary.clips_done, summary.id), summary.revealed ? t('{0}/{1} trials were rated and the results were revealed.', summary.rated, summary.trials) : t('{0}/{1} trials were rated; the results have not been revealed.', summary.rated, summary.trials), t('This cannot be undone.')].join(' ')}/>
  </div>;
}

function CreateComparison({ adapterA, adapterB, busyReason, onCreated }: { adapterA: string | null; adapterB: string | null; busyReason?: string; onCreated: (session: Session) => void }) {
  const t = useT(); const catalog = useCatalog(); const adapters = useAdapters(); const voices = useVoices(); const settings = useSettings(); const client = useQueryClient(); const [draft] = useState(readComparisonDraft);
  const [a, setA] = useState(adapterA ?? draft.a ?? ''); const [b, setB] = useState(adapterB ?? draft.b ?? ''); const [voice, setVoice] = useState(draft.voice ?? ''); const [name, setName] = useState(draft.name ?? ''); const [texts, setTexts] = useState(draft.texts ?? ''); const [repetitions, setRepetitions] = useState(draft.repetitions ?? 2); const [language, setLanguage] = useState(draft.language ?? catalog.default_language);
  useEffect(() => { try { localStorage.setItem(COMPARISON_DRAFT_KEY, JSON.stringify({ a, b, voice, name, texts, repetitions, language })); } catch { /* Private mode: edits still work for this visit. */ } }, [a, b, voice, name, texts, repetitions, language]);
  const plan = comparisonPlan(texts, repetitions); const options = [{ value: '', label: t('Original model') }, ...(adapters.data ?? []).map(item => ({ value: item.path, label: item.label }))];
  for (const path of [a, b]) if (path && !options.some(item => item.value === path)) options.push({ value: path, label: path.split(/[\\/]/).slice(-2).join(' / ') });
  const create = useMutation({ mutationFn: () => api.post<Session>('/api/abx/sessions', { name, adapter_a: a, adapter_b: b, voice, language, texts, repetitions, settings: { ...decodingSettings(readSavedForm()), use_bf16: settings?.use_bf16, use_gpt_latent: settings?.use_gpt_latent, use_torch_compile: settings?.use_torch_compile, use_flash_accel: settings?.use_flash_accel } }), onSuccess: result => { void client.invalidateQueries({ queryKey: ['abx-sessions'] }); onCreated(result); } });
  const invalid = a === b ? t('Choose two different adapters.') : !voice ? t('Choose a saved reference voice.') : !plan.sentences ? t('Enter at least one test sentence.') : undefined;
  return <Section layout="aside" title={t('Set up a fair comparison')} description={t('Each pair uses the same random seed. Listening order is shuffled and balanced.')}>
    <div className={s.fields}><div className={s.two}><Field label={t('Adapter A')}><Select value={a} onChange={setA} options={options}/></Field><Field label={t('Adapter B')}><Select value={b} onChange={setB} options={options}/></Field></div>
      <div className={s.two}><Field label={t('Saved reference voice')} hint={<Link to="/library">{t('Manage reference voices')}</Link>}><Select value={voice} onChange={setVoice} options={[{ value: '', label: t('Choose a voice…') }, ...(voices.data ?? []).map(item => ({ value: item.path, label: item.name }))]}/></Field><Field label={t('Speech language')}><Select value={language} onChange={setLanguage} options={catalog.languages.map(item => ({ value: item.code, label: t(item.label) }))}/></Field></div>
      <Field label={t('Test sentences')} hint={t('One sentence per line. Include the sounds and speaking styles you care about.')}><TextArea value={texts} onChange={e => setTexts(e.target.value)} minRows={3} showCount maxLength={100000}/></Field>
      <div className={s.two}><Field label={t('Comparison name')} hint={t('Optional; a timestamp is used if left empty.')}><TextInput value={name} onChange={e => setName(e.target.value)} maxLength={160}/></Field><Field label={t('Generations per sentence')}><NumberInput value={repetitions} onChange={setRepetitions} min={1} max={5} step={1}/></Field></div>
      <Disclosure title={t('Generation settings')} summary={t('Uses your current synthesis decoding settings')}><p className={s.muted}>{t('Decoding parameters and runtime options are copied when this comparison is created. Emotion follows the reference voice. Later changes do not affect a saved comparison.')} <Link to="/synthesis">{t('Configure synthesis')}</Link></p></Disclosure>
      {[create.error, adapters.error, voices.error].filter(Boolean).map((error, index) => <InlineAlert key={index} tone="error">{errorMessage(error)}</InlineAlert>)}
      <div className={s.toolbar}><span className={s.muted}>{t('{0} sentences × {1} repeats = {2} trials · {3} clips', plan.sentences, repetitions, plan.trials, plan.clips)}</span><Button icon={Play} pending={create.isPending} disabledReason={busyReason ?? invalid} onClick={() => create.mutate()}>{t('Create and generate')}</Button></div>
    </div>
  </Section>;
}

function ComparisonReport({ session, results }: { session: Session; results: ComparisonResults }) {
  const t = useT();
  const identityLabel = (label: string) => label === 'Base model (no adapter)' ? t('Original model') : label;
  return <Section layout="aside" title={t('Comparison results')} description={t('Two-sided sign test, excluding ties. Statistical evidence is not a guarantee of quality.')}><div className={s.fields}><InlineAlert tone={results.verdict === 'no_difference' ? 'info' : 'success'} title={t(VERDICTS[results.verdict] ?? VERDICTS.no_difference)}/><KeyValueList items={[{ label: t('Adapter A'), value: identityLabel(session.a) }, { label: t('Adapter B'), value: identityLabel(session.b) }]}/><div className={s.stats}><Stat label={t('A preferred')} value={results.wins_a}/><Stat label={t('Ties')} value={results.ties}/><Stat label={t('B preferred')} value={results.wins_b}/></div><div className={s.three}><Stat label={t('Mean preference for B')} value={results.mean_b.toFixed(3)}/><Stat label={t('Standard error')} value={results.stderr.toFixed(3)}/><Stat label={t('Sign-test p-value')} value={results.p_value.toFixed(4)}/></div>
    {results.wins_a + results.wins_b < 6 && <InlineAlert tone="warning">{t('Fewer than six non-tied trials cannot reach significance. More independent trials are needed.')}</InlineAlert>}
    <Disclosure title={t('Trial details')}><div className={s.tableWrap}><table className={s.table}><thead><tr><th>{t('Trial')}</th><th>{t('Text')}</th><th>{t('Sample 1 identity')}</th><th>{t('Rating')}</th></tr></thead><tbody>{results.rows.map(row => <tr key={row.index}><td>{row.index + 1}</td><td>{row.text}</td><td>{row.sample_1}</td><td>{row.score === null ? t('Unrated') : t(SCORE_LABELS[row.score + 2] ?? 'Unrated')}</td></tr>)}</tbody></table></div></Disclosure>
  </div></Section>;
}

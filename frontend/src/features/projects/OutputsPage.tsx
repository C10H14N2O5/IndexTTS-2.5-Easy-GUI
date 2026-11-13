import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router';
import { Boxes, Calculator, ListChecks, RefreshCw } from 'lucide-react';
import { errorMessage, mediaUrl } from '@/api/client';
import { useGpuJob, useJobs, useLive, type Job } from '@/api/live';
import { useT } from '@/i18n';
import { AudioPlayer, Badge, Button, Checkbox, EmptyState, Field, InlineAlert, LineChart, Section, Select, Skeleton, Switch, formatRelativeTime } from '@/ui';
import { useTrackedJob } from '@/features/synthesis/api';
import { JobCard } from '@/features/synthesis/JobCard';
import { BASELINE_SERIES, datasetLabel, projectRoute, useAdapters, useCheckpoints, useMetrics, useProjectContext, useStartValidationRun, useTrainingStatus, useValidationResults, type PosthocItem } from './api';
import s from './Workbench.module.css';

const BASELINE = 'base_model';
/** Prefer the result on the current validation set, otherwise the most recent one. */
function resultFor(items: PosthocItem[], name: string) { const matches = items.filter(item => item.name === name); return matches.find(item => item.status === 'current') ?? [...matches].sort((a, b) => b.computed_at - a.computed_at)[0]; }

export default function OutputsPage() {
  const t = useT(); const { project } = useProjectContext(); const name = project.project_name; const own = project.training.adapter_name || name;
  const adapters = useAdapters(name); const folders = adapters.data ?? []; const [chosen, setChosen] = useState('');
  const adapter = chosen || (!folders.length || folders.some(item => item.name === own) ? own : folders[0]!.name);
  const selector = folders.length > 0 && <Field label={t('Adapter')} hint={t('Every training output folder that contains saved adapters.')}><Select value={adapter} onChange={setChosen} options={[...folders.map(item => ({ value: item.name, label: item.is_project_adapter ? t('{0} (this project)', item.name) : item.name, description: [item.has_final ? t('{0} checkpoints · final adapter', item.checkpoints) : t('{0} checkpoints', item.checkpoints), ...(item.has_best ? [t('best adapter')] : [])].join(' · ') })), ...(folders.some(item => item.name === adapter) ? [] : [{ value: adapter, label: adapter }])]}/></Field>;
  return <div className={s.stack}>{adapters.error && <InlineAlert tone="error">{errorMessage(adapters.error)}</InlineAlert>}<AdapterOutputs key={adapter} project={name} adapter={adapter} own={adapter === own} selector={selector}/></div>;
}

function AdapterOutputs({ project, adapter, own, selector }: { project: string; adapter: string; own: boolean; selector: ReactNode }) {
  const t = useT(); const client = useQueryClient(); const query = useCheckpoints(project, adapter); const items = query.data ?? [];
  const tracked = useTrackedJob(); const jobs = useJobs(); const gpu = useGpuJob(); const training = useTrainingStatus().data?.running; const exporting = useLive(state => state.export?.status === 'running' || state.export?.status === 'cancelling');
  const matches = (job?: Job) => job?.kind === 'validation_posthoc' && job.meta?.project === project && job.meta?.adapter === adapter;
  const running = jobs.find(item => item.state === 'running' && matches(item)); const adopt = running && tracked.job?.id !== running.id ? running : undefined;
  useEffect(() => { if (adopt) tracked.track(adopt); }, [adopt, tracked]);
  const job = matches(tracked.job) ? tracked.job : running; const finished = job && job.state !== 'running' ? job.id : null;
  useEffect(() => { if (!finished) return; for (const key of ['validation-results', 'metrics', 'checkpoints']) void client.invalidateQueries({ queryKey: [key, project] }); }, [client, finished, project]);
  const results = useValidationResults(project, adapter, job?.state === 'running'); const metrics = useMetrics(project, adapter); const run = useStartValidationRun(project);
  const [selected, setSelected] = useState<string[]>([]); const [baseline, setBaseline] = useState(false); const [recompute, setRecompute] = useState(false);
  const posthoc = results.data?.items ?? []; const validation = results.data?.validation; const base = resultFor(posthoc, BASELINE); const trained = results.data?.trained_with;
  const recordLabel = (record: { version_id?: string | null; name?: string; revision?: number | null; manifest_sha256?: string | null }) => record.version_id ? datasetLabel(record, '') : `${t('working set')}${record.manifest_sha256 ? ` · ${record.manifest_sha256.slice(0, 8)}` : ''}`;
  const during = new Map(((metrics.data?.series?.find(item => item.label === 'validation/loss')?.points ?? []) as [number, number][]).map(([x, y]) => [x, y]));
  const missing = items.filter(item => resultFor(posthoc, item.name)?.status !== 'current').map(item => item.name);
  const chart = useMemo(() => {
    const labels = new Map([['validation/loss', t('Validation loss (during training)')], ['validation_posthoc/loss', t('Validation loss (post-hoc)')], [BASELINE_SERIES, t('Base model (no adapter)')]]);
    return (metrics.data?.series ?? []).flatMap(item => { const label = labels.get(item.label); return label ? [{ label, points: item.points as [number, number][], reference: item.label === BASELINE_SERIES }] : []; });
  }, [metrics.data, t]);
  /** The base-model line is only a reference, so it alone does not warrant a chart. */
  const plotted = metrics.data?.series?.some(item => (item.label === 'validation/loss' || item.label === 'validation_posthoc/loss') && item.points.length > 0);
  const mismatch = Boolean(results.data?.training_fingerprint && validation?.fingerprint && results.data.training_fingerprint !== validation.fingerprint);
  const busy = training ? t('Stop training before computing validation loss.') : exporting ? t('Wait for checkpoint export to finish.') : gpu ? t('The GPU is busy: {0}', gpu.title) : undefined;
  const computeReason = busy ?? (results.isPending ? t('Loading validation results…') : !validation ? t('Validation results are unavailable.') : !validation.ready ? t('Prepare the validation set first.') : !selected.length && !baseline ? t('Select checkpoints or include the base model.') : undefined);
  const toggle = (name: string, value: boolean) => setSelected(old => value ? [...old.filter(item => item !== name), name] : old.filter(item => item !== name));
  const compute = () => run.mutate({ adapter, checkpoints: items.filter(item => selected.includes(item.name)).map(item => item.name), include_baseline: baseline, recompute }, { onSuccess: ({ job: value }) => { tracked.track(value); setSelected([]); } });
  const all = items.length > 0 && items.every(item => selected.includes(item.name));
  return <><Section title={t('Checkpoints and adapters')} description={t('Training outputs are retained even when a project is deleted.')} actions={<Button variant="text" icon={RefreshCw} pending={query.isFetching || results.isFetching} onClick={() => { void query.refetch(); void results.refetch(); }}>{t('Refresh')}</Button>}>
    <div className={s.fields}>{selector}
      {query.error && <InlineAlert tone="error">{errorMessage(query.error)}</InlineAlert>}
      {base && <div className={s.checkRow}><span><strong>{t('Base model (no adapter)')}</strong> <span className={s.muted}>{t('Validation loss without any adapter, for reference.')}</span></span><LossValue item={base}/></div>}
      {query.isPending ? <Skeleton height={200}/> : !items.length ? <EmptyState icon={Boxes} title={t('No checkpoints yet')} description={t('Saved checkpoints and the final adapter appear here during training.')} action={<Link to="../training" className={s.linkButton}>{t('Configure training')}</Link>}/> : <div className={s.tableWrap}><table className={s.table}><thead><tr><th><Checkbox aria-label={t('Select all checkpoints')} checked={all ? true : selected.length ? 'indeterminate' : false} onCheckedChange={value => setSelected(value === true ? items.map(item => item.name) : [])}/></th><th>{t('Checkpoint')}</th><th>{t('Step')}</th><th>{t('Validation loss (post-hoc)')}</th><th>{t('Validation loss (during training)')}</th><th>{t('Saved')}</th><th>{t('Use this adapter')}</th></tr></thead><tbody>{items.map(item => { const value = (item.step != null ? during.get(item.step) : undefined) ?? item.validation_loss ?? undefined; return <tr key={item.path}><td><Checkbox aria-label={t('Select {0}', item.name)} checked={selected.includes(item.name)} onCheckedChange={checked => toggle(item.name, checked === true)}/></td><td><strong>{item.name}</strong><div className={s.actions}><Badge tone={item.kind === 'final' ? 'success' : item.kind === 'best' ? 'accent' : 'neutral'}>{item.kind === 'final' ? t('Final adapter') : item.kind === 'best' ? t('Best adapter') : t('Checkpoint')}</Badge>{item.early_stopped && <Badge tone="warning">{t('Early stop')}</Badge>}{item.auto_stopped && <Badge tone="warning">{t('Auto stopped')}</Badge>}</div></td><td>{item.step ?? '—'}</td><td><LossValue item={resultFor(posthoc, item.name)}/></td><td>{value !== undefined ? value.toFixed(4) : <span className={s.muted}>—</span>}</td><td>{formatRelativeTime(item.modified)}</td><td><div className={s.actions}><Link to={`/synthesis?adapter=${encodeURIComponent(item.path)}`} className={s.linkButton}>{t('Synthesize')}</Link><Link to={`/abx?b=${encodeURIComponent(item.path)}`} className={s.linkButton}>{t('Blind compare')}</Link><Link to={`/export?adapter=${encodeURIComponent(item.path)}`} className={s.linkButton}>{t('Export')}</Link>{own && (item.kind === 'checkpoint' || (item.kind === 'best' && item.resumable)) && <Link to={`${projectRoute(project, 'training')}?resume=${encodeURIComponent(item.path)}`} className={s.linkButton}>{t('Resume training')}</Link>}</div></td></tr>; })}</tbody></table></div>}
    </div>
  </Section>
  {items.length > 0 && <Section title={t('Compute validation loss')} description={t('Measure saved checkpoints on the current validation set after training. Results are stored next to the adapter.')}>
    <div className={s.fields}>
      {results.error && <InlineAlert tone="error">{errorMessage(results.error)}</InlineAlert>}
      {validation && <InlineAlert tone={validation.ready ? 'success' : 'warning'} title={validation.message} action={!validation.ready && <Link to="../validation" className={s.linkButton}>{t('Configure validation')}</Link>}>{validation.ready && t('{0} clips · fingerprint {1}', validation.samples, (validation.fingerprint ?? '').slice(0, 8))}{validation.version && <> · {t('validation version {0}', datasetLabel({ version_id: validation.version.id, name: validation.version.name, revision: validation.version.revision }, ''))}</>}</InlineAlert>}
      {trained && <p className={s.muted}>{t('Latest training run of this adapter: training set {0} · validation set {1}', trained.training ? recordLabel(trained.training) : t('not recorded'), trained.validation ? recordLabel(trained.validation) : t('none'))}</p>}
      {mismatch && <InlineAlert tone="warning">{t('The adapter was trained with a different validation set, so its training-time curve is not directly comparable.')}</InlineAlert>}
      <div className={s.two}><Switch checked={baseline} onCheckedChange={setBaseline} label={t('Include base model (no adapter)')} description={t('Drawn as a dashed reference line on the chart.')}/><Switch checked={recompute} onCheckedChange={setRecompute} label={t('Recompute existing results')} description={t('Otherwise items with a current result are skipped.')}/></div>
      <div className={s.toolbar}><div className={s.actions}><Button variant="outlined" icon={ListChecks} disabledReason={!missing.length ? t('Every checkpoint already has a current result.') : undefined} onClick={() => setSelected(missing)}>{t('Select missing')}</Button><span className={s.muted}>{t('{0} selected', selected.length)}</span></div><Button icon={Calculator} pending={run.isPending} disabledReason={computeReason} onClick={compute}>{t('Compute selected')}</Button></div>
      {computeReason && !results.isPending && <p className={s.muted}>{computeReason}</p>}
      {[run.error, tracked.error].filter(Boolean).map((error, index) => <InlineAlert key={index} tone="error">{errorMessage(error)}</InlineAlert>)}
      {job && <JobCard job={job}/>}
      {posthoc.some(item => item.status === 'outdated') && <p className={s.muted}>{t('Outdated results were computed on a different validation set or before the adapter files changed. Recompute them for a fair comparison.')}</p>}
      {plotted && <LineChart series={chart} height={220} xLabel={t('Step')}/>}
      {metrics.error && <InlineAlert tone="error">{errorMessage(metrics.error)}</InlineAlert>}
    </div>
  </Section>}
  {items.some(item => item.evaluation) && <Section title={t('Evaluation previews')} description={t('Listen to the same reference and text across saved checkpoints.')}><div className={s.fields}>{items.filter(item => item.evaluation).map(item => <AudioPlayer key={item.path} src={mediaUrl(item.evaluation!)} title={item.name} downloadName={`eval-${item.step}.wav`}/>)}</div></Section>}
  </>;
}

function LossValue({ item }: { item?: PosthocItem }) {
  const t = useT();
  return item ? <span className={s.actions}><span>{item.loss.toFixed(4)}</span>{item.status === 'outdated' && <Badge tone="warning">{t('Outdated')}</Badge>}{item.status === 'outdated' && item.validation_version && <span className={s.muted}>{t('on {0} r{1}', item.validation_version.name, item.validation_version.revision)}</span>}{item.same_as && <span className={s.muted}>{t('same as {0}', item.same_as)}</span>}</span> : <span className={s.muted}>—</span>;
}

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { ExternalLink, RefreshCw, Square } from 'lucide-react';
import { api, errorMessage, type Schemas } from '@/api/client';
import { useT } from '@/i18n';
import { Badge, Button, ConfirmDialog, EmptyState, InlineAlert, LineChart, ProgressBar, Section, SegmentedControl, Stat, formatDuration } from '@/ui';
import { curveSeries, useMetrics, useStopTraining, useTrainingStatus, type AutoStopStatus } from './api';
import s from './Workbench.module.css';

export function TrainingMonitor({ project, adapter }: { project: string; adapter: string }) {
  const t = useT(); const statusQuery = useTrainingStatus(); const status = statusQuery.data; const own = status?.project === project; const metrics = useMetrics(project, adapter); const stop = useStopTraining(); const [confirm, setConfirm] = useState(false); const [metric, setMetric] = useState('loss');
  const tb = useMutation({ mutationFn: (reload: boolean) => api.post<Schemas['TensorboardResponse']>('/api/training/tensorboard', { project, adapter, reload }) });
  const all = metrics.data?.series ?? [];
  const series = curveSeries(all, metric);
  return <div className={s.stack}><Section title={t('Training monitor')} actions={own && status?.running && <Button variant="outlined" icon={Square} pending={stop.isPending} disabledReason={status.status === 'stopping' ? t('Stopping after the current step…') : undefined} onClick={() => setConfirm(true)}>{t('Stop training')}</Button>}>
    <div className={s.status}>{own && status ? <><div className={s.toolbar}><Badge tone={status.failed ? 'error' : status.running ? 'accent' : status.status === 'stopped' ? 'neutral' : 'success'}>{status.failed ? t('Failed') : status.running ? status.status === 'stopping' ? t('Stopping…') : t('Training in progress') : status.status === 'stopped' ? t('Stopped') : t('Completed')}</Badge><div className={s.actions}><span className={s.muted}>{t('{0} / {1} steps', status.step, status.total)}</span>{(status.schedule_total ?? 0) > status.total && <><Badge tone="warning">{t('Early stop')}</Badge><span className={s.muted}>{t('LR schedule {0} steps', status.schedule_total)}</span></>}{status.auto_stopped && !status.auto_stop && <Badge tone="accent">{t('Stopped automatically')}</Badge>}</div></div><ProgressBar value={status.total ? status.step / status.total : status.running ? null : 0} label={t('Training progress')}/><p className={s.muted} aria-live="polite">{status.message}</p><div className={s.stats}><Stat label={t('Loss')} value={status.loss?.toFixed(4) ?? '—'}/><Stat label={t('Elapsed')} value={formatDuration(status.elapsed)}/><Stat label={t('Remaining')} value={status.running && status.eta ? formatDuration(status.eta) : '—'}/></div>{status.auto_stop && <AutoStopBlock auto={status.auto_stop} stopped={status.auto_stopped}/>}{status.failed && <InlineAlert tone="error">{status.message}</InlineAlert>}</> : <EmptyState title={t('Ready when you are')} description={t('Live progress, loss and remaining time will appear here once training starts.')}/>}
      {statusQuery.error && <InlineAlert tone="error">{errorMessage(statusQuery.error)}</InlineAlert>}{stop.error && <InlineAlert tone="error">{errorMessage(stop.error)}</InlineAlert>}
    </div>
  </Section>
  <Section title={t('Learning curves')} description={t('Read directly from TensorBoard events. At most 2,000 points per series.')} actions={<Button icon={RefreshCw} size="sm" variant="text" pending={metrics.isFetching} onClick={() => void metrics.refetch()}>{t('Refresh')}</Button>}>
    <div className={s.fields}><SegmentedControl value={metric} onChange={setMetric} options={[{ value: 'loss', label: t('Smoothed loss') }, { value: 'raw', label: t('Raw loss') }, { value: 'lr', label: t('Learning rate') }]}/>
      {series.length ? <LineChart series={series.map(item => ({ label: item.label, points: item.points as [number, number][] }))} height={220} xLabel={t('Step')} /> : <EmptyState title={t('No scalar events yet')} description={t('The chart appears after the trainer writes its first event.')}/>}
      {metrics.error && <InlineAlert tone="error">{errorMessage(metrics.error)}</InlineAlert>}{metrics.data?.error && <InlineAlert tone="warning">{metrics.data.error}</InlineAlert>}
      <div className={s.actions}><Button variant="outlined" size="sm" icon={ExternalLink} pending={tb.isPending} onClick={() => tb.mutate(false)}>{t('Start TensorBoard')}</Button><Button variant="text" size="sm" pending={tb.isPending} onClick={() => tb.mutate(true)}>{t('Reload TensorBoard')}</Button></div>
      {tb.data && <InlineAlert tone="info" title={tb.data.message}><a href={tb.data.url} target="_blank" rel="noreferrer">{t('Open TensorBoard')}</a><p>{t('TensorBoard is available only on the computer running IndexTTS. Use the built-in charts from another device.')}</p></InlineAlert>}{tb.error && <InlineAlert tone="error">{errorMessage(tb.error)}</InlineAlert>}
    </div>
  </Section><ConfirmDialog open={confirm} onOpenChange={setConfirm} title={t('Stop this training run?')} description={t('The current optimizer step will finish before the trainer exits. Existing checkpoints are kept.')} confirmLabel={t('Request stop')} tone="default" onConfirm={() => stop.mutateAsync()}/></div>;
}

function AutoStopBlock({ auto, stopped }: { auto: AutoStopStatus; stopped: boolean }) {
  const t = useT(); const phase = stopped ? 'stopped' : auto.phase; const loss = (value?: number | null) => value != null ? value.toFixed(4) : '—'; const at = (step?: number | null) => step != null ? t('at step {0}', step) : undefined;
  return <div className={s.fields} aria-label={t('Auto stop')} role="group"><div className={s.toolbar}><strong>{t('Auto stop')}</strong><Badge tone={phase === 'watching' ? 'warning' : phase === 'stopped' ? 'accent' : 'success'}>{phase === 'watching' ? t('Watching') : phase === 'stopped' ? t('Stopped automatically') : t('Normal')}</Badge></div>
    <div className={s.stats}><Stat label={t('Best validation loss')} value={loss(auto.best_loss)} hint={at(auto.best_step)}/><Stat label={t('Valley loss')} value={loss(auto.valley_loss)} hint={at(auto.valley_step)}/>{phase === 'watching' && <Stat label={t('Improvement / required')} value={`${auto.improvement.toFixed(4)} / ${auto.min_delta.toFixed(4)}`} hint={auto.watch_start_step != null ? t('Watching since step {0}', auto.watch_start_step) : undefined}/>}</div>
    {phase === 'watching' && auto.deadline_step != null && <p className={s.muted}>{t('Stops at step {0} unless validation improves by at least {1}.', auto.deadline_step, auto.min_delta.toFixed(4))}</p>}</div>;
}

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Activity, Square } from 'lucide-react';
import { Badge, Button, EmptyState, InlineAlert, Popover, ProgressBar } from '@/ui';
import { formatClock, formatDuration } from '@/ui/format';
import { api, errorMessage } from '@/api/client';
import { useJobs, useLive } from '@/api/live';
import { useT } from '@/i18n';
import { JobCard } from '@/features/synthesis/JobCard';
import styles from './Activity.module.css';

export function ActivityButton() {
  const t = useT(); const [open, setOpen] = useState(false); const jobs = useJobs();
  const status = useLive((state) => state.status); const training = useLive((state) => state.training); const exporting = useLive((state) => state.export);
  const stopTraining = useMutation({ mutationFn: () => api.post('/api/training/stop') });
  const stopExport = useMutation({ mutationFn: () => api.post('/api/export/cancel') });
  const active = jobs.filter((job) => job.state === 'running'); const finished = jobs.filter((job) => job.state !== 'running');
  const exportRunning = exporting?.status === 'running' || exporting?.status === 'cancelling';
  const count = active.length + Number(Boolean(training?.running)) + Number(exportRunning);
  return <Popover open={open} onOpenChange={setOpen} className={styles.popover} trigger={<Button variant="text" size="sm" icon={Activity} aria-label={count ? t('Activity: {0} running', count) : t('Activity')}>{t('Activity')}{count > 0 && <Badge tone="accent">{count}</Badge>}</Button>}>
      <div className={styles.panel} role="region" aria-label={t('Activity')}>
        <h2 className={styles.sectionTitle}>{t('Activity')}</h2>
        <p className={styles.meta}>{t('Current work, recent completions, and useful status details.')}</p>
        <h3 className={styles.sectionTitle}>{t('Running now')}</h3>
        {!count && <EmptyState icon={Activity} title={t('All caught up')} description={t('No tasks are running.')} />}
        {active.map((job) => <div className={styles.item} key={job.id}><JobCard job={job} /></div>)}
        {training?.running && <div className={styles.item}><div className={styles.row}><strong>{t('Training')} · {training.adapter || training.project}</strong><Button size="sm" variant="text" icon={Square} pending={stopTraining.isPending} onClick={() => stopTraining.mutate()}>{t('Stop training')}</Button></div><ProgressBar value={training.total ? training.step / training.total : null} label={t('Step {0} of {1}', training.step, training.total)} /><p>{training.message}</p><span className={styles.meta}>{formatDuration(training.elapsed)} · {t('Remaining: {0}', formatDuration(training.eta))}</span>{stopTraining.error && <InlineAlert tone="error">{errorMessage(stopTraining.error)}</InlineAlert>}</div>}
        {exportRunning && exporting && <div className={styles.item}><div className={styles.row}><strong>{t('Export')} · {exporting.phase}</strong><Button size="sm" variant="text" pending={stopExport.isPending} disabled={exporting.status === 'cancelling'} onClick={() => stopExport.mutate()}>{t('Cancel export')}</Button></div><ProgressBar value={null} label={exporting.phase} /><span className={styles.meta}>{formatDuration(exporting.elapsed)}</span>{stopExport.error && <InlineAlert tone="error">{errorMessage(stopExport.error)}</InlineAlert>}</div>}
        {!!finished.length && <><h3 className={styles.sectionTitle}>{t('Recent tasks')}</h3>{finished.slice(0, 12).map((job) => <div className={styles.item} key={job.id}><JobCard job={job} /></div>)}</>}
        {!!status.length && <><h3 className={styles.sectionTitle}>{t('Status history')}</h3>{[...status].reverse().map((item) => <article key={item.id} className={styles.item}><div className={styles.row}><strong>{item.phase}</strong><span className={styles.meta}>{formatClock(item.time)}</span></div><p>{item.message}</p>{item.detail && <details><summary>{t('Details')}</summary><p className={styles.details}>{item.detail}</p></details>}</article>)}</>}
      </div>
  </Popover>;
}

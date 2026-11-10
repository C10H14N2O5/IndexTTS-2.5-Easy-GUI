import { useEffect, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Square } from 'lucide-react';
import { Badge, Button, InlineAlert, ProgressBar } from '@/ui';
import { formatDuration } from '@/ui/format';
import { errorMessage } from '@/api/client';
import type { Job } from '@/api/live';
import { useT } from '@/i18n';
import { cancelJob } from './api';
import styles from './Synthesis.module.css';

export function JobCard({ job }: { job: Job }) {
  const t = useT();
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (job.state !== 'running') return;
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [job.state]);
  const stop = useMutation({ mutationFn: () => cancelJob(job.id) });
  const label = job.state === 'running' ? t('In progress') : job.state === 'succeeded' ? t('Completed') : job.state === 'cancelled' ? t('Cancelled') : t('Failed');
  const tone = job.state === 'failed' ? 'error' : job.state === 'succeeded' ? 'success' : 'accent';
  return <div className={styles.job} aria-live="polite">
    <div className={styles.jobHeading}><strong>{job.title}</strong><Badge tone={tone}>{label}</Badge></div>
    <ProgressBar value={job.progress ?? null} label={job.stage ?? label} />
    <div className={styles.toolbar}><span className={styles.muted}>{job.stage} · {formatDuration(Math.max(0, (job.finished_at ?? now) - job.started_at))}</span>
      {job.state === 'running' && <Button size="sm" variant="text" icon={Square} pending={stop.isPending} disabledReason={job.cancel_requested ? t('Stopping after the current step…') : !job.cancellable ? t('This task cannot be cancelled.') : undefined} onClick={() => stop.mutate()}>{job.cancel_requested ? t('Stopping…') : t('Cancel task')}</Button>}
    </div>
    {job.error && <InlineAlert tone="error" title={job.error.message}>{typeof job.error.detail === 'string' ? job.error.detail : null}</InlineAlert>}
    {stop.error && <InlineAlert tone="error">{errorMessage(stop.error)}</InlineAlert>}
  </div>;
}

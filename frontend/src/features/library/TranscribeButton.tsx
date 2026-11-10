import { useEffect, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { ScanText } from 'lucide-react';
import { Button, InlineAlert } from '@/ui';
import { errorMessage } from '@/api/client';
import { useT } from '@/i18n';
import { useTrackedJob } from '@/features/synthesis/api';
import { JobCard } from '@/features/synthesis/JobCard';
import { transcribeReference } from './api';

export function TranscribeButton({ path, onTranscript }: { path: string; onTranscript: (text: string) => void }) {
  const t = useT();
  const tracked = useTrackedJob();
  const consumed = useRef('');
  const requestedPath = useRef('');
  const start = useMutation({ mutationFn: () => transcribeReference(path), onSuccess: ({ job }) => { requestedPath.current = path; tracked.track(job); } });
  useEffect(() => {
    const job = tracked.job;
    if (!job || consumed.current === job.id || job.state !== 'succeeded') return;
    consumed.current = job.id;
    if (path !== requestedPath.current) return; // Never write an old clip's transcript into a replacement clip.
    const result = job.result as { text?: unknown } | undefined;
    if (typeof result?.text === 'string') onTranscript(result.text);
  }, [tracked.job, onTranscript, path]);
  return <>
    <Button size="sm" variant="tonal" icon={ScanText} pending={start.isPending || tracked.job?.state === 'running'} disabledReason={!path ? t('Add a reference clip first.') : undefined} onClick={() => start.mutate()}>{t('Transcribe reference')}</Button>
    {start.error && <InlineAlert tone="error">{errorMessage(start.error)}</InlineAlert>}
    {tracked.error && <InlineAlert tone="error">{errorMessage(tracked.error)}</InlineAlert>}
    {tracked.job && tracked.job.state !== 'succeeded' && <JobCard job={tracked.job} />}
  </>;
}

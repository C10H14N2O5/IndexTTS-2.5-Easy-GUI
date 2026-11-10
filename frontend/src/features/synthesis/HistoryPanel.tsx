import { useState } from 'react';
import { Clock3, RotateCcw, Trash2 } from 'lucide-react';
import { AudioPlayer, Button, ConfirmDialog, EmptyState, IconButton, InlineAlert, Section, Skeleton } from '@/ui';
import { formatRelativeTime } from '@/ui/format';
import { errorMessage, mediaUrl } from '@/api/client';
import { useT } from '@/i18n';
import { useDeleteHistory, useHistory, type HistoryItem } from './api';
import styles from './Synthesis.module.css';

export function HistoryPanel({ onReuse }: { onReuse: (item: HistoryItem) => void }) {
  const t = useT(); const history = useHistory(); const remove = useDeleteHistory();
  const [deleting, setDeleting] = useState<HistoryItem | null>(null);
  return <Section title={t('Recent results')} description={t('Saved audio and the settings used to create it.')}>
    {history.isPending ? <Skeleton height={120} /> : history.error ? <InlineAlert tone="error" action={<Button size="sm" variant="text" onClick={() => void history.refetch()}>{t('Retry')}</Button>}>{errorMessage(history.error)}</InlineAlert> : !history.data?.length ? <EmptyState icon={Clock3} title={t('Your first result starts here')} description={t('Generate speech with Save output enabled to keep it in this list.')} /> :
      <div className={styles.history}>{history.data.map((item) => <article className={styles.historyItem} key={item.path}>
        <div className={styles.toolbar}><span className={styles.historyName} title={item.name}>{item.name}</span><span className={styles.muted}>{formatRelativeTime(item.created)}</span></div>
        <AudioPlayer src={mediaUrl(item.path, item.created)} downloadName={item.name} compact />
        <div className={styles.toolbar}><Button size="sm" variant="text" icon={RotateCcw} disabledReason={!item.params ? t('This older recording has no saved settings.') : undefined} onClick={() => onReuse(item)}>{t('Reuse settings')}</Button><IconButton size="sm" icon={Trash2} label={t('Delete recording')} onClick={() => setDeleting(item)} /></div>
      </article>)}</div>}
    <ConfirmDialog open={Boolean(deleting)} onOpenChange={(open) => { if (!open) setDeleting(null); }} title={t('Delete this recording?')} description={deleting?.name} confirmLabel={t('Delete recording')} tone="danger" onConfirm={async () => { if (deleting) await remove.mutateAsync(deleting.name); setDeleting(null); }} />
  </Section>;
}

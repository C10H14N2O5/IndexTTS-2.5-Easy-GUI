import { Link } from 'react-router';
import { Boxes, RefreshCw } from 'lucide-react';
import { errorMessage, mediaUrl } from '@/api/client';
import { useT } from '@/i18n';
import { AudioPlayer, Badge, Button, EmptyState, InlineAlert, Section, Skeleton, formatRelativeTime } from '@/ui';
import { projectRoute, useCheckpoints, useProjectContext } from './api';
import s from './Workbench.module.css';

export default function OutputsPage() {
  const t = useT(); const { project } = useProjectContext(); const query = useCheckpoints(project.project_name, project.training.adapter_name); const items = query.data ?? [];
  return <div className={s.stack}><Section title={t('Checkpoints and adapters')} description={t('Training outputs are retained even when a project is deleted.')} actions={<Button variant="text" icon={RefreshCw} pending={query.isFetching} onClick={() => void query.refetch()}>{t('Refresh')}</Button>}>
    {query.error && <InlineAlert tone="error">{errorMessage(query.error)}</InlineAlert>}
    {query.isPending ? <Skeleton height={200}/> : !items.length ? <EmptyState icon={Boxes} title={t('No checkpoints yet')} description={t('Saved checkpoints and the final adapter appear here during training.')} action={<Link to="../training" className={s.linkButton}>{t('Configure training')}</Link>}/> : <div className={s.tableWrap}><table className={s.table}><thead><tr><th>{t('Checkpoint')}</th><th>{t('Step')}</th><th>{t('Saved')}</th><th>{t('Use this adapter')}</th></tr></thead><tbody>{items.map(item => <tr key={item.path}><td><strong>{item.name}</strong><div><Badge tone={item.kind === 'final' ? 'success' : 'neutral'}>{item.kind === 'final' ? t('Final adapter') : t('Checkpoint')}</Badge></div></td><td>{item.step ?? '—'}</td><td>{formatRelativeTime(item.modified)}</td><td><div className={s.actions}><Link to={`/synthesis?adapter=${encodeURIComponent(item.path)}`} className={s.linkButton}>{t('Synthesize')}</Link><Link to={`/abx?b=${encodeURIComponent(item.path)}`} className={s.linkButton}>{t('Blind compare')}</Link><Link to={`/export?adapter=${encodeURIComponent(item.path)}`} className={s.linkButton}>{t('Export')}</Link>{item.kind === 'checkpoint' && <Link to={`${projectRoute(project.project_name, 'training')}?resume=${encodeURIComponent(item.path)}`} className={s.linkButton}>{t('Resume training')}</Link>}</div></td></tr>)}</tbody></table></div>}
  </Section>
  {items.some(item => item.evaluation) && <Section title={t('Evaluation previews')} description={t('Listen to the same reference and text across saved checkpoints.')}><div className={s.fields}>{items.filter(item => item.evaluation).map(item => <AudioPlayer key={item.path} src={mediaUrl(item.evaluation!)} title={item.name} downloadName={`eval-${item.step}.wav`}/>)}</div></Section>}
  </div>;
}

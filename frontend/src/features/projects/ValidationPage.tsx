import { useRef, useState } from 'react';
import { RefreshCw, Save } from 'lucide-react';
import { api, errorMessage } from '@/api/client';
import { useT } from '@/i18n';
import { Button, Field, InlineAlert, NumberInput, PathField, Section, Stat, Switch, formatDuration, toast } from '@/ui';
import { ProofreadPanel, type ProofreadPanelHandle } from '@/features/proofread/ProofreadPanel';
import { projectUrl, stat, useProjectContext, useProjectMutation, useTrainingStatus, type Project } from './api';
import s from './Workbench.module.css';

export default function ValidationPage() {
  const t = useT(); const { project, refresh, setContextLocked } = useProjectContext(); const editor = useRef<ProofreadPanelHandle>(null);
  const prepareChange = async (commit: () => Promise<unknown>) => { if (editor.current) return editor.current.prepareContextChange(commit); try { await commit(); return true; } catch { return false; } };
  return <div className={s.stack}><ValidationSettings key={`${project.project_name}:${project.modified}`} project={project} prepareChange={prepareChange}/>
    {project.validation.source_folder ? <Section title={t('Review validation transcripts')} description={t('Validation uses the project language, but its source and manifest remain independent.')}><ProofreadPanel ref={editor} project={project.project_name} directory={project.validation.source_folder} role="validation" language={project.language ?? 'EN'} onUpdated={refresh} onContextLockedChange={setContextLocked}/></Section> : <InlineAlert tone="info">{t('Save a validation folder to open its proofreading workspace.')}</InlineAlert>}
  </div>;
}

function ValidationSettings({ project, prepareChange }: { project: Project; prepareChange: (commit: () => Promise<unknown>) => Promise<boolean> }) {
  const t = useT(); const [folder, setFolder] = useState(project.validation.source_folder || project.validation_suggested_folder || ''); const [enabled, setEnabled] = useState(project.validation.enabled ?? false); const [interval, setInterval] = useState(project.validation.interval_steps ?? 0); const running = useTrainingStatus().data?.running;
  const save = useProjectMutation(project.project_name, (prepare: boolean) => api.put<Project>(`${projectUrl(project.project_name)}/validation`, { source_folder: folder, enabled, interval_steps: interval, prepare }));
  const reason = running ? t('Stop training before editing projects.') : undefined;
  async function submit(prepare: boolean) { if (await prepareChange(() => save.mutateAsync(prepare))) toast.success(prepare ? t('Validation manifest is ready.') : t('Validation settings saved.')); }
  return <Section layout="aside" title={t('Validation loss')} description={t('Measure generalization with separate audio that is never used for optimization.')} actions={<Button variant="outlined" icon={Save} pending={save.isPending} disabledReason={reason} onClick={() => submit(false)}>{t('Save validation')}</Button>}>
    <div className={s.fields}><Switch checked={enabled} onCheckedChange={setEnabled} disabledReason={reason} label={t('Enable validation loss')} description={t('Optional. A ready validation manifest is required when enabled.')}/>
      <PathField label={t('Validation audio folder')} value={folder} onChange={setFolder} kind="dir" disabled={running} hint={t('The suggested project folder is created only when you save it. Training audio must not appear here.')}/>
      <div className={s.two}><Field label={t('Validation interval (steps)')} hint={t('0 validates at checkpoints. The final step is always validated.')} modified={interval !== 0} onReset={() => setInterval(0)}><NumberInput value={interval} onChange={setInterval} min={0} step={1} disabled={running}/></Field><div className={s.actions}><Button icon={RefreshCw} pending={save.isPending} disabledReason={reason ?? (!folder ? t('Choose a validation folder first.') : undefined)} onClick={() => submit(true)}>{t('Build validation manifest')}</Button></div></div>
      {save.error && <InlineAlert tone="error">{errorMessage(save.error)}</InlineAlert>}
      <div className={s.stats}><Stat label={t('Validation clips')} value={stat(project.validation.stats, 'valid')}/><Stat label={t('Missing transcripts')} value={stat(project.validation.stats, 'missing_text')}/><Stat label={t('Total duration')} value={formatDuration(stat(project.validation.stats, 'total_seconds'))}/></div>
      <InlineAlert tone={project.validation_status.stale ? project.validation.enabled ? 'warning' : 'info' : 'success'} title={project.validation_status.message}><span className={s.path}>{project.validation.manifest}</span></InlineAlert>
    </div>
  </Section>;
}

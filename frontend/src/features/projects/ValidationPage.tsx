import { useRef, useState } from 'react';
import { RefreshCw, Save } from 'lucide-react';
import { api, errorMessage } from '@/api/client';
import { useT } from '@/i18n';
import { Button, Field, InlineAlert, NumberInput, PathField, Section, Stat, Switch, formatDuration, toast } from '@/ui';
import { ProofreadPanel, type ProofreadPanelHandle } from '@/features/proofread/ProofreadPanel';
import { DatasetVersions } from './DatasetVersions';
import { projectUrl, stat, useProjectContext, useProjectMutation, useTrainingStatus, type Project } from './api';
import s from './Workbench.module.css';

export default function ValidationPage() {
  const t = useT(); const { project, refresh, setContextLocked } = useProjectContext(); const editor = useRef<ProofreadPanelHandle>(null);
  const prepareChange = async (commit: () => Promise<unknown>) => { if (editor.current) return editor.current.prepareContextChange(commit); try { await commit(); return true; } catch { return false; } };
  return <div className={s.stack}><ValidationSettings key={`${project.project_name}:${project.modified}`} project={project} prepareChange={prepareChange}/>
    <DatasetVersions project={project} role="validation"/>
    {project.validation.source_folder ? <Section title={t('Review validation transcripts')} description={t('Validation uses the project language, but its source and manifest remain independent.')}><ProofreadPanel ref={editor} project={project.project_name} directory={project.validation.source_folder} role="validation" language={project.language ?? 'EN'} onUpdated={refresh} onContextLockedChange={setContextLocked}/></Section> : <InlineAlert tone="info">{t('Save a validation folder to open its proofreading workspace.')}</InlineAlert>}
  </div>;
}

function ValidationSettings({ project, prepareChange }: { project: Project; prepareChange: (commit: () => Promise<unknown>) => Promise<boolean> }) {
  const t = useT(); const [folder, setFolder] = useState(project.validation.source_folder || project.validation_suggested_folder || ''); const [enabled, setEnabled] = useState(project.validation.enabled ?? false); const [interval, setInterval] = useState(project.validation.interval_steps ?? 0); const running = useTrainingStatus().data?.running;
  const [dense, setDense] = useState(project.validation.dense_enabled ?? false); const [denseStart, setDenseStart] = useState(project.validation.dense_start_step ?? NaN); const [denseEnd, setDenseEnd] = useState(project.validation.dense_end_step ?? NaN); const [denseInterval, setDenseInterval] = useState(project.validation.dense_interval_steps ?? NaN);
  const [includeBaseline, setIncludeBaseline] = useState(project.validation.include_baseline ?? false);
  const stepOrNull = (value: number) => Number.isFinite(value) ? value : null;
  const save = useProjectMutation(project.project_name, (prepare: boolean) => api.put<Project>(`${projectUrl(project.project_name)}/validation`, { source_folder: folder, enabled, interval_steps: interval, dense_enabled: dense, dense_start_step: stepOrNull(denseStart), dense_end_step: stepOrNull(denseEnd), dense_interval_steps: stepOrNull(denseInterval), include_baseline: includeBaseline, prepare }));
  const denseReason = !dense ? undefined : [denseStart, denseEnd, denseInterval].some(value => !Number.isInteger(value) || value < 1) ? t('Enter a start step, end step and interval of at least 1 for the dense range.') : denseStart > denseEnd ? t('The dense range must start at or before its end step.') : undefined;
  const reason = running ? t('Stop training before editing projects.') : undefined; const saveReason = reason ?? denseReason;
  async function submit(prepare: boolean) { if (await prepareChange(() => save.mutateAsync(prepare))) toast.success(prepare ? t('Validation manifest is ready.') : t('Validation settings saved.')); }
  return <Section layout="aside" title={t('Validation loss')} description={t('Measure generalization with separate audio that is never used for optimization.')} actions={<Button variant="outlined" icon={Save} pending={save.isPending} disabledReason={saveReason} onClick={() => submit(false)}>{t('Save validation')}</Button>}>
    <div className={s.fields}><Switch checked={enabled} onCheckedChange={setEnabled} disabledReason={reason} label={t('Enable validation loss')} description={t('Optional. A ready validation manifest is required when enabled.')}/>
      <PathField label={t('Validation audio folder')} value={folder} onChange={setFolder} kind="dir" disabled={running} hint={t('The suggested project folder is created only when you save it. Training audio must not appear here.')}/>
      <div className={s.two}><Field label={t('Validation interval (steps)')} hint={t('0 validates at checkpoints. The final step is always validated.')} modified={interval !== 0} onReset={() => setInterval(0)}><NumberInput value={interval} onChange={setInterval} min={0} step={1} disabled={running}/></Field><div className={s.actions}><Button icon={RefreshCw} pending={save.isPending} disabledReason={saveReason ?? (!folder ? t('Choose a validation folder first.') : undefined)} onClick={() => submit(true)}>{t('Build validation manifest')}</Button></div></div>
      <Switch checked={includeBaseline} onCheckedChange={setIncludeBaseline} disabledReason={reason} label={t('Include base model (no adapter)')} description={t('Validate the base model once before training starts so checkpoints can be compared with it. Skipped when the adapter folder already has a result for the current validation set.')}/>
      <div className={s.summary}><Switch checked={dense} onCheckedChange={setDense} disabledReason={reason} label={t('Dense validation range')} description={t('Inside this range, validation also runs every N steps counted from the start step (the start step included), in addition to the global interval and the final step.')}/>
        {dense && <div className={s.three}><Field label={t('Range start step')}><NumberInput value={denseStart} onChange={setDenseStart} min={1} step={1} disabled={running}/></Field><Field label={t('Range end step')}><NumberInput value={denseEnd} onChange={setDenseEnd} min={1} step={1} disabled={running}/></Field><Field label={t('Range interval (steps)')}><NumberInput value={denseInterval} onChange={setDenseInterval} min={1} step={1} disabled={running}/></Field></div>}
        {denseReason && <InlineAlert tone="warning">{denseReason}</InlineAlert>}
      </div>
      {save.error && <InlineAlert tone="error">{errorMessage(save.error)}</InlineAlert>}
      <div className={s.stats}><Stat label={t('Validation clips')} value={stat(project.validation.stats, 'valid')}/><Stat label={t('Missing transcripts')} value={stat(project.validation.stats, 'missing_text')}/><Stat label={t('Total duration')} value={formatDuration(stat(project.validation.stats, 'total_seconds'))}/></div>
      <InlineAlert tone={project.validation_status.stale ? project.validation.enabled ? 'warning' : 'info' : 'success'} title={project.validation_status.message}><span className={s.path}>{project.validation.manifest}</span></InlineAlert>
    </div>
  </Section>;
}

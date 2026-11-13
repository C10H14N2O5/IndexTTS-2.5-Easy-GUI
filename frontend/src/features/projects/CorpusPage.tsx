import { useRef, useState } from 'react';
import { Link } from 'react-router';
import { Check, RefreshCw, Save } from 'lucide-react';
import { api, errorMessage } from '@/api/client';
import { useCatalog } from '@/app/bootstrap';
import { useT } from '@/i18n';
import { Button, Disclosure, Field, InlineAlert, PathField, Section, Select, Stat, TextInput, formatDuration, toast } from '@/ui';
import { ProofreadPanel, type ProofreadPanelHandle } from '@/features/proofread/ProofreadPanel';
import { DatasetVersions } from './DatasetVersions';
import { projectUrl, stat, useProjectContext, useProjectMutation, useTrainingStatus, type Project } from './api';
import s from './Workbench.module.css';

export default function CorpusPage() {
  const { project, refresh, setContextLocked } = useProjectContext(); const t = useT(); const editor = useRef<ProofreadPanelHandle>(null);
  const prepareChange = async (commit: () => Promise<unknown>) => { if (editor.current) return editor.current.prepareContextChange(commit); try { await commit(); return true; } catch { return false; } };
  return <div className={s.stack}><CorpusSettings key={`${project.project_name}:${project.modified}`} project={project} prepareChange={prepareChange}/>
    <DatasetVersions project={project} role="training"/>
    <Section title={t('Review transcripts')} description={t('Correct the words before building the training manifest. Audio files stay untouched.')}>
      <ProofreadPanel ref={editor} project={project.project_name} directory={project.source_folder ?? ''} role="training" language={project.language ?? 'EN'} onUpdated={refresh} onContextLockedChange={setContextLocked}/>
    </Section></div>;
}

function CorpusSettings({ project, prepareChange }: { project: Project; prepareChange: (commit: () => Promise<unknown>) => Promise<boolean> }) {
  const t = useT(); const catalog = useCatalog(); const [folder, setFolder] = useState(project.source_folder ?? ''); const [language, setLanguage] = useState(project.language ?? 'ZH'); const [manifest, setManifest] = useState(project.manifest ?? ''); const running = useTrainingStatus().data?.running;
  const save = useProjectMutation(project.project_name, (prepare: boolean) => api.put<Project>(`${projectUrl(project.project_name)}/dataset`, { source_folder: folder, language, ...(manifest !== project.manifest ? { manifest } : {}), prepare }));
  async function submit(prepare: boolean) { if (await prepareChange(() => save.mutateAsync(prepare))) toast.success(prepare ? t('Training manifest is ready.') : t('Corpus settings saved.')); }
  const busyReason = running ? t('Stop training before editing projects.') : undefined;
  return <Section layout="aside" title={t('Training corpus')} description={t('Use clean, single-speaker audio with a TXT transcript beside each file.')} actions={<Button icon={Save} variant="outlined" pending={save.isPending} disabledReason={busyReason} onClick={() => submit(false)}>{t('Save folder')}</Button>}>
    <div className={s.fields}><PathField label={t('Source audio folder')} kind="dir" value={folder} onChange={setFolder} disabled={running} hint={t('Nested folders are included. Matching audio and TXT files are paired by filename.')}/>
      <div className={s.two}><Field label={t('Corpus language')}><Select value={language} onChange={setLanguage} disabled={running} options={catalog.languages.map(item => ({ value: item.code, label: t(item.label) }))}/></Field><div className={s.actions}><Button icon={RefreshCw} pending={save.isPending} disabledReason={busyReason ?? (!folder ? t('Choose a source folder first.') : undefined)} onClick={() => submit(true)}>{t('Build training manifest')}</Button></div></div>
      {save.error && <InlineAlert tone="error">{errorMessage(save.error)}</InlineAlert>}
      <div className={s.stats}><Stat label={t('Usable clips')} value={stat(project.stats, 'valid')}/><Stat label={t('Missing transcripts')} value={stat(project.stats, 'missing_text')}/><Stat label={t('Total duration')} value={formatDuration(stat(project.stats, 'total_seconds'))}/></div>
      <InlineAlert tone={project.dataset_status.stale ? 'warning' : 'success'} title={project.dataset_status.message} action={!project.dataset_status.stale && <Link to="../training" className={s.linkButton}>{t('Continue to training')} <Check size={14}/></Link>}><span className={s.path}>{project.manifest}</span></InlineAlert>
      <Disclosure title={t('Manifest and clip statistics')} summary={t('Automatic clip bounds and prepared file location')}><div className={s.fields}><div className={s.stats}><Stat label={t('Median clip')} value={formatDuration(stat(project.stats, 'median_seconds'))}/><Stat label={t('P90 clip')} value={formatDuration(stat(project.stats, 'p90_seconds'))}/><Stat label={t('Automatic range')} value={`${stat(project.stats, 'auto_min_seconds').toFixed(1)}–${stat(project.stats, 'auto_max_seconds').toFixed(1)} s`}/></div><Field label={t('Prepared manifest path')} hint={t('Normally generated automatically. Only change this if you manage manifests yourself.')}><TextInput value={manifest} onChange={e => setManifest(e.target.value)} disabled={running}/></Field></div></Disclosure>
    </div>
  </Section>;
}

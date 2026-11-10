import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router';
import { ArrowRight, FolderPlus, Folders, RefreshCw } from 'lucide-react';
import { api, errorMessage } from '@/api/client';
import { useCatalog } from '@/app/bootstrap';
import { useT } from '@/i18n';
import { Badge, Button, Dialog, EmptyState, Field, InlineAlert, PageHeader, PathField, Select, Skeleton, Stat, TextInput, formatDuration, formatRelativeTime, toast } from '@/ui';
import { projectRoute, stat, useProjects, type Project } from './api';
import s from './Workbench.module.css';

export default function ProjectsPage() {
  const t = useT(); const query = useProjects(); const [search, setSearch] = useState(''); const [create, setCreate] = useState(false);
  const projects = (query.data ?? []).filter(item => item.project_name.toLowerCase().includes(search.toLowerCase()));
  return <div className={s.stack}>
    <PageHeader title={t('Projects')} description={t('A focused workspace for your corpus, training and checkpoints.')} actions={<Button icon={FolderPlus} onClick={() => setCreate(true)}>{t('New project')}</Button>} />
    <div className={s.toolbar}><TextInput className={s.search} value={search} onChange={e => setSearch(e.target.value)} aria-label={t('Search projects')} placeholder={t('Search projects')} /><Button icon={RefreshCw} variant="text" pending={query.isFetching} onClick={() => void query.refetch()}>{t('Refresh')}</Button></div>
    {query.error && <InlineAlert tone="error" action={<Button variant="text" onClick={() => void query.refetch()}>{t('Retry')}</Button>}>{errorMessage(query.error)}</InlineAlert>}
    {query.isPending ? <div className={s.two}><Skeleton height={220}/><Skeleton height={220}/></div> : !projects.length ? <EmptyState icon={Folders} title={search ? t('No matching projects') : t('Start with a voice project')} description={search ? t('Try another project name.') : t('Create a project, choose an audio folder, then review its transcripts. Your existing data stays in place.')} action={!search && <Button icon={FolderPlus} onClick={() => setCreate(true)}>{t('Create project')}</Button>} /> : <div className={s.cards}>{projects.map(project => <article key={project.project_name} className={s.card}>
      <div className={s.toolbar}><Link className={s.cardTitle} to={projectRoute(project.project_name)}>{project.project_name}</Link><Badge tone={project.dataset_status.stale ? 'neutral' : 'success'}>{project.dataset_status.stale ? t('Needs preparation') : t('Corpus ready')}</Badge></div>
      <div className={s.stats}><Stat label={t('Usable clips')} value={stat(project.stats, 'valid')} /><Stat label={t('Duration')} value={formatDuration(stat(project.stats, 'total_seconds'))} /><Stat label={t('Language')} value={project.language} /></div>
      <div className={s.toolbar}><span className={s.muted}>{t('Updated {0}', formatRelativeTime(project.modified))}</span><Link className={s.linkButton} to={projectRoute(project.project_name)}>{t('Open project')} <ArrowRight size={14}/></Link></div>
    </article>)}</div>}
    {create && <CreateProjectDialog onClose={() => setCreate(false)} />}
  </div>;
}

function CreateProjectDialog({ onClose }: { onClose: () => void }) {
  const t = useT(); const catalog = useCatalog(); const navigate = useNavigate(); const client = useQueryClient();
  const [name, setName] = useState(''); const [language, setLanguage] = useState(catalog.default_language); const [folder, setFolder] = useState('');
  const create = useMutation({ mutationFn: () => api.post<Project>('/api/projects', { name: name.trim(), language, source_folder: folder }), onSuccess: project => { void client.invalidateQueries({ queryKey: ['projects'] }); toast.success(t('Project created')); onClose(); void navigate(projectRoute(project.project_name)); } });
  return <Dialog open onOpenChange={open => { if (!open) onClose(); }} title={t('New project')} description={t('Give this voice a name. You can add the corpus now or later.')} dismissible={!create.isPending} footer={<><Button variant="text" onClick={onClose} disabled={create.isPending}>{t('Cancel')}</Button><Button pending={create.isPending} disabledReason={!name.trim() ? t('Enter a project name.') : undefined} onClick={() => create.mutate()}>{t('Create project')}</Button></>}>
    <div className={s.fields}><Field label={t('Project name')} required><TextInput value={name} onChange={e => setName(e.target.value)} autoFocus maxLength={120} onKeyDown={e => { if (e.key === 'Enter' && name.trim()) create.mutate(); }}/></Field>
      <Field label={t('Corpus language')}><Select value={language} onChange={setLanguage} options={catalog.languages.map(item => ({ value: item.code, label: t(item.label) }))}/></Field>
      <PathField label={t('Source audio folder')} value={folder} onChange={setFolder} kind="dir" hint={t('Optional. Select a folder on the server containing audio and matching TXT files.')} />
      {create.error && <InlineAlert tone="error">{errorMessage(create.error)}</InlineAlert>}
    </div>
  </Dialog>;
}

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, NavLink, Outlet, useNavigate, useParams } from 'react-router';
import { Check, Copy, FileAudio, FolderOpen, GraduationCap, ShieldCheck, Trash2 } from 'lucide-react';
import { api, errorMessage } from '@/api/client';
import { useT } from '@/i18n';
import { Badge, Button, ConfirmDialog, EmptyState, InlineAlert, PageHeader, Skeleton, toast } from '@/ui';
import { projectRoute, projectUrl, useProject, useTrainingStatus, type Project, type ProjectContext } from './api';
import s from './Workbench.module.css';

export default function ProjectLayout() {
  const t = useT(); const { project: name = '' } = useParams(); const query = useProject(name); const client = useQueryClient(); const navigate = useNavigate(); const [remove, setRemove] = useState(false); const [contextLocked, setContextLocked] = useState(false); const training = useTrainingStatus().data;
  const clone = useMutation({ mutationFn: () => api.post<Project>(`${projectUrl(name)}/clone`, {}), onSuccess: result => { void client.invalidateQueries({ queryKey: ['projects'] }); toast.success(t('Project cloned')); void navigate(projectRoute(result.project_name)); } });
  async function deleteProject() { await api.delete(projectUrl(name), undefined, { confirm_name: name }); await client.invalidateQueries({ queryKey: ['projects'] }); toast.success(t('Project deleted; checkpoints were kept.')); void navigate('/projects'); }
  if (query.isPending) return <div className={s.stack}><Skeleton height={56}/><Skeleton height={64}/><Skeleton height={360}/></div>;
  if (!query.data) return <EmptyState title={t('Project could not be loaded')} description={errorMessage(query.error)} action={<Link to="/projects">{t('Back to projects')}</Link>}/>;
  const project = query.data; const locked = training?.running ? t('Stop training before editing projects.') : contextLocked ? t('Finish the current task and save or discard your edits before managing this project.') : undefined;
  const context: ProjectContext = { project, refresh: () => { void query.refetch(); }, setContextLocked };
  return <div>
    <PageHeader title={name} description={t('Prepare your data. Train with confidence. Compare the result.')} back={{ to: '/projects', label: t('All projects') }} actions={<><Button variant="outlined" icon={Copy} pending={clone.isPending} disabledReason={locked} onClick={() => clone.mutate()}>{t('Clone project')}</Button><Button variant="text" icon={Trash2} disabledReason={locked} onClick={() => setRemove(true)}>{t('Delete project')}</Button></>} />
    {clone.error && <InlineAlert tone="error">{errorMessage(clone.error)}</InlineAlert>}
    <div className={`${s.readiness} ${s.toolbar}`}><div className={s.actions}><Badge tone={project.dataset_status.stale ? 'warning' : 'success'} icon={Check}>{project.dataset_status.stale ? t('Corpus needs attention') : t('Corpus ready')}</Badge><Badge tone={project.validation.enabled && project.validation_status.stale ? 'warning' : 'neutral'}>{project.validation.enabled ? project.validation_status.stale ? t('Validation needs attention') : t('Validation ready') : t('Validation off')}</Badge>{training?.running && training.project === name && <Badge tone="accent">{t('Training in progress')}</Badge>}</div><span className={s.chip}>{project.language} · {project.training.adapter_name}</span></div>
    <nav className={s.nav} aria-label={t('Project sections')}><NavLink to="corpus"><FileAudio size={16}/>{t('Corpus')}</NavLink><NavLink to="validation"><ShieldCheck size={16}/>{t('Validation')}</NavLink><NavLink to="training"><GraduationCap size={16}/>{t('Training')}</NavLink><NavLink to="outputs"><FolderOpen size={16}/>{t('Outputs')}</NavLink></nav>
    <Outlet context={context}/>
    <ConfirmDialog open={remove} onOpenChange={setRemove} title={t('Delete project {0}?', name)} description={t('Project settings and prepared manifests will be removed. Source audio, validation audio and training outputs will be kept.')} requireText={name} confirmLabel={t('Delete project')} onConfirm={deleteProject}/>
  </div>;
}

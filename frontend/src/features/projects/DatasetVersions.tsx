import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Pencil, Snowflake, Trash2, Wrench } from 'lucide-react';
import { api, errorMessage, type Schemas } from '@/api/client';
import { intlLocale, useT } from '@/i18n';
import { Badge, Button, Checkbox, ConfirmDialog, Dialog, Disclosure, Field, InlineAlert, KeyValueList, Section, SegmentedControl, Select, Skeleton, Switch, TextArea, TextInput, formatDuration, toast } from '@/ui';
import { projectKey, stat, useDatasetVersions, useProjectMutation, useTrainingStatus, versionsKey, versionsUrl, type DatasetRole, type DatasetVersion, type IntegrityReport, type Project } from './api';
import s from './Workbench.module.css';

type Action = { kind: 'rename' | 'delete' | 'repair'; version: DatasetVersion } | null;

function formatDate(seconds: number) { return new Intl.DateTimeFormat(intlLocale(), { dateStyle: 'medium', timeStyle: 'short' }).format(seconds * 1000); }

/** Freeze, choose, rename, delete and repair read-only versions of one manifest (training or validation). */
export function DatasetVersions({ project, role }: { project: Project; role: DatasetRole }) {
  const t = useT(); const client = useQueryClient(); const name = project.project_name;
  const query = useDatasetVersions(name, role, project.modified); const data = query.data; const versions = data?.versions ?? [];
  const running = Boolean(useTrainingStatus().data?.running); const busyReason = running ? t('Stop training before changing the dataset version.') : undefined;
  const [freezing, setFreezing] = useState(false); const [action, setAction] = useState<Action>(null);
  const refresh = () => { void client.invalidateQueries({ queryKey: versionsKey(name, role) }); void client.invalidateQueries({ queryKey: projectKey(name) }); void client.invalidateQueries({ queryKey: ['projects'] }); };
  const select = useProjectMutation(name, (version: string) => api.put<Project>(versionsUrl(name, role, '/active'), { version_id: version || null }));
  const active = versions.find(item => item.id === data?.active);
  const training = role === 'training';
  const onSelect = (value: string) => select.mutate(value, { onSuccess: () => { void client.invalidateQueries({ queryKey: versionsKey(name, role) }); toast.success(value ? t('Now using version {0}.', versions.find(item => item.id === value)?.name ?? value) : t('Now using the working set.')); } });
  const options = [{ value: '', label: t('Working set (latest manifest)'), description: data?.working.message }, ...versions.map(item => ({ value: item.id, label: `${item.name} · ${t('{0} clips', stat(item.stats, 'valid'))}`, description: `${item.id} · r${item.revision} · ${formatDate(item.created_at)}${item.integrity.status !== 'ready' ? ` · ${t('Needs repair')}` : ''}` }))];
  return <Section layout="aside" title={t('Dataset versions')} description={training ? t('Freeze the training manifest to keep an exact, read-only copy of its rows and transcripts. Audio is not copied; each file is fingerprinted instead.') : t('Freeze the validation manifest so later checkpoints are measured on exactly the same clips. Audio is not copied; each file is fingerprinted instead.')} actions={<Button icon={Snowflake} variant="outlined" disabledReason={query.isPending ? t('Loading…') : data?.freeze_reason ?? undefined} onClick={() => setFreezing(true)}>{t('Freeze current manifest')}</Button>}>
    <div className={s.fields}>
      {query.error && <InlineAlert tone="error">{errorMessage(query.error)}</InlineAlert>}
      {query.isPending ? <Skeleton height={60}/> : <Field label={training ? t('Training uses') : t('Validation uses')} hint={training ? t('Training, suggested settings and the planned steps use this dataset.') : t('Training-time and post-hoc validation loss use this dataset.')}><Select value={data?.active ?? ''} onChange={onSelect} disabled={running || select.isPending} options={options}/></Field>}
      {busyReason && <p className={s.muted}>{busyReason}</p>}
      {data?.freeze_reason && <p className={s.muted}>{data.freeze_reason}</p>}
      {select.error && <InlineAlert tone="error">{errorMessage(select.error)}</InlineAlert>}
      {active ? <VersionDetails version={active} onAction={setAction} running={running}/> : data && <p className={s.muted}>{t('The working set follows the latest built manifest and changes whenever you rebuild it. Freeze it to keep the current rows for later runs.')}</p>}
      {active && <InlineAlert tone="info">{t('Proofreading and rebuilding the manifest below only change the working set. Version {0} stays unchanged.', active.name)}</InlineAlert>}
      {versions.length > 0 && <Disclosure title={t('All versions')} summary={t('{0} frozen versions', versions.length)}><div className={s.tableWrap}><table className={s.table}><thead><tr><th>{t('Version')}</th><th>{t('Clips')}</th><th>{t('Frozen')}</th><th>{t('Status')}</th><th>{t('Actions')}</th></tr></thead><tbody>{versions.map(item => <tr key={item.id}><td><strong>{item.name}</strong> {item.active && <Badge tone="accent">{t('In use')}</Badge>}<div className={s.muted}>{item.id} · r{item.revision}</div></td><td>{stat(item.stats, 'valid')}</td><td>{formatDate(item.created_at)}</td><td><IntegrityBadge version={item}/></td><td><VersionButtons version={item} onAction={setAction} running={running}/></td></tr>)}</tbody></table></div></Disclosure>}
    </div>
    <FreezeDialog open={freezing} onOpenChange={setFreezing} project={name} role={role} nextName={data?.next_name ?? ''} running={running} onDone={refresh}/>
    {action?.kind === 'rename' && <RenameDialog version={action.version} project={name} role={role} onClose={() => setAction(null)} onDone={refresh}/>}
    {action?.kind === 'repair' && <RepairDialog version={action.version} project={name} role={role} onClose={() => setAction(null)} onDone={refresh}/>}
    <ConfirmDialog open={action?.kind === 'delete'} onOpenChange={open => { if (!open) setAction(null); }} title={t('Delete version {0}?', action?.version.name ?? '')} confirmLabel={t('Delete version')}
      description={action?.kind === 'delete' ? [t('The frozen manifest and fingerprints are deleted. Audio files and the working set are not touched.'), action.version.used_by?.length ? t('Training records of {0} keep the version ID and SHA-256, but the version can no longer be used.', action.version.used_by.join(', ')) : '', action.version.active ? t('This version is in use; the working set is used afterwards.') : ''].filter(Boolean).join(' ') : undefined}
      onConfirm={async () => { if (!action) return; const result = await api.delete<Schemas['Ok']>(versionsUrl(name, role, `/${encodeURIComponent(action.version.id)}`)); refresh(); toast.success(result.message ?? t('Version deleted.')); }}/>
  </Section>;
}

function IntegrityBadge({ version }: { version: DatasetVersion }) {
  const t = useT(); const { status, missing, changed } = version.integrity;
  return status === 'ready' ? <Badge tone="success">{t('Intact')}</Badge> : status === 'damaged' ? <Badge tone="warning">{t('{0} missing · {1} changed', missing ?? 0, changed ?? 0)}</Badge> : <Badge tone="error">{t('Invalid')}</Badge>;
}

function VersionButtons({ version, onAction, running }: { version: DatasetVersion; onAction: (action: Action) => void; running: boolean }) {
  const t = useT(); const stop = running ? t('Stop training first.') : undefined;
  return <div className={s.actions}><Button size="sm" variant="text" icon={Pencil} onClick={() => onAction({ kind: 'rename', version })}>{t('Rename')}</Button>{version.integrity.status === 'damaged' && <Button size="sm" variant="text" icon={Wrench} disabledReason={stop} onClick={() => onAction({ kind: 'repair', version })}>{t('Repair')}</Button>}<Button size="sm" variant="text" icon={Trash2} disabledReason={stop} onClick={() => onAction({ kind: 'delete', version })}>{t('Delete')}</Button></div>;
}

function VersionDetails({ version, onAction, running }: { version: DatasetVersion; onAction: (action: Action) => void; running: boolean }) {
  const t = useT(); const diff = version.diff;
  return <div className={s.fields}>
    <div className={s.toolbar}><div className={s.actions}><strong>{version.name}</strong><IntegrityBadge version={version}/></div><VersionButtons version={version} onAction={onAction} running={running}/></div>
    {version.integrity.status !== 'ready' && <InlineAlert tone={version.integrity.status === 'damaged' ? 'warning' : 'error'} title={version.integrity.message}>{version.integrity.status === 'damaged' && t('Training and validation with this version are blocked until it is repaired.')}</InlineAlert>}
    <KeyValueList items={[
      { label: t('Version ID'), value: `${version.id} · r${version.revision}` },
      { label: t('Frozen'), value: formatDate(version.created_at) },
      { label: t('Clips'), value: `${stat(version.stats, 'valid')} · ${formatDuration(stat(version.stats, 'total_seconds'))}` },
      { label: t('Language'), value: version.language },
      { label: t('Source folder'), value: <span className={s.path}>{version.source_folder || '—'}</span> },
      { label: t('Manifest SHA-256'), value: <span className={s.path}>{version.manifest_sha256.slice(0, 16)}</span> },
      { label: t('Compared with the working set'), value: diff ? t('+{0} added · −{1} removed · {2} text changes', diff.added, diff.removed, diff.text_changed) : t('Working manifest unavailable') },
      { label: t('Used by'), value: version.used_by?.length ? version.used_by.join(', ') : '—' },
      ...(version.note ? [{ label: t('Note'), value: version.note }] : []),
    ]}/>
  </div>;
}

function FreezeDialog({ open, onOpenChange, project, role, nextName, running, onDone }: { open: boolean; onOpenChange: (open: boolean) => void; project: string; role: DatasetRole; nextName: string; running: boolean; onDone: () => void }) {
  const t = useT(); const [name, setName] = useState(''); const [note, setNote] = useState(''); const [activate, setActivate] = useState(true);
  const freeze = useMutation({ mutationFn: () => api.post<DatasetVersion>(versionsUrl(project, role), { name: name.trim() || null, note, activate: activate && !running }), onSuccess: version => { onDone(); toast.success(t('Version {0} frozen.', version.name)); setName(''); setNote(''); onOpenChange(false); } });
  return <Dialog open={open} onOpenChange={value => { if (!value) freeze.reset(); onOpenChange(value); }} title={t('Freeze the current manifest')} description={t('The rows and transcripts of the current manifest are saved as a read-only version. Audio is fingerprinted, not copied.')}
    footer={<><Button variant="text" onClick={() => onOpenChange(false)} disabled={freeze.isPending}>{t('Cancel')}</Button><Button icon={Snowflake} pending={freeze.isPending} onClick={() => freeze.mutate()}>{t('Freeze')}</Button></>}>
    <div className={s.fields}><Field label={t('Version name')} hint={t('Leave empty to use {0}.', nextName)}><TextInput value={name} onChange={event => setName(event.target.value)} placeholder={nextName} maxLength={60}/></Field>
      <Field label={t('Note')}><TextArea value={note} onChange={event => setNote(event.target.value)} minRows={2} maxLength={500} placeholder={t('Optional, e.g. what changed since the last version')}/></Field>
      <Switch checked={activate && !running} onCheckedChange={setActivate} disabledReason={running ? t('Stop training before changing the dataset version.') : undefined} label={t('Use this version after freezing')}/>
      {freeze.error && <InlineAlert tone="error">{errorMessage(freeze.error)}</InlineAlert>}</div>
  </Dialog>;
}

function RenameDialog({ version, project, role, onClose, onDone }: { version: DatasetVersion; project: string; role: DatasetRole; onClose: () => void; onDone: () => void }) {
  const t = useT(); const [name, setName] = useState(version.name); const [note, setNote] = useState(version.note ?? '');
  const save = useMutation({ mutationFn: () => api.patch<DatasetVersion>(versionsUrl(project, role, `/${encodeURIComponent(version.id)}`), { name, note }), onSuccess: () => { onDone(); toast.success(t('Version saved.')); onClose(); } });
  return <Dialog open onOpenChange={value => { if (!value) onClose(); }} title={t('Rename version {0}', version.id)}
    footer={<><Button variant="text" onClick={onClose} disabled={save.isPending}>{t('Cancel')}</Button><Button pending={save.isPending} disabledReason={!name.trim() ? t('Enter a version name.') : undefined} onClick={() => save.mutate()}>{t('Save')}</Button></>}>
    <div className={s.fields}><Field label={t('Version name')}><TextInput value={name} onChange={event => setName(event.target.value)} maxLength={60}/></Field><Field label={t('Note')}><TextArea value={note} onChange={event => setNote(event.target.value)} minRows={2} maxLength={500}/></Field>
      {save.error && <InlineAlert tone="error">{errorMessage(save.error)}</InlineAlert>}</div>
  </Dialog>;
}

function RepairDialog({ version, project, role, onClose, onDone }: { version: DatasetVersion; project: string; role: DatasetRole; onClose: () => void; onDone: () => void }) {
  const t = useT(); const [changed, setChanged] = useState<'accept' | 'drop'>('accept'); const [affected, setAffected] = useState(false);
  const report = useQuery({ queryKey: ['dataset-version-integrity', project, role, version.id, version.revision], queryFn: ({ signal }) => api.get<IntegrityReport>(versionsUrl(project, role, `/${encodeURIComponent(version.id)}/integrity`), undefined, signal) });
  const repair = useMutation({ mutationFn: () => api.post<Schemas['RepairResult']>(versionsUrl(project, role, `/${encodeURIComponent(version.id)}/repair`), { changed, scope: affected ? 'affected' : 'version' }), onSuccess: result => { onDone(); toast.success(result.message); onClose(); } });
  const data = report.data; const missing = data?.missing ?? []; const altered = data?.changed ?? []; const others = data?.affected ?? [];
  const relocated = missing.filter(item => item.relocated_to).length; const dropped = (data?.missing_count ?? 0) - (data?.relocatable ?? 0);
  return <Dialog open size="lg" onOpenChange={value => { if (!value && !repair.isPending) onClose(); }} title={t('Repair version {0}', version.name)} description={t('The version keeps its ID and becomes revision r{0}. Every change is recorded in its history and in later training records.', version.revision + 1)}
    footer={<><Button variant="text" onClick={onClose} disabled={repair.isPending}>{t('Cancel')}</Button><Button icon={Wrench} pending={repair.isPending} disabledReason={report.isPending ? t('Checking audio files…') : data?.status !== 'damaged' ? t('This version does not need repair.') : undefined} onClick={() => repair.mutate()}>{t('Repair')}</Button></>}>
    <div className={s.fields}>
      {report.isPending && <Skeleton height={80}/>}
      {report.error && <InlineAlert tone="error">{errorMessage(report.error)}</InlineAlert>}
      {data && <InlineAlert tone={data.status === 'ready' ? 'success' : 'warning'} title={data.message}/>}
      {(data?.missing_count ?? 0) > 0 && <Field label={t('Missing audio ({0})', data!.missing_count)} hint={t('{0} moved file(s) were found in the source folder by content and are relinked. {1} clip(s) that cannot be found are removed from this version.', data!.relocatable, dropped)}><FileList items={missing.map(item => item.relocated_to ? `${item.audio} → ${item.relocated_to}` : item.audio)} total={data!.missing_count}/></Field>}
      {(data?.changed_count ?? 0) > 0 && <><Field label={t('Changed audio ({0})', data!.changed_count)}><FileList items={altered.map(item => item.audio)} total={data!.changed_count}/></Field>
        <Field label={t('Changed audio')}><SegmentedControl value={changed} onChange={setChanged} options={[{ value: 'accept', label: t('Accept the new audio') }, { value: 'drop', label: t('Remove these clips') }]}/></Field></>}
      {others.length > 0 && <Checkbox checked={affected} onCheckedChange={value => setAffected(value === true)} label={t('Also repair {0} other affected version(s) the same way: {1}', others.length, others.map(item => item.name).join(', '))}/>}
      {relocated > 0 && <p className={s.muted}>{t('Relinked clips keep their transcript; only the audio path changes.')}</p>}
      {repair.error && <InlineAlert tone="error">{errorMessage(repair.error)}</InlineAlert>}
    </div>
  </Dialog>;
}

function FileList({ items, total }: { items: string[]; total: number }) {
  const t = useT();
  return <div className={s.path} style={{ maxHeight: 160, overflow: 'auto' }}>{items.map(item => <div key={item}>{item}</div>)}{total > items.length && <div>{t('… and {0} more', total - items.length)}</div>}</div>;
}

import { useEffect, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router';
import { Download, PackageCheck, Play, Square } from 'lucide-react';
import { api, errorMessage, type Schemas } from '@/api/client';
import { applyEvent, useGpuJob, useLive } from '@/api/live';
import { useT } from '@/i18n';
import { Badge, Button, ConfirmDialog, Disclosure, EmptyState, Field, InlineAlert, KeyValueList, LogConsole, PageHeader, PathField, ProgressBar, Section, SegmentedControl, Select, Stat, TextInput, formatDuration, toast } from '@/ui';
import { useAdapters } from '@/features/synthesis/api';
import { EXPORT_DRAFT_KEY, EXPORT_PHASES, EXPORT_STATES, exportActive, exportControls, exportDownloads, exportProgress, outputDestination, readExportDraft, suggestedValue, verification } from './model';
import s from '@/features/projects/Workbench.module.css';

type ExportStatus = Schemas['ExportSnapshot'];
export default function ExportPage() {
  const t = useT(); const [params] = useSearchParams(); const adapters = useAdapters(); const [draft] = useState(readExportDraft); const [adapter, setAdapter] = useState(params.get('adapter') ?? draft.adapter ?? '');
  const info = useQuery({ queryKey: ['export-adapter-info', adapter], queryFn: () => api.get<Schemas['AdapterInfo']>('/api/export/adapter-info', { adapter }), enabled: Boolean(adapter) });
  const live = useLive(state => state.export) as ExportStatus | null; const connection = useLive(state => state.connection); const training = useLive(state => state.training?.running); const gpu = useGpuJob();
  const statusQuery = useQuery({ queryKey: ['export-status'], queryFn: () => api.get<ExportStatus>('/api/export/status'), refetchInterval: connection === 'open' ? false : 1500 });
  const status = connection === 'open' ? live ?? statusQuery.data : statusQuery.data ?? live; const active = exportActive(status?.status ?? 'idle');
  const [mode, setMode] = useState(draft.mode ?? 'browser'); const [filenameOverride, setFilename] = useState<string | null>(draft.filename ?? null); const [baseOverride, setBase] = useState<string | null>(draft.base ?? null); const [configOverride, setConfig] = useState<string | null>(draft.config ?? null); const [expected, setExpected] = useState(draft.expected ?? ''); const [folder, setFolder] = useState(draft.folder ?? ''); const [cancelOpen, setCancelOpen] = useState(false);
  useEffect(() => { try { localStorage.setItem(EXPORT_DRAFT_KEY, JSON.stringify({ adapter, mode, filename: filenameOverride, base: baseOverride, config: configOverride, expected, folder })); } catch { /* Private mode: edits still work for this visit. */ } }, [adapter, mode, filenameOverride, baseOverride, configOverride, expected, folder]);
  const filename = suggestedValue(filenameOverride, info.data?.filename, 'voice_checkpoint.pth'); const base = suggestedValue(baseOverride, info.data?.base); const config = suggestedValue(configOverride, info.data?.config);
  const start = useMutation({ mutationFn: () => api.post<ExportStatus>('/api/export/start', { adapter, mode, destination: outputDestination(mode, folder, filename), base, config, expected_base: expected }), onSuccess: result => { applyEvent('export', result); toast.success(t('Export started on CPU.')); } });
  const cancel = useMutation({ mutationFn: () => api.post<ExportStatus>('/api/export/cancel'), onSuccess: result => applyEvent('export', result) });
  const reason = active ? t('An export is already running.') : training ? t('Wait for training to finish.') : gpu ? t('The GPU is busy: {0}', gpu.title) : !adapter ? t('Choose an adapter to export.') : info.isPending ? t('Checking adapter…') : !info.data?.valid ? t('Select a valid adapter folder.') : !base || !config ? t('Choose the original training base and configuration.') : !filename.toLowerCase().endsWith('.pth') || /[<>:"/\\|?*]/.test(filename) ? t('Enter a filename ending in .pth without directory separators.') : mode === 'path' && !folder ? t('Choose an output folder on the server.') : undefined;
  const options = [{ value: '', label: t('Choose an adapter…') }, ...(adapters.data ?? []).map(item => ({ value: item.path, label: item.label }))];
  if (adapter && !options.some(item => item.value === adapter)) options.push({ value: adapter, label: adapter.split(/[\\/]/).slice(-2).join(' / ') });
  return <div className={s.stack}><PageHeader title={t('Export checkpoint')} description={t('Merge a LoRA adapter into a complete FP32 GPT checkpoint on CPU. Source weights are never overwritten.')}/>
    <div className={s.grid}><Section title={t('What to export')} description={t('Select a final adapter or any saved checkpoint.')}>
      <div className={s.fields}><Field label={t('LoRA adapter')}><Select value={adapter} onChange={setAdapter} options={options} disabled={active || start.isPending}/></Field>
        <Disclosure title={t('Custom adapter folder')}><PathField label={t('Adapter folder')} kind="dir" value={adapter} onChange={setAdapter} disabled={active || start.isPending}/></Disclosure>
        {info.data && <InlineAlert tone={info.data.valid ? 'info' : 'warning'}>{info.data.notice}</InlineAlert>}
        <Field label={t('Destination')}><SegmentedControl value={mode} onChange={setMode} aria-label={t('Destination')} options={[{ value: 'browser', label: t('Download in browser'), disabled: active }, { value: 'path', label: t('Save on server'), disabled: active }]}/></Field>
        <Field label={t('Output filename')} modified={filenameOverride !== null} onReset={() => setFilename(null)} hint={t('A new file is always created. Existing files cannot be overwritten.')}><TextInput value={filename} onChange={e => setFilename(e.target.value)} disabled={active || start.isPending}/></Field>
        {mode === 'path' && <PathField label={t('Output folder on server')} kind="dir" value={folder} onChange={setFolder} disabled={active || start.isPending} hint={t('Browsing only chooses a folder; it never starts an export.')}/>}
        <Disclosure title={t('Original base and verification')} summary={t('Use the exact FP32 base from training')} modifiedCount={Number(baseOverride !== null) + Number(configOverride !== null) + Number(Boolean(expected))} defaultOpen><div className={s.fields}>
          <PathField label={t('Original FP32 training base (.pth)')} kind="pth" value={base} onChange={setBase} disabled={active || start.isPending} hint={t('A matching folder name does not prove model identity. Use the original weights, not a quantized or already-merged model.')}/>
          {baseOverride !== null && <Button variant="text" size="sm" disabled={active} onClick={() => setBase(null)}>{t('Use suggested base')}</Button>}
          <Field label={t('Model configuration (config.yaml)')} modified={configOverride !== null} onReset={() => setConfig(null)}><TextInput value={config} onChange={e => setConfig(e.target.value)} disabled={active || start.isPending}/></Field>
          <PathField label={t('Expected original base (optional)')} kind="pth" value={expected} onChange={setExpected} disabled={active || start.isPending} hint={t('An independent known-original checkpoint enables additional base identity verification.')}/>
        </div></Disclosure>
        {[adapters.error, info.error, start.error].filter(Boolean).map((error, index) => <InlineAlert tone="error" key={index}>{errorMessage(error)}</InlineAlert>)}
        <Button icon={Play} pending={start.isPending} disabledReason={reason} onClick={() => start.mutate()}>{t('Export and verify')}</Button>{reason && <p className={s.muted}>{reason}</p>}
      </div>
    </Section><div className={s.sticky}><Section title={t('Export progress')} actions={status && exportControls(status.status).canCancel && <Button icon={Square} variant="outlined" pending={cancel.isPending} onClick={() => setCancelOpen(true)}>{t('Cancel export')}</Button>}>
      {status && status.status !== 'idle' ? <ExportResult status={status}/> : <EmptyState icon={PackageCheck} title={t('Ready to create a standalone checkpoint')} description={t('The exporter validates tensor values, reloads the saved weights and checks that source files are unchanged.')}/>}
      {[statusQuery.error, cancel.error].filter(Boolean).map((error, index) => <InlineAlert tone="error" key={index}>{errorMessage(error)}</InlineAlert>)}
    </Section></div></div>
    {status?.log && <Section title={t('Export log')}><LogConsole lines={status.log.trimEnd().split('\n').map((text, i) => ({ id: i + 1, text, level: 'info' }))} height={240}/></Section>}
    <ConfirmDialog open={cancelOpen} onOpenChange={setCancelOpen} title={t('Cancel this export?')} description={t('Only this task’s temporary staging files will be removed. Source weights and completed exports are kept.')} confirmLabel={t('Cancel export')} onConfirm={() => cancel.mutateAsync()}/>
  </div>;
}

export function ExportResult({ status }: { status: ExportStatus }) {
  const t = useT(); const active = exportActive(status.status); const report = status.report ?? {}; const downloads = exportDownloads(status.status, status.downloads);
  return <div className={s.fields}><div className={s.toolbar}><Badge tone={status.status === 'succeeded' ? 'success' : status.status === 'failed' ? 'error' : active ? 'accent' : 'neutral'}>{t(EXPORT_STATES[status.status] ?? status.status)}</Badge><span className={s.muted}>{formatDuration(status.elapsed)}</span></div>
    <div className={s.phase} aria-live="polite">{t(EXPORT_PHASES[status.phase as keyof typeof EXPORT_PHASES] ?? status.phase)}</div><ProgressBar value={status.status === 'succeeded' ? 1 : active ? exportProgress(status.phase) : 0} label={t('Export progress')}/>
    {status.error && <InlineAlert tone="error">{status.error}</InlineAlert>}
    {status.status === 'cancelling' && <InlineAlert tone="info">{t('Cancellation requested. Waiting for the CPU process to exit safely.')}</InlineAlert>}
    {status.output && <p className={s.path}>{status.output}</p>}
    {status.status === 'succeeded' && <><div className={s.stats}><Stat label={t('Validated tensors')} value={String(report.tensor_count ?? '—')}/><Stat label={t('LoRA targets')} value={String(report.lora_target_count ?? '—')}/><Stat label={t('Changed tensors')} value={String(report.changed_tensor_count ?? '—')}/></div><KeyValueList items={[{ label: t('Source files unchanged'), value: t(verification(report.source_files_unchanged)) }, { label: t('Saved tensors identical'), value: t(verification(report.round_trip_tensors_identical)) }, ...(report.cpu_forward_max_abs_difference !== undefined ? [{ label: t('CPU forward max absolute difference'), value: String(report.cpu_forward_max_abs_difference) }] : [])]}/>
      <div className={s.actions}>{downloads.map(([kind, url]) => <a key={kind} href={url} download className={s.download}><Download size={16}/>{kind === 'model' ? t('Download GPT checkpoint (.pth)') : t('Download validation report (.merge.json)')}</a>)}</div>
      <Disclosure title={t('Verification hashes')}><KeyValueList items={[{ label: t('Base SHA-256'), value: <span className={s.path}>{String(report.base_sha256 ?? t('Not verified'))}</span> }, { label: t('Export SHA-256'), value: <span className={s.path}>{String(report.output_sha256 ?? t('Not verified'))}</span> }]}/></Disclosure>
    </>}
  </div>;
}

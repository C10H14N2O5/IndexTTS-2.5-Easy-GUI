import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { HardDrive, RefreshCw, Volume2 } from 'lucide-react';
import { Badge, Button, ConfirmDialog, Field, InlineAlert, KeyValueList, PageHeader, ProgressBar, Section, SegmentedControl, Select, Stat, Switch, toast } from '@/ui';
import { api, errorMessage, type Schemas } from '@/api/client';
import { useGpuJob, useLive } from '@/api/live';
import { useBootstrap, useSettings, useUpdateSettings, type SettingsPatch } from '@/app/bootstrap';
import { useT } from '@/i18n';
import { notificationsAvailable, playCompletionSound, unlockCompletionSound } from '@/features/activity/effects';
import { voicesKey } from '@/features/library/api';
import { historyKey } from '@/features/synthesis/api';
import styles from './System.module.css';

type DangerAction = 'unload' | 'outputs' | 'voices';
export default function SystemPage() {
  const t = useT(); const bootstrap = useBootstrap(); const settings = useSettings(); const update = useUpdateSettings();
  const liveSystem = useLive((state) => state.system); const training = useLive((state) => state.training); const gpu = useGpuJob();
  const status = useQuery({ queryKey: ['system-status'], queryFn: () => api.get<Schemas['SystemStatus']>('/api/system/status'), refetchInterval: 15000 });
  const system = liveSystem ?? status.data ?? bootstrap.data?.system; const capabilities = bootstrap.data?.capabilities;
  const [confirm, setConfirm] = useState<DangerAction | null>(null); const [notificationError, setNotificationError] = useState(''); const [permissionPending, setPermissionPending] = useState(false);
  const [permission, setPermission] = useState(() => typeof Notification === 'undefined' ? 'unavailable' : Notification.permission);
  const client = useQueryClient();
  const danger = useMutation({ mutationFn: async (action: DangerAction) => {
    if (action === 'unload') return api.post('/api/model/unload');
    return api.delete(action === 'outputs' ? '/api/outputs' : '/api/voices');
  }, onSuccess: async (_, action) => { await client.invalidateQueries({ queryKey: action === 'voices' ? voicesKey : action === 'outputs' ? historyKey : ['system-status'] }); } });
  const busyReason = gpu ? t('The GPU is busy: {0}', gpu.title) : training?.running ? t('Stop training before unloading models.') : undefined;
  function save(patch: SettingsPatch) { update.mutate(patch); }
  async function enableNotifications(checked: boolean) {
    setNotificationError('');
    if (!checked) { save({ notifications: false }); return; }
    if (!notificationsAvailable()) { setNotificationError(t('Notifications require HTTPS or localhost and a supported browser.')); return; }
    setPermissionPending(true);
    try {
      const next = await Notification.requestPermission(); setPermission(next);
      if (next === 'granted') save({ notifications: true });
      else setNotificationError(t('Notifications were not allowed. Change the permission in your browser site settings to try again.'));
    } catch (error) { setNotificationError(errorMessage(error)); }
    finally { setPermissionPending(false); }
  }
  if (!settings || !system) return null;
  const titles = { unload: t('Unload all models?'), outputs: t('Delete all output recordings?'), voices: t('Delete every reference voice?') };
  const descriptions = { unload: t('This frees model memory. The next synthesis will load the model again.'), outputs: t('All audio files directly inside the outputs folder and their settings sidecars will be permanently deleted. Training and export files are not affected.'), voices: t('Every reference audio file and its transcript in the voice library will be permanently deleted. This cannot be undone.') };
  const vram = system.vram; const cuda = system.cuda;
  return <div className={styles.page}>
    <PageHeader title={t('System')} description={t('A clear view of your runtime, preferences, and local storage.')} actions={<Button variant="text" icon={RefreshCw} pending={status.isFetching} onClick={() => { void status.refetch(); void bootstrap.refetch(); }}>{t('Refresh status')}</Button>} />
    {system.fake_engine && <InlineAlert tone="info" title={t('Demo engine is active')}>{t('Audio and training are simulated. No production model or GPU is required in this mode.')}</InlineAlert>}
    {update.error && <InlineAlert tone="error" title={t('Settings could not be saved')}>{errorMessage(update.error)}</InlineAlert>}
    <div className={styles.sections}>
        <Section layout="aside" title={t('Model and device')} actions={<Badge tone={system.model.loaded ? 'success' : 'neutral'}>{system.model.loaded ? t('Loaded') : t('Not loaded')}</Badge>}>
          <div className={styles.fields}>
            {!system.model_present && !system.fake_engine && <InlineAlert tone="warning" title={t('Model files are missing')}>{t('Place the IndexTTS model files in the model directory before synthesizing.')}</InlineAlert>}
            <KeyValueList items={[{ label: t('Application version'), value: bootstrap.data?.version }, { label: t('Model directory'), value: system.model_dir }, { label: t('Active adapter'), value: system.model.adapter || t('Original model') }, { label: t('CUDA device'), value: cuda?.device || t('No CUDA device detected') }, { label: t('PyTorch version'), value: cuda?.torch ?? '—' }, { label: t('CUDA version'), value: cuda?.cuda ?? '—' }]} />
            {cuda?.error && <InlineAlert tone="warning">{cuda.error}</InlineAlert>}
            {status.data?.asr && <InlineAlert tone={status.data.asr.available ? 'info' : 'warning'} title={t('Reference transcription')}>{status.data.asr.message}</InlineAlert>}
            <div className={styles.stats}><Stat label={t('Allocated VRAM')} value={`${vram.allocated_gb.toFixed(2)} GB`} /><Stat label={t('Reserved VRAM')} value={`${vram.reserved_gb.toFixed(2)} GB`} hint={vram.total_gb ? t('{0} GB total', vram.total_gb.toFixed(1)) : undefined} /></div>
            <ProgressBar value={vram.total_gb ? Math.min(1, vram.allocated_gb / vram.total_gb) : 0} label={t('GPU memory usage')} />
            <Button variant="outlined" icon={HardDrive} disabledReason={busyReason || (!system.model.loaded ? t('There are no loaded models to unload.') : undefined)} onClick={() => setConfirm('unload')}>{t('Unload models')}</Button>
            {status.error && <InlineAlert tone="error">{errorMessage(status.error)}</InlineAlert>}
          </div>
        </Section>
        <Section layout="aside" title={t('Runtime acceleration')} description={t('Changes apply when the model is next loaded. Unsupported options cannot be enabled.')}>
          <div className={styles.fields}>
            <Switch checked={settings.use_bf16} onCheckedChange={(value) => save({ use_bf16: value })} label={t('BF16 precision')} description={t('Reduce memory use with bfloat16 inference.')} disabled={update.isPending} disabledReason={settings.use_flash_accel ? t('FlashAttention requires BF16. Turn off FlashAttention first.') : undefined} />
            <Switch checked={settings.use_gpt_latent} onCheckedChange={(value) => save({ use_gpt_latent: value })} label={t('GPT latent path')} description={t('Use the IndexTTS GPT latent path with the bundled temporal-alignment fix.')} disabled={update.isPending} />
            <Switch checked={settings.use_torch_compile} onCheckedChange={(value) => save({ use_torch_compile: value })} label={t('Torch compile')} description={t('Compile the runtime. The first generation may take longer.')} disabled={update.isPending} disabledReason={!capabilities?.triton.ok ? capabilities?.triton.detail || t('Triton is unavailable.') : undefined} />
            <Switch checked={settings.use_flash_accel} onCheckedChange={(value) => save({ use_flash_accel: value })} label={t('Flash attention')} description={t('Use the optional FlashAttention acceleration package.')} disabled={update.isPending} disabledReason={!capabilities?.triton.ok ? capabilities?.triton.detail || t('Triton is unavailable.') : !capabilities?.flash_attn.ok ? capabilities?.flash_attn.detail || t('FlashAttention is unavailable.') : undefined} />
          </div>
        </Section>
        <Section layout="aside" title={t('Appearance and language')}><div className={styles.fields}>
          <Field label={t('Interface language')}><Select aria-label={t('Interface language')} value={settings.ui_language} onChange={(value) => save({ ui_language: value })} disabled={update.isPending} options={[{ value: 'zh_CN', label: '简体中文' }, { value: 'en_US', label: 'English' }]} /></Field>
          <Field label={t('Theme')}><SegmentedControl value={settings.theme} onChange={(value) => { if (!update.isPending) save({ theme: value }); }} options={[{ value: 'system', label: t('Follow system') }, { value: 'light', label: t('Light') }, { value: 'dark', label: t('Dark') }]} /></Field>
        </div></Section>
        <Section layout="aside" title={t('Completion feedback')}><div className={styles.fields}>
          <Switch checked={settings.completion_sound} onCheckedChange={(value) => { if (value) void unlockCompletionSound().catch(() => {}); save({ completion_sound: value }); }} label={t('Completion sound')} description={t('Play a short chime when a task finishes successfully.')} disabled={update.isPending} />
          <Button variant="text" size="sm" icon={Volume2} onClick={async () => { try { await unlockCompletionSound(); await playCompletionSound(); } catch (error) { toast.error(errorMessage(error)); } }}>{t('Test sound')}</Button>
          <Switch checked={settings.notifications && permission === 'granted'} onCheckedChange={(value) => void enableNotifications(value)} label={t('Browser notifications')} description={t('Notify you when work finishes while this tab is in the background.')} disabled={update.isPending || permissionPending} disabledReason={!notificationsAvailable() ? t('Notifications require HTTPS or localhost and a supported browser.') : undefined} />
          {settings.notifications && permission !== 'granted' && <InlineAlert tone="warning">{t('Notification permission is not granted in this browser.')}</InlineAlert>}
          {notificationError && <InlineAlert tone="warning">{notificationError}</InlineAlert>}
          <p className={styles.caption}>{t('Sound and notifications are handled by this browser, not by the server.')}</p>
        </div></Section>
        <Section layout="aside" title={t('Local storage cleanup')} description={t('These actions permanently remove local files. Review the confirmation carefully.')}>
          <div className={styles.danger}><div><h3 className={styles.dangerTitle}>{t('Output recordings')}</h3><p className={styles.caption}>{t('Clear saved synthesis and dialogue audio.')}</p></div><Button variant="danger" size="sm" disabledReason={gpu ? t('Wait for the active synthesis to finish.') : undefined} onClick={() => setConfirm('outputs')}>{t('Delete outputs')}</Button></div>
          <div className={styles.danger}><div><h3 className={styles.dangerTitle}>{t('Reference voices')}</h3><p className={styles.caption}>{t('Clear reference audio and transcripts.')}</p></div><Button variant="danger" size="sm" disabledReason={gpu ? t('Wait for the active synthesis to finish.') : undefined} onClick={() => setConfirm('voices')}>{t('Delete all voices')}</Button></div>
        </Section>
    </div>
    <ConfirmDialog open={Boolean(confirm)} onOpenChange={(open) => { if (!open) setConfirm(null); }} title={confirm ? titles[confirm] : ''} description={confirm ? descriptions[confirm] : ''} confirmLabel={confirm === 'unload' ? t('Unload models') : t('Delete permanently')} tone={confirm === 'unload' ? 'default' : 'danger'} requireText={confirm && confirm !== 'unload' ? 'DELETE' : undefined} onConfirm={async () => { if (confirm) await danger.mutateAsync(confirm); setConfirm(null); }} />
  </div>;
}

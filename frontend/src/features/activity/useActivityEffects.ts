import { useEffect, useRef } from 'react';
import { onLiveEvent, useLive, useRunningJobs } from '@/api/live';
import { useSettings } from '@/app/bootstrap';
import { toast } from '@/ui';
import { useT } from '@/i18n';
import { CompletionTracker, notificationsAvailable, playCompletionSound, unlockCompletionSound } from './effects';

export function useActivityEffects(): void {
  const t = useT(); const settings = useSettings(); const jobs = useRunningJobs();
  const training = useLive((state) => state.training); const exporting = useLive((state) => state.export);
  const tracker = useRef(new CompletionTracker()); const preferences = useRef(settings); preferences.current = settings;
  useEffect(() => onLiveEvent((type, data) => {
    const completion = tracker.current.accept(type, data);
    if (!completion) return;
    if (completion.failed) toast.error(completion.title || t('Task failed'), { description: completion.message });
    else if (completion.cancelled) toast.info(t('Task cancelled'), { description: completion.message });
    else toast.success(completion.title || t('Task completed'), { description: completion.message });
    if (!completion.failed && !completion.cancelled && preferences.current?.completion_sound) void playCompletionSound();
    if (document.visibilityState === 'hidden' && preferences.current?.notifications && notificationsAvailable() && Notification.permission === 'granted') {
      try { const notification = new Notification(completion.title || t('IndexTTS Easy GUI'), { body: completion.message, tag: 'indextts-completion' }); notification.onclick = () => { window.focus(); notification.close(); }; }
      catch { /* Browser/OS can deny delivery even after granting permission. */ }
    }
  }), [t]);
  useEffect(() => {
    if (!settings?.completion_sound) return;
    const unlock = () => { void unlockCompletionSound().catch(() => {}); };
    window.addEventListener('pointerdown', unlock, { once: true }); window.addEventListener('keydown', unlock, { once: true });
    return () => { window.removeEventListener('pointerdown', unlock); window.removeEventListener('keydown', unlock); };
  }, [settings?.completion_sound]);
  useEffect(() => {
    const job = jobs[0];
    const progress = job?.progress ?? (training?.running && training.total > 0 ? training.step / training.total : null);
    const title = job?.title ?? (training?.running ? t('Training') : exporting?.status === 'running' || exporting?.status === 'cancelling' ? t('Export') : null);
    document.title = title ? `${progress === null ? '●' : `${Math.round(progress * 100)}%`} ${title} — IndexTTS` : t('IndexTTS Easy GUI');
    return () => { document.title = t('IndexTTS Easy GUI'); };
  }, [jobs, training?.running, training?.step, training?.total, exporting?.status, t]);
}

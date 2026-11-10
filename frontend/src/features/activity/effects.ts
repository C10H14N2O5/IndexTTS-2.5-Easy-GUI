import type { ExportSnapshot, Job, StatusItem, TrainingSnapshot } from '@/api/live';

export interface Completion { title: string; message: string; failed: boolean; cancelled?: boolean }
/** Track transitions, not snapshots: reconnecting never replays an old completion. */
export class CompletionTracker {
  private jobs = new Map<string, Job['state']>();
  private training: TrainingSnapshot | null = null;
  private exporting: ExportSnapshot | null = null;
  private seenStatuses = new Set<number>();
  private lastStatusAt = -Infinity;
  private lastStatus: StatusItem | null = null;
  constructor(private now: () => number = Date.now) {}
  accept(type: string, data: unknown): Completion | null {
    if (type === 'snapshot') {
      const snapshot = data as { jobs?: Job[]; status?: StatusItem[]; training?: TrainingSnapshot; export?: ExportSnapshot };
      for (const job of snapshot.jobs ?? []) this.jobs.set(job.id, job.state);
      for (const status of snapshot.status ?? []) this.seenStatuses.add(status.id);
      this.training = snapshot.training ?? null; this.exporting = snapshot.export ?? null;
      return null;
    }
    if (type === 'status') {
      const status = data as StatusItem;
      if (this.seenStatuses.has(status.id)) return null;
      this.seenStatuses.add(status.id);
      if (this.seenStatuses.size > 500) this.seenStatuses.delete(this.seenStatuses.values().next().value!);
      if (status.level !== 'success' && status.level !== 'error') return null;
      this.lastStatusAt = this.now(); this.lastStatus = status;
      return { title: status.phase, message: status.message, failed: status.level === 'error' };
    }
    if (type === 'job') {
      const job = data as Job; const previous = this.jobs.get(job.id); this.jobs.set(job.id, job.state);
      if (this.jobs.size > 200) this.jobs.delete(this.jobs.keys().next().value!);
      if (job.state === 'running' || previous === job.state || (previous === undefined && job.finished_at && this.now() / 1000 - job.finished_at > 10)) return null;
      if (this.lastStatus?.phase === job.title && this.now() - this.lastStatusAt < 1200) return null; // Services often publish status just before the terminal job.
      return { title: job.title, message: job.error?.message ?? job.stage ?? job.title, failed: job.state === 'failed', cancelled: job.state === 'cancelled' };
    }
    if (type === 'training') {
      const training = data as TrainingSnapshot; const previous = this.training; this.training = training;
      if (!previous?.running || training.running || this.now() - this.lastStatusAt < 1200) return null;
      return { title: training.adapter || training.project, message: training.message, failed: training.failed, cancelled: training.outcome === 'stopped' || training.stopped === true };
    }
    if (type === 'export') {
      const exporting = data as ExportSnapshot; const previous = this.exporting; this.exporting = exporting;
      if (!previous || !['running', 'cancelling'].includes(previous.status) || !['succeeded', 'failed', 'cancelled'].includes(exporting.status) || this.now() - this.lastStatusAt < 1200) return null;
      return { title: exporting.phase, message: exporting.error || exporting.output || exporting.phase, failed: exporting.status === 'failed', cancelled: exporting.status === 'cancelled' };
    }
    return null;
  }
}
let context: AudioContext | null = null;
export async function unlockCompletionSound(): Promise<void> {
  if (typeof AudioContext === 'undefined') return;
  context ??= new AudioContext();
  if (context.state === 'suspended') await context.resume();
}
export async function playCompletionSound(): Promise<void> {
  if (!context || context.state !== 'running') return;
  for (const [index, frequency] of [660, 880].entries()) {
    const oscillator = context.createOscillator(); const gain = context.createGain();
    const start = context.currentTime + index * 0.15;
    oscillator.type = 'sine'; oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0, start); gain.gain.linearRampToValueAtTime(0.07, start + 0.015); gain.gain.exponentialRampToValueAtTime(0.001, start + 0.22);
    oscillator.connect(gain); gain.connect(context.destination); oscillator.start(start); oscillator.stop(start + 0.23);
    oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
  }
}
export function notificationsAvailable(): boolean { return Boolean(window.isSecureContext && 'Notification' in window); }

import { describe, expect, it } from 'vitest';
import type { Job, StatusItem, TrainingSnapshot, ExportSnapshot } from '@/api/live';
import { CompletionTracker } from './effects';
const job = (state: Job['state']): Job => ({ id: 'one', kind: 'synthesis', title: 'Speech', state, started_at: 1, finished_at: state === 'running' ? null : 100, progress: 1, stage: 'Done', result: null, error: null, cancellable: true, cancel_requested: false, gpu: true, meta: {} });
const status: StatusItem = { id: 1, time: 100, phase: 'Speech', message: 'Complete', level: 'success' };
const training: TrainingSnapshot = { running: true, failed: false, step: 1, total: 10, epoch: 1, loss: 1, elapsed: 1, eta: 9, message: 'Train', output: '', started_at: 1, finished_at: null, output_dir: '', project: 'P', adapter: 'A' };
const exporting: ExportSnapshot = { status: 'running', phase: 'merge', elapsed: 1, log: '', output: '', error: '', report: {}, token: '', task_id: '1', logfile: '' };
describe('completion effects', () => {
  it('does not replay completed jobs or status history on a reconnect snapshot', () => {
    const tracker = new CompletionTracker(() => 100000);
    expect(tracker.accept('snapshot', { jobs: [job('succeeded')], status: [status] })).toBeNull();
    expect(tracker.accept('job', job('succeeded'))).toBeNull(); expect(tracker.accept('status', status)).toBeNull();
  });
  it('emits a terminal job once', () => {
    const tracker = new CompletionTracker(() => 100000);
    expect(tracker.accept('job', job('running'))).toBeNull();
    expect(tracker.accept('job', job('succeeded'))).toMatchObject({ title: 'Speech', failed: false });
    expect(tracker.accept('job', job('succeeded'))).toBeNull();
  });
  it('does not toast old jobs first loaded via fallback polling', () => {
    expect(new CompletionTracker(() => 200000).accept('job', job('succeeded'))).toBeNull();
  });
  it('does not duplicate a terminal workflow status and job event', () => {
    const tracker = new CompletionTracker(() => 100000);
    expect(tracker.accept('status', status)).not.toBeNull(); expect(tracker.accept('job', job('succeeded'))).toBeNull();
  });
  it('does not swallow an unrelated job failure after another workflow completed', () => {
    const tracker = new CompletionTracker(() => 100000);
    tracker.accept('status', { ...status, phase: 'Library save' });
    expect(tracker.accept('job', { ...job('failed'), error: { message: 'Synthesis failed' } })).toMatchObject({ failed: true, message: 'Synthesis failed' });
  });
  it('never celebrates cancellations', () => {
    const tracker = new CompletionTracker(() => 100000);
    tracker.accept('job', job('running'));
    expect(tracker.accept('job', job('cancelled'))).toMatchObject({ cancelled: true });
  });
  it('surfaces the original failure text', () => {
    const tracker = new CompletionTracker(() => 100000);
    expect(tracker.accept('job', { ...job('failed'), error: { message: 'VRAM exhausted' } })).toMatchObject({ failed: true, message: 'VRAM exhausted' });
  });
  it('tracks training and export running-to-terminal transitions', () => {
    const tracker = new CompletionTracker(() => 100000);
    expect(tracker.accept('snapshot', { training, export: exporting })).toBeNull();
    expect(tracker.accept('training', { ...training, running: false, message: 'Training done' })).toMatchObject({ title: 'A', message: 'Training done' });
    expect(tracker.accept('export', { ...exporting, status: 'failed', error: 'Not enough disk space' })).toMatchObject({ failed: true, message: 'Not enough disk space' });
  });
  it('ignores working and informational statuses', () => {
    const tracker = new CompletionTracker(() => 100000);
    expect(tracker.accept('status', { ...status, level: 'working' })).toBeNull(); expect(tracker.accept('status', { ...status, id: 2, level: 'info' })).toBeNull();
  });
});

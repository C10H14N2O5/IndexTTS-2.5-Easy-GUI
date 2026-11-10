import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { applyEvent, connectLive, onLiveEvent, selectGpuJob, selectJobs, startLive, useLive, type Job } from './live';
import { setLanguage } from '@/i18n';

const initial = useLive.getState();
function job(id: string, state: Job['state'] = 'running', gpu = false): Job {
  return { id, state, gpu, kind: 'synthesis', title: 'Synthesis', progress: null, stage: null,
    started_at: Number(id) || 1, finished_at: state === 'running' ? null : 2, result: null,
    error: null, cancellable: state === 'running', cancel_requested: false, meta: {} };
}
class FakeSource {
  static instances: FakeSource[] = [];
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((message: { data: string }) => void) | null = null;
  close = vi.fn();
  constructor(public url: string) { FakeSource.instances.push(this); }
  emit(type: string, data: unknown) { this.onmessage?.({ data: JSON.stringify({ type, data }) }); }
}
beforeEach(() => { useLive.setState(initial, true); FakeSource.instances = []; vi.stubGlobal('EventSource', FakeSource); });
afterEach(() => vi.unstubAllGlobals());

describe('live snapshots and events', () => {
  it('replaces running state on a reconnect snapshot and increments generation', () => {
    applyEvent('snapshot', { jobs: [job('old')], console: { lines: [], current: '' } });
    applyEvent('snapshot', { jobs: [job('new', 'succeeded')] });
    expect(useLive.getState().generation).toBe(2);
    expect(selectJobs(useLive.getState()).map((j) => j.id)).toEqual(['new']);
    expect(useLive.getState().ready).toBe(true);
  });
  it('updates progress in place and bounds finished jobs without evicting running jobs', () => {
    applyEvent('job', job('active', 'running', true));
    for (let i = 0; i < 80; i++) applyEvent('job', job(String(i), 'succeeded'));
    applyEvent('job', { ...job('active', 'running', true), progress: .5 });
    expect(selectJobs(useLive.getState())).toHaveLength(51);
    expect(selectGpuJob(useLive.getState())?.progress).toBe(.5);
    expect(useLive.getState().jobs['0']).toBeUndefined();
  });
  it('replaces an updated console line by id and honors a clear reset', () => {
    applyEvent('console', { lines: [{ id: 1, text: '10%', level: 'progress' }], current: 'a' });
    applyEvent('console', { lines: [{ id: 1, text: '100%', level: 'progress' }], current: '' });
    expect(useLive.getState().console).toEqual([{ id: 1, text: '100%', level: 'progress' }]);
    applyEvent('console', { reset: true, lines: [], current: '' });
    expect(useLive.getState().console).toEqual([]);
  });
  it('caps console and status histories', () => {
    applyEvent('console', { lines: Array.from({ length: 1500 }, (_, id) => ({ id, text: '', level: 'plain' })), current: '' });
    for (let id = 0; id < 70; id++) applyEvent('status', { id });
    expect(useLive.getState().console).toHaveLength(1400);
    expect(useLive.getState().status).toHaveLength(60);
  });
  it('notifies raw subscribers and removes them on cleanup', () => {
    const listener = vi.fn(); const remove = onLiveEvent(listener);
    applyEvent('heartbeat', { time: 1 }); remove(); applyEvent('heartbeat', { time: 2 });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(useLive.getState().extra).toEqual({});
  });
  it('ignores late events from replaced language connections', () => {
    const firstClose = connectLive(); const first = FakeSource.instances[0]!;
    const secondClose = connectLive(); const second = FakeSource.instances[1]!;
    expect(first.close).toHaveBeenCalled();
    first.emit('job', job('stale')); first.onerror?.(); first.onopen?.();
    expect(useLive.getState().jobs).toEqual({});
    second.emit('snapshot', { jobs: [job('current')] });
    expect(useLive.getState().connection).toBe('open');
    second.onerror?.(); expect(useLive.getState().connection).toBe('reconnecting');
    firstClose(); secondClose();
  });
  it('reopens SSE on language switch and unsubscribes when stopped', () => {
    setLanguage('zh_CN'); const stop = startLive();
    setLanguage('en_US');
    expect(FakeSource.instances.at(-1)?.url).toBe('/api/events?lang=en_US');
    expect(FakeSource.instances[0]!.close).toHaveBeenCalled();
    stop(); const count = FakeSource.instances.length; setLanguage('zh_CN');
    expect(FakeSource.instances).toHaveLength(count);
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '@/api/client';
import { setLanguage } from '@/i18n';
import { ProofreadController, type UnsavedChoice } from './controller';
import { filterClips, sameContext, samePath, type AsrStatus, type Clip, type ManifestStatus, type ProofreadContext, type ProofreadTransport, type SessionSnapshot } from './api';

const context: ProofreadContext = { directory: 'D:/音频', project: 'voice', language: 'ZH', role: 'training' };
const idleAsr = (): AsrStatus => ({ available: true, job: { running: false, total: 0, completed: 0, succeeded: 0, skipped: 0, failed: 0 } });
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(fn => { resolve = fn; }); return { promise, resolve }; };
function setup() {
  let selected = { ...context }, asr = idleAsr();
  let manifest: ManifestStatus = { stale: false, status: 'ready', manifest: 'manifest.jsonl', available: true };
  let clips: Clip[] = [
    { id: '1', name: 'sub/中文.wav', text: 'Original words', version: 'v1', reviewed: false },
    { id: '2', name: 'clip2.wav', text: 'Second clip', version: 'v1', reviewed: false },
    { id: '3', name: 'clip3.wav', text: '', version: 'missing', reviewed: false },
  ];
  let lastId = '1', choice: UnsavedChoice = 'cancel';
  const snapshot = (): SessionSnapshot => ({ ...selected, token: 'memory-token', items: clips.map(({ id, name, text, reviewed }) => ({ id, name, reviewed, missing_text: !text.trim() })), last_id: lastId, asr: structuredClone(asr) });
  const calls: { operation: string; body: Record<string, unknown> }[] = [];
  const transport: ProofreadTransport = {
    get: async <T,>(operation: string, query?: Record<string, string>): Promise<T> => {
      if (operation === 'session') return snapshot() as T;
      if (operation === 'item') return { ...clips.find(clip => clip.id === query?.id)! } as T;
      if (operation === 'asr/status') return structuredClone(asr) as T;
      if (operation === 'manifest/status') return { ...manifest, role: selected.role } as T;
      throw new Error(`Unexpected read ${operation}`);
    },
    post: async <T,>(operation: string, input: unknown = {}): Promise<T> => {
      const body = input as Record<string, unknown>; calls.push({ operation, body });
      if (operation === 'session') return snapshot() as T;
      if (operation === 'select') { selected = body as unknown as ProofreadContext; return snapshot() as T; }
      if (operation === 'position') { lastId = String(body.id); return {} as T; }
      if (operation === 'save') { const current = clips.find(clip => clip.id === body.id)!; Object.assign(current, { text: body.text, reviewed: body.reviewed, version: 'v2' }); manifest.stale = true; return { ...current } as T; }
      if (operation === 'review') { const current = clips.find(clip => clip.id === body.id)!; current.reviewed = body.reviewed === true; return { ...current } as T; }
      if (operation === 'asr/item') return { id: body.id, version: body.version, text: 'Recognized draft' } as T;
      if (operation === 'asr/start') { asr.job = { ...asr.job, running: true, total: clips.length }; return structuredClone(asr) as T; }
      if (operation === 'asr/stop') { asr.job.running = false; asr.job.cancelled = true; return structuredClone(asr) as T; }
      if (operation === 'manifest/update') { manifest = { ...manifest, role: selected.role, stale: false, status: 'ready', manifest: selected.role === 'validation' ? 'validation.jsonl' : 'manifest.jsonl', stats: { samples: 3, valid: 2, missing_text: 1 } }; return { status: { ...manifest } } as T; }
      throw new Error(`Unexpected write ${operation}`);
    },
    setToken: vi.fn(),
    audioUrl: id => `/proofread/api/audio?id=${id}&token=memory-token`,
  };
  const ask = vi.fn(async () => choice);
  const controller = new ProofreadController(transport, ask);
  const updated = vi.fn(); controller.setOnUpdated(updated);
  return { controller, transport, ask, updated, calls, snapshot, clips: () => clips, setClips: (value: Clip[]) => { clips = value; }, setChoice: (value: UnsavedChoice) => { choice = value; }, setLast: (id: string) => { lastId = id; }, setAsr: (value: AsrStatus) => { asr = value; }, setManifest: (value: ManifestStatus) => { manifest = value; } };
}
type Harness = ReturnType<typeof setup>;
let h: Harness;
beforeEach(async () => { setLanguage('en_US'); h = setup(); await h.controller.start(context); });
afterEach(() => { h.controller.dispose(); vi.restoreAllMocks(); });
const settle = async () => { for (let n = 0; n < 20; n++) await Promise.resolve(); };

describe('session and context isolation', () => {
  it('opens a tokenized session and restores position without autoplay side effects', () => {
    const state = h.controller.getSnapshot(); expect(state.ready).toBe(true); expect(state.current?.id).toBe('1'); expect(state.audioSrc).toContain('token=memory-token'); expect(h.transport.setToken).toHaveBeenCalledWith('memory-token'); expect(h.calls.some(call => call.operation === 'position')).toBe(true);
  });
  it('accepts canonical server paths without repeatedly reselecting relative input', async () => {
    const relative = { ...context, directory: './audio' };
    const base = h.transport.post;
    h.transport.post = async <T,>(op: string, body?: unknown) => {
      const result = await base<T>(op, body);
      return op === 'select' ? { ...result, directory: 'D:/Project/audio' } : result;
    };
    await h.controller.start(relative); expect(h.controller.getSnapshot().ready).toBe(true); expect(h.controller.getSnapshot().context.directory).toBe('D:/Project/audio');
    const selects = h.calls.filter(call => call.operation === 'select').length;
    h.controller.setContext(relative); await settle(); expect(h.calls.filter(call => call.operation === 'select')).toHaveLength(selects);
  });
  it('keeps session tokens out of browser storage', () => { expect(JSON.stringify(localStorage)).not.toContain('memory-token'); expect(JSON.stringify(sessionStorage)).not.toContain('memory-token'); });
  it('rejects a context change on Cancel and clears the pending target', async () => {
    h.controller.setDraft('Unsaved edits'); const calls = h.calls.length; h.controller.setContext({ ...context, role: 'validation', project: 'another' }); await settle(); expect(h.controller.getSnapshot().context).toEqual(context); expect(h.controller.getSnapshot().draft).toBe('Unsaved edits'); expect(h.controller.getSnapshot().retainedContext).toBe(true); expect(h.calls.slice(calls).filter(call => call.operation === 'select')).toHaveLength(0); await h.controller.refresh(); expect(h.ask).toHaveBeenCalledTimes(1);
  });
  it('saves before accepting a role-only context switch', async () => {
    h.controller.setDraft('New transcript'); h.setChoice('save'); h.controller.setContext({ ...context, role: 'validation' }); await settle(); expect(h.controller.getSnapshot().context.role).toBe('validation'); const operations = h.calls.map(call => call.operation); expect(operations.lastIndexOf('save')).toBeLessThan(operations.lastIndexOf('select')); expect(h.clips()[0]?.text).toBe('New transcript');
  });
  it('discards before switching and does not write discarded text', async () => {
    h.controller.setDraft('Discard this'); h.setChoice('discard'); h.controller.setContext({ ...context, role: 'validation' }); await settle(); expect(h.controller.getSnapshot().draft).toBe('Original words'); expect(h.calls.filter(call => call.operation === 'save')).toHaveLength(0);
  });
  it('blocks folder and role changes while a batch runs', async () => {
    await h.controller.startBatch(); const count = h.calls.filter(call => call.operation === 'select').length; h.controller.setContext({ ...context, directory: 'D:/other', role: 'validation' }); await settle(); expect(h.calls.filter(call => call.operation === 'select')).toHaveLength(count); expect(h.controller.getSnapshot().context).toEqual(context); expect(h.controller.getSnapshot().error).toContain('Stop transcription');
  });
  it('ignores a stale session response from the old role', async () => {
    const old = h.snapshot(); const waiting = deferred<SessionSnapshot>(); const base = h.transport.get; let intercepted = false; h.transport.get = async <T,>(op: string, query?: Record<string, string>) => { if (op === 'session' && !intercepted) { intercepted = true; return waiting.promise as Promise<T>; } return base<T>(op, query); };
    const poll = h.controller.poll(); await settle(); h.controller.setContext({ ...context, role: 'validation' }); await settle(); waiting.resolve(old); await poll; expect(h.controller.getSnapshot().context.role).toBe('validation'); expect(h.controller.getSnapshot().manifest?.role).toBe('validation');
  });
  it('does not apply responses or notify after unmount', async () => {
    const waiting = deferred<Clip>(); const original = h.transport.post; h.transport.post = async <T,>(op: string, body?: unknown) => op === 'save' ? waiting.promise as Promise<T> : original<T>(op, body); h.controller.setDraft('Save in flight'); const saving = h.controller.save(); const before = h.controller.getSnapshot(); h.controller.dispose(); waiting.resolve({ ...h.clips()[0]!, text: 'Saved', version: 'v2' }); await saving; expect(h.controller.getSnapshot()).toBe(before); expect(h.updated).not.toHaveBeenCalled();
  });
  it('keeps project, folder, language and role intact when server select fails', async () => {
    const base = h.transport.post; h.transport.post = async <T,>(op: string, body?: unknown) => { if (op === 'select') throw new ApiError(409, 'conflict', 'Folder busy'); return base<T>(op, body); }; h.controller.setContext({ project: 'new', directory: 'D:/new', role: 'validation', language: 'EN' }); await settle(); expect(h.controller.getSnapshot().context).toEqual(context); expect(h.controller.getSnapshot().error).toBe('Folder busy'); expect(h.controller.getSnapshot().draft).toBe('Original words');
  });
});

describe('editing and optimistic concurrency', () => {
  it('preserves dirty text on background polling', async () => { h.controller.setDraft('Manual edits'); await h.controller.poll(); expect(h.controller.getSnapshot().draft).toBe('Manual edits'); expect(h.controller.dirty).toBe(true); });
  it('does not overwrite a draft typed while a disk reload request is in flight', async () => {
    const waiting = deferred<Clip>(); const base = h.transport.get; h.transport.get = async <T,>(op: string, query?: Record<string, string>) => op === 'item' ? waiting.promise as Promise<T> : base<T>(op, query); const poll = h.controller.poll(); await settle(); h.controller.setDraft('Typed during polling'); waiting.resolve({ ...h.clips()[0]!, text: 'Disk text' }); await poll; expect(h.controller.getSnapshot().draft).toBe('Typed during polling');
  });
  it('preserves the current clip if background snapshots have another saved position', async () => { h.setLast('2'); await h.controller.poll(); expect(h.controller.getSnapshot().current?.id).toBe('1'); });
  it('refreshes a clean editor when its disk version changes', async () => { h.clips()[0]!.text = 'Updated externally'; h.clips()[0]!.version = 'external'; await h.controller.poll(); expect(h.controller.getSnapshot().draft).toBe('Updated externally'); expect(h.controller.getSnapshot().current?.version).toBe('external'); });
  it('sends the original version and preserves edits on 409', async () => {
    h.controller.setDraft('Manual edits'); const post = vi.spyOn(h.transport, 'post').mockRejectedValueOnce(new ApiError(409, 'conflict', 'TXT changed on disk')); expect(await h.controller.save()).toBe(false); expect(post).toHaveBeenCalledWith('save', expect.objectContaining({ version: 'v1', text: 'Manual edits' })); expect(h.controller.getSnapshot().draft).toBe('Manual edits'); expect(h.controller.getSnapshot().errorStatus).toBe(409); expect(h.clips()[0]?.text).toBe('Original words');
  });
  it('preserves edits and surfaces 423 locks', async () => { h.controller.setDraft('Keep this'); vi.spyOn(h.transport, 'post').mockRejectedValueOnce(new ApiError(423, 'locked', 'Dataset locked')); await h.controller.save(); expect(h.controller.getSnapshot().draft).toBe('Keep this'); expect(h.controller.getSnapshot().errorStatus).toBe(423); });
  it('rejects blank saves locally without API writes', async () => { h.controller.setDraft('   '); const count = h.calls.length; expect(await h.controller.save()).toBe(false); expect(h.calls).toHaveLength(count); expect(h.controller.getSnapshot().error).toContain('empty transcript'); });
  it('marks the manifest stale after a successful save and emits an update', async () => { h.controller.setDraft('Saved edits'); expect(await h.controller.save()).toBe(true); expect(h.controller.dirty).toBe(false); expect(h.controller.getSnapshot().manifest?.stale).toBe(true); expect(h.updated).toHaveBeenCalledOnce(); });
  it('preserves reviewed status when saving unchanged text', async () => { await h.controller.review(); await h.controller.save(); expect(h.controller.getSnapshot().current?.reviewed).toBe(true); });
  it('unmarks previously reviewed text when it is edited and saved', async () => { await h.controller.review(); h.controller.setDraft('Changed content'); await h.controller.save(); expect(h.controller.getSnapshot().current?.reviewed).toBe(false); });
  it('saves edited text before toggling reviewed status', async () => { h.controller.setDraft('Reviewed edits'); await h.controller.review(); expect(h.clips()[0]?.text).toBe('Reviewed edits'); expect(h.controller.getSnapshot().current?.reviewed).toBe(true); const ops = h.calls.map(call => call.operation); expect(ops.lastIndexOf('save')).toBeLessThan(ops.lastIndexOf('review')); });
  it('uses the pre-save filtered ordering for Save and next', async () => { h.controller.setFilter('pending'); await h.controller.saveNext(); expect(h.controller.getSnapshot().current?.id).toBe('2'); expect(h.clips()[0]?.reviewed).toBe(true); });
  it('does not wrap past the end of the filtered list', async () => { h.controller.setQuery('中文'); await h.controller.saveNext(); expect(h.controller.getSnapshot().current?.id).toBe('1'); expect(h.controller.getSnapshot().notice).toContain('End of this list'); });
  it('cancels navigation without losing the active clip or draft', async () => { h.controller.setDraft('Stay here'); expect(await h.controller.select('2')).toBe(false); expect(h.controller.getSnapshot().current?.id).toBe('1'); expect(h.controller.getSnapshot().draft).toBe('Stay here'); });
});

describe('speech recognition and manifest parity', () => {
  it('places single-clip ASR only into a dirty draft, never writes TXT', async () => { const saved = h.calls.filter(call => call.operation === 'save').length; await h.controller.transcribe(); expect(h.controller.getSnapshot().draft).toBe('Recognized draft'); expect(h.controller.dirty).toBe(true); expect(h.clips()[0]?.text).toBe('Original words'); expect(h.calls.filter(call => call.operation === 'save')).toHaveLength(saved); });
  it('reloads explicitly discarded text before starting a batch', async () => { h.controller.setDraft('Discarded draft'); h.setChoice('discard'); await h.controller.startBatch(); expect(h.controller.getSnapshot().draft).toBe('Original words'); expect(h.controller.getSnapshot().asr.job.running).toBe(true); });
  it('batch requests do not use list search or filter parameters', async () => { h.controller.setQuery('中文'); h.controller.setFilter('missing'); await h.controller.startBatch(); expect(h.calls.find(call => call.operation === 'asr/start')?.body).toEqual({}); });
  it('stops a batch without reverting completed transcripts', async () => { await h.controller.startBatch(); h.clips()[2]!.text = 'Completed batch text'; await h.controller.stopBatch(); expect(h.controller.getSnapshot().asr.job.cancelled).toBe(true); expect(h.clips()[2]?.text).toBe('Completed batch text'); });
  it('blocks single recognition when a batch owns the session', async () => { await h.controller.startBatch(); const count = h.calls.length; expect(await h.controller.transcribe()).toBe(false); expect(h.calls).toHaveLength(count); });
  it('blocks manifest updates while training owns the corpus', async () => { h.setManifest({ stale: true, available: true, training_running: true }); await h.controller.refresh(); const count = h.calls.length; expect(await h.controller.updateManifest()).toBe(false); expect(h.calls).toHaveLength(count); });
  it('uses only the current validation role when building its manifest', async () => { h.controller.setContext({ ...context, role: 'validation' }); await settle(); await h.controller.updateManifest(); expect(h.calls.filter(call => call.operation === 'manifest/update').at(-1)?.body).toEqual({ role: 'validation' }); expect(h.controller.getSnapshot().manifest?.manifest).toBe('validation.jsonl'); expect(h.controller.getSnapshot().manifest?.stats).toEqual({ samples: 3, valid: 2, missing_text: 1 }); });
  it('ignores wrong-role manifest updates and does not emit a misleading callback', async () => { const base = h.transport.post; h.transport.post = async <T,>(op: string, body?: unknown) => op === 'manifest/update' ? { status: { stale: false, role: 'validation', manifest: 'wrong.jsonl' } } as T : base<T>(op, body); expect(await h.controller.updateManifest()).toBe(false); expect(h.controller.getSnapshot().manifest?.manifest).toBe('manifest.jsonl'); expect(h.updated).not.toHaveBeenCalled(); });
  it('ignores wrong-role status results', async () => { const base = h.transport.get; h.transport.get = async <T,>(op: string, query?: Record<string, string>) => op === 'manifest/status' ? { stale: false, role: 'validation', manifest: 'wrong.jsonl' } as T : base<T>(op, query); await h.controller.refresh(); expect(h.controller.getSnapshot().manifest?.manifest).toBe('manifest.jsonl'); });
  it('drops discarded edits before rebuilding the saved-file manifest', async () => { h.controller.setDraft('Not saved'); h.setChoice('discard'); await h.controller.updateManifest(); expect(h.controller.getSnapshot().draft).toBe('Original words'); expect(h.controller.dirty).toBe(false); expect(h.calls.filter(call => call.operation === 'save')).toHaveLength(0); });
  it('prevents navigation from orphaning a running ASR batch', async () => { await h.controller.startBatch(); expect(await h.controller.allowNavigation()).toBe(false); expect(h.controller.getSnapshot().error).toContain('Stop transcription before leaving'); });
  it('detects session expiry without clearing an unsaved transcript', async () => { h.controller.setDraft('Do not lose this'); vi.spyOn(h.transport, 'get').mockRejectedValueOnce(new ApiError(403, 'expired', 'Expired')); await h.controller.poll(); expect(h.controller.getSnapshot().draft).toBe('Do not lose this'); expect(h.controller.getSnapshot().error).toContain('Copy any unsaved transcript'); });
});

describe('parent binding mutation preflight', () => {
  it('never persists a new binding when the user cancels', async () => {
    h.controller.setDraft('Keep this draft'); const commit = vi.fn(async () => {});
    expect(await h.controller.prepareContextChange(commit)).toBe(false);
    expect(commit).not.toHaveBeenCalled(); expect(h.controller.getSnapshot().context).toEqual(context); expect(h.controller.getSnapshot().draft).toBe('Keep this draft');
  });
  it('never persists a new binding when saving the old transcript fails', async () => {
    h.controller.setDraft('Keep this draft'); h.setChoice('save');
    vi.spyOn(h.transport, 'post').mockRejectedValueOnce(new ApiError(409, 'conflict', 'TXT changed on disk'));
    const commit = vi.fn(async () => {}); expect(await h.controller.prepareContextChange(commit)).toBe(false);
    expect(commit).not.toHaveBeenCalled(); expect(h.controller.getSnapshot().draft).toBe('Keep this draft'); expect(h.controller.getSnapshot().context).toEqual(context);
  });
  it('saves to the old binding before invoking the parent PUT and switches without another prompt', async () => {
    h.controller.setDraft('Saved before rebinding'); h.setChoice('save');
    const next = { ...context, directory: 'D:/next' };
    const commit = vi.fn(async () => { expect(h.clips()[0]?.text).toBe('Saved before rebinding'); expect(h.controller.getSnapshot().context).toEqual(context); expect(h.controller.getSnapshot().busy).toBe('parent-context'); h.controller.setContext(next); });
    expect(await h.controller.prepareContextChange(commit)).toBe(true); await settle();
    expect(commit).toHaveBeenCalledOnce(); expect(h.ask).toHaveBeenCalledOnce(); expect(h.controller.getSnapshot().context).toEqual(next); expect(h.controller.dirty).toBe(false);
  });
  it('clears an explicitly discarded draft exactly once before the successful binding change', async () => {
    h.controller.setDraft('Discarded draft'); h.setChoice('discard');
    const next = { ...context, role: 'validation' as const };
    expect(await h.controller.prepareContextChange(async () => { expect(h.controller.dirty).toBe(false); expect(h.controller.getSnapshot().draft).toBe('Original words'); h.controller.setContext(next); })).toBe(true); await settle();
    expect(h.ask).toHaveBeenCalledOnce(); expect(h.calls.filter(call => call.operation === 'save')).toHaveLength(0); expect(h.controller.getSnapshot().context.role).toBe('validation');
  });
  it('restores a discarded draft when the parent PUT fails so saving and retrying remain possible', async () => {
    h.controller.setDraft('Recover this draft'); h.setChoice('discard');
    expect(await h.controller.prepareContextChange(async () => { throw new Error('Binding update failed'); })).toBe(false);
    expect(h.controller.getSnapshot().context).toEqual(context); expect(h.controller.getSnapshot().draft).toBe('Recover this draft'); expect(h.controller.dirty).toBe(true); expect(h.controller.getSnapshot().busy).toBeNull();
    h.setChoice('save'); const retry = vi.fn(async () => {}); expect(await h.controller.prepareContextChange(retry)).toBe(true); expect(retry).toHaveBeenCalledOnce(); expect(h.clips()[0]?.text).toBe('Recover this draft');
  });
  it('keeps the saved old transcript available if the subsequent parent PUT fails', async () => {
    h.controller.setDraft('Saved safely'); h.setChoice('save');
    expect(await h.controller.prepareContextChange(async () => { throw new Error('Binding update failed'); })).toBe(false);
    expect(h.controller.getSnapshot().context).toEqual(context); expect(h.controller.getSnapshot().draft).toBe('Saved safely'); expect(h.clips()[0]?.text).toBe('Saved safely'); expect(h.controller.dirty).toBe(false);
  });
  it('locks the editor throughout the decision and the pending parent request', async () => {
    h.controller.setDraft('Old edits'); const decision = deferred<UnsavedChoice>(); h.ask.mockImplementationOnce(() => decision.promise);
    const waiting = deferred<void>(); const commit = vi.fn(() => waiting.promise); const guarded = h.controller.prepareContextChange(commit);
    expect(h.controller.contextLocked).toBe(true); expect(commit).not.toHaveBeenCalled(); h.controller.setDraft('Late keystroke'); expect(h.controller.getSnapshot().draft).toBe('Old edits');
    decision.resolve('save'); await settle(); expect(commit).toHaveBeenCalledOnce(); expect(h.controller.getSnapshot().busy).toBe('parent-context');
    const competing = vi.fn(async () => {}); expect(await h.controller.prepareContextChange(competing)).toBe(false); expect(competing).not.toHaveBeenCalled(); expect(await h.controller.select('2')).toBe(false);
    waiting.resolve(); expect(await guarded).toBe(true); expect(h.controller.contextLocked).toBe(false);
  });
  it('refuses to change bindings while transcription is running', async () => {
    await h.controller.startBatch(); const commit = vi.fn(async () => {}); expect(await h.controller.prepareContextChange(commit)).toBe(false); expect(commit).not.toHaveBeenCalled(); expect(h.controller.contextLocked).toBe(true);
  });
  it('does not invoke a deferred binding mutation after the panel unmounts', async () => {
    h.controller.setDraft('Unsaved'); const choice = deferred<UnsavedChoice>(); h.ask.mockImplementationOnce(() => choice.promise);
    const commit = vi.fn(async () => {}); const guarded = h.controller.prepareContextChange(commit); h.controller.dispose(); choice.resolve('save'); expect(await guarded).toBe(false); expect(commit).not.toHaveBeenCalled();
  });
});

describe('matching and filters', () => {
  it('normalizes Windows paths while distinguishing dataset role and language', () => { expect(samePath('D:\\Audio\\', 'd:/audio')).toBe(true); expect(sameContext(context, { ...context, role: 'validation' })).toBe(false); expect(sameContext(context, { ...context, language: 'EN' })).toBe(false); });
  it('combines Unicode search and missing/reviewed filters', () => { const items = h.snapshot().items; expect(filterClips(items, ' 中文 ', 'pending').map(item => item.id)).toEqual(['1']); expect(filterClips(items, '', 'missing').map(item => item.id)).toEqual(['3']); expect(filterClips(items, '', 'reviewed')).toHaveLength(0); });
});

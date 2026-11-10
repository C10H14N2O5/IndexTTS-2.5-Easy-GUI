import { t } from '@/i18n';
import { filterClips, proofreadError, sameContext, type AsrStatus, type Clip, type ClipFilter, type ClipSummary, type ManifestResult, type ManifestStatus, type ProofreadContext, type ProofreadTransport, type SessionSnapshot, type Transcription } from './api';

export type UnsavedChoice = 'save' | 'discard' | 'cancel';
export interface ProofreadState {
  ready: boolean;
  context: ProofreadContext;
  items: ClipSummary[];
  current: Clip | null;
  draft: string;
  audioSrc: string;
  filter: ClipFilter;
  query: string;
  asr: AsrStatus;
  manifest: ManifestStatus | null;
  busy: string | null;
  notice: string;
  error: string;
  errorStatus: number;
  retainedContext: boolean;
}
const blankContext = (): ProofreadContext => ({ project: '', directory: '', language: 'EN', role: 'training' });
const emptyAsr = (): AsrStatus => ({ available: false, job: { running: false, total: 0, completed: 0, succeeded: 0, skipped: 0, failed: 0 } });
function initialState(): ProofreadState { return { ready: false, context: blankContext(), items: [], current: null, draft: '', audioSrc: '', filter: 'all', query: '', asr: emptyAsr(), manifest: null, busy: null, notice: '', error: '', errorStatus: 0, retainedContext: false }; }

/**
 * One memory-only server session per mounted panel. Every asynchronous response
 * is guarded by the lifecycle + context revision that requested it. Polling may
 * update metadata, but never replaces a dirty editor or changes the active clip.
 */
export class ProofreadController {
  private state: ProofreadState = initialState();
  private listeners = new Set<() => void>();
  private mounted = false;
  private lifecycle = 0;
  private contextVersion = 0;
  private polling = false;
  private pendingContext: ProofreadContext | null = null;
  private acceptedRequest: ProofreadContext | null = null;
  private updateCallback: () => void = () => {};
  constructor(private transport: ProofreadTransport, private askUnsaved: () => Promise<UnsavedChoice>) {}
  getSnapshot = () => this.state;
  subscribe = (fn: () => void) => { this.listeners.add(fn); return () => this.listeners.delete(fn); };
  setOnUpdated(fn: () => void) { this.updateCallback = fn; }
  get dirty() { return this.state.current !== null && this.state.draft !== this.state.current.text; }
  get contextLocked() { return this.dirty || Boolean(this.state.busy || this.state.asr.job.running || this.state.asr.directory_running || this.state.manifest?.asr_running || this.state.manifest?.training_running); }
  get visible() { return filterClips(this.state.items, this.state.query, this.state.filter); }
  private update(patch: Partial<ProofreadState>) { if (!this.mounted) return; this.state = { ...this.state, ...patch }; this.listeners.forEach(fn => fn()); }
  private valid(lifecycle: number, context = this.contextVersion) { return this.mounted && this.lifecycle === lifecycle && this.contextVersion === context; }
  private failure(error: unknown) { const result = proofreadError(error); this.update({ error: result.message, errorStatus: result.status }); }
  private notify(contextVersion: number) { if (this.mounted && this.contextVersion === contextVersion) this.updateCallback(); }
  dispose() { this.mounted = false; this.lifecycle++; this.contextVersion++; this.pendingContext = null; this.polling = false; }
  async start(context: ProofreadContext) {
    this.mounted = true;
    const lifecycle = ++this.lifecycle;
    const version = ++this.contextVersion;
    this.pendingContext = null;
    this.acceptedRequest = null;
    this.state = initialState();
    this.update({ busy: 'loading' });
    try {
      const session = await this.transport.post<SessionSnapshot>('session');
      if (!this.valid(lifecycle, version)) return;
      if (!session.token || !Array.isArray(session.items)) throw new Error(t('The server returned an invalid proofreading session.'));
      this.transport.setToken(session.token);
      const selected = await this.transport.post<SessionSnapshot>('select', context);
      if (!this.valid(lifecycle, version)) return;
      await this.accept(selected, context, lifecycle, version);
      if (this.valid(lifecycle, version)) this.update({ ready: true });
    } catch (error) { if (this.valid(lifecycle, version)) this.failure(error); }
    finally { if (this.valid(lifecycle, version)) { this.update({ busy: null }); void this.drainContext(); } }
  }
  private async accept(data: SessionSnapshot, expected: ProofreadContext, lifecycle: number, version: number) {
    // The server canonicalizes relative paths; role/project/language identify the requested selection.
    if (data.project !== expected.project || data.role !== expected.role || data.language !== expected.language) throw new Error(t('The selected dataset changed. Reload the proofreading panel.'));
    if (!this.valid(lifecycle, version)) return;
    if (data.token) this.transport.setToken(data.token);
    this.acceptedRequest = { ...expected };
    this.update({ context: { project: data.project, directory: data.directory, language: data.language, role: data.role }, items: data.items, current: null, draft: '', audioSrc: '', manifest: null, asr: data.asr ?? emptyAsr(), retainedContext: false });
    const id = data.items.some(item => item.id === data.last_id) ? data.last_id : data.items[0]?.id;
    if (id) await this.load(id, lifecycle, version);
    await this.refreshManifest(lifecycle, version);
  }
  setContext(context: ProofreadContext) { this.pendingContext = context; void this.drainContext(); }
  private async drainContext() {
    if (!this.mounted || !this.state.ready || this.state.busy || !this.pendingContext) return;
    const next = this.pendingContext;
    this.pendingContext = null;
    if (sameContext(next, this.state.context) || this.acceptedRequest && sameContext(next, this.acceptedRequest)) return;
    await this.run('context', async () => {
      if (this.state.asr.job.running) { this.update({ retainedContext: true }); throw new Error(t('Stop transcription before changing the project or folder.')); }
      if (!await this.allowLeave()) { this.update({ retainedContext: true, notice: t('The previous proofreading context and unsaved transcript were retained.') }); return false; }
      const lifecycle = this.lifecycle;
      const oldVersion = this.contextVersion;
      this.update({ retainedContext: true });
      const selected = await this.transport.post<SessionSnapshot>('select', next);
      if (!this.valid(lifecycle, oldVersion)) return false;
      if (selected.project !== next.project || selected.role !== next.role || selected.language !== next.language) throw new Error(t('The selected dataset changed. Reload the proofreading panel.'));
      const version = ++this.contextVersion;
      await this.accept(selected, next, lifecycle, version);
      this.update({ notice: t('Folder loaded. Audio will not autoplay.') });
      return true;
    });
  }
  setDraft(text: string) { if (this.state.busy !== 'parent-context') this.update({ draft: text }); }
  setQuery(query: string) { this.update({ query }); }
  setFilter(filter: ClipFilter) { this.update({ filter }); }
  setError(message: string) { this.update({ error: message, errorStatus: 0 }); }
  private async run(name: string, action: () => Promise<unknown>): Promise<boolean> {
    if (!this.mounted || this.state.busy) return false;
    const lifecycle = this.lifecycle;
    this.update({ busy: name, error: '', errorStatus: 0, notice: '' });
    try { return (await action()) !== false; }
    catch (error) { if (this.mounted && this.lifecycle === lifecycle) this.failure(error); return false; }
    finally { if (this.mounted && this.lifecycle === lifecycle) { this.update({ busy: null }); void this.drainContext(); } }
  }
  private applyItem(item: Clip) {
    this.update({ current: item, draft: item.text, items: this.state.items.map(row => row.id === item.id ? { ...row, reviewed: item.reviewed, missing_text: !item.text.trim() } : row) });
  }
  private async load(id: string, lifecycle = this.lifecycle, version = this.contextVersion) {
    const item = await this.transport.get<Clip>('item', { id });
    if (!this.valid(lifecycle, version)) return;
    this.applyItem(item);
    this.update({ audioSrc: this.transport.audioUrl(id) });
    await this.transport.post('position', { id });
  }
  private async saveInternal(reviewed: boolean): Promise<boolean> {
    const current = this.state.current;
    if (!current) return false;
    if (!this.state.draft.trim()) throw new Error(t('An empty transcript cannot be saved. Listen to the clip and enter its text.'));
    const lifecycle = this.lifecycle, version = this.contextVersion;
    reviewed = reviewed || (!this.dirty && current.reviewed);
    const saved = await this.transport.post<Clip>('save', { id: current.id, text: this.state.draft, version: current.version, reviewed });
    if (!this.valid(lifecycle, version)) return false;
    this.applyItem(saved);
    await this.refreshManifest(lifecycle, version);
    this.update({ notice: reviewed ? t('Saved and marked as reviewed.') : t('Transcript saved. Update the manifest before training.') });
    this.notify(version);
    return true;
  }
  save = () => this.run('save', () => this.saveInternal(false));
  saveNext = () => this.run('save-next', async () => {
    const visible = this.visible;
    const index = visible.findIndex(item => item.id === this.state.current?.id);
    const next = index >= 0 ? visible[index + 1] : undefined;
    if (!await this.saveInternal(true)) return false;
    if (next) { await this.load(next.id); this.update({ notice: t('The previous clip was saved and marked as reviewed.') }); }
    else this.update({ notice: t('End of this list. The transcript was saved and marked as reviewed.') });
    return true;
  });
  private async allowLeave(reloadDiscard = false, onDiscard?: () => void): Promise<boolean> {
    if (!this.dirty) return true;
    const lifecycle = this.lifecycle, version = this.contextVersion;
    const choice = await this.askUnsaved();
    if (!this.valid(lifecycle, version) || choice === 'cancel') return false;
    if (choice === 'save') return this.saveInternal(false);
    onDiscard?.();
    if (reloadDiscard && this.state.current) await this.load(this.state.current.id);
    return true;
  }
  async allowNavigation(): Promise<boolean> {
    if (this.state.asr.job.running) { this.setError(t('Stop transcription before leaving this proofreading session.')); return false; }
    if (this.state.busy) { this.setError(t('Wait for the current operation to finish before leaving.')); return false; }
    return this.run('navigation', () => this.allowLeave());
  }
  /** Resolve edits against the OLD binding, then keep the editor locked until the parent PUT completes. */
  prepareContextChange = (commit: () => Promise<unknown>): Promise<boolean> => {
    if (this.state.asr.job.running || this.state.asr.directory_running || this.state.manifest?.asr_running) { this.setError(t('Stop transcription before changing the project or folder.')); return Promise.resolve(false); }
    if (this.state.manifest?.training_running) { this.setError(t('Wait for training to finish.')); return Promise.resolve(false); }
    if (this.state.busy) { this.setError(t('Wait for the current operation to finish before leaving.')); return Promise.resolve(false); }
    return this.run('parent-context', async () => {
      const lifecycle = this.lifecycle, version = this.contextVersion, currentId = this.state.current?.id;
      const previousDraft = this.state.draft;
      let discarded = false;
      try {
        if (!await this.allowLeave(true, () => { discarded = true; }) || !this.valid(lifecycle, version)) return false;
        // An already-in-flight poll may discover a lock while the decision dialog is open.
        if (this.state.asr.job.running || this.state.asr.directory_running || this.state.manifest?.asr_running) throw new Error(t('Stop transcription before changing the project or folder.'));
        if (this.state.manifest?.training_running) throw new Error(t('Wait for training to finish.'));
        await commit();
        return true;
      } catch (error) {
        // Failed persistence leaves the previous binding intact. Restore explicitly discarded edits for retry.
        if (discarded && this.valid(lifecycle, version) && this.state.current?.id === currentId) this.update({ draft: previousDraft });
        throw error;
      }
    });
  };
  select = (id: string) => this.run('select', async () => {
    if (id === this.state.current?.id || !await this.allowLeave()) return false;
    await this.load(id);
    return true;
  });
  move = (delta: number) => {
    const list = this.visible, index = list.findIndex(item => item.id === this.state.current?.id);
    const target = index >= 0 ? list[index + delta] : undefined;
    return target ? this.select(target.id) : Promise.resolve(false);
  };
  reload = () => this.run('reload', async () => { if (!this.state.current || !await this.allowLeave()) return false; await this.load(this.state.current.id); this.update({ notice: t('Text reloaded from disk.') }); return true; });
  review = () => this.run('review', async () => {
    if (!this.state.current) return false;
    const reviewed = !(this.state.current.reviewed && !this.dirty);
    if (this.dirty && !await this.saveInternal(false)) return false;
    const current = this.state.current;
    const lifecycle = this.lifecycle, version = this.contextVersion;
    const result = await this.transport.post<Clip>('review', { id: current.id, version: current.version, reviewed });
    if (!this.valid(lifecycle, version)) return false;
    this.applyItem(result);
    this.update({ notice: reviewed ? t('Marked as reviewed.') : t('Marked as pending.') });
    this.notify(version);
    return true;
  });
  transcribe = () => this.run('transcribe', async () => {
    if (!this.state.current || this.state.asr.job.running || !this.state.asr.available || !await this.allowLeave()) return false;
    const current = this.state.current, lifecycle = this.lifecycle, version = this.contextVersion;
    const result = await this.transport.post<Transcription>('asr/item', { id: current.id, version: current.version });
    if (!this.valid(lifecycle, version) || this.state.current?.id !== result.id) return false;
    this.update({ draft: result.text, notice: t('The recognition result is a draft. Review it, then save; no TXT file was changed.') });
    return true;
  });
  startBatch = () => this.run('batch-start', async () => {
    if (!this.state.context.directory || !this.state.asr.available || this.state.asr.job.running || this.state.asr.directory_running || !await this.allowLeave(true)) return false;
    const lifecycle = this.lifecycle, version = this.contextVersion;
    const asr = await this.transport.post<AsrStatus>('asr/start');
    if (!this.valid(lifecycle, version)) return false;
    this.update({ asr, notice: t('Transcribing missing or blank TXT files throughout this folder and its subfolders.') });
    return true;
  });
  stopBatch = () => this.run('batch-stop', async () => {
    const lifecycle = this.lifecycle, version = this.contextVersion;
    const asr = await this.transport.post<AsrStatus>('asr/stop');
    if (!this.valid(lifecycle, version)) return false;
    this.update({ asr, notice: t('Stopping transcription. Completed transcripts will be kept.') });
    return true;
  });
  updateManifest = () => this.run('manifest', async () => {
    if (!this.state.context.project || !this.state.context.directory || this.state.asr.job.running || this.state.manifest?.available === false || this.state.manifest?.training_running || this.state.manifest?.asr_running || !await this.allowLeave(true)) return false;
    const lifecycle = this.lifecycle, version = this.contextVersion, role = this.state.context.role;
    const result = await this.transport.post<ManifestResult>('manifest/update', { role });
    if (!this.valid(lifecycle, version) || result.status.role && result.status.role !== role) return false;
    this.update({ manifest: result.status, notice: t('The manifest was updated from saved TXT files.') });
    this.notify(version);
    return true;
  });
  private async refreshManifest(lifecycle = this.lifecycle, version = this.contextVersion) {
    if (!this.state.context.project) { this.update({ manifest: null }); return; }
    try {
      const result = await this.transport.get<ManifestStatus>('manifest/status');
      if (this.valid(lifecycle, version) && (!result.role || result.role === this.state.context.role)) this.update({ manifest: result });
    } catch (error) { if (this.valid(lifecycle, version)) { this.update({ manifest: { stale: true, message: proofreadError(error).message } }); if (proofreadError(error).status === 403) this.failure(error); } }
  }
  private async refreshSnapshot(background: boolean, lifecycle = this.lifecycle, version = this.contextVersion) {
    const currentId = this.state.current?.id, currentVersion = this.state.current?.version;
    const snapshot = await this.transport.get<SessionSnapshot>('session');
    if (!this.valid(lifecycle, version) || !sameContext(snapshot, this.state.context) || background && this.state.busy) return;
    this.update({ items: snapshot.items, asr: snapshot.asr ?? this.state.asr });
    if (currentId && this.state.current?.id === currentId && snapshot.items.some(item => item.id === currentId) && !this.dirty) {
      const disk = await this.transport.get<Clip>('item', { id: currentId });
      if (this.valid(lifecycle, version) && this.state.current?.id === currentId && this.state.current?.version === currentVersion && !this.dirty && (!background || !this.state.busy)) this.applyItem(disk);
    }
  }
  refresh = () => this.run('refresh', async () => { await this.refreshSnapshot(false); await this.refreshManifest(); return true; });
  async poll() {
    if (!this.mounted || !this.state.ready || this.state.busy || this.polling || !this.state.context.directory) return;
    this.polling = true;
    const lifecycle = this.lifecycle, version = this.contextVersion, wasRunning = this.state.asr.job.running;
    try {
      const asr = await this.transport.get<AsrStatus>('asr/status');
      if (!this.valid(lifecycle, version)) return;
      this.update({ asr });
      if (!this.state.busy) { await this.refreshSnapshot(true, lifecycle, version); if (this.valid(lifecycle, version)) await this.refreshManifest(lifecycle, version); }
      if (this.valid(lifecycle, version) && wasRunning && !this.state.asr.job.running) this.notify(version);
    } catch (error) { if (this.valid(lifecycle, version)) this.failure(error); }
    finally { if (this.lifecycle === lifecycle) this.polling = false; }
  }
}

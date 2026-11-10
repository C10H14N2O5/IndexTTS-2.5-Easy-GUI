import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, Link, RouterProvider } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createRef, type RefObject } from 'react';
import { setLanguage } from '@/i18n';
import { ProofreadPanel, type ProofreadPanelHandle, type ProofreadPanelProps } from './ProofreadPanel';
import type { Clip, ProofreadContext } from './api';

vi.mock('./ProofreadAudio', () => ({ ProofreadAudio: ({ src, audioRef }: { src: string; audioRef: RefObject<HTMLAudioElement | null> }) => <audio ref={audioRef} src={src} controls aria-label="Proofreading audio" /> }));
let clips: Clip[];
let context: ProofreadContext;
let requests: { method: string; operation: string; body: Record<string, unknown>; token: string | null }[];
let failSave: number;
let reviewedVersion = 1;
let play: ReturnType<typeof vi.spyOn>;
let snapshot: () => unknown;

beforeEach(() => {
  setLanguage('en_US');
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  Object.defineProperty(HTMLElement.prototype, 'hasPointerCapture', { configurable: true, value: () => false });
  Object.defineProperty(HTMLElement.prototype, 'setPointerCapture', { configurable: true, value: () => {} });
  Object.defineProperty(HTMLElement.prototype, 'releasePointerCapture', { configurable: true, value: () => {} });
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: () => {} });
  play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  context = { project: 'voice', directory: 'D:/audio', role: 'training', language: 'EN' };
  clips = [{ id: '1', name: 'first.wav', text: 'Original transcript', version: 'v1', reviewed: false }, { id: '2', name: 'second.wav', text: 'Second transcript', version: 'v1', reviewed: false }];
  requests = []; failSave = 0; reviewedVersion = 1;
  let running = false;
  const asr = () => ({ available: true, job: { running, total: running ? 2 : 0, completed: 0, succeeded: 0, skipped: 0, failed: 0 } });
  const manifest = () => ({ available: true, stale: false, status: 'ready', role: context.role, manifest: context.role === 'training' ? 'training.jsonl' : 'validation.jsonl' });
  snapshot = () => ({ ...context, token: 'panel-token', items: clips.map(clip => ({ id: clip.id, name: clip.name, reviewed: clip.reviewed, missing_text: !clip.text.trim() })), last_id: '1', asr: asr() });
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options: RequestInit = {}) => {
    const url = new URL(String(input), 'http://localhost');
    const operation = url.pathname.replace('/proofread/api/', '');
    const body = options.body ? JSON.parse(String(options.body)) as Record<string, unknown> : {};
    const method = options.method ?? 'GET';
    requests.push({ method, operation, body, token: new Headers(options.headers).get('X-Session-Token') });
    let data: unknown = {}, status = 200;
    if (operation === 'session') data = snapshot();
    else if (operation === 'select') { context = { ...body } as unknown as ProofreadContext; data = snapshot(); }
    else if (operation === 'item') data = { ...clips.find(clip => clip.id === url.searchParams.get('id'))! };
    else if (operation === 'save') {
      if (failSave) { data = { error: failSave === 409 ? 'TXT changed on disk' : 'Dataset is locked' }; status = failSave; }
      else { const item = clips.find(clip => clip.id === body.id)!; Object.assign(item, { text: body.text, reviewed: body.reviewed, version: `v${++reviewedVersion}` }); data = { ...item }; }
    } else if (operation === 'review') { const item = clips.find(clip => clip.id === body.id)!; item.reviewed = body.reviewed === true; data = { ...item }; }
    else if (operation === 'manifest/status') data = manifest();
    else if (operation === 'manifest/update') data = { status: manifest() };
    else if (operation === 'asr/status') data = asr();
    else if (operation === 'asr/item') data = { id: body.id, version: body.version, text: 'ASR draft only' };
    else if (operation === 'asr/start') { running = true; data = asr(); }
    else if (operation === 'asr/stop') { running = false; data = asr(); }
    else if (operation !== 'position') throw new Error(`Unexpected operation ${operation}`);
    return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
  }));
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

async function openPanel(role: 'training' | 'validation' = 'training', extras: Pick<ProofreadPanelProps, 'ref' | 'onContextLockedChange'> = {}) {
  const router = createMemoryRouter([{ path: '/proofread', element: <><Link to="/other">Leave panel</Link><ProofreadPanel project="voice" directory="D:/audio" role={role} language="EN" {...extras} /></> }, { path: '/other', element: <h1>Other page</h1> }], { initialEntries: ['/proofread'] });
  render(<RouterProvider router={router} />);
  await screen.findByRole('textbox', { name: 'Transcript' });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save transcript' })).toBeEnabled());
  return router;
}
async function editDraft(value = 'Human revision') { const editor = screen.getByRole('textbox', { name: 'Transcript' }); await userEvent.clear(editor); await userEvent.type(editor, value); return editor; }

describe('imperative parent mutation guard', () => {
  it('keeps the editor locked through the parent request and saves the old transcript first', async () => {
    const ref = createRef<ProofreadPanelHandle>(); const onContextLockedChange = vi.fn();
    await openPanel('training', { ref, onContextLockedChange }); await editDraft('Save before parent PUT');
    let finish!: () => void; const waiting = new Promise<void>(resolve => { finish = resolve; });
    const commit = vi.fn(() => { expect(clips[0]?.text).toBe('Save before parent PUT'); return waiting; });
    let guarded!: Promise<boolean>; await act(async () => { guarded = ref.current!.prepareContextChange(commit); });
    expect(screen.getByRole('dialog', { name: 'Unsaved transcript' })).toBeVisible(); expect(commit).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Save and continue' })); await waitFor(() => expect(commit).toHaveBeenCalledOnce());
    expect(screen.getByRole('textbox', { name: 'Transcript' })).toBeDisabled(); expect(onContextLockedChange).toHaveBeenLastCalledWith(true);
    fireEvent.keyDown(document, { key: 's', ctrlKey: true }); expect(requests.filter(request => request.operation === 'save')).toHaveLength(1);
    await act(async () => { finish(); expect(await guarded).toBe(true); });
    expect(screen.getByRole('textbox', { name: 'Transcript' })).toBeEnabled(); expect(onContextLockedChange).toHaveBeenLastCalledWith(false);
  });
  it('restores discarded edits after a failed parent request and clears the layout lock on unmount', async () => {
    const ref = createRef<ProofreadPanelHandle>(); const onContextLockedChange = vi.fn();
    await openPanel('training', { ref, onContextLockedChange }); await editDraft('Recovery buffer');
    let guarded!: Promise<boolean>; await act(async () => { guarded = ref.current!.prepareContextChange(async () => { throw new Error('Parent request failed'); }); });
    await userEvent.click(screen.getByRole('button', { name: 'Discard edits' }));
    await act(async () => { expect(await guarded).toBe(false); });
    expect(screen.getByRole('textbox', { name: 'Transcript' })).toHaveValue('Recovery buffer');
    expect(screen.getByRole('textbox', { name: 'Transcript' })).toBeEnabled(); expect(onContextLockedChange).toHaveBeenLastCalledWith(true);
    expect(screen.getByRole('alert')).toHaveTextContent('Parent request failed'); cleanup(); expect(onContextLockedChange).toHaveBeenLastCalledWith(false);
  });
  it('skips the parent mutation on Cancel and retains the original context and draft', async () => {
    const ref = createRef<ProofreadPanelHandle>(); await openPanel('training', { ref }); await editDraft('Keep current binding');
    const commit = vi.fn(async () => {}); let guarded!: Promise<boolean>; await act(async () => { guarded = ref.current!.prepareContextChange(commit); });
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' })); await act(async () => { expect(await guarded).toBe(false); });
    expect(commit).not.toHaveBeenCalled(); expect(screen.getByRole('textbox', { name: 'Transcript' })).toHaveValue('Keep current binding'); expect(screen.getByText('Training set')).toBeVisible();
  });
});

describe('React proofreading workbench', () => {
  it('loads the saved text and authenticates subsequent requests without autoplay', async () => {
    await openPanel(); expect(screen.getByRole('textbox', { name: 'Transcript' })).toHaveValue('Original transcript'); expect(play).not.toHaveBeenCalled(); expect(requests.find(request => request.operation === 'select')?.token).toBe('panel-token'); expect(screen.getByLabelText('Proofreading audio')).toHaveAttribute('src', expect.stringContaining('token=panel-token'));
  });
  it('shows validation context and updates only its manifest', async () => {
    await openPanel('validation'); expect(screen.getByText('Validation set')).toBeVisible(); await userEvent.click(screen.getByRole('button', { name: 'Update validation manifest' })); await waitFor(() => expect(requests.find(request => request.operation === 'manifest/update')?.body).toEqual({ role: 'validation' })); expect(screen.getByText('validation.jsonl')).toBeVisible(); expect(screen.queryByText('training.jsonl')).not.toBeInTheDocument();
  });
  it('asks before changing clips and preserves edits on Cancel', async () => {
    await openPanel(); await editDraft(); await userEvent.click(screen.getByRole('button', { name: 'Next clip' })); const dialog = screen.getByRole('dialog', { name: 'Unsaved transcript' }); await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' })); expect(screen.getByRole('textbox', { name: 'Transcript' })).toHaveValue('Human revision'); expect(screen.getByRole('heading', { name: 'first.wav' })).toBeVisible();
  });
  it('discards only on explicit choice then selects the next clip', async () => {
    await openPanel(); await editDraft(); await userEvent.click(screen.getByRole('button', { name: 'Next clip' })); await userEvent.click(screen.getByRole('button', { name: 'Discard edits' })); await waitFor(() => expect(screen.getByRole('textbox', { name: 'Transcript' })).toHaveValue('Second transcript')); expect(clips[0]?.text).toBe('Original transcript'); expect(play).not.toHaveBeenCalled();
  });
  it('saves first when Save and continue is chosen', async () => {
    await openPanel(); await editDraft(); await userEvent.click(screen.getByRole('button', { name: 'Next clip' })); await userEvent.click(screen.getByRole('button', { name: 'Save and continue' })); await screen.findByRole('heading', { name: 'second.wav' }); expect(clips[0]?.text).toBe('Human revision');
  });
  it('keeps a conflict draft visible with recovery guidance', async () => {
    await openPanel(); await editDraft(); failSave = 409; await userEvent.click(screen.getByRole('button', { name: 'Save transcript' })); expect(await screen.findByRole('alert')).toHaveTextContent('TXT changed on disk'); expect(screen.getByRole('textbox', { name: 'Transcript' })).toHaveValue('Human revision'); expect(screen.getByRole('button', { name: 'Select transcript to copy' })).toBeEnabled();
  });
  it('treats a locked dataset as recoverable without dropping text', async () => {
    await openPanel(); await editDraft(); failSave = 423; await userEvent.click(screen.getByRole('button', { name: 'Save transcript' })); expect(await screen.findByRole('alert')).toHaveTextContent('This dataset is locked'); expect(screen.getByRole('textbox', { name: 'Transcript' })).toHaveValue('Human revision');
  });
  it('saves with Ctrl+S and advances with Ctrl+Enter', async () => {
    await openPanel(); await editDraft(); fireEvent.keyDown(document, { key: 's', ctrlKey: true }); await waitFor(() => expect(clips[0]?.text).toBe('Human revision')); await waitFor(() => expect(screen.getByRole('button', { name: 'Save and next' })).toBeEnabled()); fireEvent.keyDown(document, { key: 'Enter', ctrlKey: true }); await screen.findByRole('heading', { name: 'second.wav' }); expect(clips[0]?.reviewed).toBe(true);
  });
  it('does not submit keyboard shortcuts through the unsaved dialog', async () => {
    await openPanel(); await editDraft(); await userEvent.click(screen.getByRole('button', { name: 'Next clip' })); fireEvent.keyDown(document, { key: 's', ctrlKey: true }); expect(requests.filter(request => request.operation === 'save')).toHaveLength(0); expect(screen.getByRole('dialog', { name: 'Unsaved transcript' })).toBeVisible();
  });
  it('requires deliberate Alt+Space to start audio playback', async () => {
    await openPanel(); fireEvent.keyDown(document, { code: 'Space', altKey: true }); expect(play).toHaveBeenCalledOnce();
  });
  it('blocks route navigation with the same three-way guard', async () => {
    const router = await openPanel(); await editDraft(); await userEvent.click(screen.getByRole('link', { name: 'Leave panel' })); await userEvent.click(await screen.findByRole('button', { name: 'Cancel' })); expect(router.state.location.pathname).toBe('/proofread'); await userEvent.click(screen.getByRole('link', { name: 'Leave panel' })); await userEvent.click(await screen.findByRole('button', { name: 'Discard edits' })); await screen.findByRole('heading', { name: 'Other page' }); expect(router.state.location.pathname).toBe('/other');
  });
  it('protects unsaved text from browser reload using beforeunload', async () => {
    await openPanel(); await editDraft(); const event = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(event); expect(event.defaultPrevented).toBe(true);
  });
  it('keeps single ASR as an unsaved draft until the user saves', async () => {
    await openPanel(); await userEvent.click(screen.getByRole('button', { name: 'Transcribe this clip again' })); await waitFor(() => expect(screen.getByRole('textbox', { name: 'Transcript' })).toHaveValue('ASR draft only')); expect(clips[0]?.text).toBe('Original transcript'); expect(requests.filter(request => request.operation === 'save')).toHaveLength(0); expect(screen.getByText('Unsaved changes')).toBeVisible();
  });
  it('keeps a batch session on the page until transcription stops', async () => {
    const router = await openPanel(); await userEvent.click(screen.getByRole('button', { name: 'Transcribe missing text' })); await screen.findByRole('button', { name: 'Stop transcription' }); await userEvent.click(screen.getByRole('link', { name: 'Leave panel' })); await waitFor(() => expect(router.state.location.pathname).toBe('/proofread')); expect(await screen.findByRole('alert')).toHaveTextContent('Stop transcription before leaving'); await userEvent.click(screen.getByRole('button', { name: 'Stop transcription' })); await screen.findByRole('button', { name: 'Transcribe missing text' });
  });
  it('searches without changing the active editor', async () => {
    await openPanel(); await userEvent.type(screen.getByRole('searchbox', { name: 'Search relative file paths' }), 'second'); expect(within(screen.getByRole('navigation', { name: 'Audio clips' })).queryByText('first.wav')).not.toBeInTheDocument(); expect(screen.getByRole('heading', { name: 'first.wav' })).toBeVisible(); expect(screen.getByText('This clip is outside the active filter.')).toBeVisible();
  });
  it('updates the reviewed marker and progress after explicit review', async () => {
    await openPanel(); await userEvent.click(screen.getByRole('button', { name: 'Mark as reviewed' })); await screen.findByRole('button', { name: 'Mark as pending' }); expect(clips[0]?.reviewed).toBe(true); expect(screen.getByText('1 / 2')).toBeVisible();
  });
  it('changes UI language without discarding a draft or selecting another session', async () => {
    await openPanel(); await editDraft(); const selects = requests.filter(request => request.operation === 'select').length; act(() => setLanguage('zh_CN')); expect(screen.getByRole('textbox', { name: '转写文本' })).toHaveValue('Human revision'); expect(screen.getByRole('button', { name: '保存并下一条' })).toBeVisible(); expect(requests.filter(request => request.operation === 'select')).toHaveLength(selects);
  });
});

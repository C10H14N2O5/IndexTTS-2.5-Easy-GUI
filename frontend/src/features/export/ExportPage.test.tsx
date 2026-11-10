import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useLive } from '@/api/live';
import { setLanguage, t } from '@/i18n';
import type { Schemas } from '@/api/client';
import ExportPage, { ExportResult } from './ExportPage';
import { EXPORT_PHASES, exportControls, exportDownloads, exportProgress, outputDestination, suggestedValue, verification } from './model';
let client: QueryClient;
const idle: Schemas['ExportSnapshot'] = { status: 'idle', phase: 'idle', elapsed: 0, log: '', output: '', error: '', report: {}, token: '', task_id: '', logfile: '', downloads: {} };
beforeEach(() => {
  setLanguage('en_US'); localStorage.clear(); useLive.setState({ ready: true, connection: 'open', jobs: {}, jobOrder: [], training: null, export: null });
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  Element.prototype.scrollIntoView = vi.fn();
  vi.stubGlobal('ResizeObserver', class { observe = vi.fn(); unobserve = vi.fn(); disconnect = vi.fn(); });
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (url === '/api/adapters') return Response.json([{ path: '/adapters/a', label: 'Adapter A', project: 'voice', kind: 'final', checkpoint: 'final_adapter' }]);
    if (url.startsWith('/api/export/adapter-info')) { const other = String(url).includes('%2Fb'); return Response.json({ filename: other ? 'b.pth' : 'a.pth', base: other ? '/base/b.pth' : '/base/a.pth', config: '/base/config.yaml', valid: true, error: '', training_base: '', notice: 'Ready to export.' }); }
    if (url === '/api/export/status') return Response.json(idle);
    if (url === '/api/export/start') return Response.json({ ...idle, status: 'running', phase: 'starting', task_id: 'task', output: 'out.pth' });
    if (url === '/api/fs/roots') return Response.json([{ label: 'Disk', path: 'D:/' }]);
    if (url.startsWith('/api/fs/list')) return Response.json({ error: { code: 'not_found', message: 'Folder is unavailable' } }, { status: 404 });
    if (init?.method === 'POST') throw new Error(`Unexpected mutation ${url}`);
    return Response.json([]);
  }));
});
afterEach(() => { client.clear(); vi.unstubAllGlobals(); setLanguage('en_US'); });
function mount() { return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/export?adapter=%2Fadapters%2Fa']}><ExportPage/></MemoryRouter></QueryClientProvider>); }
async function changeAdapter() { await userEvent.click(screen.getByRole('button', { name: 'Custom adapter folder' })); const input = screen.getByRole('textbox', { name: 'Adapter folder' }); await userEvent.clear(input); await userEvent.type(input, '/adapters/b'); }
describe('export safety model', () => {
  it('follows suggestions until the user explicitly edits a value', () => { expect(suggestedValue(null, 'b.pth')).toBe('b.pth'); expect(suggestedValue('custom.pth', 'b.pth')).toBe('custom.pth'); expect(suggestedValue('', 'b.pth')).toBe(''); });
  it('maps browser and both server path separators without modifying filenames', () => { expect(outputDestination('browser', 'ignored', 'voice.pth')).toBe('voice.pth'); expect(outputDestination('path', 'D:\\out\\', 'voice.pth')).toBe('D:\\out\\voice.pth'); expect(outputDestination('path', '/out/', 'voice.pth')).toBe('/out/voice.pth'); });
  it.each(['idle', 'running', 'cancelling', 'failed', 'cancelled'])('never exposes links for %s', status => { expect(exportDownloads(status, { model: '/exports/id/model?token=token' })).toEqual([]); });
  it('rejects non-download URLs even for success', () => { expect(exportDownloads('succeeded', { model: 'javascript:alert(1)', report: 'https://other.test/report' })).toEqual([]); });
  it('enables start and cancel only in their correct lifecycle states', () => { expect(exportControls('running')).toEqual({ canStart: false, canCancel: true }); expect(exportControls('cancelling')).toEqual({ canStart: false, canCancel: false }); expect(exportControls('succeeded')).toEqual({ canStart: true, canCancel: false }); });
  it.each([undefined, null, false, 1, 'true', 'false'])('does not infer verification from %s', value => expect(verification(value)).toBe('Not verified'));
  it('accepts actual true evidence only', () => expect(verification(true)).toBe('Yes'));
  it('localizes all phase labels and has bounded or indeterminate progress', () => { setLanguage('zh_CN'); for (const [phase, label] of Object.entries(EXPORT_PHASES)) { expect(t(label)).not.toBe(label); const progress = exportProgress(phase); if (progress !== null) { expect(progress).toBeGreaterThanOrEqual(0); expect(progress).toBeLessThanOrEqual(1); } } expect(exportProgress('unknown')).toBeNull(); });
});
describe('export interface', () => {
  it('follows adapter filename and base suggestions before user edits', async () => { mount(); await waitFor(() => expect(screen.getByRole('textbox', { name: 'Output filename' })).toHaveValue('a.pth')); await changeAdapter(); await waitFor(() => expect(screen.getByRole('textbox', { name: 'Output filename' })).toHaveValue('b.pth')); expect(screen.getByRole('textbox', { name: 'Original FP32 training base (.pth)' })).toHaveValue('/base/b.pth'); });
  it('retains explicitly edited filename and base when the adapter changes', async () => {
    mount(); await waitFor(() => expect(screen.getByRole('textbox', { name: 'Output filename' })).toHaveValue('a.pth'));
    const filename = screen.getByRole('textbox', { name: 'Output filename' }); await userEvent.clear(filename); await userEvent.type(filename, 'keep.pth'); const base = screen.getByRole('textbox', { name: 'Original FP32 training base (.pth)' }); await userEvent.clear(base); await userEvent.type(base, '/custom/original.pth'); await changeAdapter();
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes('adapter=%2Fadapters%2Fb'))).toBe(true)); expect(filename).toHaveValue('keep.pth'); expect(base).toHaveValue('/custom/original.pth');
  });
  it('browser failure or cancellation preserves destination and never starts export', async () => {
    mount(); await screen.findByText('Ready to export.'); await userEvent.click(screen.getByRole('radio', { name: 'Save on server' })); const folder = screen.getByRole('textbox', { name: 'Output folder on server' }); await userEvent.type(folder, 'D:/keep');
    await userEvent.click(within(folder.parentElement!).getByRole('button', { name: 'Browse…' })); await screen.findByText('Folder is unavailable'); await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Cancel' }));
    expect(folder).toHaveValue('D:/keep'); expect(vi.mocked(fetch).mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
  });
  it('retains explicit export settings across navigation', async () => {
    const view = mount(); await screen.findByText('Ready to export.'); const filename = screen.getByRole('textbox', { name: 'Output filename' }); await userEvent.clear(filename); await userEvent.type(filename, 'persisted.pth'); view.unmount(); mount(); expect(await screen.findByRole('textbox', { name: 'Output filename' })).toHaveValue('persisted.pth');
  });
  it('starts with explicit destination and selected original base', async () => {
    mount(); await screen.findByText('Ready to export.'); await userEvent.click(screen.getByRole('button', { name: 'Export and verify' })); await screen.findByText('Starting CPU exporter');
    const call = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/export/start'); expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({ adapter: '/adapters/a', mode: 'browser', destination: 'a.pth', base: '/base/a.pth', config: '/base/config.yaml' }); expect(screen.getByRole('button', { name: 'Export and verify' })).toBeDisabled(); expect(screen.getByRole('button', { name: 'Cancel export' })).toBeEnabled();
  });
  it('renders unsafe output names as text and marks missing evidence unverified', () => {
    const output = '<img src=x onerror=alert(1)>.pth'; const view = render(<ExportResult status={{ ...idle, status: 'succeeded', phase: 'completed', output, downloads: { model: '/exports/id/model?token=test', report: '/exports/id/report?token=test' } }}/>);
    expect(screen.getByText(output)).toBeInTheDocument(); expect(view.container.querySelector('img')).toBeNull(); expect(screen.getAllByText('Not verified').length).toBeGreaterThanOrEqual(2); expect(screen.getAllByRole('link')).toHaveLength(2);
  });
  it('hides download links even if a failed snapshot contains stale URLs', () => { render(<ExportResult status={{ ...idle, status: 'failed', phase: 'failed', downloads: { model: '/exports/id/model?token=test' } }}/>); expect(screen.queryByRole('link')).not.toBeInTheDocument(); });
});

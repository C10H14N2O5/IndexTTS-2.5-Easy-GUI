import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { bootstrapKey } from '@/app/bootstrap';
import { useLive } from '@/api/live';
import { setLanguage } from '@/i18n';
import SynthesisPage from './SynthesisPage';
import { defaultForm, FORM_STORAGE_KEY } from './model';
import { testBootstrap } from './testFixtures';
let client: QueryClient;
beforeEach(() => {
  localStorage.clear(); setLanguage('en_US');
  useLive.setState({ settings: null, ready: true, connection: 'open', jobs: {}, jobOrder: [], training: null, export: null });
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  client.setQueryData(bootstrapKey, testBootstrap());
  Element.prototype.scrollIntoView = vi.fn();
  vi.stubGlobal('ResizeObserver', class { observe = vi.fn(); unobserve = vi.fn(); disconnect = vi.fn(); });
  vi.stubGlobal('fetch', vi.fn(async (input: string, init?: RequestInit) => {
    if (init?.method === 'POST') return Response.json({ error: { code: 'busy', message: 'GPU occupied' } }, { status: 409 });
    if (input.includes('/api/voices') || input.includes('/api/history') || input.includes('/api/adapters') || input.includes('/api/emotion-descriptions')) return Response.json([]);
    throw new Error(`Unexpected request ${input}`);
  }));
});
afterEach(() => { client.clear(); vi.unstubAllGlobals(); });
function mount(path = '/synthesis') { return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[path]}><Routes><Route path="synthesis" element={<SynthesisPage />} /><Route path="synthesis/:mode" element={<SynthesisPage />} /></Routes></MemoryRouter></QueryClientProvider>); }
describe('synthesis page', () => {
  it('renders a real blank form and validates before making a generation request', async () => {
    mount();
    await userEvent.click(screen.getByRole('button', { name: 'Generate speech' }));
    expect(screen.getByText('Enter the words you want to hear.')).toBeInTheDocument();
    expect(screen.getByText('Choose a voice or upload a reference clip.')).toBeInTheDocument();
    const calls = vi.mocked(fetch).mock.calls;
    expect(calls.some(([, options]) => options?.method === 'POST')).toBe(false);
  });
  it('persists text edits across unmounting and restores after reload', async () => {
    const first = mount();
    await userEvent.type(screen.getByRole('textbox', { name: 'Target text' }), 'Hello from the draft');
    await waitFor(() => expect(JSON.parse(localStorage.getItem(FORM_STORAGE_KEY)!).text).toBe('Hello from the draft'));
    first.unmount(); mount();
    expect(screen.getByRole('textbox', { name: 'Target text' })).toHaveValue('Hello from the draft');
  });
  it('surfaces a structured backend error without clearing the user draft', async () => {
    localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify({ ...defaultForm(), voice_path: 'voices/test.wav', text: 'Keep my text' }));
    mount();
    await userEvent.click(screen.getByRole('button', { name: 'Generate speech' }));
    await screen.findByText('GPU occupied');
    expect(screen.getByRole('textbox', { name: 'Target text' })).toHaveValue('Keep my text');
    const request = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/synthesis');
    expect(JSON.parse(String(request?.[1]?.body))).toMatchObject({ voice_path: 'voices/test.wav', text: 'Keep my text', temperature: 0.8 });
  });
  it('renders two dialogue turns and can insert, duplicate, and remove them', async () => {
    mount('/synthesis/dialogue');
    expect(screen.getAllByRole('article', { name: /^Turn / })).toHaveLength(2);
    await userEvent.type(screen.getByRole('textbox', { name: 'Text for turn 1' }), 'Hello');
    await userEvent.click(screen.getAllByRole('button', { name: 'Duplicate turn' })[0]!);
    expect(screen.getAllByRole('article', { name: /^Turn / })).toHaveLength(3);
    expect(screen.getByRole('textbox', { name: 'Text for turn 2' })).toHaveValue('Hello');
    await userEvent.click(screen.getAllByRole('button', { name: 'Remove turn' })[0]!);
    expect(screen.getAllByRole('article', { name: /^Turn / })).toHaveLength(2);
    expect(screen.queryByText('Drop reference audio here')).not.toBeInTheDocument(); // Dialogue accepts saved library voices only.
  });
  it('requires explicit confirmation before resetting the persisted draft', async () => {
    mount(); await userEvent.type(screen.getByRole('textbox', { name: 'Target text' }), 'Do not lose this');
    await userEvent.click(screen.getByRole('button', { name: 'Reset form' }));
    expect(screen.getByRole('dialog', { name: 'Reset this draft?' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.getByRole('textbox', { name: 'Target text' })).toHaveValue('Do not lose this');
  });
});

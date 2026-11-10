import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { bootstrapKey } from '@/app/bootstrap';
import { useLive } from '@/api/live';
import { setLanguage, t } from '@/i18n';
import { testBootstrap } from '@/features/synthesis/testFixtures';
import { defaultForm, FORM_STORAGE_KEY } from '@/features/synthesis/model';
import type { Schemas } from '@/api/client';
import AbxPage from './AbxPage';
import { comparisonPlan, decodingSettings } from './model';
vi.mock('@/ui', async importOriginal => ({ ...await importOriginal<typeof import('@/ui')>(), AudioPlayer: ({ src, title }: { src: string; title: string }) => <audio aria-label={title} src={src}/> }));
let client: QueryClient; let session: Schemas['AbxSession'];
beforeEach(() => {
  setLanguage('en_US'); localStorage.clear(); useLive.setState({ ready: true, connection: 'open', jobs: {}, jobOrder: [], training: null, export: null, settings: null });
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } }); client.setQueryData(bootstrapKey, testBootstrap());
  Element.prototype.scrollIntoView = vi.fn(); Element.prototype.hasPointerCapture = vi.fn(() => false); Element.prototype.setPointerCapture = vi.fn(); Element.prototype.releasePointerCapture = vi.fn();
  vi.stubGlobal('ResizeObserver', class { observe = vi.fn(); unobserve = vi.fn(); disconnect = vi.fn(); });
  session = { id: 'demo', name: 'Demo comparison', a: 'Original model', b: 'Voice adapter', clips_done: 4, clips_total: 4, rated: 0, trials: 2, revealed: false, generation_complete: true, texts: ['One sentence'], repetitions: 2, language: 'ZH', next_unrated: 0, trial: { index: 0, total: 2, text: 'One sentence', rep: 0, reference: '/voice.wav', sample_1: '/one.wav', sample_2: '/two.wav', score: null }, results: null };
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (url === '/api/adapters') return Response.json([{ path: '/adapters/voice', label: 'Voice adapter', project: 'voice', checkpoint: 'final_adapter', kind: 'final' }]);
    if (url === '/api/voices') return Response.json([{ path: '/voices/a.wav', name: 'Reference A', duration: 5, size: 100, modified: 0 }]);
    if (url === '/api/abx/sessions') return Response.json(init?.method === 'POST' ? { ...session, generation_complete: false, trial: null } : [session]);
    if (String(url).includes('/generate') || String(url).includes('/api/jobs/')) return Response.json(String(url).includes('/generate') ? { job: { id: 'job1', kind: 'abx_generate', title: 'Generating samples', state: 'running', progress: 0, started_at: Date.now() / 1000, finished_at: null, cancellable: true, cancel_requested: false, meta: { session_id: 'demo' } } } : useLive.getState().jobs.job1);
    if (String(url).includes('/rate')) { const body = JSON.parse(String(init?.body)); session = { ...session, rated: body.score === null ? 0 : 1, trial: { ...session.trial!, score: body.score }, next_unrated: body.score === null ? 0 : 1 }; return Response.json(session); }
    if (String(url).includes('/reveal')) { session = { ...session, revealed: true, results: { rated: 1, total: 2, wins_a: 1, wins_b: 0, ties: 0, mean_b: -1, stderr: 0, p_value: 1, verdict: 'no_difference', rows: [{ index: 0, text: 'One sentence', rep: 0, sample_1: 'A', score: -1, preference_b: -1 }] } }; return Response.json(session); }
    if (url.startsWith('/api/abx/sessions/demo')) { const index = Number(new URL(url, 'http://local').searchParams.get('trial') ?? 0); return Response.json({ ...session, trial: session.trial ? { ...session.trial, index, rep: index } : null }); }
    throw new Error(`Unexpected request ${url}`);
  }));
});
afterEach(() => { client.clear(); vi.unstubAllGlobals(); });
function mount(path = '/abx?session=demo') { return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[path]}><AbxPage/></MemoryRouter></QueryClientProvider>); }
describe('blind comparison', () => {
  it('counts nonempty sentences and paired independent repetitions', () => expect(comparisonPlan('First\n\nSecond\r\n ', 3)).toEqual({ sentences: 2, trials: 6, clips: 12 }));
  it('freezes decoding without carrying a single synthesis adapter or emotions', () => { const form = defaultForm(); form.temperature = 1.1; form.adapter_path = 'not part of pair'; const result = decodingSettings(form); expect(result.temperature).toBe(1.1); expect(result).not.toHaveProperty('adapter_path'); expect(result).not.toHaveProperty('emotion_vector'); });
  it('saves a rating immediately without exposing adapter identities', async () => {
    mount(); const rating = await screen.findByRole('button', { name: 'Sample 1 much better' }); await userEvent.click(rating); await waitFor(() => expect(rating).toHaveAttribute('aria-pressed', 'true'));
    expect(screen.queryByText('Adapter A')).not.toBeInTheDocument(); expect(screen.queryByText('Comparison results')).not.toBeInTheDocument(); const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/rate')); expect(JSON.parse(String(call?.[1]?.body))).toEqual({ index: 0, score: -2 });
  });
  it('can clear a rating and jump to the next unrated trial', async () => {
    mount(); await userEvent.click(await screen.findByRole('button', { name: 'Sample 2 slightly better' })); await userEvent.click(screen.getByRole('button', { name: 'Next unrated' })); expect(await screen.findByText('Trial 2 of 2')).toBeInTheDocument(); await userEvent.click(screen.getByRole('button', { name: 'Clear rating' })); await waitFor(() => expect(session.rated).toBe(0));
  });
  it('requires reveal confirmation, then locks ratings and displays statistics', async () => {
    mount(); await userEvent.click(await screen.findByRole('button', { name: 'Sample 1 slightly better' })); await userEvent.click(screen.getByRole('button', { name: 'Reveal results' })); const dialog = screen.getByRole('dialog'); expect(within(dialog).getByText('This locks all ratings. 1 trial(s) are still unrated.')).toBeInTheDocument(); expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes('/reveal'))).toBe(false);
    await userEvent.click(within(dialog).getByRole('button', { name: 'Reveal and lock ratings' })); await screen.findByText('Comparison results'); expect(screen.getByRole('button', { name: 'Sample 1 slightly better' })).toBeDisabled(); expect(screen.getByText('No significant difference yet')).toBeInTheDocument(); expect(screen.getByText('Original model')).toBeInTheDocument();
  });
  it.each(['a', 'b'] as const)('localizes the reserved base identity on side %s only after reveal, preserving user labels and stored data', async side => {
    const reserved = 'Base model (no adapter)'; const custom = 'My Base model (no adapter) / final_adapter';
    session[side] = reserved; session[side === 'a' ? 'b' : 'a'] = custom; setLanguage('zh_CN'); mount();
    await screen.findByText(t('Blind listening')); expect(screen.queryByText('原始模型')).not.toBeInTheDocument(); expect(screen.queryByText(reserved)).not.toBeInTheDocument(); expect(screen.queryByText(custom)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: t('Sample 1 slightly better') })); await userEvent.click(screen.getByRole('button', { name: t('Reveal results') })); await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: t('Reveal and lock ratings') }));
    expect(await screen.findByText('原始模型')).toBeInTheDocument(); expect(screen.queryByText(reserved)).not.toBeInTheDocument(); expect(screen.getByText(custom)).toBeInTheDocument(); expect(session[side]).toBe(reserved); expect(session[side === 'a' ? 'b' : 'a']).toBe(custom);
  });
  it('resumes incomplete generation without creating a new session', async () => {
    session = { ...session, clips_done: 1, generation_complete: false, trial: null }; mount(); await userEvent.click(await screen.findByRole('button', { name: 'Continue generating' })); await screen.findByText('Generating samples'); expect(vi.mocked(fetch).mock.calls.some(([url, init]) => url === '/api/abx/sessions' && init?.method === 'POST')).toBe(false); expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/abx/sessions/demo/generate')).toBe(true);
  });
  it('retains setup text when navigating away to change synthesis settings', async () => {
    const view = mount('/abx?b=%2Fadapters%2Fvoice'); await userEvent.type(await screen.findByRole('textbox', { name: 'Test sentences' }), 'Keep this listening plan'); view.unmount(); mount('/abx'); expect(await screen.findByRole('textbox', { name: 'Test sentences' })).toHaveValue('Keep this listening plan'); expect(screen.getByRole('combobox', { name: 'Adapter B' })).toHaveTextContent('Voice adapter');
  });
  it('creates from current decoding settings, then starts a separate generation job', async () => {
    localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify({ ...defaultForm(), temperature: 1.2 })); mount('/abx?b=%2Fadapters%2Fvoice');
    await userEvent.click(await screen.findByRole('combobox', { name: 'Saved reference voice' })); await userEvent.click(await screen.findByRole('option', { name: 'Reference A' })); await userEvent.type(screen.getByRole('textbox', { name: 'Test sentences' }), 'First\nSecond'); await userEvent.click(screen.getByRole('button', { name: 'Create and generate' }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/generate'))).toBe(true)); const call = vi.mocked(fetch).mock.calls.find(([url, init]) => url === '/api/abx/sessions' && init?.method === 'POST'); expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({ adapter_a: '', adapter_b: '/adapters/voice', voice: '/voices/a.wav', repetitions: 2, settings: { temperature: 1.2, use_bf16: true } });
  });
});

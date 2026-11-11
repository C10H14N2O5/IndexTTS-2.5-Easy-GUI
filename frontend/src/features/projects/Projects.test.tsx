import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { useEffect, useImperativeHandle, type Ref } from 'react';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { bootstrapKey } from '@/app/bootstrap';
import { applyEvent, useLive } from '@/api/live';
import { setLanguage } from '@/i18n';
import { testBootstrap } from '@/features/synthesis/testFixtures';
import { adapterFolders, checkpointsFixture, posthocFixture, posthocJob, projectFixture, idleTraining, trainingDefaults } from './testFixtures';
import ProjectsPage from './ProjectsPage';
import ProjectLayout from './ProjectLayout';
import TrainingPage from './TrainingPage';
import CorpusPage from './CorpusPage';
import ValidationPage from './ValidationPage';
import OutputsPage from './OutputsPage';
import type { Project } from './api';

const guard = vi.hoisted(() => ({ allow: true, locked: false, order: [] as string[] }));
vi.mock('@/features/proofread/ProofreadPanel', () => ({ ProofreadPanel: function MockPanel(props: { project: string; directory: string; role: string; language: string; ref?: Ref<{ prepareContextChange: (commit: () => Promise<unknown>) => Promise<boolean> }>; onContextLockedChange?: (locked: boolean) => void; onUpdated?: () => void }) {
  useImperativeHandle(props.ref, () => ({ prepareContextChange: async commit => { guard.order.push('guard'); if (!guard.allow) return false; try { await commit(); return true; } catch { return false; } } }));
  const onContextLockedChange = props.onContextLockedChange;
  useEffect(() => { onContextLockedChange?.(guard.locked); return () => onContextLockedChange?.(false); }, [onContextLockedChange]);
  return <div><div data-testid="proofreader">{JSON.stringify({ project: props.project, directory: props.directory, role: props.role, language: props.language })}</div><button onClick={() => props.onContextLockedChange?.(false)}>Resolve mock edits</button><button onClick={() => { project.dataset_status = { ...project.dataset_status, stale: true, message: 'Training manifest needs updating.' }; props.onUpdated?.(); }}>Mock saved transcript</button></div>;
} }));
let client: QueryClient; let project: Project; let metricSeries: { label: string; points: [number, number][]; count: number }[] = [];
beforeEach(() => {
  setLanguage('en_US'); localStorage.clear(); project = projectFixture(); guard.allow = true; guard.locked = false; guard.order = [];
  useLive.setState({ ready: true, connection: 'open', jobs: {}, jobOrder: [], training: null, export: null });
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } }); client.setQueryData(bootstrapKey, testBootstrap());
  Element.prototype.scrollIntoView = vi.fn(); Element.prototype.hasPointerCapture = vi.fn(() => false); Element.prototype.setPointerCapture = vi.fn(); Element.prototype.releasePointerCapture = vi.fn(); metricSeries = [];
  vi.stubGlobal('ResizeObserver', class { observe = vi.fn(); unobserve = vi.fn(); disconnect = vi.fn(); });
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (url === '/api/training/defaults') return Response.json(trainingDefaults);
    if (url === '/api/training/status') return Response.json(idleTraining);
    if (url === '/api/training/start') return Response.json({ ...idleTraining, running: true, status: 'running', project: 'voice', total: 1000 });
    const adapter = new URL(url, 'http://local').searchParams.get('adapter') ?? 'voice';
    if (url.includes('/checkpoints')) return Response.json(checkpointsFixture(adapter));
    if (url.includes('/metrics')) return Response.json({ project: 'voice', adapter, logdir: '', available: metricSeries.length > 0, error: '', series: metricSeries });
    if (url.endsWith('/adapters')) return Response.json(adapterFolders);
    if (url.includes('/validation-results')) return Response.json(posthocFixture(adapter));
    if (url.endsWith('/validation-runs')) return Response.json({ job: posthocJob }, { status: 202 });
    if (url.startsWith('/api/jobs/')) return Response.json(useLive.getState().jobs[posthocJob.id] ?? posthocJob);
    if (init?.method === 'PUT') { const body = JSON.parse(String(init.body)); if (String(url).endsWith('/training')) project = { ...project, training: body }; else if (String(url).endsWith('/validation')) project = { ...project, validation: { ...project.validation, ...body } }; else project = { ...project, ...body }; return Response.json(project); }
    if (init?.method === 'DELETE') return Response.json({ deleted: 'voice' });
    if (url === '/api/projects/voice/clone') return Response.json({ ...project, project_name: 'voice_copy' });
    if (url === '/api/projects') return Response.json(init?.method === 'POST' ? { ...project, project_name: 'new_voice' } : [project]);
    if (url.startsWith('/api/projects/')) return Response.json(project);
    throw new Error(`Unexpected request ${url}`);
  }));
});
afterEach(() => { client.clear(); vi.unstubAllGlobals(); });
function mount(path = '/projects/voice/training') {
  const router = createMemoryRouter([{ path: '/projects', element: <ProjectsPage/> }, { path: '/projects/:project', element: <ProjectLayout/>, children: [{ path: 'training', element: <TrainingPage/> }, { path: 'corpus', element: <CorpusPage/> }, { path: 'validation', element: <ValidationPage/> }, { path: 'outputs', element: <OutputsPage/> }] }], { initialEntries: [path] });
  render(<QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider>); return router;
}
describe('project workbench', () => {
  it('sends the edited seed in persisted config before starting training', async () => {
    mount(); await screen.findByText('Training setup'); await userEvent.click(screen.getByRole('button', { name: /Advanced hyperparameters/ }));
    const seed = screen.getByRole('spinbutton', { name: 'Training seed' }); await userEvent.clear(seed); await userEvent.type(seed, '9876');
    await userEvent.click(screen.getByRole('button', { name: 'Save and start training' }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/training/start')).toBe(true));
    const saves = vi.mocked(fetch).mock.calls.filter(([url, init]) => String(url).endsWith('/training') && init?.method === 'PUT');
    expect(JSON.parse(String(saves[0]?.[1]?.body))).toMatchObject({ seed: 9876, learning_rate: .00004, lora_alpha: 32, grad_clip: .5 });
    const calls = vi.mocked(fetch).mock.calls.map(([url]) => url); expect(calls.indexOf('/api/projects/voice/training')).toBeLessThan(calls.indexOf('/api/training/start'));
  });
  it('blocks training for stale training or enabled validation manifests', async () => {
    project.validation.enabled = true; mount();
    expect(await screen.findByRole('button', { name: 'Save and start training' })).toBeDisabled(); expect(screen.getAllByText('Prepare validation data or disable validation first.').length).toBeGreaterThan(0);
    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/training/start')).toBe(false);
  });
  it('requires confirmation to discard a dirty training form when navigating', async () => {
    const router = mount(); const adapter = await screen.findByRole('textbox', { name: 'Adapter name' }); await userEvent.type(adapter, '_changed');
    await userEvent.click(screen.getByRole('link', { name: 'Outputs' })); expect(await screen.findByRole('dialog', { name: 'Keep your training changes?' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Stay here' })); expect(router.state.location.pathname).toContain('training'); expect(adapter).toHaveValue('voice_changed');
    await userEvent.click(screen.getByRole('link', { name: 'Outputs' })); await userEvent.click(screen.getByRole('button', { name: 'Discard changes' })); await waitFor(() => expect(router.state.location.pathname).toContain('outputs'));
  });
  it('save-and-leave persists changes then follows the blocked navigation', async () => {
    const router = mount(); await userEvent.type(await screen.findByRole('textbox', { name: 'Adapter name' }), '_saved'); await userEvent.click(screen.getByRole('link', { name: 'Outputs' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Save and leave' })); await waitFor(() => expect(router.state.location.pathname).toContain('outputs')); expect(project.training.adapter_name).toBe('voice_saved');
  });
  it('prefills a resume checkpoint from the outputs deep link', async () => {
    mount('/projects/voice/training?resume=D%3A%2Fout%2Fcheckpoint-100'); await screen.findByText('Training setup'); expect(screen.getByRole('combobox', { name: 'Resume checkpoint' })).toHaveTextContent('checkpoint-100');
  });
  it('resets parameters only after explicit confirmation', async () => {
    project.training.seed = 42; mount(); await screen.findByText('Training setup'); await userEvent.click(screen.getByRole('button', { name: 'Reset defaults' }));
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Cancel' })); await userEvent.click(screen.getByRole('button', { name: /Advanced hyperparameters/ })); expect(screen.getByRole('spinbutton', { name: 'Training seed' })).toHaveValue(42);
    await userEvent.click(screen.getByRole('button', { name: 'Reset defaults' })); await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Reset defaults' })); expect(screen.getByRole('spinbutton', { name: 'Training seed' })).toHaveValue(1234);
  });
  it('passes the exact project language and training role to proofreading', async () => {
    mount('/projects/voice/corpus'); expect(await screen.findByTestId('proofreader')).toHaveTextContent('"role":"training"'); expect(screen.getByTestId('proofreader')).toHaveTextContent('"language":"ZH"'); expect(screen.getByTestId('proofreader')).toHaveTextContent('D:/corpus');
  });
  it('refreshes project readiness after a saved transcript callback settles', async () => {
    mount('/projects/voice/corpus'); await screen.findByTestId('proofreader'); expect(screen.getByText('Corpus ready')).toBeInTheDocument(); await userEvent.click(screen.getByRole('button', { name: 'Mock saved transcript' })); expect(await screen.findByText('Corpus needs attention')).toBeInTheDocument(); expect(screen.getByText('Training manifest needs updating.')).toBeInTheDocument();
  });
  it('advertises the default validation folder but binds it only on save', async () => {
    mount('/projects/voice/validation'); expect(await screen.findByRole('textbox', { name: 'Validation audio folder' })).toHaveValue('D:/validation/voice'); expect(screen.queryByTestId('proofreader')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('switch', { name: 'Enable validation loss' })); await userEvent.click(screen.getByRole('button', { name: 'Save validation' }));
    expect(await screen.findByTestId('proofreader')).toHaveTextContent('"role":"validation"'); expect(project.validation).toMatchObject({ source_folder: 'D:/validation/voice', enabled: true, interval_steps: 0 }); expect(project.source_folder).toBe('D:/corpus');
  });
  it('requires the exact project name before deleting', async () => {
    mount('/projects/voice/outputs'); await userEvent.click(await screen.findByRole('button', { name: 'Delete project' })); const dialog = screen.getByRole('dialog'); const confirm = within(dialog).getByRole('button', { name: 'Delete project' }); expect(confirm).toBeDisabled();
    await userEvent.type(within(dialog).getByRole('textbox'), 'wrong'); expect(confirm).toBeDisabled(); await userEvent.clear(within(dialog).getByRole('textbox')); await userEvent.type(within(dialog).getByRole('textbox'), 'voice'); await userEvent.click(confirm);
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, init]) => url === '/api/projects/voice' && init?.method === 'DELETE')).toBe(true));
  });
  it.each(['corpus', 'validation'])('does not mutate %s binding when the proofreader preflight is cancelled or rejected', async page => {
    project.validation.source_folder = 'D:/old-validation'; guard.allow = false; mount(`/projects/voice/${page}`); const folder = await screen.findByRole('textbox', { name: page === 'corpus' ? 'Source audio folder' : 'Validation audio folder' }); await userEvent.clear(folder); await userEvent.type(folder, 'D:/new');
    await userEvent.click(screen.getByRole('button', { name: page === 'corpus' ? 'Save folder' : 'Save validation' })); await waitFor(() => expect(guard.order).toEqual(['guard'])); expect(vi.mocked(fetch).mock.calls.some(([, init]) => init?.method === 'PUT')).toBe(false); expect(screen.getByTestId('proofreader')).not.toHaveTextContent('D:/new'); expect(folder).toHaveValue('D:/new');
  });
  it('guards manifest builds as well as binding changes', async () => {
    guard.allow = false; mount('/projects/voice/corpus'); await userEvent.click(await screen.findByRole('button', { name: 'Build training manifest' })); await waitFor(() => expect(guard.order).toEqual(['guard'])); expect(vi.mocked(fetch).mock.calls.some(([, init]) => init?.method === 'PUT')).toBe(false);
  });
  it('commits only after preflight and retains the old binding after a failed PUT', async () => {
    const original = vi.mocked(fetch).getMockImplementation()!; vi.mocked(fetch).mockImplementation(async (url, init) => { if (init?.method === 'PUT') { guard.order.push('put'); return Response.json({ error: { code: 'bad_request', message: 'Folder save failed' } }, { status: 400 }); } return original(url, init); });
    mount('/projects/voice/corpus'); const folder = await screen.findByRole('textbox', { name: 'Source audio folder' }); await userEvent.clear(folder); await userEvent.type(folder, 'D:/new'); await userEvent.click(screen.getByRole('button', { name: 'Save folder' })); await screen.findByText('Folder save failed'); expect(guard.order).toEqual(['guard', 'put']); expect(folder).toHaveValue('D:/new'); expect(screen.getByTestId('proofreader')).toHaveTextContent('D:/corpus'); expect(project.source_folder).toBe('D:/corpus');
  });
  it('disables project management while proofreader edits or tasks lock its context, then unlocks', async () => {
    guard.locked = true; mount('/projects/voice/corpus'); await screen.findByTestId('proofreader'); const clone = screen.getByRole('button', { name: 'Clone project' }); const remove = screen.getByRole('button', { name: 'Delete project' }); expect(clone).toBeDisabled(); expect(remove).toBeDisabled(); await userEvent.click(clone); await userEvent.click(remove); expect(vi.mocked(fetch).mock.calls.some(([, init]) => init?.method === 'DELETE' || init?.method === 'POST')).toBe(false);
    await userEvent.click(screen.getByRole('button', { name: 'Resolve mock edits' })); expect(screen.getByRole('button', { name: 'Clone project' })).toBeEnabled(); expect(screen.getByRole('button', { name: 'Delete project' })).toBeEnabled();
  });
  it('disables clone and deletion until dirty training settings are saved', async () => {
    mount(); await userEvent.type(await screen.findByRole('textbox', { name: 'Adapter name' }), '_dirty'); expect(screen.getByRole('button', { name: 'Clone project' })).toBeDisabled(); expect(screen.getByRole('button', { name: 'Delete project' })).toBeDisabled(); await userEvent.click(screen.getByRole('button', { name: 'Save settings' })); await waitFor(() => expect(screen.getByRole('button', { name: 'Clone project' })).toBeEnabled()); expect(screen.getByRole('button', { name: 'Delete project' })).toBeEnabled();
  });
  it('creates a project from the list with explicit pending mutation', async () => {
    mount('/projects'); await userEvent.click(screen.getByRole('button', { name: 'New project' })); await userEvent.type(screen.getByRole('textbox', { name: 'Project name' }), 'new_voice'); await userEvent.click(screen.getByRole('button', { name: 'Create project' }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, init]) => url === '/api/projects' && init?.method === 'POST')).toBe(true));
  });
});

const requests = (match: (url: string, init?: RequestInit) => boolean) => vi.mocked(fetch).mock.calls.filter(([url, init]) => match(String(url), init));
const rowOf = (text: string) => screen.getByText(text).closest('tr')!;
describe('early stop and dense validation', () => {
  it('saves early stop settings and blocks a stop step at or beyond the planned schedule', async () => {
    mount(); await screen.findByText('Training setup'); await userEvent.click(screen.getByRole('switch', { name: 'Early stop' }));
    const stop = screen.getByRole('spinbutton', { name: 'Stop at step' }); expect(screen.getByText(/still follows the planned 1000 steps\./)).toBeInTheDocument();
    await userEvent.clear(stop); await userEvent.type(stop, '1000');
    expect(screen.getAllByText('The early stop step must be smaller than the planned 1000 training steps.').length).toBeGreaterThan(0); expect(screen.getByRole('button', { name: 'Save settings' })).toBeDisabled(); expect(screen.getByRole('button', { name: 'Save and start training' })).toBeDisabled();
    await userEvent.clear(stop); await userEvent.type(stop, '400'); await userEvent.click(screen.getByRole('button', { name: 'Save settings' }));
    await waitFor(() => expect(project.training).toMatchObject({ early_stop_enabled: true, early_stop_step: 400 }));
  });
  it('estimates the planned schedule in epoch mode from usable clips', async () => {
    project.training = { ...project.training, mode: 'Epochs', early_stop_enabled: true, early_stop_step: 40 }; mount(); await screen.findByText('Training setup');
    expect(screen.getByText(/planned 40 steps \(estimated from 12 usable clips\)/)).toBeInTheDocument(); expect(screen.getAllByText('The early stop step must be smaller than the planned 40 training steps.').length).toBeGreaterThan(0);
  });
  it('rejects an early stop step that the resume checkpoint already passed', async () => {
    project.training = { ...project.training, early_stop_enabled: true, early_stop_step: 150, resume_checkpoint: 'D:/out/voice/checkpoint-step-000200' }; mount();
    expect((await screen.findAllByText('The resume checkpoint is already at step 200; choose an early stop step after it.')).length).toBeGreaterThan(0); expect(screen.getByRole('button', { name: 'Save settings' })).toBeDisabled();
  });
  it('shows the LR schedule length when a run stops early', async () => {
    useLive.setState({ training: { ...idleTraining, running: true, status: 'running', project: 'voice', adapter: 'voice', step: 10, total: 400, schedule_total: 1000 } }); mount();
    expect(await screen.findByText('10 / 400 steps')).toBeInTheDocument(); expect(screen.getByText('LR schedule 1000 steps')).toBeInTheDocument();
  });
  it('round-trips the dense validation range and rejects an inverted range', async () => {
    project.validation = { ...project.validation, dense_enabled: true, dense_start_step: 200, dense_end_step: 400, dense_interval_steps: 20 }; mount('/projects/voice/validation');
    const end = await screen.findByRole('spinbutton', { name: 'Range end step' }); expect(screen.getByRole('spinbutton', { name: 'Range start step' })).toHaveValue(200); expect(end).toHaveValue(400); expect(screen.getByRole('spinbutton', { name: 'Range interval (steps)' })).toHaveValue(20);
    await userEvent.clear(end); await userEvent.type(end, '100'); expect(screen.getByText('The dense range must start at or before its end step.')).toBeInTheDocument(); expect(screen.getByRole('button', { name: 'Save validation' })).toBeDisabled(); expect(screen.getByRole('button', { name: 'Build validation manifest' })).toBeDisabled();
    await userEvent.clear(end); await userEvent.type(end, '600'); await userEvent.click(screen.getByRole('button', { name: 'Save validation' }));
    await waitFor(() => expect(project.validation).toMatchObject({ dense_enabled: true, dense_start_step: 200, dense_end_step: 600, dense_interval_steps: 20 }));
  });
  it('requires every dense field once the range is enabled', async () => {
    mount('/projects/voice/validation'); await userEvent.click(await screen.findByRole('switch', { name: 'Dense validation range' }));
    expect(screen.getByText('Enter a start step, end step and interval of at least 1 for the dense range.')).toBeInTheDocument(); expect(screen.getByRole('button', { name: 'Save validation' })).toBeDisabled();
    for (const [name, value] of [['Range start step', '300'], ['Range end step', '500'], ['Range interval (steps)', '25']] as const) await userEvent.type(screen.getByRole('spinbutton', { name }), value);
    await userEvent.click(screen.getByRole('button', { name: 'Save validation' }));
    await waitFor(() => expect(requests((url, init) => url.endsWith('/validation') && init?.method === 'PUT')).toHaveLength(1));
    expect(JSON.parse(String(requests((url, init) => url.endsWith('/validation') && init?.method === 'PUT')[0]?.[1]?.body))).toMatchObject({ dense_enabled: true, dense_start_step: 300, dense_end_step: 500, dense_interval_steps: 25, prepare: false });
  });
});

describe('post-hoc validation loss', () => {
  it('shows post-hoc and training-time validation loss for each checkpoint', async () => {
    metricSeries = [{ label: 'validation/loss', points: [[100, 1.3], [200, 1.2]], count: 2 }]; mount('/projects/voice/outputs'); await screen.findByText('1.2346');
    expect(within(rowOf('checkpoint-step-000100')).getByText('1.3000')).toBeInTheDocument(); expect(within(rowOf('checkpoint-step-000100')).queryByText('Outdated')).not.toBeInTheDocument();
    const late = rowOf('checkpoint-step-000200'); expect(within(late).getByText('1.1000')).toBeInTheDocument(); expect(within(late).getByText('Outdated')).toBeInTheDocument(); expect(within(late).getByText('Early stop')).toBeInTheDocument(); expect(within(late).getByText('1.2000')).toBeInTheDocument();
    expect(screen.getByText('Base model (no adapter)')).toBeInTheDocument(); expect(screen.getByText('2.5000')).toBeInTheDocument(); expect(screen.getByText('6 clips · fingerprint abcdef12')).toBeInTheDocument();
    expect(screen.getByText('The adapter was trained with a different validation set, so its training-time curve is not directly comparable.')).toBeInTheDocument(); expect(screen.getAllByRole('link', { name: 'Resume training' })).toHaveLength(2);
  });
  it('selects missing results and posts the chosen checkpoints, then tracks the job', async () => {
    mount('/projects/voice/outputs'); await screen.findByText('1.2346'); const compute = screen.getByRole('button', { name: 'Compute selected' }); expect(compute).toBeDisabled(); expect(screen.getByText('Select checkpoints or include the base model.')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Select missing' }));
    expect(screen.getByRole('checkbox', { name: 'Select checkpoint-step-000200' })).toBeChecked(); expect(screen.getByRole('checkbox', { name: 'Select final_adapter' })).toBeChecked(); expect(screen.getByRole('checkbox', { name: 'Select checkpoint-step-000100' })).not.toBeChecked();
    await userEvent.click(screen.getByRole('switch', { name: 'Include base model (no adapter)' })); await userEvent.click(screen.getByRole('button', { name: 'Compute selected' }));
    await waitFor(() => expect(requests((url, init) => url.endsWith('/validation-runs') && init?.method === 'POST')).toHaveLength(1));
    expect(JSON.parse(String(requests(url => url.endsWith('/validation-runs'))[0]?.[1]?.body))).toEqual({ adapter: 'voice', checkpoints: ['checkpoint-step-000200', 'final_adapter'], include_baseline: true, recompute: false });
    expect(await screen.findByText('Computing validation loss.')).toBeInTheDocument(); expect(screen.getByRole('button', { name: 'Cancel task' })).toBeInTheDocument();
  });
  it('switches adapters for checkpoints and results and hides resume for other adapters', async () => {
    mount('/projects/voice/outputs'); await screen.findByText('checkpoint-step-000100');
    await userEvent.click(screen.getByRole('combobox', { name: 'Adapter' })); await userEvent.click(await screen.findByRole('option', { name: /^old_voice/ }));
    expect(await screen.findByText('checkpoint-step-000050')).toBeInTheDocument(); expect(screen.queryByText('checkpoint-step-000100')).not.toBeInTheDocument(); expect(screen.queryByRole('link', { name: 'Resume training' })).not.toBeInTheDocument();
    expect(requests(url => url.includes('/validation-results?adapter=old_voice')).length).toBeGreaterThan(0); expect(requests(url => url.includes('/checkpoints?adapter=old_voice')).length).toBeGreaterThan(0);
  });
  it('shows an already running post-hoc job after a refresh and reloads results when it ends', async () => {
    useLive.setState({ jobs: { [posthocJob.id]: posthocJob }, jobOrder: [posthocJob.id] }); mount('/projects/voice/outputs');
    expect(await screen.findByText('Computing validation loss.')).toBeInTheDocument(); expect(screen.getByRole('button', { name: 'Compute selected' })).toBeDisabled();
    const before = requests(url => url.includes('/validation-results')).length;
    act(() => applyEvent('job', { ...posthocJob, state: 'succeeded', progress: 1, finished_at: Date.now() / 1000 }));
    await waitFor(() => expect(requests(url => url.includes('/validation-results')).length).toBeGreaterThan(before)); expect(screen.getByText('Computing validation loss.')).toBeInTheDocument();
  });
});

describe('project transitions', () => {
  it('fades only the tab content and keeps the project header mounted', async () => {
    const animate = vi.fn(() => ({ cancel: vi.fn() }) as unknown as Animation);
    Object.defineProperty(HTMLElement.prototype, 'animate', { configurable: true, writable: true, value: animate });
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: false, media: query, addEventListener() {}, removeEventListener() {} }));
    try {
      const router = mount('/projects/voice/outputs'); const heading = await screen.findByRole('heading', { level: 1, name: 'voice' });
      expect(animate).not.toHaveBeenCalled();
      await userEvent.click(screen.getByRole('link', { name: 'Corpus' })); await waitFor(() => expect(router.state.location.pathname).toBe('/projects/voice/corpus'));
      expect(screen.getByRole('heading', { level: 1, name: 'voice' })).toBe(heading);
      expect(animate).toHaveBeenCalledTimes(1);
      expect(heading.closest('[data-transition-scope]')).toBeNull();
    } finally { delete (HTMLElement.prototype as Partial<HTMLElement>).animate; }
  });
});

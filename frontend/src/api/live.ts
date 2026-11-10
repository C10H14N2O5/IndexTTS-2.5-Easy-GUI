/**
 * Live server state over Server-Sent Events (`GET /api/events?lang=`).
 *
 * The first event is a full `snapshot`; later events replace one slice:
 * `job` (one job record), `status` (one workflow message), `console` (changed
 * lines), `training`, `export`, `system`, `settings`. The browser reconnects
 * automatically; every reconnect receives a fresh snapshot. Messages inside
 * events are already translated for the subscribed language.
 */
import { create } from 'zustand';
import { useShallow } from 'zustand/react/shallow';
import { getLanguage, subscribeLanguage } from '@/i18n';
import type { Schemas } from './client';

export type Job = Schemas['JobInfo'];
export type StatusItem = Schemas['StatusItem'];
export type ConsoleLine = Schemas['ConsoleLine'];
export type Settings = Schemas['Settings'];
export type SystemSnapshot = Schemas['SystemSnapshot'];

/** indextts_training.manager.TrainingManager.snapshot() */
export interface TrainingSnapshot {
  running: boolean;
  failed: boolean;
  step: number;
  total: number;
  epoch: number;
  loss: number | null;
  elapsed: number;
  eta: number;
  message: string;
  output: string;
  started_at: number | null;
  finished_at: number | null;
  output_dir: string;
  project: string;
  adapter: string;
  [key: string]: unknown;
}

/** indextts_training.export.ExportManager.snapshot() */
export interface ExportSnapshot {
  status: 'idle' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled' | string;
  phase: string;
  elapsed: number;
  log: string;
  output: string;
  error: string;
  report: Record<string, unknown>;
  token: string;
  task_id: string;
  logfile: string;
  [key: string]: unknown;
}

export type Connection = 'connecting' | 'open' | 'reconnecting';

interface LiveState {
  connection: Connection;
  /** Set once the first snapshot arrived. */
  ready: boolean;
  /** Increments on every (re)connect snapshot; queries may refetch on change. */
  generation: number;
  jobs: Record<string, Job>;
  jobOrder: string[];
  status: StatusItem[];
  console: ConsoleLine[];
  consoleCurrent: string;
  training: TrainingSnapshot | null;
  export: ExportSnapshot | null;
  system: SystemSnapshot | null;
  settings: Settings | null;
  /** Other provider slices (e.g. future keys) by name. */
  extra: Record<string, unknown>;
}

const CONSOLE_LIMIT = 1400;
const STATUS_LIMIT = 60;
const FINISHED_JOB_LIMIT = 50;

function compactJobs(jobs: Record<string, Job>, order: string[]) {
  let finished = 0;
  const jobOrder = order.filter((id) => jobs[id] &&
    (jobs[id].state === 'running' || ++finished <= FINISHED_JOB_LIMIT));
  return { jobs: Object.fromEntries(jobOrder.map((id) => [id, jobs[id]!])), jobOrder };
}

export const useLive = create<LiveState>(() => ({
  connection: 'connecting',
  ready: false,
  generation: 0,
  jobs: {},
  jobOrder: [],
  status: [],
  console: [],
  consoleCurrent: '',
  training: null,
  export: null,
  system: null,
  settings: null,
  extra: {},
}));

type Listener = (type: string, data: unknown) => void;
const listeners = new Set<Listener>();

/** Observe raw events (e.g. to show toasts or play the completion sound). */
export function onLiveEvent(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function mergeConsole(lines: ConsoleLine[], changed: ConsoleLine[]): ConsoleLine[] {
  if (!changed.length) return lines;
  const next = lines.slice();
  const index = new Map(next.map((line, i) => [line.id, i]));
  for (const line of changed) {
    const at = index.get(line.id);
    if (at === undefined) {
      index.set(line.id, next.length);
      next.push(line);
    } else {
      next[at] = line;
    }
  }
  return next.length > CONSOLE_LIMIT ? next.slice(next.length - CONSOLE_LIMIT) : next;
}

function applySnapshot(data: Record<string, unknown>): void {
  const jobs = (data.jobs as Job[] | undefined) ?? [];
  const consoleData = data.console as { lines: ConsoleLine[]; current: string } | undefined;
  const { jobs: _j, status, console: _c, training, export: exportData, system, settings, ...extra } = data;
  useLive.setState((state) => ({
    ready: true,
    connection: 'open',
    generation: state.generation + 1,
    ...compactJobs(Object.fromEntries(jobs.map((job) => [job.id, job])), jobs.map((job) => job.id)),
    status: (status as StatusItem[] | undefined) ?? [],
    console: consoleData?.lines ?? [],
    consoleCurrent: consoleData?.current ?? '',
    training: (training as TrainingSnapshot | null) ?? null,
    export: (exportData as ExportSnapshot | null) ?? null,
    system: (system as SystemSnapshot | null) ?? null,
    settings: (settings as Settings | null) ?? state.settings,
    extra,
  }));
}

export function applyEvent(type: string, data: unknown): void {
  switch (type) {
    case 'snapshot':
      applySnapshot(data as Record<string, unknown>);
      break;
    case 'job': {
      const job = data as Job;
      useLive.setState((state) => compactJobs(
        { ...state.jobs, [job.id]: job },
        state.jobOrder.includes(job.id) ? state.jobOrder : [job.id, ...state.jobOrder],
      ));
      break;
    }
    case 'status':
      useLive.setState((state) => ({ status: [...state.status, data as StatusItem].slice(-STATUS_LIMIT) }));
      break;
    case 'console': {
      const batch = data as { lines: ConsoleLine[]; current: string; reset?: boolean };
      useLive.setState((state) => ({ console: mergeConsole(batch.reset ? [] : state.console, batch.lines), consoleCurrent: batch.current }));
      break;
    }
    case 'training':
      useLive.setState({ training: data as TrainingSnapshot });
      break;
    case 'export':
      useLive.setState({ export: data as ExportSnapshot });
      break;
    case 'system':
      useLive.setState({ system: data as SystemSnapshot });
      break;
    case 'settings':
      useLive.setState({ settings: data as Settings });
      break;
    case 'heartbeat':
      break;
    default:
      useLive.setState((state) => ({ extra: { ...state.extra, [type]: data } }));
  }
  listeners.forEach((listener) => listener(type, data));
}

let source: EventSource | null = null;
if (import.meta.hot) import.meta.hot.dispose(() => source?.close());

/** Open (or reopen, e.g. after a language change) the event stream. */
export function connectLive(): () => void {
  source?.close();
  const es = new EventSource(`/api/events?lang=${encodeURIComponent(getLanguage())}`);
  source = es;
  useLive.setState((state) => ({ connection: state.ready ? 'reconnecting' : 'connecting' }));
  es.onopen = () => {
    if (source === es) useLive.setState({ connection: 'open' });
  };
  es.onerror = () => {
    if (source === es) useLive.setState({ connection: 'reconnecting' });
  };
  es.onmessage = (message) => {
    if (source !== es) return; // A closing stream may still deliver a queued old-language event.
    try {
      const event = JSON.parse(message.data as string) as { type: string; data: unknown };
      applyEvent(event.type, event.data);
    } catch (error) {
      console.error('Bad event', error);
    }
  };
  return () => {
    es.close();
    if (source === es) source = null;
  };
}

/** Keep one stream open for the app lifetime; reconnect in the new language on switch. */
export function startLive(): () => void {
  let close = connectLive();
  const unsubscribe = subscribeLanguage(() => {
    close();
    close = connectLive();
  });
  return () => {
    unsubscribe();
    close();
  };
}

// Selectors ------------------------------------------------------------------
// Selectors that build arrays must be used through useShallow (see the hooks
// below); a fresh array on every store read would loop React's renderer.

export const selectJobs = (state: LiveState): Job[] =>
  state.jobOrder.map((id) => state.jobs[id]).filter((job): job is Job => Boolean(job));

export const selectRunningJobs = (state: LiveState): Job[] => selectJobs(state).filter((job) => job.state === 'running');

export const selectGpuJob = (state: LiveState): Job | undefined =>
  selectJobs(state).find((job) => job.gpu && job.state === 'running');

export function useJobs(): Job[] {
  return useLive(useShallow(selectJobs));
}

export function useRunningJobs(): Job[] {
  return useLive(useShallow(selectRunningJobs));
}

export function useGpuJob(): Job | undefined {
  return useLive(selectGpuJob);
}

export function useJob(id: string | null | undefined): Job | undefined {
  return useLive((state) => (id ? state.jobs[id] : undefined));
}

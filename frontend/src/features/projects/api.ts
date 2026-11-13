import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useOutletContext } from 'react-router';
import { api, type Schemas } from '@/api/client';
import { applyEvent, useLive, type Job } from '@/api/live';

export type TrainingConfig = Required<Schemas['TrainingConfig']>;
export type Project = Schemas['ProjectInfo'] & { training: TrainingConfig };
export type TrainingStatus = Required<Schemas['TrainingSnapshot']>;
export type Checkpoint = Schemas['CheckpointInfo'];
export type ValidationConfig = Schemas['ValidationConfig'];
export type AdapterFolder = Schemas['AdapterFolder'];
export type PosthocResults = Schemas['PosthocResults'];
export type PosthocItem = Schemas['PosthocItem'];
export type ValidationRunRequest = Schemas['ValidationRunRequest'];
export type DatasetRole = 'training' | 'validation';
export type DatasetSelection = Schemas['DatasetSelection'];
export type DatasetVersion = Schemas['DatasetVersionInfo'];
export type DatasetVersionList = Schemas['DatasetVersionList'];
export type IntegrityReport = Schemas['IntegrityReport'];
export const projectUrl = (name: string) => `/api/projects/${encodeURIComponent(name)}`;
export const versionsUrl = (name: string, role: DatasetRole, suffix = '') => `${projectUrl(name)}/dataset-versions/${role}${suffix}`;
export const versionsKey = (name: string, role: DatasetRole) => ['dataset-versions', name, role] as const;
export function useDatasetVersions(name: string, role: DatasetRole, revision?: number) { return useQuery({ queryKey: [...versionsKey(name, role), revision], queryFn: ({ signal }) => api.get<DatasetVersionList>(versionsUrl(name, role), undefined, signal), enabled: Boolean(name) }); }
/** Label of the dataset a run uses: a frozen version, or the working set. */
export function datasetLabel(selection: { version_id?: string | null; name?: string; revision?: number | null }, working: string) { return selection.version_id ? `${selection.name || selection.version_id}${selection.revision && selection.revision > 1 ? ` (r${selection.revision})` : ''}` : working; }
export const projectRoute = (name: string, page = 'corpus') => `/projects/${encodeURIComponent(name)}/${page}`;
export const projectKey = (name: string) => ['project', name] as const;
export function useProjects() { return useQuery({ queryKey: ['projects'], queryFn: () => api.get<Project[]>('/api/projects') }); }
export function useProject(name: string) { return useQuery({ queryKey: projectKey(name), queryFn: ({ signal }) => api.get<Project>(projectUrl(name), undefined, signal), enabled: Boolean(name) }); }
export function useProjectMutation<T>(name: string, action: (input: T) => Promise<Project>) {
  const client = useQueryClient();
  return useMutation({ mutationFn: action, onMutate: () => client.cancelQueries({ queryKey: projectKey(name) }), onSuccess: (project) => { client.setQueryData(projectKey(name), project); void client.invalidateQueries({ queryKey: ['projects'] }); } });
}
export interface ProjectContext { project: Project; refresh: () => void; setContextLocked: (locked: boolean) => void }
export function useProjectContext() { return useOutletContext<ProjectContext>(); }
export function useTrainingStatus() {
  const live = useLive(s => s.training) as TrainingStatus | null;
  const connection = useLive(s => s.connection);
  const query = useQuery({ queryKey: ['training-status'], queryFn: () => api.get<TrainingStatus>('/api/training/status'), refetchInterval: connection === 'open' ? false : 2000 });
  return { ...query, data: connection === 'open' ? live ?? query.data : query.data ?? live };
}
export function useTrainingDefaults() { return useQuery({ queryKey: ['training-defaults'], queryFn: () => api.get<TrainingConfig>('/api/training/defaults'), staleTime: Infinity }); }
export function useCheckpoints(name: string, adapter?: string) {
  const status = useTrainingStatus().data;
  return useQuery({ queryKey: ['checkpoints', name, adapter, status?.finished_at], queryFn: () => api.get<Checkpoint[]>(`${projectUrl(name)}/checkpoints`, { adapter }), refetchInterval: status?.running && status.project === name ? 5000 : false });
}
export function useMetrics(name: string, adapter?: string) {
  const status = useTrainingStatus().data;
  return useQuery({ queryKey: ['metrics', name, adapter, status?.finished_at], queryFn: () => api.get<Schemas['MetricsResponse']>(`${projectUrl(name)}/metrics`, { adapter, max_points: 2000 }), refetchInterval: status?.running && status.project === name ? 2000 : false });
}
export function useAdapters(name: string) {
  const status = useTrainingStatus().data;
  return useQuery({ queryKey: ['project-adapters', name, status?.finished_at], queryFn: ({ signal }) => api.get<AdapterFolder[]>(`${projectUrl(name)}/adapters`, undefined, signal), enabled: Boolean(name) });
}
export function useValidationResults(name: string, adapter: string, poll = false) { return useQuery({ queryKey: ['validation-results', name, adapter], queryFn: ({ signal }) => api.get<PosthocResults>(`${projectUrl(name)}/validation-results`, { adapter }, signal), enabled: Boolean(name && adapter), refetchInterval: poll ? 3000 : false }); }
export function useStartValidationRun(name: string) { return useMutation({ mutationFn: (body: ValidationRunRequest) => api.post<{ job: Job }>(`${projectUrl(name)}/validation-runs`, body) }); }
/** LR schedule length the server will plan (mirrors services/training.start). */
export function plannedSteps(training: TrainingConfig, validClips: number) { const updates = Math.max(1, Math.ceil(Math.max(1, validClips) / Math.max(1, training.micro_batch * training.grad_accum))); return training.mode === 'Steps' ? training.training_steps : training.training_epochs * updates; }
/** Python's round(): half to even, so warmup matches the trainer exactly. */
export function roundHalfEven(value: number) { const floor = Math.floor(value); const rest = value - floor; return rest > .5 ? floor + 1 : rest < .5 ? floor : floor % 2 === 0 ? floor : floor + 1; }
/** LR warmup length the trainer uses for a planned schedule (mirrors the trainer's max(20, min(100, round(total * 0.05)))). */
export function warmupSteps(planned: number) { return Math.max(20, Math.min(100, roundHalfEven(planned * .05))); }
export function checkpointStep(path: string, checkpoints?: Checkpoint[]) { const known = checkpoints?.find(item => item.path === path)?.step; if (known != null) return known; const suffix = path.split(/[\\/]/).filter(Boolean).pop()?.split('-').pop() ?? ''; return /^\d+$/.test(suffix) ? Number(suffix) : null; }
export function useStopTraining() { return useMutation({ mutationFn: () => api.post<TrainingStatus>('/api/training/stop'), onSuccess: data => applyEvent('training', data) }); }
export type AutoStopStatus = Schemas['AutoStopStatus'];
/** One-point metrics series holding the base model's (no adapter) validation loss; drawn as a reference line. */
export const BASELINE_SERIES = 'validation_baseline/loss';
/** Series for a learning-curve view; auto-stop bookkeeping series never mix into the loss charts. */
export function curveSeries<T extends { label: string }>(series: T[], view: string) { return series.filter(({ label }) => !label.startsWith('auto_stop/') && (view === 'loss' ? ['train/loss_ema', 'validation/loss', 'validation_posthoc/loss', BASELINE_SERIES].includes(label) : view === 'raw' ? label.includes('loss') && !label.includes('ema') && !label.startsWith('validation') : label === 'train/lr')); }
export function stat(values: Record<string, unknown> | undefined, key: string) { const value = Number(values?.[key] ?? 0); return Number.isFinite(value) ? value : 0; }

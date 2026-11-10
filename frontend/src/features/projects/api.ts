import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useOutletContext } from 'react-router';
import { api, type Schemas } from '@/api/client';
import { applyEvent, useLive } from '@/api/live';

export type TrainingConfig = Required<Schemas['TrainingConfig']>;
export type Project = Schemas['ProjectInfo'] & { training: TrainingConfig };
export type TrainingStatus = Required<Schemas['TrainingSnapshot']>;
export type Checkpoint = Schemas['CheckpointInfo'];
export const projectUrl = (name: string) => `/api/projects/${encodeURIComponent(name)}`;
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
export function useStopTraining() { return useMutation({ mutationFn: () => api.post<TrainingStatus>('/api/training/stop'), onSuccess: data => applyEvent('training', data) }); }
export function stat(values: Record<string, unknown> | undefined, key: string) { const value = Number(values?.[key] ?? 0); return Number.isFinite(value) ? value : 0; }

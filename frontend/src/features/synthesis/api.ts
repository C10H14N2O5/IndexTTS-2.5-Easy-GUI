import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type Schemas } from '@/api/client';
import { applyEvent, useJob, useLive, type Job } from '@/api/live';
import { dialoguePayload, synthesisPayload, type SynthesisForm } from './model';

export type Adapter = Schemas['Adapter'];
export type HistoryItem = Schemas['HistoryItem'];
export type SynthesisResult = Schemas['AudioResult'];
export const historyKey = ['history'] as const;
export function useAdapters() { return useQuery({ queryKey: ['adapters'], queryFn: () => api.get<Adapter[]>('/api/adapters') }); }
export function useHistory() { return useQuery({ queryKey: historyKey, queryFn: () => api.get<HistoryItem[]>('/api/history', { limit: 20 }) }); }
export function useDeleteHistory() {
  const client = useQueryClient();
  return useMutation({ mutationFn: (name: string) => api.delete(`/api/history/${encodeURIComponent(name)}`), onSuccess: () => client.invalidateQueries({ queryKey: historyKey }) });
}
export async function generateSpeech(form: SynthesisForm, dialogue: boolean) {
  return api.post<{ job: Job }>(dialogue ? '/api/dialogue' : '/api/synthesis', dialogue ? dialoguePayload(form) : synthesisPayload(form));
}
export async function cancelJob(id: string) {
  const job = await api.post<Job>(`/api/jobs/${encodeURIComponent(id)}/cancel`);
  applyEvent('job', job);
  return job;
}

/** SSE is primary; low-frequency polling recovers responses lost during reconnects. */
export function useTrackedJob() {
  const [id, setId] = useState<string | null>(null);
  const live = useJob(id);
  const query = useQuery({
    queryKey: ['job', id], queryFn: () => api.get<Job>(`/api/jobs/${encodeURIComponent(id!)}`), enabled: Boolean(id),
    retry: false, refetchInterval: () => live && live.state !== 'running' ? false : 1500,
  });
  useEffect(() => {
    if (!query.data) return;
    const current = useLive.getState().jobs[query.data.id];
    if (current && current.state !== 'running' && query.data.state === 'running') return;
    if (current?.state === 'running' && query.data.state === 'running' && (current.progress ?? 0) > (query.data.progress ?? 0)) return;
    applyEvent('job', query.data);
  }, [query.data]);
  const job = live ?? query.data;
  return { job, error: query.error, track: (next: Job) => { applyEvent('job', next); setId(next.id); }, clear: () => setId(null) };
}
export function isSynthesisResult(result: unknown): result is SynthesisResult {
  return Boolean(result && typeof result === 'object' && 'path' in result && typeof result.path === 'string');
}

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type Schemas } from '@/api/client';
import type { Job } from '@/api/live';

export type Voice = Schemas['Voice'];
export type EmotionDescription = Schemas['EmotionDescription'];
export type VoiceInput = Pick<Schemas['SaveVoice'], 'name' | 'source_path'> & Partial<Pick<Schemas['SaveVoice'], 'transcript' | 'overwrite'>>;
export const voicesKey = ['voices'] as const;
export const descriptionsKey = ['emotion-descriptions'] as const;
export function useVoices() { return useQuery({ queryKey: voicesKey, queryFn: () => api.get<Voice[]>('/api/voices') }); }
export function useDescriptions() { return useQuery({ queryKey: descriptionsKey, queryFn: () => api.get<EmotionDescription[]>('/api/emotion-descriptions') }); }
export function useSaveVoice() {
  const client = useQueryClient();
  return useMutation({ mutationFn: (body: VoiceInput) => api.post<Voice>('/api/voices', body), onSuccess: () => client.invalidateQueries({ queryKey: voicesKey }) });
}
export function useRenameVoice() {
  const client = useQueryClient();
  return useMutation({ mutationFn: ({ name, new_name }: { name: string; new_name: string }) => api.patch<Voice>(`/api/voices/${encodeURIComponent(name)}`, { new_name }), onSuccess: () => client.invalidateQueries({ queryKey: voicesKey }) });
}
export function useDeleteVoice() {
  const client = useQueryClient();
  return useMutation({ mutationFn: (name: string) => api.delete(`/api/voices/${encodeURIComponent(name)}`), onSuccess: () => client.invalidateQueries({ queryKey: voicesKey }) });
}
export function useSaveDescription() {
  const client = useQueryClient();
  return useMutation({ mutationFn: (body: { name: string; text: string; overwrite?: boolean }) => api.post<EmotionDescription>('/api/emotion-descriptions', body), onSuccess: () => client.invalidateQueries({ queryKey: descriptionsKey }) });
}
export function useDeleteDescription() {
  const client = useQueryClient();
  return useMutation({ mutationFn: (name: string) => api.delete(`/api/emotion-descriptions/${encodeURIComponent(name)}`), onSuccess: () => client.invalidateQueries({ queryKey: descriptionsKey }) });
}
export function transcribeReference(path: string) { return api.post<{ job: Job }>('/api/asr/transcribe', { path }); }

/**
 * App-wide server data: `/api/bootstrap` (version, settings, capabilities,
 * catalog of choices and defaults) and settings updates.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type Schemas } from '@/api/client';
import { useLive } from '@/api/live';
import { setLanguage, type Language } from '@/i18n';

export type Bootstrap = Schemas['Bootstrap'];
export type Catalog = Schemas['Catalog'];
export type Settings = Schemas['Settings'];
export type SettingsPatch = Schemas['SettingsPatch'];

export const bootstrapKey = ['bootstrap'] as const;

export function useBootstrap() {
  return useQuery({
    queryKey: bootstrapKey,
    queryFn: () => api.get<Bootstrap>('/api/bootstrap'),
    staleTime: Infinity,
  });
}

/** The catalog is static for the server lifetime; throws if used before bootstrap resolved. */
export function useCatalog(): Catalog {
  const { data } = useBootstrap();
  if (!data) throw new Error('useCatalog() used before bootstrap finished');
  return data.catalog;
}

/** Current settings: live SSE value first, bootstrap value as fallback. */
export function useSettings(): Settings | undefined {
  const live = useLive((state) => state.settings);
  const { data } = useBootstrap();
  return live ?? data?.settings;
}

export function useUpdateSettings() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (patch: SettingsPatch) => api.put<Settings>('/api/settings', patch),
    onMutate: (patch) => {
      if (patch.ui_language) setLanguage(patch.ui_language as Language);
    },
    onSuccess: (settings) => {
      useLive.setState({ settings });
      client.setQueryData<Bootstrap>(bootstrapKey, (old) => (old ? { ...old, settings } : old));
    },
  });
}

import { useEffect, useRef } from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router';
import { queryClient } from '@/app/queryClient';
import { router } from '@/app/router';
import { useBootstrap, useSettings } from '@/app/bootstrap';
import { useThemeSync, type ThemeSetting } from '@/app/theme';
import { startLive, useLive } from '@/api/live';
import { getLanguage, setLanguage, useT, type Language } from '@/i18n';
import { errorMessage } from '@/api/client';
import { BootScreen } from '@/app/shell/BootScreen';

function Root() {
  const t = useT();
  const bootstrap = useBootstrap();
  const settings = useSettings();
  const generation = useLive((state) => state.generation);
  useThemeSync(settings?.theme as ThemeSetting | undefined);

  useEffect(() => startLive(), []);

  // The server setting is authoritative for the interface language. The saved
  // value may arrive over SSE before the settings request resolves, so changes
  // after the first one crossfade here as well.
  const languageSynced = useRef(false);
  useEffect(() => {
    const language = settings?.ui_language as Language | undefined;
    if (!language) return;
    if (language !== getLanguage()) setLanguage(language, { animate: languageSynced.current });
    languageSynced.current = true;
  }, [settings?.ui_language]);

  // After a reconnect (e.g. the server restarted) refresh everything cached.
  useEffect(() => {
    if (generation > 1) void queryClient.invalidateQueries();
  }, [generation]);

  if (bootstrap.isPending) return <BootScreen />;
  if (bootstrap.isError)
    return <BootScreen error={errorMessage(bootstrap.error)} onRetry={() => void bootstrap.refetch()} title={t('Cannot start the interface')} />;
  return <RouterProvider router={router} />;
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Root />
    </QueryClientProvider>
  );
}

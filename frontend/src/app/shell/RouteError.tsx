/** Router error boundary: a failed lazy chunk usually means the GUI was updated. */
import { isRouteErrorResponse, useRouteError } from 'react-router';
import { useT } from '@/i18n';
import { BootScreen } from './BootScreen';

export function RouteError() {
  const t = useT();
  const error = useRouteError();
  const detail = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : error instanceof Error
      ? error.message
      : String(error);
  return <BootScreen title={t('This page failed to load')} error={detail} onRetry={() => window.location.reload()} />;
}

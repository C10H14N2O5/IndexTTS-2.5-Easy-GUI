/** Full-screen state while /api/bootstrap loads or after it failed. */
import { useT } from '@/i18n';
import styles from './BootScreen.module.css';

interface BootScreenProps {
  title?: string;
  error?: string;
  onRetry?: () => void;
}

export function BootScreen({ title, error, onRetry }: BootScreenProps) {
  const t = useT();
  return (
    <div className={styles.screen} role={error ? 'alert' : 'status'} aria-live="polite">
      <div className={styles.mark} aria-hidden data-error={error ? '' : undefined} />
      <h1 className={styles.title}>{title ?? t('Starting IndexTTS Easy GUI…')}</h1>
      {error ? <p className={styles.detail}>{error}</p> : null}
      {onRetry ? (
        <button type="button" className={styles.retry} onClick={onRetry}>
          {t('Retry')}
        </button>
      ) : null}
    </div>
  );
}

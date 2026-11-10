import { RefreshCw } from 'lucide-react';
import { Link } from 'react-router';
import { AudioInput, AudioPlayer, Field, IconButton, InlineAlert, Select } from '@/ui';
import { errorMessage, mediaUrl } from '@/api/client';
import { useT } from '@/i18n';
import { useVoices } from '@/features/library/api';
import styles from './Synthesis.module.css';

export function VoicePicker({ path, name, onChange, error, label, compact = false, libraryOnly = false }: {
  path: string; name?: string; onChange: (path: string, name: string, transcript?: string) => void; error?: string; label?: string; compact?: boolean; libraryOnly?: boolean;
}) {
  const t = useT(); const voices = useVoices();
  const selected = voices.data?.find((voice) => voice.path === path);
  return <div className={styles.fields}>
    <Field label={label ?? t('Reference voice')} error={error}>
      <div className={styles.row}>
        <div style={{ flex: 1, minWidth: 180 }}><Select aria-label={label ?? t('Reference voice')} value={selected ? selected.path : '__upload__'} onChange={(value) => {
          const voice = voices.data?.find((item) => item.path === value);
          onChange(voice?.path ?? '', voice?.name ?? '', voice?.transcript ?? '');
        }} options={[{ value: '__upload__', label: libraryOnly ? t('Choose a saved voice') : t('Upload or record a clip') }, ...(voices.data ?? []).map((voice) => ({ value: voice.path, label: voice.name }))]} /></div>
        <IconButton icon={RefreshCw} label={t('Refresh voices')} pending={voices.isFetching} onClick={() => void voices.refetch()} />
      </div>
    </Field>
    {voices.error && <InlineAlert tone="error">{errorMessage(voices.error)}</InlineAlert>}
    {selected ? <AudioPlayer src={mediaUrl(selected.path, selected.modified)} title={selected.name} compact={compact} /> : libraryOnly ? <p className={styles.muted}><Link to="/library">{t('Add a reference voice in the library')}</Link></p> :
      <AudioInput label={t('Reference clip')} hint={compact ? undefined : t('A short, clean recording with one speaker works best.')} value={path ? { path, name: name || path.split(/[\\/]/).pop() || t('Reference clip') } : null} onChange={(audio) => onChange(audio?.path ?? '', audio?.name ?? '', '')} allowRecord allowTrim />}
  </div>;
}

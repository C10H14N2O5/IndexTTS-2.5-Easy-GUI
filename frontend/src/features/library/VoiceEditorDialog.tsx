import { useEffect, useState } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { AudioInput, Button, ConfirmDialog, Dialog, Field, InlineAlert, TextArea, TextInput } from '@/ui';
import { errorMessage, isApiError } from '@/api/client';
import { useT } from '@/i18n';
import { useSaveVoice, type VoiceInput } from './api';
import { TranscribeButton } from './TranscribeButton';
import styles from './Library.module.css';

interface Audio { path: string; name: string; duration?: number | null }
export function VoiceEditorDialog({ open, onOpenChange, initial, onSaved }: {
  open: boolean; onOpenChange: (open: boolean) => void; initial?: { path: string; name?: string; transcript?: string }; onSaved?: (path: string, name: string) => void;
}) {
  const t = useT(); const save = useSaveVoice();
  const [audio, setAudio] = useState<Audio | null>(null);
  const [overwrite, setOverwrite] = useState<VoiceInput | null>(null);
  const [failure, setFailure] = useState('');
  const { register, reset, handleSubmit, control, setValue, formState: { errors } } = useForm<{ name: string; transcript: string }>({ defaultValues: { name: '', transcript: '' } });
  const transcript = useWatch({ control, name: 'transcript' });
  const initialPath = initial?.path, initialName = initial?.name, initialTranscript = initial?.transcript;
  useEffect(() => {
    if (!open) return;
    reset({ name: initialName?.replace(/\.[^.]+$/, '') ?? '', transcript: initialTranscript ?? '' });
    setAudio(initialPath ? { path: initialPath, name: initialName ?? initialPath.split(/[\\/]/).pop() ?? '' } : null);
    setFailure(''); setOverwrite(null);
  }, [open, initialPath, initialName, initialTranscript, reset]);
  async function persist(body: VoiceInput) {
    const result = await save.mutateAsync(body);
    onSaved?.(result.path, result.name); onOpenChange(false);
  }
  async function submit(values: { name: string; transcript: string }) {
    if (!audio) { setFailure(t('Add a reference clip first.')); return; }
    const body = { name: values.name.trim(), source_path: audio.path, transcript: values.transcript.trim(), overwrite: false };
    setFailure('');
    try { await persist(body); }
    catch (error) { if (isApiError(error) && error.status === 409 && error.code === 'conflict') setOverwrite(body); else setFailure(errorMessage(error)); }
  }
  return <>
    <Dialog open={open} onOpenChange={(next) => { if (!save.isPending) onOpenChange(next); }} title={t('Save a reference voice')} description={t('Audio and its optional transcript stay in your local voice library.')} footer={<><Button variant="text" onClick={() => onOpenChange(false)} disabled={save.isPending}>{t('Cancel')}</Button><Button pending={save.isPending} onClick={() => void handleSubmit(submit)()}>{t('Save voice')}</Button></>}>
      <div className={styles.stack}>
        <Field label={t('Voice name')} error={errors.name?.message}><TextInput aria-label={t('Voice name')} autoFocus {...register('name', { required: t('Enter a name.'), validate: (value) => Boolean(value.trim()) || t('Enter a name.') })} /></Field>
        <AudioInput label={t('Reference clip')} value={audio} onChange={setAudio} allowRecord allowTrim disabled={save.isPending} />
        <Field label={t('Reference transcript')} hint={t('Optional. Use the exact words spoken in the recording.')}><TextArea aria-label={t('Reference transcript')} value={transcript} onChange={(event) => setValue('transcript', event.target.value)} minRows={3} /></Field>
        <TranscribeButton path={audio?.path ?? ''} onTranscript={(text) => setValue('transcript', text)} />
        {failure && <InlineAlert tone="error">{failure}</InlineAlert>}
      </div>
    </Dialog>
    <ConfirmDialog open={Boolean(overwrite)} onOpenChange={(next) => { if (!next) setOverwrite(null); }} title={t('Replace the existing voice?')} description={t('A voice with this name already exists. Its audio and transcript will be replaced.')} confirmLabel={t('Replace voice')} tone="danger" onConfirm={async () => { if (overwrite) await persist({ ...overwrite, overwrite: true }); setOverwrite(null); }} />
  </>;
}

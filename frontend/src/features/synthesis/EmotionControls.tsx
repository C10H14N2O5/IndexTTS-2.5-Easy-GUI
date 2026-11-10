import { AudioInput, Button, Disclosure, Field, InlineAlert, ParamField, Select, Switch, TextArea } from '@/ui';
import { errorMessage } from '@/api/client';
import { useCatalog } from '@/app/bootstrap';
import { useT } from '@/i18n';
import { useDescriptions } from '@/features/library/api';
import type { SynthesisForm } from './model';
import styles from './Synthesis.module.css';

export function EmotionControls({ form, onChange, error }: { form: SynthesisForm; onChange: (patch: Partial<SynthesisForm>) => void; error?: string }) {
  const t = useT(); const catalog = useCatalog(); const descriptions = useDescriptions();
  return <Disclosure title={t('Emotion and delivery')} summary={t(form.emotion_mode)} modifiedCount={form.emotion_mode === 'Same as voice reference' ? 0 : 1}>
    <div className={styles.fields}>
      <Field label={t('Emotion source')}><Select aria-label={t('Emotion source')} value={form.emotion_mode} onChange={(value) => onChange({ emotion_mode: value })} options={catalog.emotion_modes.map((mode) => ({ value: mode as SynthesisForm['emotion_mode'], label: t(mode) }))} /></Field>
      {form.emotion_mode !== 'Same as voice reference' && <ParamField label={t('Emotion strength')} value={form.emotion_alpha} defaultValue={0.65} onChange={(value) => onChange({ emotion_alpha: value })} min={0} max={1} step={0.05} />}
      {form.emotion_mode === 'Emotion reference audio' && <Field label={t('Emotion reference')} error={error}><AudioInput label={t('Emotion reference')} value={form.emotion_audio ? { path: form.emotion_audio, name: form.emotion_audio.split(/[\\/]/).pop() ?? '' } : null} onChange={(audio) => onChange({ emotion_audio: audio?.path ?? '' })} allowRecord allowTrim /></Field>}
      {form.emotion_mode === 'Emotion vector' && <>
        <Field label={t('Emotion preset')}><Select aria-label={t('Emotion preset')} value={form.emotion_preset || '__custom__'} onChange={(value) => onChange({ emotion_preset: value, emotion_vector: [...(catalog.emotion_presets[value] ?? form.emotion_vector)] })} options={[{ value: '__custom__', label: t('Custom vector'), disabled: true }, ...Object.keys(catalog.emotion_presets).map((preset) => ({ value: preset, label: t(preset) }))]} /></Field>
        <div className={styles.two}>{catalog.emotions.map((emotion, i) => <ParamField key={emotion} label={t(emotion)} value={form.emotion_vector[i] ?? 0} defaultValue={catalog.emotion_presets[catalog.default_preset]?.[i] ?? 0} onChange={(value) => onChange({ emotion_preset: '', emotion_vector: form.emotion_vector.map((old, index) => index === i ? value : old) })} min={0} max={1} step={0.01} />)}</div>
      </>}
      {form.emotion_mode === 'Emotion description' && <>
        {descriptions.error && <InlineAlert tone="error" action={<Button variant="text" size="sm" onClick={() => void descriptions.refetch()}>{t('Retry')}</Button>}>{errorMessage(descriptions.error)}</InlineAlert>}
        <Field label={t('Saved description')}><Select aria-label={t('Saved description')} value="__custom__" onChange={(path) => { const found = descriptions.data?.find((d) => d.path === path); if (found) onChange({ emotion_text: found.text }); }} options={[{ value: '__custom__', label: t('Choose a description to fill below') }, ...(descriptions.data ?? []).map((d) => ({ value: d.path, label: d.name }))]} /></Field>
        <Field label={t('Delivery description')} hint={t('Describe emotion and prosody. Leave empty to infer it from the target text.')}><TextArea aria-label={t('Delivery description')} value={form.emotion_text} onChange={(event) => onChange({ emotion_text: event.target.value })} minRows={3} /></Field>
      </>}
      {(form.emotion_mode === 'Emotion vector' || form.emotion_mode === 'Emotion description') && <Switch checked={form.random_emotion} onCheckedChange={(value) => onChange({ random_emotion: value })} label={t('Random emotion sampling')} description={t('Adds variation, but may reduce voice-cloning fidelity.')} />}
    </div>
  </Disclosure>;
}

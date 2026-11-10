import { Link } from 'react-router';
import { Button, Disclosure, Field, InlineAlert, ParamField, Select, Switch } from '@/ui';
import { errorMessage } from '@/api/client';
import { useCatalog } from '@/app/bootstrap';
import { useT } from '@/i18n';
import { useAdapters } from './api';
import { DECODING_KEYS, GENERATION_DEFAULTS, modifiedDecodingCount, PARAM_LABELS, type NumericParam, type SynthesisForm } from './model';
import styles from './Synthesis.module.css';

export function GenerationOptions({ form, onChange }: { form: SynthesisForm; onChange: (patch: Partial<SynthesisForm>) => void }) {
  const t = useT(); const catalog = useCatalog(); const adapters = useAdapters();
  function parameter(key: NumericParam) {
    const range = catalog.synthesis_params[key];
    if (!range) return null;
    return <ParamField key={key} label={t(PARAM_LABELS[key])} value={form[key]} defaultValue={range.default} onChange={(value) => onChange({ [key]: value })} min={range.min} max={range.max} step={range.step} unit={key === 'interval_silence' ? t('ms') : undefined} hint={key === 'duration_factor' ? t('Below 1 is faster; above 1 is slower.') : undefined} />;
  }
  const advanced = modifiedDecodingCount(form);
  return <div className={styles.fields}>
    <div className={styles.two}>
      <Field label={t('Speech language')}><Select aria-label={t('Speech language')} value={form.language} onChange={(language) => onChange({ language })} options={catalog.languages.map(({ code, label }) => ({ value: code as SynthesisForm['language'], label: t(label) }))} /></Field>
      <Field label={t('LoRA adapter')} hint={t('Leave off to use the original model.')}><Select aria-label={t('LoRA adapter')} value={form.adapter_path || '__base__'} onChange={(value) => onChange({ adapter_path: value === '__base__' ? '' : value })} options={[{ value: '__base__', label: t('Original model') }, ...(adapters.data ?? []).map((adapter) => ({ value: adapter.path, label: adapter.label })), ...(form.adapter_path && !adapters.data?.some((a) => a.path === form.adapter_path) ? [{ value: form.adapter_path, label: form.adapter_path.split(/[\\/]/).slice(-2).join(' / ') }] : [])]} /></Field>
    </div>
    {adapters.error && <InlineAlert tone="error" action={<Button variant="text" size="sm" onClick={() => void adapters.refetch()}>{t('Retry')}</Button>}>{errorMessage(adapters.error)}</InlineAlert>}
    <Disclosure title={t('Timing and pronunciation')} modifiedCount={Number(form.duration_factor !== GENERATION_DEFAULTS.duration_factor) + Number(form.interval_silence !== GENERATION_DEFAULTS.interval_silence) + Number(!form.text_normalization)}>
      <div className={styles.fields}><div className={styles.two}>{parameter('duration_factor')}{parameter('interval_silence')}</div><Switch checked={form.text_normalization} onCheckedChange={(value) => onChange({ text_normalization: value })} label={t('Text normalization')} description={t('Expand numbers and abbreviations before synthesis.')} /></div>
    </Disclosure>
    <Disclosure title={t('Advanced decoding')} modifiedCount={advanced} summary={t('Sampling, repetition and token limits')}>
      <div className={styles.fields}><Switch checked={form.do_sample} onCheckedChange={(value) => onChange({ do_sample: value })} label={t('Sampling')} description={t('Turn off for deterministic decoding; sampling controls are ignored when off.')} /><div className={styles.two}>{DECODING_KEYS.map(parameter)}</div></div>
    </Disclosure>
    <p className={styles.muted}>{t('Engine options apply to all synthesis tasks.')} <Link to="/system">{t('Configure runtime')}</Link></p>
  </div>;
}

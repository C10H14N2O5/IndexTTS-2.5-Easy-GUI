import { ArrowDown, ArrowUp, Copy, Eraser, Plus, Trash2 } from 'lucide-react';
import { Button, Field, IconButton, InlineAlert, Select, TextArea } from '@/ui';
import { useCatalog } from '@/app/bootstrap';
import { errorMessage } from '@/api/client';
import { useT } from '@/i18n';
import { useDescriptions } from '@/features/library/api';
import { editTurns, MAX_TURNS, type DialogueTurn, type FormErrors } from './model';
import { VoicePicker } from './VoicePicker';
import styles from './Synthesis.module.css';

export function DialogueEditor({ turns, onChange, errors }: { turns: DialogueTurn[]; onChange: (turns: DialogueTurn[]) => void; errors: FormErrors }) {
  const t = useT(); const catalog = useCatalog(); const descriptions = useDescriptions();
  const full = turns.length >= MAX_TURNS;
  function update(index: number, patch: Partial<DialogueTurn>) { onChange(turns.map((turn, i) => i === index ? { ...turn, ...patch } : turn)); }
  return <div className={styles.fields}>
    <div className={styles.toolbar}><span className={styles.muted}>{t('{0} of {1} turns', turns.length, MAX_TURNS)}</span><Button variant="text" size="sm" onClick={() => onChange(editTurns(turns, 'compact'))}>{t('Remove empty turns')}</Button></div>
    {errors.turns && <InlineAlert tone="error">{t(errors.turns)}</InlineAlert>}
    {descriptions.error && turns.some((turn) => turn.emotion_source === 'Emotion Description') && <InlineAlert tone="error" action={<Button variant="text" size="sm" onClick={() => void descriptions.refetch()}>{t('Retry')}</Button>}>{errorMessage(descriptions.error)}</InlineAlert>}
    <div className={styles.turnList}>{turns.map((turn, index) => <article className={styles.turn} key={turn.id} aria-label={t('Turn {0}', index + 1)}>
      <div className={styles.toolbar}><h3 className={styles.turnTitle}>{t('Turn {0}', index + 1)}</h3><div className={styles.row}>
        <IconButton size="sm" icon={ArrowUp} label={t('Move turn up')} disabledReason={index === 0 ? t('Already the first turn.') : undefined} onClick={() => onChange(editTurns(turns, 'up', index))} />
        <IconButton size="sm" icon={ArrowDown} label={t('Move turn down')} disabledReason={index === turns.length - 1 ? t('Already the last turn.') : undefined} onClick={() => onChange(editTurns(turns, 'down', index))} />
        <IconButton size="sm" icon={Copy} label={t('Duplicate turn')} disabledReason={full ? t('Maximum of {0} turns reached.', MAX_TURNS) : undefined} onClick={() => onChange(editTurns(turns, 'duplicate', index))} />
        <IconButton size="sm" icon={Eraser} label={t('Clear turn')} onClick={() => onChange(editTurns(turns, 'clear', index))} />
        <IconButton size="sm" icon={Trash2} label={t('Remove turn')} onClick={() => onChange(editTurns(turns, 'remove', index))} />
      </div></div>
      <VoicePicker path={turn.voice_path} onChange={(voice_path) => update(index, { voice_path })} error={errors[`turns.${index}.voice_path`] ? t(errors[`turns.${index}.voice_path`]!) : undefined} compact libraryOnly />
      <Field label={t('Turn text')}><TextArea aria-label={t('Text for turn {0}', index + 1)} value={turn.text} onChange={(e) => update(index, { text: e.target.value })} minRows={2} showCount placeholder={t('Empty turns are skipped.')} /></Field>
      <div className={styles.two}>
        <Field label={t('Emotion source')}><Select aria-label={t('Emotion source for turn {0}', index + 1)} value={turn.emotion_source} onChange={(emotion_source) => update(index, { emotion_source })} options={catalog.dialogue_emotion_sources.map((source) => ({ value: source as DialogueTurn['emotion_source'], label: t(source) }))} /></Field>
        {turn.emotion_source === 'Emotion Vector Preset' ? <Field label={t('Emotion preset')}><Select aria-label={t('Emotion preset for turn {0}', index + 1)} value={turn.preset} onChange={(preset) => update(index, { preset })} options={Object.keys(catalog.emotion_presets).map((preset) => ({ value: preset, label: t(preset) }))} /></Field> :
          <Field label={t('Saved description')} error={errors[`turns.${index}.description_path`] ? t(errors[`turns.${index}.description_path`]!) : undefined}><Select aria-label={t('Description for turn {0}', index + 1)} value={turn.description_path || '__none__'} onChange={(path) => update(index, { description_path: path === '__none__' ? '' : path })} options={[{ value: '__none__', label: t('Choose a description') }, ...(descriptions.data ?? []).map((item) => ({ value: item.path, label: item.name }))]} /></Field>}
      </div>
      <Button variant="text" size="sm" icon={Plus} disabledReason={full ? t('Maximum of {0} turns reached.', MAX_TURNS) : undefined} onClick={() => onChange(editTurns(turns, 'add', index))}>{t('Insert turn after this')}</Button>
    </article>)}</div>
    <Button variant="outlined" icon={Plus} disabledReason={full ? t('Maximum of {0} turns reached.', MAX_TURNS) : undefined} onClick={() => onChange(editTurns(turns, 'add'))}>{t('Add dialogue turn')}</Button>
  </div>;
}

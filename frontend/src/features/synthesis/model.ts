import { z } from 'zod';
import type { Schemas } from '@/api/client';
import { tk } from '@/i18n';

export const FORM_STORAGE_KEY = 'indextts.synthesis.v2';
export const EMOTION_MODES = ['Same as voice reference', 'Emotion reference audio', 'Emotion vector', 'Emotion description'] as const;
export const CHUNK_MODES = ['None', 'Paragraph/Sentence Auto', 'Periods', 'Paragraphs', 'Lines', 'Speaker turns'] as const;
export const MAX_TURNS = 12;
export const GENERATION_DEFAULTS = {
  duration_factor: 1, interval_silence: 200, temperature: 0.8, top_p: 0.8, top_k: 30,
  num_beams: 3, repetition_penalty: 10, length_penalty: 0, max_mel_tokens: 1500, max_text_tokens: 120,
  do_sample: true, text_normalization: true,
};
export const PARAM_LABELS = {
  duration_factor: tk('Duration factor'), interval_silence: tk('Segment silence'), temperature: tk('Temperature'),
  top_p: tk('Top P'), top_k: tk('Top K'), num_beams: tk('Beam count'), repetition_penalty: tk('Repetition penalty'),
  length_penalty: tk('Length penalty'), max_mel_tokens: tk('Maximum mel tokens'), max_text_tokens: tk('Text tokens per segment'),
};
export type NumericParam = keyof typeof PARAM_LABELS;
export const DECODING_KEYS: NumericParam[] = ['temperature', 'top_p', 'top_k', 'num_beams', 'repetition_penalty', 'length_penalty', 'max_mel_tokens', 'max_text_tokens'];
const generationShape = {
  duration_factor: z.number().min(0.5).max(2).default(1), interval_silence: z.number().int().min(0).max(1000).default(200),
  temperature: z.number().min(0.1).max(2).default(0.8), top_p: z.number().min(0).max(1).default(0.8),
  top_k: z.number().int().min(0).max(200).default(30), num_beams: z.number().int().min(1).max(10).default(3),
  repetition_penalty: z.number().min(0.1).max(30).default(10), length_penalty: z.number().min(-3).max(3).default(0),
  max_mel_tokens: z.number().int().min(50).max(3000).default(1500), max_text_tokens: z.number().int().min(20).max(300).default(120),
  do_sample: z.boolean().default(true), text_normalization: z.boolean().default(true),
};
const turnSchema = z.object({
  id: z.string(), voice_path: z.string().default(''), text: z.string().default(''),
  emotion_source: z.enum(['Emotion Vector Preset', 'Emotion Description']).default('Emotion Vector Preset'),
  preset: z.string().default('Neutral / balanced'), description_path: z.string().default(''),
});
export type DialogueTurn = z.infer<typeof turnSchema>;
export function newTurn(source?: Partial<DialogueTurn>): DialogueTurn {
  // Local row keys, not authorization tokens. Unlike randomUUID(), this API
  // is available on ordinary LAN HTTP origins as well as secure contexts.
  const id = Array.from(crypto.getRandomValues(new Uint32Array(4)), value => value.toString(16).padStart(8, '0')).join('');
  return { voice_path: '', text: '', emotion_source: 'Emotion Vector Preset', preset: 'Neutral / balanced', description_path: '', ...source, id };
}
export const formSchema = z.object({
  ...generationShape,
  voice_path: z.string().default(''), voice_name: z.string().default(''), text: z.string().default(''),
  reference_transcript: z.string().default(''), reference_auto_transcribe: z.boolean().default(false),
  adapter_path: z.string().default(''), language: z.enum(['EN', 'ES', 'JA', 'ZH', 'AR']).default('ZH'),
  emotion_mode: z.enum(EMOTION_MODES).default('Same as voice reference'), emotion_audio: z.string().default(''),
  emotion_alpha: z.number().min(0).max(1).default(0.65), emotion_text: z.string().default(''),
  emotion_vector: z.array(z.number().min(0).max(1)).length(8).default([0, 0, 0, 0, 0, 0, 0, 0.55]),
  emotion_preset: z.string().default('Neutral / balanced'), random_emotion: z.boolean().default(false),
  chunk_mode: z.enum(CHUNK_MODES).default('None'), chunk_silence: z.number().min(0).max(5).default(0.5),
  dialogue_silence: z.number().min(0).max(5).default(0.5), save_outputs: z.boolean().default(true),
  turns: z.array(turnSchema).min(1).max(MAX_TURNS),
});
export type SynthesisForm = z.infer<typeof formSchema>;
export function defaultForm(): SynthesisForm {
  return formSchema.parse({ turns: [newTurn(), newTurn()] });
}

/** Restore known valid fields independently; one bad saved value must not erase the rest. */
export function restoreForm(value: unknown): SynthesisForm {
  const defaults = defaultForm();
  if (!value || typeof value !== 'object' || Array.isArray(value)) return defaults;
  const source = value as Record<string, unknown>;
  const result: Record<string, unknown> = { ...defaults };
  for (const [key, validator] of Object.entries(formSchema.shape)) {
    if (!(key in source)) continue;
    const parsed = validator.safeParse(source[key]);
    if (parsed.success) result[key] = parsed.data;
  }
  return result as SynthesisForm;
}
export function readSavedForm(): SynthesisForm {
  try { return restoreForm(JSON.parse(localStorage.getItem(FORM_STORAGE_KEY) ?? 'null')); }
  catch { return defaultForm(); }
}
export type FormErrors = Record<string, string>;
export function validateForm(form: SynthesisForm, dialogue: boolean): FormErrors {
  const errors: FormErrors = {};
  const parsed = formSchema.safeParse(form);
  if (!parsed.success) for (const issue of parsed.error.issues) errors[issue.path.join('.')] = tk('Check this value.');
  if (!dialogue) {
    if (!form.voice_path) errors.voice_path = tk('Choose a voice or upload a reference clip.');
    if (!form.text.trim()) errors.text = tk('Enter the words you want to hear.');
    if (form.emotion_mode === 'Emotion reference audio' && !form.emotion_audio) errors.emotion_audio = tk('Add an emotion reference clip.');
  } else {
    if (!form.turns.some((turn) => turn.text.trim())) errors.turns = tk('Add text to at least one dialogue turn.');
    form.turns.forEach((turn, i) => {
      if (!turn.text.trim()) return;
      if (!turn.voice_path) errors[`turns.${i}.voice_path`] = tk('Choose a voice for this turn.');
      if (turn.emotion_source === 'Emotion Description' && !turn.description_path) errors[`turns.${i}.description_path`] = tk('Choose an emotion description for this turn.');
    });
  }
  return errors;
}
export function editTurns(turns: DialogueTurn[], action: 'add' | 'duplicate' | 'remove' | 'clear' | 'up' | 'down' | 'compact', index = turns.length - 1): DialogueTurn[] {
  const next = turns.slice();
  if (action === 'compact') return next.filter((turn) => turn.text.trim()).length ? next.filter((turn) => turn.text.trim()) : [newTurn()];
  if (action === 'add' || action === 'duplicate') {
    if (next.length >= MAX_TURNS) return turns;
    next.splice(index + 1, 0, newTurn(action === 'duplicate' ? next[index] : undefined));
  } else if (action === 'remove') {
    if (next.length === 1) return [newTurn()];
    next.splice(index, 1);
  } else if (action === 'clear') next[index] = newTurn();
  else {
    const target = index + (action === 'up' ? -1 : 1);
    const from = next[index]; const to = next[target];
    if (!from || !to) return turns;
    next[index] = to; next[target] = from;
  }
  return next;
}
export function modifiedDecodingCount(form: SynthesisForm): number {
  return DECODING_KEYS.filter((key) => form[key] !== GENERATION_DEFAULTS[key]).length + Number(!form.do_sample);
}
export function synthesisPayload(form: SynthesisForm): Schemas['SynthesisRequest'] {
  const { turns: _turns, dialogue_silence: _silence, voice_name: _name, emotion_preset: _preset, ...payload } = form;
  return { ...payload, text: payload.text.trim(), reference_transcript: payload.reference_transcript.trim() };
}
export function dialoguePayload(form: SynthesisForm): Schemas['DialogueRequest'] {
  return {
    duration_factor: form.duration_factor, interval_silence: form.interval_silence,
    temperature: form.temperature, top_p: form.top_p, top_k: form.top_k, num_beams: form.num_beams,
    repetition_penalty: form.repetition_penalty, length_penalty: form.length_penalty,
    max_mel_tokens: form.max_mel_tokens, max_text_tokens: form.max_text_tokens,
    do_sample: form.do_sample, text_normalization: form.text_normalization,
    language: form.language, adapter_path: form.adapter_path, dialogue_silence: form.dialogue_silence,
    save_outputs: form.save_outputs,
    turns: form.turns.map(({ id: _id, ...turn }) => ({ ...turn, text: turn.text.trim() })),
  };
}

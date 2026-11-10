import { describe, expect, it, vi } from 'vitest';
import { defaultForm, dialoguePayload, editTurns, formSchema, MAX_TURNS, modifiedDecodingCount, newTurn, restoreForm, synthesisPayload, validateForm } from './model';

describe('synthesis form model', () => {
  it('initializes and duplicates rows on LAN HTTP without secure-context randomUUID', () => {
    const getRandomValues = globalThis.crypto.getRandomValues.bind(globalThis.crypto);
    vi.stubGlobal('crypto', { getRandomValues });
    try {
      const form = defaultForm();
      const duplicate = editTurns(form.turns, 'duplicate', 0);
      expect(duplicate).toHaveLength(3);
      expect(new Set(duplicate.map(turn => turn.id)).size).toBe(3);
      expect(duplicate.every(turn => /^[0-9a-f]{32}$/.test(turn.id))).toBe(true);
      expect(restoreForm({ text: 'LAN draft' }).text).toBe('LAN draft');
    } finally {
      vi.unstubAllGlobals();
    }
  });
  it('preserves 1.x widget defaults', () => {
    const form = defaultForm();
    expect(form).toMatchObject({ language: 'ZH', temperature: 0.8, top_p: 0.8, top_k: 30, num_beams: 3, repetition_penalty: 10, length_penalty: 0, max_mel_tokens: 1500, max_text_tokens: 120, duration_factor: 1, interval_silence: 200, emotion_alpha: 0.65, do_sample: true, save_outputs: true, text_normalization: true });
    expect(form.emotion_vector).toEqual([0, 0, 0, 0, 0, 0, 0, 0.55]);
    expect(form.turns).toHaveLength(2);
    expect(form.turns[0]!.id).not.toEqual(form.turns[1]!.id);
  });
  it('requires target and voice for single synthesis', () => {
    expect(Object.keys(validateForm(defaultForm(), false))).toEqual(['voice_path', 'text']);
    expect(validateForm({ ...defaultForm(), text: 'hello', voice_path: 'voices/a.wav' }, false)).toEqual({});
  });
  it('requires a separate emotion audio only in the reference mode', () => {
    const form = { ...defaultForm(), voice_path: 'a', text: 'hello' };
    expect(validateForm({ ...form, emotion_mode: 'Emotion reference audio' }, false)).toHaveProperty('emotion_audio');
    expect(validateForm({ ...form, emotion_mode: 'Emotion description', emotion_text: '' }, false)).toEqual({});
  });
  it('validates vector length, range and nonfinite values', () => {
    expect(formSchema.safeParse({ ...defaultForm(), emotion_vector: [1] }).success).toBe(false);
    expect(formSchema.safeParse({ ...defaultForm(), temperature: Infinity }).success).toBe(false);
    expect(formSchema.safeParse({ ...defaultForm(), top_k: 2.5 }).success).toBe(false);
    expect(formSchema.safeParse({ ...defaultForm(), emotion_vector: [2, 0, 0, 0, 0, 0, 0, 0] }).success).toBe(false);
  });
  it('skips empty dialogue turns but validates every active turn', () => {
    const form = defaultForm();
    expect(validateForm(form, true)).toHaveProperty('turns');
    form.turns = [newTurn(), newTurn({ text: 'hello' })];
    expect(validateForm(form, true)).toHaveProperty('turns.1.voice_path');
    form.turns[1]!.voice_path = 'voices/a.wav';
    expect(validateForm(form, true)).toEqual({});
    form.turns[1]!.emotion_source = 'Emotion Description';
    expect(validateForm(form, true)).toHaveProperty('turns.1.description_path');
  });
  it('restores valid fields independently and strips unknown saved settings', () => {
    const restored = restoreForm({ text: 'keep me', top_k: -99, max_mel_tokens: 'invalid', newField: true, language: 'EN' });
    expect(restored.text).toBe('keep me'); expect(restored.language).toBe('EN');
    expect(restored.top_k).toBe(30); expect(restored.max_mel_tokens).toBe(1500);
    expect(restored).not.toHaveProperty('newField');
    expect(restoreForm(null).turns).toHaveLength(2);
  });
  it('counts modified decoding values without counting unrelated text edits', () => {
    expect(modifiedDecodingCount(defaultForm())).toBe(0);
    expect(modifiedDecodingCount({ ...defaultForm(), temperature: 1, do_sample: false, text: 'Hi' })).toBe(2);
  });
  it('sends single inputs without local-only fields or dialogue state', () => {
    const payload = synthesisPayload({ ...defaultForm(), text: '  test  ', voice_name: 'A', reference_transcript: ' words ' });
    expect(payload.text).toBe('test'); expect(payload.reference_transcript).toBe('words');
    expect(payload).not.toHaveProperty('turns'); expect(payload).not.toHaveProperty('voice_name'); expect(payload).not.toHaveProperty('emotion_preset');
  });
  it('sends dialogue inputs with global decoding but without single emotion controls', () => {
    const payload = dialoguePayload({ ...defaultForm(), turns: [newTurn({ text: '  test  ', voice_path: 'voice.wav' })] });
    expect(payload.turns[0]).toMatchObject({ text: 'test', voice_path: 'voice.wav' });
    expect(payload.turns[0]).not.toHaveProperty('id'); expect(payload).toHaveProperty('temperature', 0.8); expect(payload).not.toHaveProperty('emotion_alpha');
  });
});

describe('dialogue turn editing', () => {
  it('never exceeds twelve turns', () => {
    const turns = Array.from({ length: MAX_TURNS }, () => newTurn());
    expect(editTurns(turns, 'add')).toBe(turns); expect(editTurns(turns, 'duplicate', 0)).toBe(turns);
  });
  it('duplicates content but assigns a new identity', () => {
    const original = newTurn({ text: 'Hello', voice_path: 'voice.wav' });
    const result = editTurns([original], 'duplicate', 0);
    expect(result[1]).toMatchObject({ text: 'Hello', voice_path: 'voice.wav' }); expect(result[1]!.id).not.toBe(original.id);
  });
  it('reorders entire rows without mutating the source', () => {
    const rows = [newTurn({ text: 'A' }), newTurn({ text: 'B' })];
    expect(editTurns(rows, 'down', 0).map((r) => r.text)).toEqual(['B', 'A']); expect(rows.map((r) => r.text)).toEqual(['A', 'B']);
    expect(editTurns(rows, 'up', 0)).toBe(rows); expect(editTurns(rows, 'down', 1)).toBe(rows);
  });
  it('clears the last turn rather than removing every row', () => {
    const result = editTurns([newTurn({ text: 'A' })], 'remove', 0);
    expect(result).toHaveLength(1); expect(result[0]!.text).toBe('');
  });
  it('compacts blank rows and retains content order', () => {
    expect(editTurns([newTurn({ text: 'A' }), newTurn({ text: ' ' }), newTurn({ text: 'B' })], 'compact').map((r) => r.text)).toEqual(['A', 'B']);
    expect(editTurns([newTurn()], 'compact')).toHaveLength(1);
  });
});

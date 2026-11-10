import { tk } from '@/i18n';
import { GENERATION_DEFAULTS, type SynthesisForm } from '@/features/synthesis/model';
export const SCORE_LABELS = [tk('Sample 1 much better'), tk('Sample 1 slightly better'), tk('About the same'), tk('Sample 2 slightly better'), tk('Sample 2 much better')];
export const VERDICTS = { a_better: tk('Adapter A is preferred'), b_better: tk('Adapter B is preferred'), no_difference: tk('No significant difference yet') };
export function comparisonPlan(text: string, repetitions: number) { const sentences = text.split(/\r?\n/).filter(line => line.trim()).length; return { sentences, trials: sentences * repetitions, clips: sentences * repetitions * 2 }; }
export function decodingSettings(form: SynthesisForm) { return Object.fromEntries(Object.keys(GENERATION_DEFAULTS).map(key => [key, form[key as keyof SynthesisForm]])); }
export const COMPARISON_DRAFT_KEY = 'indextts.abx.draft.v2';
export interface ComparisonDraft { a: string; b: string; voice: string; name: string; texts: string; repetitions: number; language: string }
export function readComparisonDraft(): Partial<ComparisonDraft> {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(COMPARISON_DRAFT_KEY) ?? 'null');
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
    const source = value as Record<string, unknown>; const result: Partial<ComparisonDraft> = {};
    for (const key of ['a', 'b', 'voice', 'name', 'texts', 'language'] as const) if (typeof source[key] === 'string') result[key] = source[key];
    if (typeof source.repetitions === 'number' && Number.isInteger(source.repetitions) && source.repetitions >= 1 && source.repetitions <= 5) result.repetitions = source.repetitions;
    return result;
  } catch { return {}; }
}
export interface ComparisonResults { rated: number; total: number; wins_a: number; wins_b: number; ties: number; mean_b: number; stderr: number; p_value: number; verdict: keyof typeof VERDICTS; rows: { index: number; text: string; rep: number; sample_1: string; score: number | null; preference_b: number | null }[] }

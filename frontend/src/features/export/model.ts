import { tk } from '@/i18n';
export const EXPORT_PHASES = {
  idle: tk('Waiting for an export'), starting: tk('Starting CPU exporter'), hashing: tk('Hashing and checking source files'),
  building: tk('Building and merging the CPU model'), checking: tk('Checking tensor structure and values'), saving: tk('Saving the checkpoint'),
  verifying: tk('Verifying saved weights'), publishing: tk('Publishing the completed export'), completed: tk('Validation complete'),
  failed: tk('Export failed'), cancelled: tk('Export cancelled'),
};
export const EXPORT_STATES = { idle: tk('Export idle'), running: tk('Export running'), cancelling: tk('Export cancellation requested'), succeeded: tk('Export complete'), failed: tk('Export failed'), cancelled: tk('Export cancelled') };
export const EXPORT_DRAFT_KEY = 'indextts.export.draft.v2';
export interface ExportDraft { adapter: string; mode: 'browser' | 'path'; filename: string | null; base: string | null; config: string | null; expected: string; folder: string }
export function readExportDraft(): Partial<ExportDraft> {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(EXPORT_DRAFT_KEY) ?? 'null');
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
    const source = value as Record<string, unknown>; const result: Partial<ExportDraft> = {};
    for (const key of ['adapter', 'expected', 'folder'] as const) if (typeof source[key] === 'string') result[key] = source[key];
    for (const key of ['filename', 'base', 'config'] as const) if (typeof source[key] === 'string' || source[key] === null) result[key] = source[key];
    if (source.mode === 'browser' || source.mode === 'path') result.mode = source.mode;
    return result;
  } catch { return {}; }
}
export function exportActive(status: string) { return status === 'running' || status === 'cancelling'; }
export function exportControls(status: string) { return { canStart: !exportActive(status), canCancel: status === 'running' }; }
export function exportProgress(phase: string) { const steps = ['starting', 'hashing', 'building', 'checking', 'saving', 'verifying', 'publishing', 'completed']; const index = steps.indexOf(phase); return index < 0 ? null : (index + 1) / steps.length; }
export function verification(value: unknown) { return value === true ? tk('Yes') : tk('Not verified'); }
export function suggestedValue(custom: string | null, suggestion: string | undefined, fallback = '') { return custom ?? suggestion ?? fallback; }
export function outputDestination(mode: string, folder: string, filename: string) { if (mode === 'browser') return filename; const separator = folder.includes('\\') ? '\\' : '/'; return folder.replace(/[\\/]+$/, '') + separator + filename; }
export function exportDownloads(status: string, downloads: Record<string, string> | undefined) { return status === 'succeeded' ? Object.entries(downloads ?? {}).filter(([kind, url]) => ['model', 'report'].includes(kind) && /^\/exports\/.+\/(?:model|report)\?token=/.test(url)) : []; }

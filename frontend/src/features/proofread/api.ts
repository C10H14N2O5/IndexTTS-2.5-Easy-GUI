import { ApiError, request } from '@/api/client';
import { t } from '@/i18n';

export type DatasetRole = 'training' | 'validation';
export interface ProofreadContext { project: string; directory: string; language: string; role: DatasetRole }
export interface ClipSummary { id: string; name: string; reviewed: boolean; missing_text: boolean; error?: string }
export interface Clip extends Omit<ClipSummary, 'missing_text'> { text: string; version: string }
export interface AsrJob { running: boolean; total: number; completed: number; succeeded: number; skipped: number; failed: number; current?: string; error?: string; cancelled?: boolean }
export interface AsrStatus { available: boolean; message?: string; directory_running?: boolean; job: AsrJob }
export interface ManifestStatus { role?: DatasetRole; available?: boolean; stale: boolean; status?: 'missing' | 'stale' | 'ready'; message?: string; manifest?: string; training_running?: boolean; asr_running?: boolean; stats?: { samples: number; valid: number; missing_text?: number; [key: string]: unknown } }
export interface SessionSnapshot extends ProofreadContext { token: string; last_id: string | null; items: ClipSummary[]; asr?: AsrStatus }
export interface Transcription { id: string; version: string; text: string }
export interface ManifestResult { status: ManifestStatus; meta?: Record<string, unknown> }
export interface ProofreadTransport {
  get<T>(operation: string, query?: Record<string, string>): Promise<T>;
  post<T>(operation: string, body?: unknown): Promise<T>;
  audioUrl(id: string): string;
  setToken(token: string): void;
}
/** Session credentials exist only in this transport instance, never in browser storage. */
export class ProofreadApi implements ProofreadTransport {
  private token = '';
  setToken(token: string) { this.token = token; }
  private headers(): Record<string, string> { return this.token ? { 'X-Session-Token': this.token } : {}; }
  async get<T>(operation: string, query?: Record<string, string>): Promise<T> {
    return request<T>('GET', `/proofread/api/${operation}`, { query, headers: { ...this.headers(), 'Cache-Control': 'no-cache' } });
  }
  async post<T>(operation: string, body: unknown = {}): Promise<T> {
    return request<T>('POST', `/proofread/api/${operation}`, { body, headers: this.headers() });
  }
  audioUrl(id: string) { return `/proofread/api/audio?${new URLSearchParams({ id, token: this.token })}`; }
}
export function proofreadError(error: unknown): { message: string; status: number } {
  if (error instanceof ApiError) return { message: error.status === 403 ? t('The proofreading session expired. Copy any unsaved transcript before reloading the page.') : error.message, status: error.status };
  return { message: error instanceof Error ? error.message : String(error), status: 0 };
}
export function samePath(a: string, b: string) { return a.replace(/\\/g, '/').replace(/\/+$/, '').toLocaleLowerCase() === b.replace(/\\/g, '/').replace(/\/+$/, '').toLocaleLowerCase(); }
export function sameContext(a: ProofreadContext, b: ProofreadContext) { return samePath(a.directory, b.directory) && a.project === b.project && a.language === b.language && a.role === b.role; }
export type ClipFilter = 'all' | 'pending' | 'reviewed' | 'missing';
export function filterClips(items: ClipSummary[], query: string, filter: ClipFilter): ClipSummary[] {
  const search = query.trim().toLocaleLowerCase();
  return items.filter(item => item.name.toLocaleLowerCase().includes(search) && (filter === 'all' || (filter === 'missing' ? item.missing_text : filter === 'reviewed' ? item.reviewed : !item.reviewed)));
}

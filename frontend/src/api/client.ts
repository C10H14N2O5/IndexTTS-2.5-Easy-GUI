/**
 * Thin fetch wrapper for the FastAPI back end.
 *
 * - Sends `X-UI-Language` so server messages come back in the UI language.
 * - Parses the uniform error body `{error: {code, message, detail, ...extra}}`
 *   into an ApiError (status, code, message, detail, extra).
 * - JSON in / JSON out; use `upload()` for multipart.
 */
import { getLanguage, t } from '@/i18n';
import type { components } from './schema';

export type Schemas = components['schemas'];

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: unknown;
  readonly extra: Record<string, unknown>;

  constructor(status: number, code: string, message: string, detail?: unknown, extra: Record<string, unknown> = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.detail = detail;
    this.extra = extra;
  }

  /** 409 with a running GPU job attached (synthesis/dialogue/ABX share one GPU slot). */
  get busyJob(): Schemas['JobInfo'] | undefined {
    return this.extra.busy_job as Schemas['JobInfo'] | undefined;
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

/** Human-readable text for any thrown value. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof TypeError) return t('Cannot reach the GUI server. Check that the console window is still running.');
  if (error instanceof Error) return error.message;
  return String(error);
}

type Query = Record<string, string | number | boolean | null | undefined>;

export function withQuery(path: string, query?: Query): string {
  if (!query) return path;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== '') params.set(key, String(value));
  }
  const text = params.toString();
  return text ? `${path}${path.includes('?') ? '&' : '?'}${text}` : path;
}

async function parse<T>(response: Response): Promise<T> {
  const type = response.headers.get('content-type') ?? '';
  const body: unknown = type.includes('application/json') ? await response.json().catch(() => null) : await response.text();
  if (!response.ok) {
    const error = (body as { error?: unknown } | null)?.error;
    if (error && typeof error === 'object') {
      const { code, message, detail, ...extra } = error as Record<string, unknown>;
      throw new ApiError(response.status, String(code ?? 'error'), String(message ?? response.statusText), detail, extra);
    }
    if (typeof error === 'string') throw new ApiError(response.status, 'error', error);
    throw new ApiError(response.status, 'http_error', t('Request failed ({0}).', response.status));
  }
  return body as T;
}

export interface RequestOptions {
  query?: Query;
  body?: unknown;
  signal?: AbortSignal;
  headers?: Record<string, string>;
}

export async function request<T>(method: string, path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { 'X-UI-Language': getLanguage(), ...options.headers };
  let body: BodyInit | undefined;
  if (options.body instanceof FormData) {
    body = options.body;
  } else if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(options.body);
  }
  const response = await fetch(withQuery(path, options.query), {
    method,
    headers,
    body,
    signal: options.signal,
    credentials: 'same-origin',
  });
  return parse<T>(response);
}

export const api = {
  get: <T>(path: string, query?: Query, signal?: AbortSignal) => request<T>('GET', path, { query, signal }),
  post: <T>(path: string, body?: unknown, query?: Query) => request<T>('POST', path, { body: body ?? {}, query }),
  put: <T>(path: string, body?: unknown, query?: Query) => request<T>('PUT', path, { body: body ?? {}, query }),
  patch: <T>(path: string, body?: unknown) => request<T>('PATCH', path, { body: body ?? {} }),
  delete: <T>(path: string, query?: Query, body?: unknown) => request<T>('DELETE', path, { query, body }),
};

/** Multipart upload with progress (XMLHttpRequest, since fetch has no upload progress). */
export function upload<T>(path: string, form: FormData, onProgress?: (fraction: number) => void, signal?: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', path);
    xhr.setRequestHeader('X-UI-Language', getLanguage());
    xhr.responseType = 'json';
    if (onProgress) xhr.upload.onprogress = (event) => event.lengthComputable && onProgress(event.loaded / event.total);
    xhr.onerror = () => reject(new TypeError('network error'));
    xhr.onabort = () => reject(new DOMException('aborted', 'AbortError'));
    xhr.onload = () => {
      const body = xhr.response as { error?: Record<string, unknown> } | null;
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body as T);
      const { code, message, detail, ...extra } = body?.error ?? {};
      reject(new ApiError(xhr.status, String(code ?? 'error'), String(message ?? t('Upload failed.')), detail, extra));
    };
    signal?.addEventListener('abort', () => xhr.abort());
    xhr.send(form);
  });
}

/** URL for an audio/image file served by /api/media (whitelisted server paths only). */
export function mediaUrl(path: string, version?: string | number): string {
  return withQuery('/api/media', { path, v: version });
}

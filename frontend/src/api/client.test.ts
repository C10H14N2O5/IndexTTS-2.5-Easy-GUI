import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError, errorMessage, mediaUrl, request, upload, withQuery } from './client';
import { setLanguage } from '@/i18n';

beforeEach(() => setLanguage('en_US'));
afterEach(() => vi.unstubAllGlobals());

describe('API transport', () => {
  it('encodes paths, booleans, Unicode and existing query strings', () => {
    expect(withQuery('/api/x?a=1', { path: 'D:/中文 a.wav', none: null, zero: 0, flag: false, empty: '' }))
      .toBe('/api/x?a=1&path=D%3A%2F%E4%B8%AD%E6%96%87+a.wav&zero=0&flag=false');
    expect(mediaUrl('voices/a #.wav', 2)).toBe('/api/media?path=voices%2Fa+%23.wav&v=2');
  });
  it('sends JSON and current UI language with same-origin credentials', async () => {
    const fetcher = vi.fn().mockResolvedValue(Response.json({ ok: true })); vi.stubGlobal('fetch', fetcher);
    await api.put('/api/settings', { theme: 'dark' });
    expect(fetcher).toHaveBeenCalledWith('/api/settings', expect.objectContaining({ method: 'PUT',
      credentials: 'same-origin', body: '{"theme":"dark"}', headers: { 'Content-Type': 'application/json', 'X-UI-Language': 'en_US' } }));
  });
  it('preserves structured busy errors and detail', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ error: { code: 'busy', message: 'GPU busy', detail: 'wait', busy_job: { id: 'j' } } }, { status: 409 })));
    try { await api.post('/api/synthesis'); throw new Error('expected rejection'); }
    catch (error) { expect(error).toBeInstanceOf(ApiError); expect((error as ApiError).busyJob?.id).toBe('j'); expect((error as ApiError).detail).toBe('wait'); }
  });
  it('maps a non-JSON gateway failure to a readable HTTP error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<h1>bad gateway</h1>', { status: 502 })));
    await expect(api.get('/api/bootstrap')).rejects.toThrow('Request failed (502).');
  });
  it('does not set multipart content-type or swallow cancellation', async () => {
    const form = new FormData(); const signal = new AbortController().signal;
    const fetcher = vi.fn().mockRejectedValue(new DOMException('Aborted', 'AbortError')); vi.stubGlobal('fetch', fetcher);
    await expect(request('POST', '/api/uploads', { body: form, signal })).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetcher.mock.calls[0]![1].headers).not.toHaveProperty('Content-Type');
    expect(fetcher.mock.calls[0]![1].signal).toBe(signal);
  });
  it('rejects pre-aborted uploads without starting a network request', async () => {
    const controller = new AbortController(); controller.abort();
    const constructor = vi.fn(); vi.stubGlobal('XMLHttpRequest', constructor);
    await expect(upload('/api/uploads', new FormData(), undefined, controller.signal)).rejects.toMatchObject({ name: 'AbortError' });
    expect(constructor).not.toHaveBeenCalled();
  });
  it('explains fetch network failures without exposing browser internals', () => {
    expect(errorMessage(new TypeError('Failed to fetch'))).toContain('Cannot reach the GUI server');
    expect(errorMessage(new ApiError(400, 'invalid', 'Select a voice.'))).toBe('Select a voice.');
  });
});

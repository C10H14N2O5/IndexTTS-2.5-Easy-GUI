import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { setLanguage } from '@/i18n';
import LibraryPage from './LibraryPage';
let client: QueryClient;
const description = { name: 'gentle.txt', label: 'gentle', path: 'emotion_descriptions/gentle.txt', text: 'Warm and gentle' };
beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class { observe = vi.fn(); unobserve = vi.fn(); disconnect = vi.fn(); });
  setLanguage('en_US'); client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.stubGlobal('fetch', vi.fn(async (url: string, options?: RequestInit) => {
    if (url === '/api/voices') return Response.json([]);
    if (url === '/api/emotion-descriptions') return Response.json(options?.method === 'POST' ? description : [description]);
    if (url.startsWith('/api/emotion-descriptions/')) return Response.json({ deleted: 1 });
    throw new Error(`Unexpected request ${url}`);
  }));
});
afterEach(() => { client.clear(); vi.unstubAllGlobals(); });
function mount() { render(<QueryClientProvider client={client}><MemoryRouter><LibraryPage /></MemoryRouter></QueryClientProvider>); }
describe('library workflows', () => {
  it('offers an actionable empty state instead of a blank table', async () => {
    mount(); expect(await screen.findByText('Make this library yours')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Add your first item' }));
    expect(screen.getByRole('dialog', { name: 'Save a reference voice' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Save voice' }));
    expect(await screen.findByText('Enter a name.')).toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([, options]) => options?.method === 'POST')).toBe(false);
  });
  it('filters descriptions by name without changing their stored text', async () => {
    mount(); await userEvent.click(screen.getByRole('radio', { name: 'Emotion descriptions' }));
    await screen.findByRole('button', { name: /gentle.txt/ });
    await userEvent.type(screen.getByRole('textbox', { name: 'Search library' }), 'nonexistent');
    expect(screen.getByText('No matching items')).toBeInTheDocument();
    await userEvent.clear(screen.getByRole('textbox', { name: 'Search library' }));
    await userEvent.click(screen.getByRole('button', { name: /gentle.txt/ }));
    expect(screen.getByText('Warm and gentle')).toBeInTheDocument();
  });
  it('updates an existing description by stem rather than appending a second extension', async () => {
    mount(); await userEvent.click(screen.getByRole('radio', { name: 'Emotion descriptions' }));
    await userEvent.click(await screen.findByRole('button', { name: /gentle.txt/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Edit description' }));
    const dialog = screen.getByRole('dialog', { name: 'Edit description' });
    expect(within(dialog).getByRole('textbox', { name: 'Description name' })).toHaveValue('gentle');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Save description' }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, options]) => url === '/api/emotion-descriptions' && options?.method === 'POST')).toBe(true));
    const request = vi.mocked(fetch).mock.calls.find(([, options]) => options?.method === 'POST');
    expect(JSON.parse(String(request?.[1]?.body))).toEqual({ name: 'gentle', text: 'Warm and gentle', overwrite: true });
  });
  it('confirms deletion of a single description', async () => {
    mount(); await userEvent.click(screen.getByRole('radio', { name: 'Emotion descriptions' }));
    await userEvent.click(await screen.findByRole('button', { name: /gentle.txt/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Delete description' }));
    const dialog = screen.getByRole('dialog', { name: 'Delete this description?' });
    expect(vi.mocked(fetch).mock.calls.some(([, options]) => options?.method === 'DELETE')).toBe(false);
    await userEvent.click(within(dialog).getByRole('button', { name: 'Delete description' }));
    expect(vi.mocked(fetch).mock.calls.some(([url, options]) => url === '/api/emotion-descriptions/gentle.txt' && options?.method === 'DELETE')).toBe(true);
  });
});

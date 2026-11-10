import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { bootstrapKey } from '@/app/bootstrap';
import { useLive } from '@/api/live';
import { setLanguage } from '@/i18n';
import { testBootstrap } from '@/features/synthesis/testFixtures';
import SystemPage from './SystemPage';
let client: QueryClient;
beforeEach(() => {
  setLanguage('en_US'); useLive.setState({ settings: null, system: null, jobs: {}, jobOrder: [], training: null });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); client.setQueryData(bootstrapKey, testBootstrap());
  vi.stubGlobal('fetch', vi.fn(async (url: string, options?: RequestInit) => url === '/api/system/status' ? Response.json(testBootstrap().system) : options?.method === 'DELETE' ? Response.json({ deleted: 2 }) : Response.json(testBootstrap().settings)));
});
afterEach(() => { client.clear(); vi.unstubAllGlobals(); });
function mount() { render(<QueryClientProvider client={client}><SystemPage /></QueryClientProvider>); }
describe('system preferences and safety', () => {
  it('disables unsupported acceleration and explains why', () => {
    mount(); expect(screen.getByRole('switch', { name: 'Torch compile' })).toBeDisabled(); expect(screen.getByRole('switch', { name: 'Flash attention' })).toBeDisabled();
    expect(screen.getAllByText('No Triton').length).toBeGreaterThan(0);
  });
  it('requires typing DELETE before permanent bulk cleanup', async () => {
    mount(); await userEvent.click(screen.getByRole('button', { name: 'Delete all voices' }));
    const dialog = screen.getByRole('dialog', { name: 'Delete every reference voice?' });
    const confirm = within(dialog).getByRole('button', { name: 'Delete permanently' });
    expect(confirm).toBeDisabled();
    await userEvent.type(within(dialog).getByRole('textbox'), 'DELETE'); expect(confirm).toBeEnabled();
    await userEvent.click(confirm);
    expect(vi.mocked(fetch).mock.calls.some(([url, options]) => url === '/api/voices' && options?.method === 'DELETE')).toBe(true);
  });
  it('requests notification permission only after an explicit user action', async () => {
    const requestPermission = vi.fn().mockResolvedValue('granted');
    vi.stubGlobal('isSecureContext', true);
    vi.stubGlobal('Notification', class { static permission = 'default'; static requestPermission = requestPermission; });
    mount(); expect(requestPermission).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('switch', { name: 'Browser notifications' }));
    expect(requestPermission).toHaveBeenCalledTimes(1);
    expect(vi.mocked(fetch).mock.calls.some(([url, options]) => url === '/api/settings' && options?.method === 'PUT' && JSON.parse(String(options.body)).notifications === true)).toBe(true);
  });
  it('does not send destructive requests when the confirmation is cancelled', async () => {
    mount(); await userEvent.click(screen.getByRole('button', { name: 'Delete outputs' })); await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(vi.mocked(fetch).mock.calls.some(([, options]) => options?.method === 'DELETE')).toBe(false);
  });
});

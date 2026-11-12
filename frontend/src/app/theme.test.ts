import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { applyTheme } from './theme';

type Update = () => void;

/** A controllable stand-in for document.startViewTransition. */
function stubViewTransition() {
  const transitions: { update: Update; skip: ReturnType<typeof vi.fn>; finish: () => void }[] = [];
  const start = vi.fn((update: Update) => {
    let finish!: () => void;
    const finished = new Promise<void>((resolve) => { finish = resolve; });
    const skip = vi.fn();
    transitions.push({ update, skip, finish });
    return { finished, ready: Promise.resolve(), updateCallbackDone: Promise.resolve(), skipTransition: skip } as unknown as ViewTransition;
  });
  Object.defineProperty(document, 'startViewTransition', { configurable: true, value: start });
  return { start, transitions };
}

function stubMedia({ reduce = false, dark = false } = {}) {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: query.includes('reduced-motion') ? reduce : query.includes('dark') ? dark : false,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  stubMedia();
  document.documentElement.dataset.theme = 'light';
  applyTheme('light');
});

afterEach(() => {
  Reflect.deleteProperty(document, 'startViewTransition');
  vi.unstubAllGlobals();
  document.querySelectorAll('.theme-scan').forEach((node) => node.remove());
  delete document.documentElement.dataset.themeTransition;
});

describe('theme sweep', () => {
  it('sweeps to a different theme and cleans up afterwards', async () => {
    const { start, transitions } = stubViewTransition();
    applyTheme('dark', { animate: true });
    expect(start).toHaveBeenCalledTimes(1);
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(document.documentElement).toHaveAttribute('data-theme-transition');
    transitions[0]!.update();
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(document.querySelector('.theme-scan')).toHaveAttribute('aria-hidden', 'true');
    transitions[0]!.finish();
    await settle();
    expect(document.querySelector('.theme-scan')).toBeNull();
    expect(document.documentElement).not.toHaveAttribute('data-theme-transition');
    expect(localStorage.getItem('indextts.theme')).toBe('dark');
  });

  it('does nothing visual when the resolved theme does not change', () => {
    const { start } = stubViewTransition();
    applyTheme('light', { animate: true });
    expect(start).not.toHaveBeenCalled();
  });

  it('switches at once without animate, with reduced motion, without the API or while hidden', () => {
    const { start } = stubViewTransition();
    applyTheme('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');

    stubMedia({ reduce: true });
    applyTheme('light', { animate: true });
    expect(document.documentElement.dataset.theme).toBe('light');

    stubMedia();
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
    applyTheme('dark', { animate: true });
    expect(document.documentElement.dataset.theme).toBe('dark');
    vi.restoreAllMocks();

    Reflect.deleteProperty(document, 'startViewTransition');
    applyTheme('light', { animate: true });
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(start).not.toHaveBeenCalled();
  });

  it('skips a running sweep and lets its late update apply only the newest theme', () => {
    const { transitions } = stubViewTransition();
    applyTheme('dark', { animate: true });
    applyTheme('light');
    expect(transitions[0]!.skip).toHaveBeenCalled();
    expect(document.documentElement.dataset.theme).toBe('light');
    transitions[0]!.update();
    expect(document.documentElement.dataset.theme).toBe('light');
  });
});

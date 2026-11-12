import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getLanguage, setLanguage, subscribeLanguage, t } from './index';

type Update = () => void;

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

function stubMedia(reduce = false) {
  vi.stubGlobal('matchMedia', (query: string) => ({ matches: query.includes('reduced-motion') ? reduce : false, addEventListener: () => {}, removeEventListener: () => {} }));
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  stubMedia();
  setLanguage('en_US');
});

afterEach(() => {
  Reflect.deleteProperty(document, 'startViewTransition');
  vi.unstubAllGlobals();
  setLanguage('en_US');
  delete document.documentElement.dataset.languageTransition;
});

describe('language crossfade', () => {
  it('keeps the old language until the transition update, then re-renders subscribers', async () => {
    const { start, transitions } = stubViewTransition();
    const listener = vi.fn();
    const unsubscribe = subscribeLanguage(listener);
    setLanguage('zh_CN', { animate: true });
    expect(start).toHaveBeenCalledTimes(1);
    expect(getLanguage()).toBe('en_US');
    expect(t('Projects')).toBe('Projects');
    expect(listener).not.toHaveBeenCalled();
    expect(document.documentElement).toHaveAttribute('data-language-transition');
    transitions[0]!.update();
    expect(getLanguage()).toBe('zh_CN');
    expect(t('Projects')).toBe('训练项目');
    expect(document.documentElement.lang).toBe('zh-CN');
    expect(listener).toHaveBeenCalled();
    transitions[0]!.finish();
    await settle();
    expect(document.documentElement).not.toHaveAttribute('data-language-transition');
    unsubscribe();
  });

  it('ignores a repeated request while the crossfade is pending', () => {
    const { start } = stubViewTransition();
    setLanguage('zh_CN', { animate: true });
    setLanguage('zh_CN');
    expect(start).toHaveBeenCalledTimes(1);
    expect(getLanguage()).toBe('en_US');
  });

  it('switches at once without animate or with reduced motion', () => {
    const { start } = stubViewTransition();
    setLanguage('zh_CN');
    expect(getLanguage()).toBe('zh_CN');
    stubMedia(true);
    setLanguage('en_US', { animate: true });
    expect(getLanguage()).toBe('en_US');
    expect(start).not.toHaveBeenCalled();
  });

  it('lets a skipped crossfade apply only the newest language', () => {
    const { transitions } = stubViewTransition();
    setLanguage('zh_CN', { animate: true });
    setLanguage('en_US');
    expect(transitions[0]!.skip).toHaveBeenCalled();
    transitions[0]!.update();
    expect(getLanguage()).toBe('en_US');
  });
});

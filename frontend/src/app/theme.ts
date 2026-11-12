/** Apply the theme setting ('system' | 'light' | 'dark') as data-theme on <html>. */
import { useEffect, useRef } from 'react';

export type ThemeSetting = 'system' | 'light' | 'dark';

export function resolveTheme(setting: ThemeSetting): 'light' | 'dark' {
  if (setting !== 'system') return setting;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

let sweep: ViewTransition | null = null;
/** The theme most recently asked for; a pending sweep applies this, not its own target. */
let wanted: 'light' | 'dark' | null = null;

/**
 * The frosted sweep needs same-document View Transitions, a visible page and a
 * user who has not asked for reduced motion; otherwise the theme switches at once.
 */
function canSweep(): boolean {
  return typeof document.startViewTransition === 'function'
    && document.visibilityState === 'visible'
    && !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Switches the theme. With `animate`, a frosted band (`.theme-scan`, styled in
 * global.css) crosses the page from right to left and reveals the new theme
 * behind it. A newer switch skips any sweep still running.
 */
export function applyTheme(setting: ThemeSetting, { animate = false }: { animate?: boolean } = {}): void {
  try {
    localStorage.setItem('indextts.theme', setting);
  } catch {
    /* ignore */
  }
  const root = document.documentElement;
  const next = resolveTheme(setting);
  if (root.dataset.theme === next && wanted === next) return;
  wanted = next;
  sweep?.skipTransition();
  if (!animate || !canSweep() || root.dataset.theme === next) {
    root.dataset.theme = next;
    return;
  }
  const scan = document.createElement('div');
  scan.className = 'theme-scan';
  scan.setAttribute('aria-hidden', 'true');
  root.dataset.themeTransition = '';
  const transition = document.startViewTransition(() => {
    // A skipped sweep still runs its update later; always apply the newest theme.
    if (wanted) root.dataset.theme = wanted;
    document.body.append(scan);
  });
  sweep = transition;
  void transition.finished.catch(() => undefined).finally(() => {
    scan.remove();
    if (sweep !== transition) return;
    sweep = null;
    delete root.dataset.themeTransition;
  });
}

/** Before React renders: avoid a light flash for dark-theme users. */
export function applyStoredTheme(): void {
  let stored: ThemeSetting = 'system';
  try {
    const value = localStorage.getItem('indextts.theme');
    if (value === 'light' || value === 'dark' || value === 'system') stored = value;
  } catch {
    /* ignore */
  }
  applyTheme(stored);
}

/** Follows the saved setting; changes after the first one sweep across the page. */
export function useThemeSync(setting: ThemeSetting | undefined): void {
  const synced = useRef(false);
  useEffect(() => {
    if (!setting) return;
    applyTheme(setting, { animate: synced.current });
    synced.current = true;
    if (setting !== 'system') return;
    const media = window.matchMedia?.('(prefers-color-scheme: dark)');
    const onChange = () => applyTheme('system', { animate: true });
    media?.addEventListener('change', onChange);
    return () => media?.removeEventListener('change', onChange);
  }, [setting]);
}

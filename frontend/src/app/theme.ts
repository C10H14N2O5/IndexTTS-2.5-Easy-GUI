/** Apply the theme setting ('system' | 'light' | 'dark') as data-theme on <html>. */
import { useEffect } from 'react';

export type ThemeSetting = 'system' | 'light' | 'dark';

export function resolveTheme(setting: ThemeSetting): 'light' | 'dark' {
  if (setting !== 'system') return setting;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function applyTheme(setting: ThemeSetting): void {
  document.documentElement.dataset.theme = resolveTheme(setting);
  try {
    localStorage.setItem('indextts.theme', setting);
  } catch {
    /* ignore */
  }
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

export function useThemeSync(setting: ThemeSetting | undefined): void {
  useEffect(() => {
    if (!setting) return;
    applyTheme(setting);
    if (setting !== 'system') return;
    const media = window.matchMedia?.('(prefers-color-scheme: dark)');
    const onChange = () => applyTheme('system');
    media?.addEventListener('change', onChange);
    return () => media?.removeEventListener('change', onChange);
  }, [setting]);
}

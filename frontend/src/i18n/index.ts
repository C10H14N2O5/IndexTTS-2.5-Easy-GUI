/**
 * Front-end translations. Keys are the English source strings, with `{0}`-style
 * placeholders. zh_CN.json maps them to Chinese; English uses the key itself.
 * `pnpm check:i18n` verifies that every literal `t('...')` key is translated.
 *
 * Messages that come from the server are already rendered in the request
 * language (X-UI-Language header / SSE ?lang=) and must NOT be passed to t().
 */
import { useSyncExternalStore } from 'react';

export type Language = 'zh_CN' | 'en_US';

// One dictionary file per feature area (zh_CN/<area>.json) so parallel work never edits the same file.
const zhModules = import.meta.glob<Record<string, string>>('./zh_CN/*.json', { eager: true, import: 'default' });

const dictionaries: Record<Language, Record<string, string>> = {
  zh_CN: Object.assign({}, ...Object.values(zhModules)) as Record<string, string>,
  en_US: {},
};

let language: Language = initialLanguage();
const listeners = new Set<() => void>();

function initialLanguage(): Language {
  try {
    const stored = localStorage.getItem('indextts.language');
    if (stored === 'zh_CN' || stored === 'en_US') return stored;
  } catch {
    /* storage unavailable */
  }
  return navigator.language?.toLowerCase().startsWith('zh') ? 'zh_CN' : 'en_US';
}

export function getLanguage(): Language {
  return language;
}

export function setLanguage(next: Language): void {
  if (next === language) return;
  language = next;
  try {
    localStorage.setItem('indextts.language', next);
  } catch {
    /* ignore */
  }
  document.documentElement.lang = next === 'zh_CN' ? 'zh-CN' : 'en';
  listeners.forEach((listener) => listener());
}

export function subscribeLanguage(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function format(template: string, args: readonly unknown[]): string {
  if (!args.length) return template;
  return template.replace(/\{(\d+)\}/g, (match, index: string) => {
    const value = args[Number(index)];
    return value === undefined || value === null ? match : String(value);
  });
}

/** Translate an English source string in the current language. */
export function t(source: string, ...args: unknown[]): string {
  const translated = language === 'zh_CN' ? (dictionaries.zh_CN[source] ?? source) : source;
  return format(translated, args);
}

/** Mark a string for extraction without translating it now (e.g. constant tables). */
export function tk<T extends string>(source: T): T {
  return source;
}

/** Re-render on language change and return the translator. */
export function useT(): typeof t {
  useSyncExternalStore(subscribeLanguage, getLanguage, getLanguage);
  return t;
}

export function useLanguage(): Language {
  return useSyncExternalStore(subscribeLanguage, getLanguage, getLanguage);
}

/** Locale for Intl formatting. */
export function intlLocale(): string {
  return language === 'zh_CN' ? 'zh-CN' : 'en-US';
}

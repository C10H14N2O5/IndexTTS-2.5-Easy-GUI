import { useEffect, useRef, useState } from 'react';
import { FORM_STORAGE_KEY, readSavedForm, type SynthesisForm } from './model';

function persist(form: SynthesisForm) {
  try { localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(form)); }
  catch { /* A full or private storage must not prevent synthesis. */ }
}
export function usePersistedForm() {
  const [form, setForm] = useState(readSavedForm);
  const latest = useRef(form); latest.current = form;
  useEffect(() => {
    const timer = window.setTimeout(() => persist(form), 300);
    return () => window.clearTimeout(timer);
  }, [form]);
  useEffect(() => {
    const flush = () => persist(latest.current);
    window.addEventListener('pagehide', flush);
    return () => { window.removeEventListener('pagehide', flush); flush(); };
  }, []);
  return [form, setForm] as const;
}

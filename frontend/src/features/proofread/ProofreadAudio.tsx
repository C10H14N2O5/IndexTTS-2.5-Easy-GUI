import { useEffect, useRef, type RefObject } from 'react';
import { RotateCcw } from 'lucide-react';
import type WaveSurfer from 'wavesurfer.js';
import { Button, Select } from '@/ui';
import { useT } from '@/i18n';
import styles from './Proofread.module.css';

export function ProofreadAudio({ src, audioRef, speed, onSpeedChange, onError }: { src: string; audioRef: RefObject<HTMLAudioElement | null>; speed: string; onSpeedChange: (speed: string) => void; onError: (message: string) => void }) {
  const t = useT();
  const canvas = useRef<HTMLDivElement>(null);
  const onFailure = useRef(onError);
  onFailure.current = onError;
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    let disposed = false, wave: WaveSurfer | undefined;
    let observer: MutationObserver | undefined;
    const colors = () => { const css = getComputedStyle(document.documentElement); return { waveColor: css.getPropertyValue('--color-separator').trim(), progressColor: css.getPropertyValue('--color-accent').trim(), cursorColor: css.getPropertyValue('--color-accent').trim() }; };
    void import('wavesurfer.js').then(({ default: WS }) => {
      if (disposed || !canvas.current) return;
      wave = WS.create({ container: canvas.current, media: audio, height: 64, barWidth: 2, barGap: 2, barRadius: 1, ...colors(), dragToSeek: true });
      void wave.load(src).catch(() => { /* Native audio exposes playback failures; a waveform decode failure alone need not block playback. */ });
      observer = new MutationObserver(() => wave?.setOptions(colors()));
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    }).catch(() => { /* Native controls remain available if the optional waveform chunk fails. */ });
    return () => { disposed = true; observer?.disconnect(); audio.pause(); wave?.destroy(); };
  }, [src, audioRef]);
  useEffect(() => { if (audioRef.current) audioRef.current.playbackRate = Number(speed); }, [speed, src, audioRef]);
  async function replay() {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = 0;
    try { await audio.play(); } catch (error) { onFailure.current(error instanceof Error ? error.message : String(error)); }
  }
  return <div className={styles.audio}>
    <div ref={canvas} className={styles.waveform} aria-hidden />
    <audio ref={audioRef} src={src} controls preload="metadata" aria-label={t('Proofreading audio')} onLoadedMetadata={() => { if (audioRef.current) audioRef.current.playbackRate = Number(speed); }} onError={() => onFailure.current(t('Audio could not be decoded. Check that the file still exists and is playable.'))} />
    <div className={styles.audioTools}><Button variant="text" size="sm" icon={RotateCcw} onClick={() => void replay()}>{t('Replay from start')}</Button><Select value={speed} onChange={onSpeedChange} aria-label={t('Playback speed')} options={['0.5', '0.75', '1', '1.25', '1.5', '2'].map(value => ({ value, label: `${value} ×` }))} /></div>
  </div>;
}

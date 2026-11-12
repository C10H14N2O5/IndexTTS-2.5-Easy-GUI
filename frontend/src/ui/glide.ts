import { useCallback, useLayoutEffect, useRef } from 'react';

/** Maps the selected element to the CSS custom properties that place an indicator. */
export type IndicatorMeasure = (target: HTMLElement) => Record<string, string>;

/**
 * Keeps a decorative indicator (sidebar marker, segmented-control thumb) on the
 * element in `container` that matches `selector`.
 *
 * Positions are written as CSS custom properties; the stylesheet decides how the
 * indicator glides between them. The first placement, placements caused by the
 * container resizing (sidebar collapse, language change) and the reappearance
 * of a hidden indicator jump without a transition (`data-instant`); only a
 * change of `selection` glides. Without a match the indicator gets
 * `data-hidden`. `measure` must be stable (define it at module level).
 */
export function useGlideIndicator<C extends HTMLElement, I extends HTMLElement>(selector: string, measure: IndicatorMeasure, selection: unknown) {
  const container = useRef<C>(null);
  const indicator = useRef<I>(null);
  const placed = useRef(false);
  const place = useCallback((instant: boolean) => {
    const root = container.current;
    const marker = indicator.current;
    if (!root || !marker) return;
    const target = root.querySelector<HTMLElement>(selector);
    if (!target) {
      marker.setAttribute('data-hidden', '');
      return;
    }
    const jump = instant || !placed.current || marker.hasAttribute('data-hidden');
    if (jump) marker.setAttribute('data-instant', '');
    for (const [name, value] of Object.entries(measure(target))) marker.style.setProperty(name, value);
    marker.removeAttribute('data-hidden');
    placed.current = true;
    if (jump) {
      // Commit the new position with transitions disabled, then re-enable them.
      void marker.offsetWidth;
      marker.removeAttribute('data-instant');
    }
  }, [selector, measure]);
  useLayoutEffect(() => { place(false); }, [selection, place]);
  useLayoutEffect(() => {
    const root = container.current;
    if (!root || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => place(true));
    observer.observe(root);
    return () => observer.disconnect();
  }, [place]);
  return { container, indicator };
}

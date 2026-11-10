import { useLayoutEffect, useRef, type ReactNode } from 'react';

/** Page entrance: fade in while rising by 8px. Mirrors --duration-page / --page-shift. */
const PAGE_KEYFRAMES: Keyframe[] = [
  { opacity: 0, transform: 'translateY(8px)' },
  { opacity: 1, transform: 'none' },
];
const PAGE_TIMING: KeyframeAnimationOptions = {
  duration: 400,
  easing: 'cubic-bezier(0.05, 0.7, 0.1, 1)',
  fill: 'none',
};

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined'
    && typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Plays a short entrance animation whenever `scope` changes.
 *
 * The wrapper is never keyed, so children keep their React state (unsaved
 * drafts, persisted forms, guards). The first render is not animated, the
 * animation leaves no transform behind (`fill: none`, so sticky and fixed
 * descendants behave normally afterwards), and it is skipped entirely when the
 * user prefers reduced motion or the platform lacks the Web Animations API.
 */
export function RouteTransition({ scope, children, className }: { scope: string; children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const previous = useRef(scope);
  const running = useRef<Animation | null>(null);
  useLayoutEffect(() => {
    if (previous.current === scope) return;
    previous.current = scope;
    const element = ref.current;
    running.current?.cancel();
    running.current = null;
    if (!element || typeof element.animate !== 'function' || prefersReducedMotion()) return;
    running.current = element.animate(PAGE_KEYFRAMES, PAGE_TIMING);
  }, [scope]);
  useLayoutEffect(() => () => running.current?.cancel(), []);
  return <div ref={ref} className={className} data-transition-scope={scope}>{children}</div>;
}

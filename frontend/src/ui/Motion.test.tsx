import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useState } from 'react';
import { RouteTransition } from './Motion';

function mockMotion(reduce: boolean) {
  vi.stubGlobal('matchMedia', (query: string) => ({ matches: reduce && query.includes('reduce'), media: query, addEventListener() {}, removeEventListener() {} }));
}

function Draft() {
  const [value, setValue] = useState('');
  return <input aria-label="Draft" value={value} onChange={(event) => setValue(event.target.value)} />;
}

const animate = vi.fn(() => ({ cancel: vi.fn() }) as unknown as Animation);

beforeEach(() => {
  animate.mockClear();
  Object.defineProperty(HTMLElement.prototype, 'animate', { configurable: true, writable: true, value: animate });
});
afterEach(() => {
  vi.unstubAllGlobals();
  delete (HTMLElement.prototype as Partial<HTMLElement>).animate;
});

describe('RouteTransition', () => {
  it('does not animate the first render and plays an entrance on every scope change', () => {
    mockMotion(false);
    const { rerender } = render(<RouteTransition scope="/synthesis"><p>Page</p></RouteTransition>);
    expect(animate).not.toHaveBeenCalled();
    rerender(<RouteTransition scope="/library"><p>Page</p></RouteTransition>);
    expect(animate).toHaveBeenCalledTimes(1);
    const [frames, timing] = animate.mock.calls[0] as unknown as [Keyframe[], KeyframeAnimationOptions];
    expect(frames[0]).toMatchObject({ opacity: 0 });
    expect(frames[1]).toMatchObject({ opacity: 1, transform: 'none' });
    expect(timing).toMatchObject({ duration: 220, fill: 'none' });
    rerender(<RouteTransition scope="/library"><p>Page</p></RouteTransition>);
    expect(animate).toHaveBeenCalledTimes(1);
  });

  it('cancels an unfinished entrance before starting the next one', () => {
    mockMotion(false);
    const cancel = vi.fn();
    animate.mockImplementation(() => ({ cancel }) as unknown as Animation);
    const { rerender } = render(<RouteTransition scope="a"><p>Page</p></RouteTransition>);
    rerender(<RouteTransition scope="b"><p>Page</p></RouteTransition>);
    rerender(<RouteTransition scope="c"><p>Page</p></RouteTransition>);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(animate).toHaveBeenCalledTimes(2);
  });

  it('respects reduced motion', () => {
    mockMotion(true);
    const { rerender } = render(<RouteTransition scope="a"><p>Page</p></RouteTransition>);
    rerender(<RouteTransition scope="b"><p>Page</p></RouteTransition>);
    expect(animate).not.toHaveBeenCalled();
  });

  it('degrades safely without the Web Animations API', () => {
    mockMotion(false);
    delete (HTMLElement.prototype as Partial<HTMLElement>).animate;
    const { rerender } = render(<RouteTransition scope="a"><p>Page</p></RouteTransition>);
    expect(() => rerender(<RouteTransition scope="b"><p>Next</p></RouteTransition>)).not.toThrow();
    expect(screen.getByText('Next')).toBeInTheDocument();
  });

  it('keeps child state across scope changes', async () => {
    mockMotion(false);
    const { rerender } = render(<RouteTransition scope="corpus"><Draft /></RouteTransition>);
    await userEvent.type(screen.getByLabelText('Draft'), 'unsaved');
    rerender(<RouteTransition scope="training"><Draft /></RouteTransition>);
    expect(screen.getByLabelText('Draft')).toHaveValue('unsaved');
    expect(animate).toHaveBeenCalledTimes(1);
  });
});

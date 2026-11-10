/// <reference types="node" />
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from '@/api/client';
import { setLanguage } from '@/i18n';
import { Field, SegmentedControl } from './Controls';
import { AudioInput, AudioPlayer } from './Audio';
import { FileBrowserDialog } from './FileBrowser';
import { readFileSync } from 'node:fs';
import { URL as NodeURL } from 'node:url';

const tokens = readFileSync(new NodeURL('../styles/tokens.css', import.meta.url), 'utf8');

vi.mock('wavesurfer.js', () => ({ default: { create: () => ({ on: () => {}, load: async () => {}, destroy: () => {}, setOptions: () => {} }) } }));
beforeEach(() => {
  setLanguage('en_US');
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });
const options = [{ value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }];

describe('composite accessible names', () => {
  it('uses the enclosing Field label and hint for a segmented group', () => {
    render(<Field label="Theme" hint="Choose the display appearance"><SegmentedControl value="light" onChange={() => {}} options={options}/></Field>);
    const group = screen.getByRole('radiogroup', { name: 'Theme' });
    expect(group).toHaveAccessibleDescription('Choose the display appearance');
    expect(group).toHaveAttribute('aria-labelledby');
    expect(within(group).getByRole('radio', { name: 'Light' })).toBeChecked();
  });
  it('gives an explicit segmented-group label precedence over the Field label', () => {
    render(<Field label="Theme"><SegmentedControl aria-label="Display appearance" value="dark" onChange={() => {}} options={options}/></Field>);
    const group = screen.getByRole('radiogroup', { name: 'Display appearance' });
    expect(group).not.toHaveAttribute('aria-labelledby');
    expect(screen.queryByRole('radiogroup', { name: 'Theme' })).not.toBeInTheDocument();
  });
  it('names standalone segmented groups from their already-localized choices', () => {
    render(<SegmentedControl value="light" onChange={() => {}} options={options}/>);
    expect(screen.getByRole('radiogroup', { name: 'Light / Dark' })).toBeVisible();
  });
  it('associates segmented-group validation errors with the group', () => {
    render(<Field label="Theme" error="Pick an available appearance"><SegmentedControl value="light" onChange={() => {}} options={options}/></Field>);
    const group = screen.getByRole('radiogroup', { name: 'Theme' });
    expect(group).toHaveAttribute('aria-invalid', 'true');
    expect(group).toHaveAccessibleDescription('Pick an available appearance');
  });
  it('names each player using its title, filename or localized fallback', () => {
    render(<><AudioPlayer src="/first.wav" title="Sample 1"/><AudioPlayer src="/second.wav" downloadName="second.wav"/><AudioPlayer src="/third.wav"/></>);
    expect(screen.getByRole('group', { name: 'Sample 1' })).toBeVisible();
    expect(screen.getByRole('group', { name: 'second.wav' })).toBeVisible();
    expect(screen.getByRole('group', { name: 'Audio playback' })).toBeVisible();
  });
  it('uses an explicit AudioPlayer label instead of its visible title', () => {
    render(<AudioPlayer src="/sample.wav" title="Sample" aria-label="Reference voice for this comparison"/>);
    expect(screen.getByRole('group', { name: 'Reference voice for this comparison' })).toBeVisible();
  });
  it('localizes the generic playback group name', () => {
    setLanguage('zh_CN'); render(<AudioPlayer src="/sample.wav"/>);
    expect(screen.getByRole('group', { name: '音频播放' })).toBeVisible();
  });
  it('names audio upload/record controls as a group and associates its visible label', () => {
    render(<AudioInput value={null} onChange={() => {}} label="Evaluation reference" hint="Choose one clean speaker"/>);
    const group = screen.getByRole('group', { name: 'Evaluation reference' });
    expect(group).toHaveAccessibleDescription('Choose one clean speaker');
    const file = within(group).getByLabelText('Choose audio file');
    expect(screen.getByText('Evaluation reference', { selector: 'label' })).toHaveAttribute('for', file.id);
  });
});

describe('file selection announcements', () => {
  it('exposes file selection as pressed state and announces the selected path', async () => {
    vi.spyOn(api, 'get').mockImplementation(async path => path === '/api/fs/roots' ? [] : { path: 'D:/data', parent: 'D:/', entries: [
      { name: 'first.pth', path: 'D:/data/first.pth', type: 'file', size: 1024, modified: 1 },
      { name: 'second.pth', path: 'D:/data/second.pth', type: 'file', size: 1024, modified: 1 },
      { name: 'nested', path: 'D:/data/nested', type: 'dir', size: null, modified: 1 },
    ] });
    render(<FileBrowserDialog open onOpenChange={() => {}} kind="pth" title="Choose checkpoint" onSelect={() => {}}/>);
    const first = await screen.findByRole('button', { name: /first\.pth/ });
    const second = screen.getByRole('button', { name: /second\.pth/ });
    expect(first).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', { name: 'nested' })).not.toHaveAttribute('aria-pressed');
    await userEvent.click(first);
    expect(first).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('status')).toHaveTextContent('D:/data/first.pth');
    await userEvent.click(second);
    expect(first).toHaveAttribute('aria-pressed', 'false');
    expect(second).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('status')).toHaveTextContent('D:/data/second.pth');
    expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite');
  });
  it('exposes directory selection in folder-selection mode', async () => {
    vi.spyOn(api, 'get').mockImplementation(async path => path === '/api/fs/roots' ? [] : { path: 'D:/data', parent: 'D:/', entries: [{ name: 'nested', path: 'D:/data/nested', type: 'dir', size: null, modified: 1 }] });
    render(<FileBrowserDialog open onOpenChange={() => {}} kind="dir" title="Choose corpus" onSelect={() => {}}/>);
    const folder = await screen.findByRole('button', { name: 'nested' });
    expect(folder).toHaveAttribute('aria-pressed', 'false');
    await userEvent.click(folder);
    expect(folder).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('status')).toHaveTextContent('D:/data/nested');
  });
});

function luminance(hex: string) {
  const channels = [1, 3, 5].map(index => Number.parseInt(hex.slice(index, index + 2), 16) / 255).map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
  return channels[0]! * 0.2126 + channels[1]! * 0.7152 + channels[2]! * 0.0722;
}
function contrast(a: string, b: string) { const first = luminance(a), second = luminance(b); return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05); }
describe('caption token WCAG contrast', () => {
  for (const theme of ['light', 'dark']) {
    for (const surface of ['--color-bg', '--color-surface-2', '--color-surface-3']) {
      it(`keeps ${theme} small captions at least 4.5:1 on ${surface}`, () => {
        const block = tokens.match(new RegExp(`\\[data-theme='${theme}'\\]\\s*\\{([^}]+)\\}`))?.[1];
        expect(block).toBeDefined();
        const colors = Object.fromEntries([...block!.matchAll(/(--[\w-]+):\s*(#[\da-f]{6});/gi)].map(match => [match[1], match[2]]));
        expect(contrast(colors['--color-text-3']!, colors[surface]!)).toBeGreaterThanOrEqual(4.5);
      });
    }
  }
});

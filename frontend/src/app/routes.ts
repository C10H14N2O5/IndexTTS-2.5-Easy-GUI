/**
 * Navigation model shared by the router, sidebar and command shortcuts.
 * Labels are English i18n keys (rendered with t()).
 */
import type { LucideIcon } from 'lucide-react';
import { AudioLines, Download, FolderKanban, Library, Scale, Settings2 } from 'lucide-react';
import { tk } from '@/i18n';

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
  /** Also active for these path prefixes. */
  match?: string[];
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

export const NAV: NavGroup[] = [
  {
    label: tk('Create'),
    items: [
      { path: '/synthesis', label: tk('Synthesis'), icon: AudioLines },
      { path: '/library', label: tk('Voice library'), icon: Library },
    ],
  },
  {
    label: tk('Train'),
    items: [
      { path: '/projects', label: tk('Projects'), icon: FolderKanban },
      { path: '/abx', label: tk('Blind A/B'), icon: Scale },
      { path: '/export', label: tk('Export'), icon: Download },
    ],
  },
  {
    label: tk('System'),
    items: [{ path: '/system', label: tk('System'), icon: Settings2 }],
  },
];

/**
 * Route table. Each feature page is a lazy module with a default export:
 *
 *   /synthesis, /synthesis/dialogue        features/synthesis/SynthesisPage
 *   /library                               features/library/LibraryPage
 *   /projects                              features/projects/ProjectsPage
 *   /projects/:project/(corpus|validation|training|outputs)
 *                                          features/projects/ProjectLayout + children
 *   /abx                                   features/abx/AbxPage
 *   /export                                features/export/ExportPage
 *   /system                                features/system/SystemPage
 */
import { createBrowserRouter, Navigate } from 'react-router';
import { AppShell } from '@/app/shell/AppShell';
import { RouteError } from '@/app/shell/RouteError';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Navigate to="/synthesis" replace /> },
      { path: 'synthesis', lazy: () => import('@/features/synthesis/SynthesisPage').then((m) => ({ Component: m.default })) },
      { path: 'synthesis/:mode', lazy: () => import('@/features/synthesis/SynthesisPage').then((m) => ({ Component: m.default })) },
      { path: 'library', lazy: () => import('@/features/library/LibraryPage').then((m) => ({ Component: m.default })) },
      { path: 'projects', lazy: () => import('@/features/projects/ProjectsPage').then((m) => ({ Component: m.default })) },
      {
        path: 'projects/:project',
        lazy: () => import('@/features/projects/ProjectLayout').then((m) => ({ Component: m.default })),
        children: [
          { index: true, element: <Navigate to="corpus" replace /> },
          { path: 'corpus', lazy: () => import('@/features/projects/CorpusPage').then((m) => ({ Component: m.default })) },
          { path: 'validation', lazy: () => import('@/features/projects/ValidationPage').then((m) => ({ Component: m.default })) },
          { path: 'training', lazy: () => import('@/features/projects/TrainingPage').then((m) => ({ Component: m.default })) },
          { path: 'outputs', lazy: () => import('@/features/projects/OutputsPage').then((m) => ({ Component: m.default })) },
        ],
      },
      { path: 'abx', lazy: () => import('@/features/abx/AbxPage').then((m) => ({ Component: m.default })) },
      { path: 'export', lazy: () => import('@/features/export/ExportPage').then((m) => ({ Component: m.default })) },
      { path: 'system', lazy: () => import('@/features/system/SystemPage').then((m) => ({ Component: m.default })) },
      { path: '*', element: <Navigate to="/synthesis" replace /> },
    ],
  },
]);

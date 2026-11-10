/**
 * P0 placeholder shell (owned by the ui-kit teammate from P1): sidebar
 * navigation plus the routed page. The final shell adds the top bar, activity
 * center, console drawer, toasts and the SSE reconnect banner.
 */
import { NavLink, Outlet } from 'react-router';
import { NAV } from '@/app/routes';
import { useT } from '@/i18n';
import styles from './AppShell.module.css';

export function AppShell() {
  const t = useT();
  return (
    <div className={styles.shell}>
      <nav className={styles.sidebar} aria-label={t('Main navigation')}>
        <div className={styles.brand}>IndexTTS</div>
        {NAV.map((group) => (
          <div key={group.label} className={styles.group}>
            <div className={styles.groupLabel}>{t(group.label)}</div>
            {group.items.map((item) => (
              <NavLink key={item.path} to={item.path} className={({ isActive }) => (isActive ? styles.active : styles.link)}>
                <item.icon size={18} aria-hidden />
                <span>{t(item.label)}</span>
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <main className={styles.main}>
        <Outlet />
      </main>
    </div>
  );
}

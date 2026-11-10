// P0 placeholder; replaced by the owning teammate.
import { useT } from '@/i18n';

export default function Page() {
  const t = useT();
  return <h1>{t('Blind A/B')}</h1>;
}

import { useState } from 'react';
import { Dialog } from 'radix-ui';
import { Terminal, Trash2, X } from 'lucide-react';
import { api, errorMessage } from '@/api/client';
import { useLive } from '@/api/live';
import { useT } from '@/i18n';
import { Button, IconButton, LogConsole, toast } from '@/ui';
import styles from './AppShell.module.css';
export function ConsoleDrawer({open,onOpenChange}:{open:boolean;onOpenChange:(open:boolean)=>void}){const t=useT();const lines=useLive(state=>state.console);const current=useLive(state=>state.consoleCurrent);const [clearing,setClearing]=useState(false);async function clear(){setClearing(true);try{await api.delete('/api/console');}catch(error){toast.error(errorMessage(error));}finally{setClearing(false);}}return <Dialog.Root open={open} onOpenChange={onOpenChange}><Dialog.Portal><Dialog.Overlay className={styles.drawerOverlay}/><Dialog.Content className={styles.drawer} aria-describedby={undefined}><div className={styles.drawerHeader}><Dialog.Title><Terminal size={18}/>{t('Console')}</Dialog.Title><span>{t('Live diagnostic output from the server')}</span><Button variant="text" size="sm" icon={Trash2} pending={clearing} disabled={!lines.length&&!current} onClick={()=>void clear()}>{t('Clear console')}</Button><Dialog.Close asChild><IconButton icon={X} label={t('Close console')}/></Dialog.Close></div><LogConsole lines={lines} current={current} height={Math.min(320,Math.round(window.innerHeight*0.43))}/></Dialog.Content></Dialog.Portal></Dialog.Root>;}

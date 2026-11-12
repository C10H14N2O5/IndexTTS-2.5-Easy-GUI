import { useLayoutEffect, useRef, useState } from 'react';
import { NavLink, Outlet, useLocation, useNavigation } from 'react-router';
import { AudioLines, Cpu, FlaskConical, Languages, Monitor, Moon, Sun, Terminal, WifiOff } from 'lucide-react';
import { NAV } from '@/app/routes';
import { useBootstrap, useSettings, useUpdateSettings } from '@/app/bootstrap';
import { resolveTheme, type ThemeSetting } from '@/app/theme';
import { useGpuJob, useLive } from '@/api/live';
import { errorMessage } from '@/api/client';
import { useLanguage, useT } from '@/i18n';
import { ActivityButton, useActivityEffects } from '@/features/activity';
import { Badge, Button, IconButton, PageEyebrow, Popover, ProgressBar, RouteTransition, SegmentedControl, Spinner, Toaster, Tooltip, toast, useGlideIndicator, type IndicatorMeasure } from '@/ui';
import { ConsoleDrawer } from './ConsoleDrawer';
import styles from './AppShell.module.css';

/**
 * Transition scope for the page area: a project workspace counts as one page (its
 * tabs animate inside ProjectLayout), every other path is its own page.
 */
function pageScope(pathname:string){const project=/^\/projects\/[^/]+/.exec(pathname);return project?project[0]:pathname.replace(/\/+$/,'')||'/';}

/** The sidebar marker spans the active link's height minus 10px at each end. */
const measureNavItem:IndicatorMeasure=link=>({'--nav-x':`${link.offsetLeft}px`,'--nav-y':`${link.offsetTop+10}px`,'--nav-h':`${Math.max(link.offsetHeight-20,0)}px`});

export function AppShell(){
 const t=useT();const location=useLocation();const navigation=useNavigation();const {data:bootstrap}=useBootstrap();const connection=useLive(state=>state.connection);const system=useLive(state=>state.system)??bootstrap?.system;const training=useLive(state=>state.training);const exporting=useLive(state=>state.export);const gpuJob=useGpuJob();const language=useLanguage();const settings=useUpdateSettings();const savedSettings=useSettings();const pendingTheme=settings.isPending?settings.variables?.theme:undefined;const theme=(pendingTheme??savedSettings?.theme??'system') as ThemeSetting;const resolved=resolveTheme(theme);const setTheme=(value:ThemeSetting)=>settings.mutate({theme:value},{onError:error=>toast.error(errorMessage(error))});const [consoleOpen,setConsoleOpen]=useState(false);
 useActivityEffects();
 const current=NAV.flatMap(group=>group.items).find(item=>location.pathname.startsWith(item.path));
 const currentGroup=NAV.find(group=>group.items.some(item=>item===current));const eyebrow=currentGroup&&currentGroup.items.length>1?t(currentGroup.label):undefined;
 const {container:navRef,indicator:navIndicator}=useGlideIndicator<HTMLElement,HTMLSpanElement>('[data-nav-active]',measureNavItem,current?.path);
 const scope=pageScope(location.pathname);const shownScope=useRef(scope);
 useLayoutEffect(()=>{if(shownScope.current===scope)return;shownScope.current=scope;if(typeof window.scrollTo==='function'&&window.scrollY>0)window.scrollTo({top:0});},[scope]);
 const busy=Boolean(gpuJob||training?.running||exporting?.status==='running');const busyLabel=gpuJob?.title??(training?.running?t('Training'):exporting?.status==='running'?t('Export'):undefined);
 const gpuDetail=busyLabel?t('GPU busy · {0}',busyLabel):system?.cuda?.available?t('GPU ready · {0}',system.cuda.device??'CUDA'):t('GPU is not available');
 async function changeLanguage(){try{await settings.mutateAsync({ui_language:language==='zh_CN'?'en_US':'zh_CN'});}catch(err){toast.error(errorMessage(err));}}
 return <div className={styles.shell}><a className={styles.skip} href="#main-content">{t('Skip to content')}</a><aside className={styles.sidebar}><div className={styles.brand}><div className={styles.brandMark} aria-hidden><AudioLines size={22}/></div><div className={styles.brandText}><strong>IndexTTS</strong><span>Easy GUI</span></div></div><nav ref={navRef} aria-label={t('Main navigation')} className={styles.nav}><span ref={navIndicator} className={styles.navIndicator} aria-hidden data-hidden=""/>{NAV.map(group=><div key={group.label} className={styles.group}><div className={styles.groupLabel}>{t(group.label)}</div>{group.items.map(item=><Tooltip key={item.path} content={t(item.label)} side="right"><NavLink to={item.path} aria-label={t(item.label)} className={location.pathname.startsWith(item.path)?styles.active:styles.link} data-nav-active={location.pathname.startsWith(item.path)||undefined}><item.icon size={18} aria-hidden/><span>{t(item.label)}</span></NavLink></Tooltip>)}</div>)}</nav><div className={styles.sidebarFooter}><span className={styles.connectionDot} data-status={connection} aria-hidden/><span>{connection==='open'?t('Connected'):t('Connecting…')}</span><small>{bootstrap?.version?`v${bootstrap.version}`:''}</small></div></aside><div className={styles.workspace}><header className={styles.topbar}><div className={styles.pageIdentity}>{current&&<><current.icon size={16} aria-hidden/><span>{t(current.label)}</span></>}</div><div className={styles.topbarActions}><Tooltip content={gpuDetail}><span className={styles.gpuStatus} tabIndex={0} data-busy={busy||undefined}><Cpu size={15}/>{busy?<><Spinner size={12}/><span>{t('GPU busy')}</span></>:<span>{system?.cuda?.available?t('GPU ready'):t('CPU only')}</span>}</span></Tooltip><div className={styles.toolbarDivider}/><IconButton icon={Languages} label={language==='zh_CN'?t('Switch to English'):t('Switch to Chinese')} onClick={()=>void changeLanguage()} pending={settings.isPending}/><Popover trigger={<IconButton icon={resolved==='dark'?Moon:Sun} label={t('Appearance')}/>}><div className={styles.themePanel}><strong>{t('Appearance')}</strong><SegmentedControl value={theme} onChange={setTheme} aria-label={t('Appearance')} options={[{value:'light',label:t('Light'),icon:Sun},{value:'dark',label:t('Dark'),icon:Moon},{value:'system',label:t('Automatic'),icon:Monitor}]}/></div></Popover><ActivityButton/><IconButton icon={Terminal} label={t('Open console')} onClick={()=>setConsoleOpen(true)}/></div>{navigation.state!=='idle'&&<div className={styles.navigationProgress}><ProgressBar value={null} label={t('Loading page')} size="sm"/></div>}</header>{connection!=='open'&&<div className={styles.reconnect} role="status"><WifiOff size={16}/><span>{t('Connection lost. Reconnecting automatically…')}</span><Button variant="text" size="sm" onClick={()=>window.location.reload()}>{t('Reload interface')}</Button></div>}{bootstrap?.fake_engine&&<div className={styles.fakeBanner} role="status"><FlaskConical size={16}/><span>{t('Demo mode — generated audio and training results are simulated.')}</span><Badge tone="neutral">{t('No model required')}</Badge></div>}<main id="main-content" tabIndex={-1} className={styles.main}><PageEyebrow value={eyebrow}><RouteTransition scope={scope} className={styles.content}><Outlet/></RouteTransition></PageEyebrow></main></div><ConsoleDrawer open={consoleOpen} onOpenChange={setConsoleOpen}/><Toaster/></div>;
}

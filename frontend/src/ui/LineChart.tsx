import { useEffect, useRef, useState } from 'react';
import type uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { useT } from '@/i18n';
import { EmptyState, InlineAlert } from './Layout';
import s from './Media.module.css';
import { alignSeries, countDefined, expandReferences, showMarkers, type ChartSeries } from './chart-data';
export type { ChartSeries } from './chart-data';
export interface LineChartProps {series:ChartSeries[];height?:number;xLabel?:string;yLabel?:string;yFormat?:(value:number)=>string}
export function LineChart({series,height=260,xLabel,yLabel,yFormat}:LineChartProps){const t=useT();const node=useRef<HTMLDivElement>(null);const [theme,setTheme]=useState(document.documentElement.dataset.theme);const [error,setError]=useState(false);const empty=!series.some(line=>line.points.length>0);
  useEffect(()=>{const observer=new MutationObserver(()=>setTheme(document.documentElement.dataset.theme));observer.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});return()=>observer.disconnect();},[]);
  useEffect(()=>{if(empty||!node.current)return;let disposed=false;let plot:uPlot|undefined;let resize:ResizeObserver|undefined;setError(false);void import('uplot').then(({default:UPlot})=>{if(disposed||!node.current)return;
    const css=getComputedStyle(document.documentElement);const palette=['--color-accent','--color-success','--color-warning','--color-text-2','--color-error'].map(name=>css.getPropertyValue(name).trim());const text=css.getPropertyValue('--color-text-2').trim();const grid=css.getPropertyValue('--color-separator-soft').trim();const bg=css.getPropertyValue('--color-bg').trim()||'#fff';
    const format=yFormat??((value:number)=>Number.isFinite(value)?new Intl.NumberFormat(undefined,{maximumSignificantDigits:4}).format(value):'—');
    const lines=expandReferences(series);const colors=lines.map((line,i)=>line.color??palette[i%palette.length]??'#0071e3');
    // uPlot series 0 is the x axis, so line i is uPlot series i + 1.
    const reference=(i:number)=>Boolean(lines[i-1]?.reference);const pxRatio=UPlot.pxRatio||window.devicePixelRatio||1;
    /** Legend value; when the cursor snapped a sparse series to another step, name that step. */
    const value=(u:uPlot,raw:number|null|undefined,i:number,idx:number|null)=>{if(raw==null)return '—';if(reference(i))return format(raw);const cursor=u.cursor.idx;return idx!=null&&cursor!=null&&idx!==cursor?t('{0} (step {1})',format(raw),u.data[0][idx]??''):format(raw);};
    const options:uPlot.Options={width:node.current.clientWidth||500,height,scales:{x:{time:false}},
      cursor:{focus:{prox:24},hover:{skip:[null,undefined]},points:{size:(_u,i)=>reference(i)?0:11,width:2,fill:(_u,i)=>colors[i-1]??bg,stroke:bg}},
      legend:{show:true,live:true},
      axes:[{label:xLabel,stroke:text,grid:{stroke:grid,width:1},ticks:{stroke:grid},font:'12px Inter, system-ui',size:44},{label:yLabel,stroke:text,grid:{stroke:grid,width:1},ticks:{stroke:grid},font:'12px Inter, system-ui',size:64,values:(_plot,values)=>values.map(format)}],
      series:[{label:xLabel??t('Step')},...lines.map((line,i):uPlot.Series=>line.reference
        ?{label:line.label,stroke:colors[i],width:1.5,dash:[6,4],spanGaps:true,points:{show:false},value}
        :{label:line.label,stroke:colors[i],width:1.75,spanGaps:true,value,points:{show:(u,si,i0,i1)=>showMarkers(countDefined(u.data[si]??[],i0,i1),u.bbox.width/pxRatio),size:7,width:2,fill:bg,stroke:colors[i]}})]};
    plot=new UPlot(options,alignSeries(lines),node.current);resize=new ResizeObserver(entries=>{const width=entries[0]?.contentRect.width;if(width&&width>0)plot?.setSize({width:Math.round(width),height});});resize.observe(node.current);}).catch(()=>{if(!disposed)setError(true);});return()=>{disposed=true;resize?.disconnect();plot?.destroy();};},[series,height,xLabel,yLabel,yFormat,theme,empty,t]);
  return <div className={s.chart}>{empty?<EmptyState title={t('Metrics will appear after the first training step.')} description={t('The chart updates while training runs.')}/>:<><div ref={node} role="img" aria-label={[yLabel,xLabel,...series.map(line=>line.label)].filter(Boolean).join(' · ')}/><div className="visually-hidden">{series.map(line=><div key={line.label}>{line.label}: {line.points.at(-1)?.[1]??'—'}</div>)}</div></>}{error&&<InlineAlert tone="warning">{t('The chart could not be rendered. Open TensorBoard for detailed metrics.')}</InlineAlert>}</div>;
}

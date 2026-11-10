import type { ReactNode } from 'react';
export interface ToastOptions {description?:ReactNode;action?:{label:string;onClick:()=>void};duration?:number}
export type ToastTone='success'|'error'|'info'|'warning';
export interface ToastItem extends ToastOptions {id:number;message:string;tone:ToastTone}
let items:ToastItem[]=[];
let sequence=0;
const subscriptions=new Set<()=>void>();
function emit(){subscriptions.forEach(fn=>fn());}
function dismiss(id:number){items=items.filter(item=>item.id!==id);emit();}
function push(tone:ToastTone,message:string,options:ToastOptions={}){
 const id=++sequence;
 items=[...items,{id,tone,message,...options}].slice(-5);
 emit();
 return id;
}
export const toast={
 success:(message:string,options?:ToastOptions)=>push('success',message,options),
 error:(message:string,options?:ToastOptions)=>push('error',message,options),
 info:(message:string,options?:ToastOptions)=>push('info',message,options),
 warning:(message:string,options?:ToastOptions)=>push('warning',message,options),
 dismiss,
};
export const subscribeToasts=(fn:()=>void)=>{subscriptions.add(fn);return()=>subscriptions.delete(fn);};
export const getToasts=()=>items;

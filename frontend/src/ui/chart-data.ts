/** `reference` series are drawn as a horizontal line at their first value across the other series' x-range. */
export interface ChartSeries {label:string;points:[number,number][];color?:string;reference?:boolean}
const finite=([x,y]:[number,number])=>Number.isFinite(x)&&Number.isFinite(y);
/** Union-align step series without inventing values between logged measurements. */
export function alignSeries(series:ChartSeries[]):[number[],...(number|null)[][]]{
 const cleaned=series.map(line=>new Map(line.points.filter(([x,y])=>Number.isFinite(x)&&Number.isFinite(y))));
 const xs=[...new Set(cleaned.flatMap(line=>[...line.keys()]))].sort((a,b)=>a-b);
 return [xs,...cleaned.map(line=>xs.map(x=>line.get(x)??null))];
}
/** Stretch each reference series to a flat line spanning the x-range of the non-reference series. */
export function expandReferences(series:ChartSeries[]):ChartSeries[]{
 const xs=series.filter(line=>!line.reference).flatMap(line=>line.points.filter(finite).map(([x])=>x));
 if(!xs.length)return series;
 const minX=Math.min(...xs);const maxX=Math.max(...xs);
 return series.map(line=>{if(!line.reference)return line;const first=line.points.find(finite);if(!first)return line;const v=first[1];return {...line,points:minX===maxX?[[minX,v]]:[[minX,v],[maxX,v]]};});
}
/** Number of finite values in the inclusive index range [i0, i1]. */
export function countDefined(values:ArrayLike<number|null|undefined>,i0:number,i1:number):number{
 let count=0;for(let i=Math.max(0,i0);i<=Math.min(i1,values.length-1);i++){const value=values[i];if(typeof value==='number'&&Number.isFinite(value))count++;}return count;
}
/** Draw point markers only when they are spaced at least `minSpacing` CSS pixels apart on average. */
export function showMarkers(count:number,widthCss:number,minSpacing=10):boolean{return count>0&&widthCss/count>=minSpacing;}

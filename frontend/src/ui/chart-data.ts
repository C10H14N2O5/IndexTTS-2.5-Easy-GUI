export interface ChartSeries {label:string;points:[number,number][];color?:string}
/** Union-align step series without inventing values between logged measurements. */
export function alignSeries(series:ChartSeries[]):[number[],...(number|null)[][]]{
 const cleaned=series.map(line=>new Map(line.points.filter(([x,y])=>Number.isFinite(x)&&Number.isFinite(y))));
 const xs=[...new Set(cleaned.flatMap(line=>[...line.keys()]))].sort((a,b)=>a-b);
 return [xs,...cleaned.map(line=>xs.map(x=>line.get(x)??null))];
}

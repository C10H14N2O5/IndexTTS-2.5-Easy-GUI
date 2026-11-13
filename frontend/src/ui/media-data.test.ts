import { describe, expect, it } from 'vitest';
import { encodeWav, type PcmBuffer } from './wav';
import { alignSeries, countDefined, expandReferences, showMarkers, type ChartSeries } from './chart-data';
import { formatBytes, formatDuration } from './format';

const pcm=(channels:number[][],sampleRate=8000):PcmBuffer=>({numberOfChannels:channels.length,sampleRate,length:channels[0]?.length??0,getChannelData:channel=>new Float32Array(channels[channel]??[])});
describe('WAV encoder',()=>{
 it('encodes a valid PCM16 RIFF header with clipped samples',()=>{
  const output=encodeWav(pcm([[-2,-1,0,1,2]]));const view=new DataView(output);expect(new TextDecoder().decode(output.slice(0,4))).toBe('RIFF');expect(new TextDecoder().decode(output.slice(8,12))).toBe('WAVE');expect(view.getUint32(4,true)).toBe(46);expect(view.getUint16(22,true)).toBe(1);expect(view.getUint32(24,true)).toBe(8000);expect(view.getUint16(34,true)).toBe(16);expect(view.getUint32(40,true)).toBe(10);expect([0,1,2,3,4].map(i=>view.getInt16(44+2*i,true))).toEqual([-32768,-32768,0,32767,32767]);
 });
 it('trims precisely and interleaves stereo channels',()=>{
  const output=encodeWav(pcm([[0,0.5,1,0],[0,-0.5,-1,0]],4),0.25,0.75);const view=new DataView(output);expect(output.byteLength).toBe(52);expect(view.getUint16(22,true)).toBe(2);expect([0,1,2,3].map(i=>view.getInt16(44+i*2,true))).toEqual([16384,-16384,32767,-32768]);
 });
 it('clamps ranges and rejects empty or non-finite selections',()=>{
  expect(encodeWav(pcm([[0,1]]),-1,100).byteLength).toBe(48);expect(()=>encodeWav(pcm([[0,1]]),1,0)).toThrow(RangeError);expect(()=>encodeWav(pcm([[0,1]]),NaN,1)).toThrow(RangeError);
 });
});
describe('chart alignment',()=>{
 it('aligns independently logged steps without fabrication and drops invalid values',()=>{
  expect(alignSeries([{label:'Train',points:[[3,0.3],[1,1],[2,NaN]]},{label:'Eval',points:[[2,0.5],[3,0.2],[3,0.1]]}])).toEqual([[1,2,3],[1,null,0.3],[null,0.5,0.1]]);
 });
 it('supports empty histories',()=>{expect(alignSeries([])).toEqual([[]]);expect(alignSeries([{label:'Loss',points:[]}])).toEqual([[],[]]);});
});
describe('chart reference lines',()=>{
 const base:ChartSeries={label:'Base',points:[[0,2.5]],reference:true};
 it('stretches a reference series across the x-range of the other series',()=>{
  const train:ChartSeries={label:'Train',points:[[10,1],[NaN,3],[50,0.8]]};const valid:ChartSeries={label:'Valid',points:[[100,0.9]]};
  expect(expandReferences([train,base,valid])).toEqual([train,{...base,points:[[10,2.5],[100,2.5]]},valid]);
 });
 it('uses the first finite value and keeps the series when it has no points',()=>{
  const train:ChartSeries={label:'Train',points:[[5,1],[9,1]]};const empty:ChartSeries={label:'Empty',points:[[1,NaN]],reference:true};
  expect(expandReferences([train,{...base,points:[[0,NaN],[3,2]]},empty])).toEqual([train,{...base,points:[[5,2],[9,2]]},empty]);
 });
 it('keeps references unchanged without other points and collapses a single x',()=>{
  expect(expandReferences([base])).toEqual([base]);expect(expandReferences([base,{label:'Empty',points:[]}])).toEqual([base,{label:'Empty',points:[]}]);
  expect(expandReferences([{label:'One',points:[[7,1]]},base])[1]).toEqual({...base,points:[[7,2.5]]});
 });
 it('counts finite values in an inclusive index range',()=>{
  const values=[1,null,undefined,NaN,2,3];expect(countDefined(values,0,5)).toBe(3);expect(countDefined(values,1,4)).toBe(1);expect(countDefined(values,2,2)).toBe(0);expect(countDefined(values,-3,99)).toBe(3);expect(countDefined([],0,0)).toBe(0);
 });
 it('shows markers only when points are spaced apart',()=>{
  expect(showMarkers(0,500)).toBe(false);expect(showMarkers(50,500)).toBe(true);expect(showMarkers(51,500)).toBe(false);expect(showMarkers(20,500,30)).toBe(false);expect(showMarkers(10,500,30)).toBe(true);
 });
});
describe('UI formats',()=>{
 it('formats durations and handles unknown values',()=>{expect(formatDuration(65)).toBe('1:05');expect(formatDuration(3661)).toBe('1:01:01');expect(formatDuration(NaN)).toBe('—');expect(formatDuration(-5)).toBe('0:00');});
 it('formats bounded binary file sizes',()=>{expect(formatBytes(1024)).toBe('1.0 KB');expect(formatBytes(1024**3)).toBe('1.00 GB');expect(formatBytes(null)).toBe('—');});
});

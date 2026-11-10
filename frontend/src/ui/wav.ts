/** Encode a decoded PCM buffer as interleaved, little-endian, 16-bit WAV. */
export interface PcmBuffer {numberOfChannels:number;sampleRate:number;length:number;getChannelData(channel:number):Float32Array}
export function encodeWav(buffer:PcmBuffer,startSeconds=0,endSeconds=buffer.length/buffer.sampleRate):ArrayBuffer {
 if(!Number.isFinite(startSeconds)||!Number.isFinite(endSeconds)||!Number.isFinite(buffer.sampleRate)||buffer.sampleRate<=0||buffer.numberOfChannels<1)throw new RangeError('Invalid audio range');
 const start=Math.max(0,Math.min(buffer.length,Math.floor(startSeconds*buffer.sampleRate)));const end=Math.max(start,Math.min(buffer.length,Math.floor(endSeconds*buffer.sampleRate)));const frames=end-start;const channels=buffer.numberOfChannels;const bytes=frames*channels*2;
 if(frames<1)throw new RangeError('Audio selection is empty');
 const output=new ArrayBuffer(44+bytes);const view=new DataView(output);const str=(offset:number,value:string)=>{for(let i=0;i<value.length;i++)view.setUint8(offset+i,value.charCodeAt(i));};
 str(0,'RIFF');view.setUint32(4,36+bytes,true);str(8,'WAVE');str(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,channels,true);view.setUint32(24,buffer.sampleRate,true);view.setUint32(28,buffer.sampleRate*channels*2,true);view.setUint16(32,channels*2,true);view.setUint16(34,16,true);str(36,'data');view.setUint32(40,bytes,true);
 const data=Array.from({length:channels},(_,i)=>buffer.getChannelData(i));let offset=44;
 for(let i=start;i<end;i++)for(const channel of data){const sample=Math.max(-1,Math.min(1,channel[i]??0));view.setInt16(offset,Math.round((Number.isFinite(sample)?sample:0)*(sample<0?32768:32767)),true);offset+=2;}
 return output;
}
export async function decodeAudio(blob:Blob):Promise<AudioBuffer>{const context=new AudioContext();try{return await context.decodeAudioData(await blob.arrayBuffer());}finally{await context.close();}}

import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useState } from 'react';
import { setLanguage } from '@/i18n';
import { api } from '@/api/client';
import { Button, Disclosure, Field, ParamField, Select, Switch, TextInput } from './Controls';
import { ConfirmDialog, Toaster } from './Feedback';
import { getToasts, toast } from './toast';
import { AudioInput } from './Audio';
import { FileBrowserDialog } from './FileBrowser';
import { LogConsole } from './LogConsole';

beforeEach(()=>{
 setLanguage('en_US');
 vi.stubGlobal('ResizeObserver', class { observe(){} unobserve(){} disconnect(){} });
 Object.defineProperty(HTMLElement.prototype,'hasPointerCapture',{configurable:true,value:()=>false});
 Object.defineProperty(HTMLElement.prototype,'setPointerCapture',{configurable:true,value:()=>{}});
 Object.defineProperty(HTMLElement.prototype,'releasePointerCapture',{configurable:true,value:()=>{}});
});
afterEach(()=>{vi.restoreAllMocks();vi.unstubAllGlobals();for(const item of getToasts())toast.dismiss(item.id);});

describe('shared controls',()=>{
 it('disables a pending button and exposes its progress state',()=>{
  const click=vi.fn();render(<Button pending pendingLabel="Generating…" onClick={click}>Generate</Button>);
  const button=screen.getByRole('button',{name:'Generating…'});expect(button).toBeDisabled();expect(button).toHaveAttribute('aria-busy','true');fireEvent.click(button);expect(click).not.toHaveBeenCalled();
 });
 it('provides a focusable reason for unavailable actions',()=>{
  render(<Button disabledReason="Wait for the GPU">Generate</Button>);
  expect(screen.getByRole('button',{name:'Generate'})).toBeDisabled();expect(screen.getByLabelText('Wait for the GPU')).toHaveAttribute('tabindex','0');
 });
 it('connects labels, hints and validation errors to the actual input',()=>{
  render(<Field label="Project name" hint="A local folder name" error="Name is required"><TextInput/></Field>);
  const input=screen.getByLabelText('Project name');expect(input).toHaveAttribute('aria-invalid','true');expect(input).toHaveAccessibleDescription('A local folder name Name is required');expect(screen.getByRole('alert')).toHaveTextContent('Name is required');
 });
 it('supports empty Select options without the Radix empty-item error',()=>{
  render(<Field label="Adapter"><Select value="" onChange={()=>{}} options={[{value:'',label:'Original model'},{value:'voice-a',label:'Voice A'}]}/></Field>);
  expect(screen.getByRole('combobox',{name:'Adapter'})).toHaveTextContent('Original model');
 });
 it('resets a changed parameter and removes its modified affordance',async()=>{
  function Example(){const [value,setValue]=useState(0.8);return <ParamField label="Temperature" value={value} defaultValue={0.5} min={0} max={1} step={0.1} onChange={setValue}/>;}
  render(<Example/>);await userEvent.click(screen.getByRole('button',{name:'Reset to default'}));expect(screen.getByRole('spinbutton',{name:'Temperature'})).toHaveValue(0.5);expect(screen.queryByRole('button',{name:'Reset to default'})).not.toBeInTheDocument();
 });
 it('shows disabled switch reasons alongside the switch',()=>{
  const changed=vi.fn();render(<Switch label="Acceleration" checked={false} disabledReason="Not installed" onCheckedChange={changed}/>);expect(screen.getByRole('switch',{name:'Acceleration'})).toBeDisabled();expect(screen.getByText('Not installed')).toBeVisible();
 });
 it('discloses advanced controls without losing keyboard semantics',async()=>{
  render(<Disclosure title="Advanced" modifiedCount={2}><TextInput aria-label="Beam count"/></Disclosure>);const trigger=screen.getByRole('button',{name:/Advanced/});expect(trigger).toHaveAttribute('aria-expanded','false');await userEvent.click(trigger);expect(trigger).toHaveAttribute('aria-expanded','true');expect(screen.getByRole('textbox',{name:'Beam count'})).toBeVisible();
 });
});

describe('confirmation and feedback',()=>{
 it('requires exact confirmation text and prevents closing during a pending action',async()=>{
  let resolve!:()=>void;const confirm=vi.fn(()=>new Promise<void>(r=>{resolve=r;}));const changed=vi.fn();render(<ConfirmDialog open onOpenChange={changed} title="Delete project" description="Saved files will be removed." confirmLabel="Delete permanently" requireText="project-a" onConfirm={confirm}/>);
  const action=screen.getByRole('button',{name:'Delete permanently'});expect(action).toBeDisabled();await userEvent.type(screen.getByRole('textbox'),'project-a');expect(action).toBeEnabled();await userEvent.click(action);expect(action).toHaveAttribute('aria-busy','true');expect(screen.getByRole('button',{name:'Cancel'})).toBeDisabled();await userEvent.keyboard('{Escape}');expect(changed).not.toHaveBeenCalled();await act(async()=>resolve());expect(changed).toHaveBeenCalledWith(false);
 });
 it('keeps the confirmation open and shows the actual server failure',async()=>{
  const changed=vi.fn();render(<ConfirmDialog open onOpenChange={changed} title="Delete voice" confirmLabel="Delete" onConfirm={async()=>{throw new Error('Voice is in use');}}/>);await userEvent.click(screen.getByRole('button',{name:'Delete'}));expect(await screen.findByRole('alert')).toHaveTextContent('Voice is in use');expect(changed).not.toHaveBeenCalled();expect(screen.getByRole('button',{name:'Delete'})).toBeEnabled();
 });
 it('renders and dismisses translated toast notifications',async()=>{
  render(<Toaster/>);act(()=>{toast.success('Voice saved',{description:'Available in your library'});});expect(await screen.findByText('Voice saved')).toBeVisible();expect(screen.getByText('Available in your library')).toBeVisible();await userEvent.click(screen.getByRole('button',{name:'Dismiss notification'}));expect(screen.queryByText('Voice saved')).not.toBeInTheDocument();
 });
});

describe('audio, files and console',()=>{
 it('disables recording explicitly on an insecure LAN origin without requesting permission',()=>{
  vi.stubGlobal('isSecureContext',false);render(<AudioInput label="Reference audio" value={null} onChange={()=>{}}/>);expect(screen.getByRole('button',{name:'Record audio'})).toBeDisabled();expect(screen.getByLabelText('Recording requires HTTPS or localhost. Upload an audio file on this connection.')).toBeInTheDocument();expect(screen.getByRole('button',{name:'Choose file'})).toBeEnabled();
 });
 it('releases microphone tracks if recorder construction fails',async()=>{
  const stop=vi.fn();vi.stubGlobal('isSecureContext',true);vi.stubGlobal('navigator',{...navigator,mediaDevices:{getUserMedia:vi.fn().mockResolvedValue({getTracks:()=>[{stop}]})}});vi.stubGlobal('MediaRecorder',class{constructor(){throw new Error('Unsupported recorder');}});
  render(<AudioInput label="Reference audio" value={null} onChange={()=>{}}/>);await userEvent.click(screen.getByRole('button',{name:'Record audio'}));expect(await screen.findByRole('alert')).toHaveTextContent('Unsupported recorder');expect(stop).toHaveBeenCalledOnce();expect(screen.getByRole('button',{name:'Record audio'})).toBeEnabled();
 });
 it('rejects over-limit audio before uploading',async()=>{
  render(<AudioInput label="Reference audio" value={null} onChange={()=>{}}/>);const file=new File(['audio'],'large.wav',{type:'audio/wav'});Object.defineProperty(file,'size',{value:201*1024*1024});fireEvent.change(screen.getByLabelText('Choose audio file'),{target:{files:[file]}});expect(await screen.findByRole('alert')).toHaveTextContent('Audio must be smaller than 200 MB.');
 });
 it('selects a server-side file with no destructive side effects',async()=>{
  vi.spyOn(api,'get').mockImplementation(async(path)=>path==='/api/fs/roots'?[{label:'Workspace',path:'D:/Workspace'}]:{path:'D:/Workspace',parent:'D:/',entries:[{name:'model.pth',path:'D:/Workspace/model.pth',type:'file',size:2048,modified:1}]});const selected=vi.fn();const changed=vi.fn();render(<FileBrowserDialog open onOpenChange={changed} title="Choose checkpoint" kind="pth" onSelect={selected}/>);await userEvent.click(await screen.findByRole('button',{name:/model.pth/}));await userEvent.click(screen.getByRole('button',{name:'Select file'}));expect(selected).toHaveBeenCalledWith('D:/Workspace/model.pth');expect(changed).toHaveBeenCalledWith(false);
 });
 it('shows a browse failure with a retry action',async()=>{
  vi.spyOn(api,'get').mockImplementation(async(path)=>{if(path==='/api/fs/roots')return [];throw new Error('Folder does not exist');});render(<FileBrowserDialog open onOpenChange={()=>{}} title="Choose folder" kind="dir" onSelect={()=>{}}/>);expect(await screen.findByRole('alert')).toHaveTextContent('Folder does not exist');expect(screen.getByRole('button',{name:'Retry'})).toBeEnabled();expect(screen.getByRole('button',{name:'Select folder'})).toBeDisabled();
 });
 it('replaces progress lines in place and clears stale output on reset',async()=>{
  const {rerender}=render(<LogConsole lines={[{id:1,text:'Starting',level:'info'},{id:2,text:'10%',level:'progress'}]}/>);rerender(<LogConsole lines={[{id:1,text:'Starting',level:'info'},{id:2,text:'100%',level:'progress'}]}/>);expect(screen.queryByText('10%')).not.toBeInTheDocument();expect(screen.getByText('100%')).toBeVisible();rerender(<LogConsole lines={[]}/>);await waitFor(()=>expect(screen.getByText('No log output yet.')).toBeVisible());expect(screen.queryByText('Starting')).not.toBeInTheDocument();
 });
});

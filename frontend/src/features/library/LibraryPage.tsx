import { useState } from 'react';
import { useNavigate } from 'react-router';
import { AudioLines, FileText, Pencil, Play, Plus, RefreshCw, Save, Search, Trash2 } from 'lucide-react';
import { AudioPlayer, Badge, Button, ConfirmDialog, Dialog, EmptyState, Field, IconButton, InlineAlert, PageHeader, RouteTransition, Section, SegmentedControl, Skeleton, TextArea, TextInput, toast } from '@/ui';
import { formatBytes, formatDuration } from '@/ui/format';
import { errorMessage, isApiError, mediaUrl } from '@/api/client';
import { useT } from '@/i18n';
import { FORM_STORAGE_KEY, readSavedForm } from '@/features/synthesis/model';
import { VoiceEditorDialog } from './VoiceEditorDialog';
import { TranscribeButton } from './TranscribeButton';
import { useDeleteDescription, useDeleteVoice, useDescriptions, useRenameVoice, useSaveDescription, useSaveVoice, useVoices, type EmotionDescription, type Voice } from './api';
import styles from './Library.module.css';

export default function LibraryPage() {
  const t = useT(); const voices = useVoices(); const descriptions = useDescriptions();
  const [tab, setTab] = useState<'voices' | 'descriptions'>('voices'); const [search, setSearch] = useState('');
  const [selected, setSelected] = useState(''); const [createVoice, setCreateVoice] = useState(false); const [editVoice, setEditVoice] = useState<Voice | null>(null);
  const [editingDescription, setEditingDescription] = useState<EmotionDescription | 'new' | null>(null);
  const items = tab === 'voices' ? voices.data : descriptions.data;
  const visible = (items ?? []).filter((item) => item.name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()));
  const voice = tab === 'voices' ? voices.data?.find((item) => item.path === selected) : undefined;
  const description = tab === 'descriptions' ? descriptions.data?.find((item) => item.path === selected) : undefined;
  const query = tab === 'voices' ? voices : descriptions;
  const add = () => tab === 'voices' ? setCreateVoice(true) : setEditingDescription('new');
  return <div className={styles.page}>
    <PageHeader title={t('Voice library')} description={t('Reusable voices and delivery notes. Stored locally, ready for your next creation.')} actions={<Button icon={Plus} onClick={add}>{tab === 'voices' ? t('Add voice') : t('Add description')}</Button>} />
    <SegmentedControl value={tab} onChange={(value) => { setTab(value); setSearch(''); setSelected(''); }} options={[{ value: 'voices', label: t('Reference voices') }, { value: 'descriptions', label: t('Emotion descriptions') }]} />
    <div className={styles.layout}>
      <Section title={tab === 'voices' ? t('Reference voices') : t('Emotion descriptions')} actions={<IconButton icon={RefreshCw} label={t('Refresh library')} pending={query.isFetching} onClick={() => void query.refetch()} />}>
        <div className={styles.stack}><TextInput aria-label={t('Search library')} value={search} onChange={(event) => setSearch(event.target.value)} placeholder={t('Search by name…')} />
          {query.isPending ? <Skeleton height={180} /> : query.error ? <InlineAlert tone="error">{errorMessage(query.error)}</InlineAlert> : !visible.length ? <EmptyState icon={search ? Search : AudioLines} title={search ? t('No matching items') : t('Make this library yours')} description={search ? t('Try another name.') : t('Add a reference voice or emotion description to get started.')} action={!search ? <Button variant="tonal" icon={Plus} onClick={add}>{t('Add your first item')}</Button> : undefined} /> :
            <div className={styles.list} role="list" aria-label={tab === 'voices' ? t('Reference voices') : t('Emotion descriptions')}>{visible.map((item) => <button key={item.path} type="button" className={styles.item} data-selected={item.path === selected} aria-pressed={item.path === selected} onClick={() => setSelected(item.path)}>
              {tab === 'voices' ? <AudioLines size={20} /> : <FileText size={20} />}<span className={styles.itemText}><span className={styles.name}>{item.name}</span><span className={styles.meta}>{'duration' in item ? `${formatDuration(item.duration ?? 0)} · ${formatBytes(item.size)}` : t('{0} characters', item.text.length)}</span></span>
            </button>)}</div>}
        </div>
      </Section>
      <RouteTransition scope={`${tab}:${selected}`}>{voice ? <VoiceDetail key={voice.path} voice={voice} onEdit={() => setEditVoice(voice)} onDeleted={() => setSelected('')} /> : description ? <DescriptionDetail key={description.path} description={description} onEdit={() => setEditingDescription(description)} onDeleted={() => setSelected('')} /> : <Section><EmptyState icon={tab === 'voices' ? AudioLines : FileText} title={t('Select an item to preview')} description={t('Listen, edit, or reuse a saved reference without leaving the page.')} /></Section>}</RouteTransition>
    </div>
    <VoiceEditorDialog open={createVoice || Boolean(editVoice)} onOpenChange={(open) => { if (!open) { setCreateVoice(false); setEditVoice(null); } }} initial={editVoice ?? undefined} onSaved={(path) => setSelected(path)} />
    {editingDescription && <DescriptionEditor value={editingDescription} onClose={() => setEditingDescription(null)} onSaved={setSelected} />}
  </div>;
}

function VoiceDetail({ voice, onEdit, onDeleted }: { voice: Voice; onEdit: () => void; onDeleted: () => void }) {
  const t = useT(); const navigate = useNavigate(); const save = useSaveVoice(); const rename = useRenameVoice(); const remove = useDeleteVoice();
  const [transcript, setTranscript] = useState(voice.transcript ?? ''); const [renameOpen, setRenameOpen] = useState(false); const [name, setName] = useState(voice.name.replace(/\.[^.]+$/, '')); const [deleteOpen, setDeleteOpen] = useState(false);
  const [failure, setFailure] = useState('');
  function useVoice() {
    try { localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify({ ...readSavedForm(), voice_path: voice.path, voice_name: voice.name, reference_transcript: transcript })); navigate('/synthesis'); }
    catch { toast.error(t('Browser storage is unavailable. Choose this voice on the synthesis page.')); }
  }
  return <Section title={voice.name} actions={<div className={styles.row}><IconButton icon={Pencil} label={t('Rename voice')} onClick={() => setRenameOpen(true)} /><IconButton icon={Trash2} label={t('Delete voice')} onClick={() => setDeleteOpen(true)} /></div>}>
    <div className={styles.stack}><AudioPlayer src={mediaUrl(voice.path, voice.modified)} title={voice.name} downloadName={voice.name} />
      <div className={styles.row}><Badge>{formatDuration(voice.duration ?? 0)}</Badge><Badge>{formatBytes(voice.size)}</Badge></div>
      <Field label={t('Reference transcript')} hint={t('Optional. Use the exact words spoken in the recording.')}><TextArea aria-label={t('Reference transcript')} value={transcript} onChange={(event) => setTranscript(event.target.value)} minRows={4} /></Field>
      <div className={styles.row}><TranscribeButton path={voice.path} onTranscript={setTranscript} /><Button variant="outlined" size="sm" icon={Save} pending={save.isPending} disabled={transcript === (voice.transcript ?? '')} onClick={async () => { setFailure(''); try { await save.mutateAsync({ name: voice.name.replace(/\.[^.]+$/, ''), source_path: voice.path, transcript, overwrite: true }); } catch (error) { setFailure(errorMessage(error)); } }}>{t('Save transcript')}</Button></div>
      {failure && <InlineAlert tone="error">{failure}</InlineAlert>}
      <div className={styles.row}><Button icon={Play} onClick={useVoice}>{t('Use in synthesis')}</Button><Button variant="text" onClick={onEdit}>{t('Edit recording')}</Button></div>
    </div>
    <Dialog open={renameOpen} onOpenChange={setRenameOpen} title={t('Rename voice')} footer={<><Button variant="text" onClick={() => setRenameOpen(false)}>{t('Cancel')}</Button><Button pending={rename.isPending} disabled={!name.trim()} onClick={async () => { try { await rename.mutateAsync({ name: voice.name, new_name: name.trim() }); setRenameOpen(false); onDeleted(); } catch { /* Mutation error is rendered in the dialog. */ } }}>{t('Rename')}</Button></>}>
      <Field label={t('Voice name')}><TextInput aria-label={t('Voice name')} autoFocus value={name} onChange={(event) => setName(event.target.value)} /></Field>{rename.error && <InlineAlert tone="error">{errorMessage(rename.error)}</InlineAlert>}
    </Dialog>
    <ConfirmDialog open={deleteOpen} onOpenChange={setDeleteOpen} title={t('Delete this voice?')} description={t('The audio and its transcript will be permanently deleted. Existing synthesized audio is kept.')} tone="danger" confirmLabel={t('Delete voice')} onConfirm={async () => { await remove.mutateAsync(voice.name); setDeleteOpen(false); onDeleted(); }} />
  </Section>;
}
function DescriptionDetail({ description, onEdit, onDeleted }: { description: EmotionDescription; onEdit: () => void; onDeleted: () => void }) {
  const t = useT(); const remove = useDeleteDescription(); const navigate = useNavigate(); const [deleteOpen, setDeleteOpen] = useState(false);
  return <Section title={description.name} actions={<IconButton icon={Trash2} label={t('Delete description')} onClick={() => setDeleteOpen(true)} />}><div className={styles.stack}>
    <p className={styles.description}>{description.text}</p><div className={styles.row}><Button icon={Pencil} variant="tonal" onClick={onEdit}>{t('Edit description')}</Button><Button variant="text" onClick={() => { try { localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify({ ...readSavedForm(), emotion_mode: 'Emotion description', emotion_text: description.text })); navigate('/synthesis'); } catch { toast.error(t('Browser storage is unavailable. Choose this voice on the synthesis page.')); } }}>{t('Use in synthesis')}</Button></div>
  </div><ConfirmDialog open={deleteOpen} onOpenChange={setDeleteOpen} title={t('Delete this description?')} description={description.name} confirmLabel={t('Delete description')} tone="danger" onConfirm={async () => { await remove.mutateAsync(description.name); setDeleteOpen(false); onDeleted(); }} /></Section>;
}
function DescriptionEditor({ value, onClose, onSaved }: { value: EmotionDescription | 'new'; onClose: () => void; onSaved: (path: string) => void }) {
  const t = useT(); const [name, setName] = useState(value === 'new' ? '' : value.name.replace(/\.txt$/i, '')); const [text, setText] = useState(value === 'new' ? '' : value.text);
  const [overwrite, setOverwrite] = useState(false); const [failure, setFailure] = useState(''); const save = useSaveDescription();
  async function persist(replace: boolean) { const result = await save.mutateAsync({ name: name.trim(), text: text.trim(), overwrite: replace }); onSaved(result.path); onClose(); }
  return <><Dialog open onOpenChange={(open) => { if (!open && !save.isPending) onClose(); }} title={value === 'new' ? t('Add emotion description') : t('Edit description')} footer={<><Button variant="text" onClick={onClose}>{t('Cancel')}</Button><Button pending={save.isPending} disabled={!name.trim() || !text.trim()} onClick={async () => { setFailure(''); try { await persist(value !== 'new' && name.trim() === value.name.replace(/\.txt$/i, '')); } catch (error) { if (isApiError(error) && error.status === 409) setOverwrite(true); else setFailure(errorMessage(error)); } }}>{t('Save description')}</Button></>}>
    <div className={styles.stack}><Field label={t('Description name')}><TextInput aria-label={t('Description name')} autoFocus value={name} onChange={(event) => setName(event.target.value)} /></Field><Field label={t('Delivery description')}><TextArea aria-label={t('Delivery description')} value={text} onChange={(event) => setText(event.target.value)} minRows={5} placeholder={t('Warm, reassuring, with a gentle pace…')} /></Field>{failure && <InlineAlert tone="error">{failure}</InlineAlert>}</div>
  </Dialog><ConfirmDialog open={overwrite} onOpenChange={setOverwrite} title={t('Replace the existing description?')} description={name} confirmLabel={t('Replace description')} tone="danger" onConfirm={() => persist(true)} /></>;
}

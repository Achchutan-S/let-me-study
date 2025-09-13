import React, { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useQAStore, QAItem } from './store'
import { api, Paper } from './api'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'

const BACKEND_BASE = 'http://localhost:8000'

function stripMetaFromMarkdown(md: string): { markdown: string; topicTitle?: string; tags?: any } {
	if (!md) return { markdown: md }
	const re = /```(?:json)?\n([\s\S]*?)\n```\s*$/
	const m = md.match(re)
	if (!m) return { markdown: md }
	let topicTitle: string | undefined
	let tags: any
	try {
		const obj = JSON.parse(m[1])
		if (typeof obj.topicTitle === 'string') topicTitle = obj.topicTitle
		if (obj.tags && typeof obj.tags === 'object') tags = obj.tags
	} catch {}
	return { markdown: md.replace(re, '').trim(), topicTitle, tags }
}

function TagFilter(): JSX.Element {
	const { computeAvailableTags, setTagFilter, clearTagFilters, filters, refreshTags } = useQAStore() as any
	const [open, setOpen] = useState(true)
	const tags = computeAvailableTags()
	const activeChips: { group: 'subject'|'topic'|'concept'|'ai_keywords'; value: string }[] = []
	for (const v of Array.from(filters.subject) as string[]) activeChips.push({ group: 'subject', value: v })
	for (const v of Array.from(filters.topic) as string[]) activeChips.push({ group: 'topic', value: v })
	for (const v of Array.from(filters.concept) as string[]) activeChips.push({ group: 'concept', value: v })
	for (const v of Array.from(filters.ai_keywords) as string[]) activeChips.push({ group: 'ai_keywords', value: v })
	const totalTags = tags.subject.length + tags.topic.length + tags.concept.length + tags.ai_keywords.length
	return (
		<div className="bg-white border rounded mb-3">
			<button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between px-3 py-2">
				<span className="text-sm font-medium text-gray-900">Filters</span>
				<span className="text-xs text-gray-500">{open ? '▾' : '▸'} {totalTags} tags</span>
			</button>
			{open && (
				<div className="p-3 border-t">
					<div className="flex items-center justify-between mb-2">
						<div className="flex flex-wrap gap-2">
							{activeChips.map((c, idx) => (
								<button key={idx} onClick={() => setTagFilter(c.group, c.value, false)} className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 hover:bg-indigo-100">{c.group}:{c.value} ×</button>
							))}
						</div>
						<div className="flex items-center gap-2">
							<button onClick={() => void refreshTags()} title="Refresh from DB" className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200">⟳</button>
							<button onClick={clearTagFilters} className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200">Clear</button>
						</div>
					</div>
					<div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
						<div>
							<div className="font-semibold mb-1">Subject</div>
							<div className="flex flex-wrap gap-2 max-h-28 overflow-auto">
												{tags.subject.map((t: string) => (
													<label key={`sub-${t}`} className="inline-flex items-center gap-1">
														<input type="checkbox" checked={filters.subject.has(t)} onChange={(e) => setTagFilter('subject', t, e.target.checked)} aria-label={`Filter by subject ${t}`} /> <span className="truncate max-w-[9rem]">{t}</span>
													</label>
												))}
							</div>
						</div>
						<div>
							<div className="font-semibold mb-1">Topic</div>
							<div className="flex flex-wrap gap-2 max-h-28 overflow-auto">
												{tags.topic.map((t: string) => (
													<label key={`top-${t}`} className="inline-flex items-center gap-1">
														<input type="checkbox" checked={filters.topic.has(t)} onChange={(e) => setTagFilter('topic', t, e.target.checked)} aria-label={`Filter by topic ${t}`} /> <span className="truncate max-w-[9rem]">{t}</span>
													<span className="sr-only">Filter by topic {t}</span></label>
												))}
							</div>
						</div>
						<div>
							<div className="font-semibold mb-1">Concept</div>
							<div className="flex flex-wrap gap-2 max-h-28 overflow-auto">
												{tags.concept.map((t: string) => (
													<label key={`con-${t}`} className="inline-flex items-center gap-1">
														<input type="checkbox" checked={filters.concept.has(t)} onChange={(e) => setTagFilter('concept', t, e.target.checked)} aria-label={`Filter by concept ${t}`} /> <span className="truncate max-w-[9rem]">{t}</span>
													<span className="sr-only">Filter by concept {t}</span></label>
												))}
							</div>
						</div>
						<div>
							<div className="font-semibold mb-1">Keywords</div>
							<div className="flex flex-wrap gap-2 max-h-28 overflow-auto">
												{tags.ai_keywords.map((t: string) => (
													<label key={`kw-${t}`} className="inline-flex items-center gap-1">
														<input type="checkbox" checked={filters.ai_keywords.has(t)} onChange={(e) => setTagFilter('ai_keywords', t, e.target.checked)} aria-label={`Filter by keyword ${t}`} /> <span className="truncate max-w-[9rem]">{t}</span>
													<span className="sr-only">Filter by keyword {t}</span></label>
												))}
							</div>
						</div>
					</div>
				</div>
			)}
		</div>
	)
}

// --- Text-to-Speech helpers ---
function mdToPlainText(md: string): string {
	return md
		.replace(/```[\s\S]*?```/g, '')
		.replace(/`([^`]+)`/g, '$1')
		.replace(/^#+\s+/gm, '')
		.replace(/\*\*([^*]+)\*\*/g, '$1')
		.replace(/\*([^*]+)\*/g, '$1')
		.replace(/_([^_]+)_/g, '$1')
		.replace(/\[(.*?)\]\([^)]*\)/g, '$1')
}

let currentUtterance: SpeechSynthesisUtterance | null = null
function speakMarkdown(md: string) {
	const text = mdToPlainText(md)
	if (!('speechSynthesis' in window)) return
	if (currentUtterance) { window.speechSynthesis.cancel(); currentUtterance = null }
	const utter = new SpeechSynthesisUtterance(text)
	utter.rate = 1
	utter.pitch = 1
	utter.lang = 'en-US'
	currentUtterance = utter
	window.speechSynthesis.speak(utter)
}
function stopSpeaking() {
	if (!('speechSynthesis' in window)) return
	window.speechSynthesis.cancel(); currentUtterance = null
}

// --- Main component ---
export default function App(): JSX.Element {
	// Local transient only used immediately after processing
	const [file, setFile] = useState(null as File | null)
	const [imagePreview, setImagePreview] = useState(null as string | null)
	const [transientQuestionText, setTransientQuestionText] = useState('')
	const [transientMarkdown, setTransientMarkdown] = useState('')
	const [transientTitle, setTransientTitle] = useState('')
	const [isLoading, setIsLoading] = useState(false)
	const [error, setError] = useState('')
	const [showUploader, setShowUploader] = useState(false)
	const dropRef = useRef(null as HTMLDivElement | null)

	// Semantic search state
	const [semanticQuery, setSemanticQuery] = useState('')
	const [isSearching, setIsSearching] = useState(false)
	const [semanticResults, setSemanticResults] = useState<Array<{
		question_number?: number;
		topicTitle?: string;
		tags?: { [key: string]: string[] };
		cleaned_question_text: string;
		gemini_markdown?: string;
		similarity_score: number;
	}>>([])

	const handleSemanticSearch = async () => {
		if (!semanticQuery.trim() || !currentPaperId) return
		setIsSearching(true)
		try {
			const results = await api.semanticSearch(semanticQuery.trim(), currentPaperId)
			setSemanticResults(results)
			// Update the filtered items to show search results
			if (results.length > 0) {
				const questionNumbers = results.map(r => r.question_number).filter(n => n !== undefined) as number[]
				// Create a mapping of question numbers to similarity scores
				const scoreMap = new Map(results.map(r => [r.question_number, r.similarity_score]))
				
				// Sort items based on semantic search results
				const currentItems = getFilteredItems()
				currentItems.sort((a: QAItem, b: QAItem) => {
					const aNumber = parseInt(a.id.split('-')[1])
					const bNumber = parseInt(b.id.split('-')[1])
					const aScore = scoreMap.get(aNumber) || -1
					const bScore = scoreMap.get(bNumber) || -1
					return bScore - aScore // Higher scores first
				})
			}
		} catch (err: any) {
			setError(err?.message || 'Search failed')
		} finally {
			setIsSearching(false)
		}
	}

	const clearSemanticSearch = () => {
		setSemanticQuery('')
		setSemanticResults([])
	}

	const {
		papers, currentPaperId, loading, saveMessage,
		items, currentId, selectedIds,
		fetchPapers, createPaper, loadQuestions, addFromProcess, select,
		toggleSelect, selectAll, clearSelection, deleteSelected, getSelectedItems,
		setCurrentPaper, getFilteredItems,
	} = useQAStore()

	const filteredItems = getFilteredItems()
	const currentItem: QAItem | null = useMemo(() => filteredItems.find((i: QAItem) => i.id === currentId) || filteredItems[0] || null, [filteredItems, currentId])

	const displayTitle = currentItem?.topicTitle || transientTitle || currentPaperId || 'Select a paper'
	const displayQuestion = currentItem?.questionText || transientQuestionText
	const displayMarkdown = currentItem?.markdown || transientMarkdown

	useEffect(() => { void fetchPapers() }, [])
	useEffect(() => { if (currentPaperId) void loadQuestions(currentPaperId) }, [currentPaperId])

	// Global paste handler to open uploader and set file
	useEffect(() => {
		function onPaste(e: ClipboardEvent) {
			const items = e.clipboardData?.items
			if (!items) return
			for (const it of items as unknown as DataTransferItemList) {
				if (it.type.startsWith('image/')) {
					const blob = (it as DataTransferItem).getAsFile()
					if (blob) {
						const f = new File([blob], 'pasted.png', { type: blob.type })
						setShowUploader(true)
						setFile(f)
						setImagePreview(URL.createObjectURL(f))
					}
					break
				}
			}
		}
		document.addEventListener('paste', onPaste as any)
		return () => document.removeEventListener('paste', onPaste as any)
	}, [])

	function onFileChange(e: React.ChangeEvent<HTMLInputElement>): void {
		const f = e.target.files?.[0]
		if (!f) return
		if (!['image/png', 'image/jpeg', 'image/jpg'].includes(f.type)) { setError('Please use PNG or JPG image.'); return }
		setError('')
		setFile(f)
		setImagePreview(URL.createObjectURL(f))
	}

	async function handleOCR(f?: File): Promise<void> {
		const source = f ?? file
		if (!source || !currentPaperId) { setError('Select a paper first'); return }
		setIsLoading(true)
		setError('')
		setTransientQuestionText('')
		setTransientMarkdown('')
		setTransientTitle('')
		try {
			const res = await api.processScreenshot(source, currentPaperId)
			let md = res.gemini_markdown || ''
			let title = res.topicTitle || ''
			let tags = res.tags || undefined
			if (!title || !tags) {
				const parsed = stripMetaFromMarkdown(md)
				md = parsed.markdown
				if (parsed.topicTitle) title = parsed.topicTitle
			}
			setTransientQuestionText(res.cleaned_question_text || '')
			setTransientMarkdown(md)
			setTransientTitle(title)
			await addFromProcess(currentPaperId, { ...res, gemini_markdown: md, topicTitle: title, tags })
		} catch (err: any) {
			setError(err?.message || 'Processing failed')
		} finally {
			setIsLoading(false)
		}
	}

	async function exportPDFSelected(): Promise<void> {
		const toExport = getSelectedItems()
		if (toExport.length === 0) return
		await exportSpecific(toExport)
	}

	async function exportSpecific(list: QAItem[]): Promise<void> {
		const doc = new jsPDF({ unit: 'pt', format: 'a4' })
		const pageWidth = doc.internal.pageSize.getWidth()
		const pageHeight = doc.internal.pageSize.getHeight()
		for (let idx = 0; idx < list.length; idx++) {
			const item = list[idx]
			const container = document.createElement('div')
			container.style.width = '595pt'
			container.style.padding = '24px'
			container.style.fontFamily = 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji", "Segoe UI Emoji"'
			container.innerHTML = `
				<h1 style=\"font-size:18px;margin:0 0 8px 0;\">${escapeHtml(item.topicTitle)}</h1>
				<h2 style=\"font-size:13px;margin:0 0 8px 0;color:#374151;\">Question</h2>
				<pre style=\"white-space:pre-wrap;font-size:12px;line-height:1.4;border:1px solid #e5e7eb;border-radius:6px;padding:8px;margin:0 0 12px 0;\">${escapeHtml(item.questionText)}</pre>
				<h2 style=\"font-size:13px;margin:0 0 8px 0;color:#374151;\">Explanation</h2>
				<div style=\"font-size:12px;line-height:1.5;\">${markdownToHtmlBasic(item.markdown)}</div>
			`
			document.body.appendChild(container)
			const canvas = await html2canvas(container as HTMLElement, { scale: 2 })
			document.body.removeChild(container)
			const imgData = canvas.toDataURL('image/png')
			if (idx > 0) doc.addPage()
			doc.addImage(imgData, 'PNG', 0, 0, pageWidth, pageHeight)
		}
		doc.save(`${currentPaperId || 'paper'}.pdf`)
	}

	const components = {
		code({ inline, className, children, ...props }: any) {
			const match = /language-(\w+)/.exec(className || '')
			return !inline && match ? (
				<SyntaxHighlighter style={oneDark as any} language={match[1]} PreTag="div" {...(props as any)}>
					{String(children).replace(/\n$/, '')}
				</SyntaxHighlighter>
			) : (
				<code className={className} {...(props as any)}>{children}</code>
			)
		},
	}

	return (
		<div className="min-h-screen grid grid-rows-[auto,1fr]">
			<header className="flex items-center justify-between px-4 py-3 border-b bg-white">
				<div className="flex items-center gap-2">
					<h1 className="text-lg font-semibold text-gray-900">Exam Helper</h1>
					{loading && <span className="text-xs text-gray-500">Loading…</span>}
					{saveMessage && <span className="text-xs text-green-600">{saveMessage}</span>}
				</div>
				<div className="flex items-center gap-2">
					<button onClick={() => setShowUploader(v => !v)} className="text-sm px-3 py-1 rounded bg-gray-100 hover:bg-gray-200">{showUploader ? 'Hide Upload' : 'Upload/Paste'}</button>
					<select value={currentPaperId || ''} onChange={(e) => setCurrentPaper(e.target.value)} className="text-sm border rounded px-2 py-1">
						<option value="" disabled>Select Paper</option>
						{papers.map((p: Paper) => (
							<option key={p.paperId} value={p.paperId}>{p.title}</option>
						))}
					</select>
					<button onClick={async () => {
						const id = prompt('New Paper ID (e.g., CS-2025-Paper-I)') || ''
						if (!id) return
						const title = prompt('Paper title', id) || id
						await createPaper(id, title, ['CS','GATE'])
					}} className="text-sm px-3 py-1 rounded bg-indigo-600 text-white">New Paper</button>
				</div>
			</header>

			<div className="grid grid-cols-1 md:grid-cols-[280px,1fr]">
				<aside className="border-r bg-white p-4 hidden md:block overflow-y-auto">
					<div className="mb-4">
						<div className="flex items-center gap-2 mb-2">
							<input 
								type="text" 
								placeholder="Semantic search..." 
								value={semanticQuery}
								onChange={(e) => setSemanticQuery(e.target.value)}
								className="flex-1 text-sm border rounded px-3 py-1.5"
								onKeyDown={(e) => e.key === 'Enter' && handleSemanticSearch()}
							/>
							<button 
								onClick={handleSemanticSearch} 
								disabled={!semanticQuery.trim() || isSearching}
								className="text-sm px-3 py-1.5 rounded bg-indigo-600 text-white disabled:opacity-50">
								{isSearching ? 'Searching...' : 'Search'}
							</button>
						</div>
						{semanticResults.length > 0 && (
							<div className="text-xs text-gray-500 mb-2">
								Found {semanticResults.length} results. Showing questions ordered by relevance.
								<button onClick={clearSemanticSearch} className="ml-2 text-indigo-600 hover:underline">Clear results</button>
							</div>
						)}
					</div>
					<TagFilter />
					<div className="flex items-center justify-between mb-3">
						<h2 className="font-semibold text-gray-900">Questions</h2>
						<div className="flex items-center gap-2">
							<button onClick={exportPDFSelected} className="text-xs px-2 py-1 rounded bg-indigo-50 text-indigo-700 hover:bg-indigo-100">Export Selected</button>
							<button onClick={deleteSelected} className="text-xs px-2 py-1 rounded bg-red-50 text-red-700 hover:bg-red-100" aria-label="Delete selected">🗑️</button>
						</div>
					</div>
					<div className="flex items-center justify-between mb-2 text-xs">
						<button onClick={selectAll} className="px-2 py-1 rounded bg-gray-100 hover:bg-gray-200">Select All</button>
						<button onClick={clearSelection} className="px-2 py-1 rounded bg-gray-100 hover:bg-gray-200">Clear</button>
					</div>
					<div className="space-y-2">
										{filteredItems.map((q: QAItem) => (
											<div key={q.id} className={`w-full p-2 rounded border ${currentId === q.id ? 'border-indigo-400 bg-indigo-50' : 'border-gray-200 hover:bg-gray-50'}`}>
												<div className="flex items-start gap-2">
													<input type="checkbox" checked={selectedIds.has(q.id)} onChange={() => toggleSelect(q.id)} className="mt-1" aria-label={`Select question ${q.topicTitle || 'Untitled'}`} />
													<button onClick={() => select(q.id)} className="text-left flex-1">
														<div className="text-sm font-medium text-gray-900 truncate overflow-hidden whitespace-nowrap max-w-[180px]">{q.topicTitle || 'Untitled'}</div>
														<div className="text-xs text-gray-500 max-h-10 overflow-hidden line-clamp-2">{q.questionText}</div>
														{q.tags && (
															<div className="text-[10px] text-gray-500 mt-1 truncate overflow-hidden whitespace-nowrap max-w-[180px]">{[...(q.tags.subject||[]), ...(q.tags.topic||[]), ...(q.tags.concept||[])].slice(0,3).join(' • ')}</div>
														)}
													</button>
												</div>
											</div>
										))}
						{filteredItems.length === 0 && <div className="text-xs text-gray-500">No questions match filters</div>}
					</div>
				</aside>

				<main className="p-4">
					<div className="max-w-3xl mx-auto">
						{showUploader && (
							<div className="bg-white rounded-xl border p-4 shadow-sm mb-4">
								<div ref={dropRef} className="border-2 border-dashed rounded-lg p-4 text-center">
									<input type="file" accept="image/png, image/jpeg" onChange={onFileChange} className="block w-full text-sm text-gray-900 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100" aria-label="Upload image file" />
									<div className="text-xs text-gray-500 mt-2">Paste an image (Cmd/Ctrl+V) or choose a file</div>
								</div>
								{imagePreview && <img src={imagePreview} alt="preview" className="mt-3 max-h-48 rounded border" />}
								<div className="mt-3 flex gap-2">
									<button onClick={() => handleOCR()} disabled={!file || isLoading || !currentPaperId} className="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2 text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-indigo-700 transition">{isLoading ? 'Processing…' : 'Process Image'}</button>
									{error && <span className="text-sm text-red-600">{error}</span>}
								</div>
							</div>
						)}

						<div className="bg-white rounded-xl border p-6 shadow-sm">
							<h2 className="text-lg font-semibold text-gray-900 mb-2">{displayTitle}</h2>
							{displayQuestion && (
								<div>
									<h3 className="mt-4 mb-2 font-medium text-gray-900">Question</h3>
									<div className="max-h-48 overflow-auto rounded-md border p-3 text-sm whitespace-pre-wrap">{displayQuestion}</div>
								</div>
							)}
							{displayMarkdown && (
								<div className="mt-4">
									<div className="flex items-center justify-between mb-1">
										<h3 className="font-medium text-gray-900">AI Explanation</h3>
										<div className="flex gap-2">
											<button onClick={() => speakMarkdown(displayMarkdown)} className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200">Listen</button>
											<button onClick={() => stopSpeaking()} className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200">Stop</button>
										</div>
									</div>
									<div className="prose prose-sm max-w-none">
										<ReactMarkdown components={{
											code({ inline, className, children, ...props }: any) {
												const match = /language-(\w+)/.exec(className || '')
												return !inline && match ? (
													<SyntaxHighlighter style={oneDark as any} language={match[1]} PreTag="div" {...(props as any)}>
														{String(children).replace(/\n$/, '')}
													</SyntaxHighlighter>
												) : (
													<code className={className} {...(props as any)}>{children}</code>
												)
										}
									}}>
										{displayMarkdown}
									</ReactMarkdown>
									</div>
								</div>
							)}
							{isLoading && <div className="mt-4 text-sm text-gray-500">Processing, please wait…</div>}
						</div>
					</div>
				</main>
			</div>
		</div>
	)
}

function escapeHtml(s: string): string { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') }
function markdownToHtmlBasic(md: string): string { return md.replace(/^###\s(.+)$/gm,'<h3>$1</h3>').replace(/^##\s(.+)$/gm,'<h2>$1</h2>').replace(/^#\s(.+)$/gm,'<h1>$1</h1>').replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\*(.+?)\*/g,'<em>$1</em>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\n/g,'<br/>') }

import { create } from 'zustand'
import { api, Paper } from './api'

export type QAItem = {
	id: string
	topicTitle: string
	questionText: string
	markdown: string
	createdAt: number
	tags?: { subject?: string[]; topic?: string[]; concept?: string[]; ai_keywords?: string[] }
}

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

type TagFilters = {
	subject: Set<string>
	topic: Set<string>
	concept: Set<string>
	ai_keywords: Set<string>
}

type StoreState = {
	papers: Paper[]
	currentPaperId: string | null
	loading: boolean
	saveMessage: string
	items: QAItem[]
	currentId: string | null
	selectedIds: Set<string>
	filters: TagFilters
	setTagFilter: (group: keyof TagFilters, tag: string, checked: boolean) => void
	clearTagFilters: () => void
	getFilteredItems: () => QAItem[]
	computeAvailableTags: () => { subject: string[]; topic: string[]; concept: string[]; ai_keywords: string[] }
	refreshTags: () => Promise<void>
	setCurrentPaper: (paperId: string) => void
	fetchPapers: () => Promise<void>
	createPaper: (paperId: string, title: string, tags?: string[]) => Promise<void>
	loadQuestions: (paperId: string) => Promise<void>
	addFromProcess: (paperId: string, result: any) => Promise<void>
	addItem: (item: Omit<QAItem, 'id' | 'createdAt'>) => string
	select: (id: string) => void
	clear: () => void
	toggleSelect: (id: string) => void
	selectAll: () => void
	clearSelection: () => void
	deleteSelected: () => Promise<void>
	deleteQuestion: (paperId: string, question_number: number) => Promise<void>
	removeTag: (paperId: string, question_number: number, group: keyof TagFilters, value: string) => Promise<void>
	getSelectedItems: () => QAItem[]
}

const emptyFilters = (): TagFilters => ({ subject: new Set(), topic: new Set(), concept: new Set(), ai_keywords: new Set() })

export const useQAStore = create<StoreState>((set, get) => ({
	papers: [],
	currentPaperId: null,
	loading: false,
	saveMessage: '',
	items: [],
	currentId: null,
	selectedIds: new Set<string>(),
	filters: emptyFilters(),
	setTagFilter: (group, tag, checked) => {
		const filters = get().filters
		const next = new Set(filters[group])
		if (checked) next.add(tag); else next.delete(tag)
		set({ filters: { ...filters, [group]: next } })
	},
	clearTagFilters: () => set({ filters: emptyFilters() }),
	getFilteredItems: () => {
		const { items, filters } = get()
		const hasAny = (arr: Set<string>) => arr.size > 0
		return items.filter((it) => {
			const tg = it.tags || {}
			const groups: (keyof TagFilters)[] = ['subject','topic','concept','ai_keywords']
			for (const g of groups) {
				const selected = filters[g]
				if (!hasAny(selected)) continue
				const values = new Set((tg as any)[g] || [])
				for (const s of selected) {
					if (!values.has(s)) return false
				}
			}
			return true
		})
	},
	computeAvailableTags: () => {
		const all = { subject: new Set<string>(), topic: new Set<string>(), concept: new Set<string>(), ai_keywords: new Set<string>() }
		for (const it of get().items) {
			if (!it.tags) continue
			for (const v of it.tags.subject || []) all.subject.add(v)
			for (const v of it.tags.topic || []) all.topic.add(v)
			for (const v of it.tags.concept || []) all.concept.add(v)
			for (const v of it.tags.ai_keywords || []) all.ai_keywords.add(v)
		}
		return {
			subject: Array.from(all.subject).sort(),
			topic: Array.from(all.topic).sort(),
			concept: Array.from(all.concept).sort(),
			ai_keywords: Array.from(all.ai_keywords).sort(),
		}
	},
	refreshTags: async () => {
		const pid = get().currentPaperId
		if (!pid) return
		await get().loadQuestions(pid)
	},
	setCurrentPaper: (paperId) => set({ currentPaperId: paperId }),
	fetchPapers: async () => {
		set({ loading: true })
		try {
			const res = await api.listPapers()
			set({ papers: res.papers })
			if (!get().currentPaperId && res.papers.length) set({ currentPaperId: res.papers[0].paperId })
		} finally {
			set({ loading: false })
		}
	},
	createPaper: async (paperId, title, tags) => {
		await api.createPaper({ paperId, title, tags })
		await get().fetchPapers()
		set({ currentPaperId: paperId, items: [], currentId: null, filters: emptyFilters() })
	},
	loadQuestions: async (paperId) => {
		set({ loading: true, currentPaperId: paperId })
		try {
			const res = await api.listQuestions(paperId)
			const qs = res.questions || []
			const items = qs.map((q: any) => {
				let md: string = q.gemini_markdown || ''
				let topicTitle: string = q.topicTitle || ''
				let tags = q.tags || {}
				if (!topicTitle || !tags || !tags.subject) {
					const parsed = stripMetaFromMarkdown(md)
					md = parsed.markdown
					if (!topicTitle && parsed.topicTitle) topicTitle = parsed.topicTitle
					if (!tags && parsed.tags) tags = parsed.tags
				}
				// normalize keywords -> ai_keywords
				if (tags && tags.keywords && !tags.ai_keywords) tags.ai_keywords = tags.keywords
				return {
					id: `${q.paperId}-${q.question_number}`,
					topicTitle: (topicTitle || tags?.topic?.[0] || 'Question'),
					questionText: q.cleaned_question_text,
					markdown: md,
					createdAt: new Date(q.updatedAt || Date.now()).getTime(),
					tags,
				} as QAItem
			})
			set({ items, currentId: items[0]?.id || null })
		} finally {
			set({ loading: false })
		}
	},
	addFromProcess: async (paperId, result) => {
		const id = crypto.randomUUID()
		const full: QAItem = {
			id,
			createdAt: Date.now(),
			topicTitle: (result.topicTitle || result.tags?.topic?.[0] || 'Question'),
			questionText: result.cleaned_question_text,
			markdown: result.gemini_markdown,
			tags: result.tags && (result.tags.keywords && !result.tags.ai_keywords ? { ...result.tags, ai_keywords: result.tags.keywords } : result.tags),
		}
		set({ items: [full, ...get().items], currentId: id })
		try {
			await api.upsertQuestion({
				paperId,
				question_number: Number(result.question_number) || 0,
				cleaned_question_text: result.cleaned_question_text,
				provided_key: result.provided_key,
				key_confidence: result.key_confidence || 0,
				gemini_markdown: result.gemini_markdown,
				topicTitle: full.topicTitle,
				tags: full.tags || {},
			})
			set({ saveMessage: 'Saved to database' })
			setTimeout(() => set({ saveMessage: '' }), 2000)
		} catch (_) {
			set({ saveMessage: 'Save failed' })
			setTimeout(() => set({ saveMessage: '' }), 2000)
		}
	},
	addItem: (item) => {
		const id = crypto.randomUUID()
		const full: QAItem = { id, createdAt: Date.now(), ...item }
		const nextItems = [full, ...get().items]
		set({ items: nextItems, currentId: id })
		return id
	},
	select: (id) => set({ currentId: id }),
	clear: () => set({ items: [], currentId: null, selectedIds: new Set<string>(), filters: emptyFilters() }),
	toggleSelect: (id) => {
		const setSel = new Set(get().selectedIds)
		if (setSel.has(id)) setSel.delete(id); else setSel.add(id)
		set({ selectedIds: setSel })
	},
	selectAll: () => set({ selectedIds: new Set(get().getFilteredItems().map(i => i.id)) }),
	clearSelection: () => set({ selectedIds: new Set<string>() }),
	deleteSelected: async () => {
		const pid = get().currentPaperId
		if (!pid) return
		const toDelete = get().getSelectedItems()
		const nums: number[] = []
		for (const it of toDelete) {
			const parts = it.id.split('-')
			const maybe = Number(parts[parts.length - 1])
			if (!Number.isNaN(maybe)) nums.push(maybe)
		}
		if (nums.length) await api.deleteQuestions(pid, nums)
		const ids = new Set(toDelete.map(i => i.id))
		const remaining = get().items.filter(i => !ids.has(i.id))
		set({ items: remaining, currentId: remaining[0]?.id || null, selectedIds: new Set<string>() })
	},
	deleteQuestion: async (paperId, question_number) => {
		await api.deleteQuestions(paperId, [question_number])
		await get().loadQuestions(paperId)
	},
	removeTag: async (paperId, question_number, group, value) => {
		await api.removeTag(paperId, question_number, group, value)
		await get().loadQuestions(paperId)
	},
	getSelectedItems: () => get().getFilteredItems().filter(i => get().selectedIds.has(i.id)),
}))

export type Paper = { paperId: string; title: string; tags?: string[]; createdAt?: string }
export type Question = any

const BASE = 'http://localhost:8000'

async function http<T>(path: string, options?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE}${path}`, options)
	if (!res.ok) throw new Error(await res.text())
	return res.json() as Promise<T>
}

export const api = {
	listPapers: () => http<{ papers: Paper[] }>(`/papers`),
	createPaper: (paper: Paper) => http<{ ok: boolean }>(`/papers`, {
		method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(paper)
	}),
	listQuestions: (paperId: string) => http<{ questions: Question[] }>(`/questions?paperId=${encodeURIComponent(paperId)}`),
	upsertQuestion: (payload: any) => http<{ ok: boolean }>(`/questions`, {
		method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
	}),
	deleteQuestions: (paperId: string, question_numbers: number[]) => http<{ deleted: number }>(`/questions`, {
		method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ paperId, question_numbers })
	}),
	removeTag: (paperId: string, question_number: number, group: string, value: string) => http<{ ok: boolean }>(`/questions/tags/remove`, {
		method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ paperId, question_number, group, value })
	}),
	processScreenshot: (file: File, paperId?: string) => {
		const form = new FormData()
		form.append('image', file)
		const q = paperId ? `?paperId=${encodeURIComponent(paperId)}` : ''
		return http<any>(`/process-screenshot${q}`, { method: 'POST', body: form })
	},
	distinctTags: (paperId?: string) => http<{ subject: string[]; topic: string[]; concept: string[]; ai_keywords: string[] }>(`/tags/distinct${paperId ? `?paperId=${encodeURIComponent(paperId)}` : ''}`),
	searchQuestions: (payload: any) => http<{ questions: Question[] }>(`/questions/search`, {
		method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
	}),
}

/// <reference types="vite/client" />

declare module 'react/jsx-runtime' {
	export const jsx: any
	export const jsxs: any
	export const Fragment: any
}

declare module 'react-dom/client' {
	export type Root = {
		render(children: any): void
	}
	export function createRoot(container: Element | DocumentFragment): Root
}

declare module 'react' {
	const React: any
	export default React
	export function useState<T>(initial: T): [T, (value: T | ((prev: T) => T)) => void]
	export function useMemo<T>(factory: () => T, deps: any[]): T
	export function useEffect(effect: () => void | (() => void), deps?: any[]): void
	export function useRef<T>(initial: T | null): { current: T | null }
	export const StrictMode: any
}

declare module './App' {
	const App: any
	export default App
}

declare module 'zustand' {
	export function create<T>(creator: (set: any, get: () => T) => T): (selector?: any) => any
}

declare module 'react-markdown' {
	const ReactMarkdown: any
	export default ReactMarkdown
}

declare module 'react-syntax-highlighter' {
	export const Prism: any
}

declare module 'react-syntax-highlighter/dist/esm/styles/prism' {
	export const oneDark: any
}

declare module 'jspdf' {
	const jsPDF: any
	export default jsPDF
}

declare module 'html2canvas' {
	const html2canvas: any
	export default html2canvas
}

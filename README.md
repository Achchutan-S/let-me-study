---
noteId: "dc237dd08bcf11f0871115987ea0e2e8"
tags: []
---

## Exam Helper

Minimal full-stack app to OCR an exam image and get a structured explanation via Gemini.

### Tech

- Frontend: React + Vite + TailwindCSS
- Backend: FastAPI (Python)
- OCR: pytesseract (Pillow)

### Prerequisites

- Node 18+
- Python 3.10+
- Tesseract OCR installed on your system (pytesseract wrapper)
  - macOS (Homebrew): `brew install tesseract`
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`
  - Windows: Install from `https://github.com/tesseract-ocr/tesseract` and ensure it is on PATH
- Gemini API key

### Setup

1. Clone and create env file

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY
```

2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend dev server runs at `http://localhost:5173`. Backend runs at `http://localhost:8000`.

### Usage

- Select or create a Paper from the header
- Paste or upload a PNG/JPG exam image
- Click "Process Image" to run OCR and display extracted text + Gemini explanation
- Items save automatically into the selected paper (text-only)

### Tag filters and PDF export

- In the left sidebar, use the Filters panel to multi-select tags by Subject, Topic, Concept, or Keywords.
- The question list is filtered with AND logic across selected tags.
- Export Selected generates a PDF for the selected items within the current filtered list. Use Select All to export all filtered items.

### Project Structure

```
backend/
  main.py
  pipeline.py
  db.py
  requirements.txt
frontend/
  index.html
  package.json
  src/
    App.tsx
    main.tsx
    index.css
    api.ts
    store.ts
  tailwind.config.js
  postcss.config.js
  tsconfig.json
.env.example
README.md
```

### Notes

- OCR quality depends on the clarity of the image
- If Tesseract is not found, ensure it is installed and on PATH or set `TESSERACT_CMD`

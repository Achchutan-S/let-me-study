import os
import io
import re
import json
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
import httpx
import cv2
import numpy as np
from dotenv import load_dotenv

# Support running as package or as a script
try:
    from .db import get_database  # type: ignore
except Exception:
    from db import get_database  # type: ignore

# Load environment variables from a local .env if present
load_dotenv()


class AskRequest(BaseModel):
    text: str


class PaperCreate(BaseModel):
    paperId: str
    title: str
    tags: Optional[List[str]] = None


class QuestionCreate(BaseModel):
    paperId: str
    question_number: Optional[int] = None
    cleaned_question_text: str
    provided_key: Optional[str] = None
    key_confidence: float = 0.0
    gemini_markdown: Optional[str] = None
    topicTitle: Optional[str] = None
    tags: Optional[Dict[str, List[str]]] = None


class QuestionDelete(BaseModel):
    paperId: str
    question_numbers: List[int]


class TagRemove(BaseModel):
    paperId: str
    question_number: int
    group: str  # subject | topic | concept | ai_keywords
    value: str


class SearchRequest(BaseModel):
    paperId: Optional[str] = None
    subject: Optional[List[str]] = None
    topic: Optional[List[str]] = None
    concept: Optional[List[str]] = None
    ai_keywords: Optional[List[str]] = None


BASE_GATE_SYSTEM_PROMPT = (
    "You are my GATE tutor. Explain step by step, teaching from absolute basics but aiming for mastery. "
    "Follow the exact six-section structure and formatting rules given by the user. "
    "Avoid conversational filler; output Markdown only."
)


def build_simple_user_prompt(cleaned_text: str, provided_key: Optional[str]) -> str:
    key_line = f"Provided Key: {provided_key}" if provided_key else "Provided Key: null"
    return (
        "You are my GATE tutor. Explain the following GATE question step by step, teaching from absolute basics but aiming for mastery.\n\n"
        f"Question (raw):\n{cleaned_text}\n\n"
        f"{key_line}\n\n"
        "Output strictly in Markdown with this structure:\n\n"
        "VERDICT: Provided key CORRECT/INCORRECT/AMBIGUOUS (and if incorrect, suggest the correct key)\n\n"
        "(1) Fundamental theory to recall\n"
        "(2) The exact concept tested here\n"
        "(3) Detailed working with reasoning\n"
        "(4) Common mistakes & traps (and why other options tempt you)\n"
        "(5) Arriving at the correct answer\n"
        "(6) Quick recap for revision\n\n"
        "At the very end, append a JSON code fence with EXACT shape: {\n"
        "  \"topicTitle\": string,\n"
        "  \"tags\": { \"subject\": string[], \"topic\": string[], \"concept\": string[], \"keywords\": string[] }\n"
        "}\n"
    )


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    return value


_TESS_CMD = get_env("TESSERACT_CMD")
if _TESS_CMD:
    try:
        pytesseract.pytesseract.tesseract_cmd = _TESS_CMD
    except Exception:
        pass

app = FastAPI(title="Exam Helper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


async def gemini_generate(user_text: str, system_text: str, api_key: str, timeout_sec: int = 90) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    body = {
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
    }
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        resp = await client.post(url, json=body, headers=headers)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            raise HTTPException(
                status_code=502, detail=f"Gemini API error: {resp.text}") from http_err
        data = resp.json()
        model_text = ""
        try:
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    model_text = parts[0].get("text", "")
        except Exception:
            model_text = ""
        return model_text or ""


# --- Simple OCR helpers ---

def _basic_clean(s: str) -> str:
    if not s:
        return ''
    s = s.replace('\u2018', "'").replace('\u2019', "'").replace(
        '\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2013', '-').replace('\u2014', '-')
    s = s.replace('we server', 'web server')
    s = re.sub(r"\b(1/0|I/0|l/0)\b", 'I/O', s)
    s = re.sub(r"¢\.g\.?", 'e.g.', s, flags=re.IGNORECASE)
    s = re.sub(r"\b(SAMPLE|WATERMARK|PROOF|TEMPLATE)\b",
               '', s, flags=re.IGNORECASE)
    # C-language common OCR fixes
    s = re.sub(r"\bInt\b", 'int', s)
    s = re.sub(r"\bprint[f£]\(", 'printf(', s)
    s = s.replace('8d', '%d')
    s = re.sub(r"\binclude\s*<", '#include <', s)
    s = re.sub(r"\bmain\s*\(\s*\)\s*\{?", 'main() {', s, flags=re.IGNORECASE)
    # Ensure Key variants consistent
    s = re.sub(r"\bKey\s*[:：]?", 'Key:', s, flags=re.IGNORECASE)
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join(line.strip() for line in s.splitlines())
    return s.strip()


def _extract_qnum(text: str) -> Optional[int]:
    m = re.search(
        r"^(\s*Q(?:uestion)?\s*[:#-]?\s*)?(\d{1,3})[).\-:]?\s", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return None
    return None


def _extract_key(text: str) -> (Optional[str], float):
    m = re.search(r"\bKey\s*:?\s*\(?\s*([A-Da-d])\s*\)?", text)
    if m:
        return m.group(1).upper(), 0.9
    m2 = re.search(r"\bAnswer\s*:?\s*\(?\s*([A-Da-d])\s*\)?", text)
    if m2:
        return m2.group(1).upper(), 0.7
    return None, 0.0


def _extract_red_key_from_bytes(img_bytes: bytes) -> Tuple[Optional[str], float]:
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low1 = np.array([0, 70, 50])
        high1 = np.array([10, 255, 255])
        low2 = np.array([170, 70, 50])
        high2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8), iterations=2)
        # OCR mask
        pil = Image.fromarray(mask)
        raw = pytesseract.image_to_string(pil)
        m = re.search(r"([A-Da-d])", raw)
        if m:
            return m.group(1).upper(), 0.95
        return None, 0.0
    except Exception:
        return None, 0.0


# ---------- Papers & Questions (minimal, async, text-only) ----------

@app.post("/papers")
async def create_paper(payload: PaperCreate, db=Depends(get_database)) -> dict:
    from datetime import datetime
    doc = {
        'paperId': payload.paperId,
        'title': payload.title,
        'tags': payload.tags or [],
        'createdAt': datetime.utcnow(),
        'updatedAt': datetime.utcnow(),
    }
    try:
        await db.papers.insert_one(doc)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Create paper failed: {exc}")
    return {'ok': True}


@app.get("/papers")
async def list_papers(db=Depends(get_database)) -> dict:
    cursor = db.papers.find({}, {'_id': 0}).sort('createdAt', -1)
    data = [doc async for doc in cursor]
    return {'papers': data}


@app.post("/questions")
async def upsert_question(payload: QuestionCreate, db=Depends(get_database)) -> dict:
    from datetime import datetime
    qnum = payload.question_number or 0
    qkey = {'paperId': payload.paperId, 'question_number': qnum}
    update = {
        '$set': {
            'paperId': payload.paperId,
            'question_number': qnum,
            'cleaned_question_text': payload.cleaned_question_text,
            'provided_key': payload.provided_key,
            'key_confidence': float(payload.key_confidence or 0.0),
            'gemini_markdown': payload.gemini_markdown or '',
            'topicTitle': payload.topicTitle or '',
            'tags': payload.tags or {},
            'updatedAt': datetime.utcnow(),
        },
        '$setOnInsert': {'createdAt': datetime.utcnow()},
    }
    await db.questions.update_one(qkey, update, upsert=True)
    return {'ok': True}


@app.get("/questions")
async def list_questions(paperId: str = Query(...), db=Depends(get_database)) -> dict:
    cursor = db.questions.find({'paperId': paperId}, {'_id': 0}).sort(
        [('question_number', 1), ('updatedAt', -1)])
    data = [doc async for doc in cursor]
    return {'questions': data}


@app.delete("/questions")
async def delete_questions(payload: QuestionDelete, db=Depends(get_database)) -> dict:
    result = await db.questions.delete_many({'paperId': payload.paperId, 'question_number': {'$in': payload.question_numbers}})
    return {'deleted': result.deleted_count}


@app.patch("/questions/tags/remove")
async def remove_question_tag(payload: TagRemove, db=Depends(get_database)) -> dict:
    group = payload.group
    if group not in ('subject', 'topic', 'concept', 'ai_keywords'):
        raise HTTPException(status_code=400, detail='invalid tag group')
    update = {'$pull': {f'tags.{group}': payload.value}}
    result = await db.questions.update_one({'paperId': payload.paperId, 'question_number': payload.question_number}, update)
    return {'ok': True, 'modified': result.modified_count}


# --------------------- Simple OCR -> Gemini -------------------------

@app.post('/process-screenshot')
async def process_screenshot(image: UploadFile = File(...), paperId: Optional[str] = None, db=Depends(get_database)) -> dict:
    if image.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400, detail="Unsupported file type. Use png or jpg/jpeg.")

    data = await image.read()
    try:
        os.makedirs('/tmp', exist_ok=True)
        with open('/tmp/input.png', 'wb') as f:
            f.write(data)
    except Exception:
        pass

    # Try red key from image first
    red_key, red_conf = _extract_red_key_from_bytes(data)

    try:
        pil_img = Image.open(io.BytesIO(data))
        raw_text = pytesseract.image_to_string(pil_img)
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(
            status_code=500, detail="tesseract not found; install it or set TESSERACT_CMD")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    cleaned = _basic_clean(raw_text)
    qnum = _extract_qnum(cleaned)
    key, key_conf = _extract_key(cleaned)
    if red_key:
        key, key_conf = red_key, max(key_conf, red_conf)

    api_key = get_env("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY not configured")

    user_prompt = build_simple_user_prompt(cleaned, key)
    gemini_markdown = await gemini_generate(user_prompt, BASE_GATE_SYSTEM_PROMPT, api_key)

    # Parse trailing JSON code fence for topicTitle + tags
    topic_title = ''
    tags: Dict[str, List[str]] = {'subject': [],
                                  'topic': [], 'concept': [], 'keywords': []}
    try:
        m = re.search(r"```(?:json)?\n([\s\S]*?)\n```", gemini_markdown)
        if m:
            meta = json.loads(m.group(1))
            topic_title = str(meta.get('topicTitle', '') or '')
            mtags = meta.get('tags') or {}
            for k in tags.keys():
                if isinstance(mtags.get(k), list):
                    tags[k] = [str(x) for x in mtags[k]]
            gemini_markdown = gemini_markdown[:m.start()].rstrip()
    except Exception:
        pass

    # Normalize tags to ensure ai_keywords exists for frontend
    tags_out = {
        'subject': tags.get('subject', []),
        'topic': tags.get('topic', []),
        'concept': tags.get('concept', []),
        'ai_keywords': tags.get('keywords', []),
    }

    if paperId:
        from datetime import datetime
        try:
            await db.questions.update_one(
                {'paperId': paperId, 'question_number': qnum or 0},
                {'$set': {
                    'paperId': paperId,
                    'question_number': qnum or 0,
                    'cleaned_question_text': cleaned,
                    'provided_key': key,
                    'key_confidence': key_conf,
                    'gemini_markdown': gemini_markdown,
                    'topicTitle': topic_title,
                    'tags': tags_out,
                    'updatedAt': datetime.utcnow(),
                }, '$setOnInsert': {'createdAt': datetime.utcnow()}},
                upsert=True
            )
        except Exception:
            pass

    return {
        'question_number': qnum,
        'raw_ocr_text': raw_text,
        'cleaned_question_text': cleaned,
        'provided_key': key,
        'key_confidence': key_conf,
        'gemini_markdown': gemini_markdown,
        'topicTitle': topic_title,
        'tags': tags_out,
    }


@app.get('/tags/distinct')
async def tags_distinct(paperId: Optional[str] = None, db=Depends(get_database)) -> dict:
    match: Dict[str, Any] = {}
    if paperId:
        match['paperId'] = paperId
    pipeline = [
        {'$match': match},
        {'$group': {
            '_id': None,
            'subject': {'$addToSet': '$tags.subject'},
            'topic': {'$addToSet': '$tags.topic'},
            'concept': {'$addToSet': '$tags.concept'},
            'ai_keywords': {'$addToSet': '$tags.ai_keywords'},
        }},
        {'$project': {
            '_id': 0,
            'subject': {'$setUnion': '$subject'},
            'topic': {'$setUnion': '$topic'},
            'concept': {'$setUnion': '$concept'},
            'ai_keywords': {'$setUnion': '$ai_keywords'},
        }},
    ]
    cursor = db.questions.aggregate(pipeline)
    result = await cursor.to_list(length=1)
    if not result:
        return {'subject': [], 'topic': [], 'concept': [], 'ai_keywords': []}
    # Flatten nested arrays

    def flatten(lst):
        out = []
        for x in lst:
            if isinstance(x, list):
                out.extend(x)
        return sorted(list({str(v) for v in out}))
    row = result[0]
    return {
        'subject': flatten(row.get('subject', [])),
        'topic': flatten(row.get('topic', [])),
        'concept': flatten(row.get('concept', [])),
        'ai_keywords': flatten(row.get('ai_keywords', [])),
    }


@app.post('/questions/search')
async def search_questions(payload: SearchRequest, db=Depends(get_database)) -> dict:
    filt: Dict[str, Any] = {}
    if payload.paperId:
        filt['paperId'] = payload.paperId
    # AND logic within each group for selected values
    if payload.subject:
        filt['tags.subject'] = {'$all': payload.subject}
    if payload.topic:
        filt['tags.topic'] = {'$all': payload.topic}
    if payload.concept:
        filt['tags.concept'] = {'$all': payload.concept}
    if payload.ai_keywords:
        filt['tags.ai_keywords'] = {'$all': payload.ai_keywords}
    proj = {'_id': 0}
    cursor = db.questions.find(filt, proj).sort(
        [('updatedAt', -1), ('question_number', 1)])
    data = [doc async for doc in cursor]
    return {'questions': data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

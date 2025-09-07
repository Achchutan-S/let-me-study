import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import cv2
import pytesseract


TESS_CONFIG_MAIN = '--psm 6 --oem 1'
TESS_CONFIG_RED = '--psm 6 --oem 1'


def _bytes_to_cv(img_bytes: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def _deskew_and_crop(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Auto-crop margins via largest contour
    gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rotated
    x, y, w2, h2 = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = rotated[y:y + h2, x:x + w2]
    return cropped


def _denoise_and_binarize(image: np.ndarray) -> np.ndarray:
    den = cv2.medianBlur(image, 3)
    gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
    return bin_img


def _mask_red_regions(image: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red has two ranges in HSV
    low1 = np.array([0, 70, 50])
    high1 = np.array([10, 255, 255])
    low2 = np.array([170, 70, 50])
    high2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, low1, high1)
    mask2 = cv2.inRange(hsv, low2, high2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(image[:, :, 0]), None
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    masked = np.zeros_like(image[:, :, 0])
    masked[mask > 0] = 255
    return masked, (x, y, w, h)


def _remove_watermark_like(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    low_sat = cv2.inRange(sat, 0, 40)
    kernel = np.ones((5, 5), np.uint8)
    low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_OPEN, kernel, iterations=1)
    inpainted = cv2.inpaint(image, low_sat, 3, cv2.INPAINT_TELEA)
    return inpainted


def _ocr_pytesseract(img_gray_or_bin: np.ndarray, config: str) -> Tuple[str, Dict[str, Any]]:
    pil = Image.fromarray(img_gray_or_bin)
    raw = pytesseract.image_to_data(
        pil, output_type=pytesseract.Output.DICT, config=config)
    lines: Dict[int, List[float]] = {}
    texts: Dict[int, List[str]] = {}
    for i, _ in enumerate(raw.get('level', [])):
        line_num = raw['line_num'][i]
        text = raw['text'][i]
        conf = raw['conf'][i]
        try:
            c = float(conf) if conf != '-1' else 0.0
        except Exception:
            c = 0.0
        lines.setdefault(line_num, []).append(c)
        texts.setdefault(line_num, []).append(text)
    per_line_conf = [float(np.mean(v)) if v else 0.0 for _,
                     v in sorted(lines.items())]
    full_text = "\n".join(" ".join(t).strip()
                          for _, t in sorted(texts.items()))
    overall = float(np.mean(per_line_conf)) / 100.0 if per_line_conf else 0.0
    return full_text.strip(), {'overall': overall, 'per_line': per_line_conf}


def _clean_text(s: str) -> str:
    if not s:
        return ''
    # Normalize glyphs and frequent mistakes
    repls = [
        ('we server', 'web server'),
        ('1/0', 'I/O'), ('I/0', 'I/O'), ('l/0', 'I/O'),
        ('¢.g.', 'e.g.'), ('¢.g', 'e.g.'),
    ]
    for a, b in repls:
        s = re.sub(rf"\b{re.escape(a)}\b", b, s, flags=re.IGNORECASE)
    s = re.sub(r"\b(SAMPLE|TEMPLATE|WATERMARK|PROOF)\b",
               '', s, flags=re.IGNORECASE)
    s = s.replace('\u2018', "'").replace('\u2019', "'").replace(
        '\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2013', '-').replace('\u2014', '-')
    # Ensure spacing before markers like (i) and (A)
    s = re.sub(r"(?<!\s)\((i{1,3}|iv)\)", r" (\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<!\s)\(([A-Da-d])\)", r" (\1)", s)
    # Normalize whitespace
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"\t", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join(line.strip() for line in s.splitlines())
    return s.strip()


def _parse_structure(cleaned: str, raw: str, key_from_red: Optional[str], key_conf: float) -> Dict[str, Any]:
    # Question number
    question_number: Optional[str] = None
    mnum = re.search(
        r"^(?:\s*Q(?:uestion)?\s*[:#-]?\s*)?(\d{1,3})[).\-:]?\s", cleaned, re.IGNORECASE | re.MULTILINE)
    if mnum:
        question_number = mnum.group(1)

    # Split into question body and options section by the first (A)
    opt_start = re.search(r"\([Aa]\)\s", cleaned)
    body_text = cleaned
    options_text = ''
    if opt_start:
        idx = opt_start.start()
        body_text = cleaned[:idx].strip()
        options_text = cleaned[idx:].strip()

    # Extract roman numeral statements from body
    statements: List[str] = []
    for m in re.finditer(r"\((i{1,3}|iv)\)\s+(.+?)(?=\n\((i{1,3}|iv)\)|$)", body_text, re.IGNORECASE | re.DOTALL):
        roman = m.group(1)
        text = m.group(2).strip()
        statements.append(f"({roman}) {text}")
    # Question stem is body without the statements block (fallback: body_text)
    question_text = body_text
    if statements:
        first_stmt = statements[0]
        pos = body_text.find(first_stmt.split(' ', 1)[1])
        if pos > 0:
            question_text = body_text[:max(0, pos - 1)].strip()

    # Strict A-D options from options_text
    options: List[Dict[str, str]] = []
    for m in re.finditer(r"\(([A-Da-d])\)\s+(.+?)(?=\n\([A-Da-d]\)|$)", options_text, re.DOTALL):
        label = m.group(1).upper()
        text = m.group(2).strip()
        options.append({'label': label, 'text': text})
    # Ensure labels A..D present in order if partially detected
    labels_seen = {o['label'] for o in options}
    for lbl in ['A', 'B', 'C', 'D']:
        if lbl not in labels_seen:
            options.append({'label': lbl, 'text': ''})
    options.sort(key=lambda o: o['label'])

    # Key detection preference: red first
    provided_key = None
    key_source = 'none'
    if key_from_red:
        provided_key = key_from_red.upper()
        key_source = 'red_region'
    else:
        inline = re.search(r"\bKey\s*:?\s*\(?\s*([A-Da-d])\s*\)?", cleaned)
        if inline:
            provided_key = inline.group(1).upper()
            key_source = 'inline_text'

    # Cross-check key against options
    labels_present = {o['label'] for o in options}
    if provided_key and provided_key not in labels_present:
        provided_key = None
        key_source = 'uncertain'

    # Confidence heuristic
    score = 0.4
    if statements:
        score += 0.2
    if any(o['text'] for o in options):
        score += 0.2
    if provided_key:
        score += 0.2
    parsing_confidence = float(max(0.0, min(1.0, score)))

    return {
        'question_number': question_number or '',
        'question_text': question_text,
        'statements': statements,
        'options': options,
        'provided_key': provided_key,
        'key_source': key_source,
        'parsing_confidence': parsing_confidence,
        'raw_ocr_text': raw,
        'cleaned_text': cleaned,
    }


def process_image(img_bytes: bytes) -> Dict[str, Any]:
    # Save input.png for debugging
    try:
        with open('/tmp/input.png', 'wb') as f:
            f.write(img_bytes)
    except Exception:
        pass

    image = _bytes_to_cv(img_bytes)
    image = _deskew_and_crop(image)
    image = _remove_watermark_like(image)
    img_h, img_w = image.shape[:2]
    red_mask, red_bbox = _mask_red_regions(image)
    bin_main = _denoise_and_binarize(image)
    try:
        cv2.imwrite('/tmp/red_regions.png', red_mask)
        cv2.imwrite('/tmp/main_region.png', bin_main)
    except Exception:
        pass

    raw_main, conf_main = _ocr_pytesseract(bin_main, TESS_CONFIG_MAIN)
    red_binary = red_mask if red_mask is not None else np.zeros_like(bin_main)
    raw_red, conf_red = _ocr_pytesseract(red_binary, TESS_CONFIG_RED)
    cleaned_main = _clean_text(raw_main)

    candidate_key = None
    m_red = re.search(r"([A-Da-d])", raw_red)
    if m_red:
        candidate_key = m_red.group(1)

    parsed = _parse_structure(cleaned_main, raw_main,
                              candidate_key, conf_red.get('overall', 0.0))
    debug = {
        'red_region_bbox': red_bbox,
        'image_shape': [img_w, img_h],
        'ocr_confidences': {'overall': conf_main.get('overall', 0.0), 'per_line': conf_main.get('per_line', [])},
        'raw_red_text': raw_red,
        'raw_main_text': raw_main,
    }
    return {
        'parsed': parsed,
        'debug': debug,
    }

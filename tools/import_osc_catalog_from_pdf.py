"""Import Apollo OSC/action-code catalog from SOP PDF into MongoDB.

Run from project root:
    pip install pymupdf
    python tools/import_osc_catalog_from_pdf.py --pdf "path/to/SOP-GQ&BE-001.pdf" --publish --replace

What this script does:
1. Reads table text using PDF word coordinates, not OCR.
2. Crops document images per catalog section and stores local image paths.
3. Imports versioned rows/images into MongoDB using src.COMMON.action_code_catalog_db.
4. Writes a review JSON beside the imported images so QC can verify the parsed master once.

Important production note:
PDF extraction can contain spelling/format mistakes from the source text layer. After first import,
review the generated JSON once, correct if required, then keep the reviewed JSON as the master seed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyMuPDF is required. Install with: pip install pymupdf") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HEADER = {
    "document_name": "Global Off Standard Catalogue for PCR Tyres",
    "document_no": "SOP-GQ&BE-001",
    "revision_no": "03",
    "document_status": "Approved",
    "date_of_release": "05/07/2023",
    "date_of_applicability": "17/07/2023",
    "process_owner": "Corporate",
    "security_classification": "Internal",
}


def build_version_id(revision_no: str, local_version_no: str = "00") -> str:
    return f"OSC_REV_{str(revision_no).strip().replace(' ', '_')}_V{str(local_version_no).strip().replace(' ', '_')}"


def infer_side_from_catalog_code(code: str) -> str:
    code = str(code).strip()
    return {
        "1": "tread",
        "2": "shoulder",
        "3": "sidewall",
        "4": "bead",
        "5": "innerliner",
        "6": "curing",
        "7": "foreign_material",
    }.get(code[:1], "general")

SECTION_RE = re.compile(r"^(\d{3})\s+(.+?)\s+Action\s+code\s+OE\s+Replacement\s+Scrap", re.I)
CODE_RE = re.compile(r"^\d{1,2}(?:\s*,\s*\d{1,2})*$")

# Column bands in PDF points. These match the uploaded SOP layout.
DESC_X0, DESC_X1 = 150.0, 535.0
ACTION_X0, ACTION_X1 = 535.0, 625.0
OE_X0, OE_X1 = 625.0, 675.0
REPL_X0, REPL_X1 = 675.0, 790.0
SCRAP_X0, SCRAP_X1 = 790.0, 890.0
TABLE_Y_MIN = 270.0
LOGO_MAX_HEIGHT = 80.0


def _line_key(y: float, tolerance: float = 4.5) -> int:
    return int(round(y / tolerance))


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.replace("≤", "<=")
    return text


def _row_has_x(tokens: List[str]) -> bool:
    return any(t.strip().upper() == "X" for t in tokens)


def _action_code_from_tokens(tokens: List[str]) -> str:
    joined = _clean_text(" ".join(tokens))
    joined = joined.replace(" ,", ",").replace(", ", ",")
    if joined and CODE_RE.match(joined):
        return joined
    # Some rows have action code tokens like "3," "4".
    only_code_tokens = []
    for t in tokens:
        tt = t.strip()
        if re.match(r"^[0-9,]+$", tt):
            only_code_tokens.append(tt)
    code = _clean_text(" ".join(only_code_tokens)).replace(" ", "")
    return code if CODE_RE.match(code) else ""


def _is_inside_any_image(x: float, y: float, image_rects: List[fitz.Rect]) -> bool:
    # Ignore text printed inside images/diagrams so it does not become fake table rows.
    p = fitz.Point(x, y)
    for r in image_rects:
        if r.height > LOGO_MAX_HEIGHT and r.contains(p):
            return True
    return False


@dataclass
class ParsedLine:
    y: float
    text: str
    desc: str
    action_code: str
    oe: bool
    replacement: bool
    scrap: bool
    raw_words: List[Tuple[float, float, str]]

    @property
    def has_cells(self) -> bool:
        return bool(self.action_code or self.oe or self.replacement or self.scrap)


@dataclass
class SectionBand:
    page_no: int
    catalog_code: str
    section_name: str
    start_y: float
    end_y: float
    critical: bool = False


def extract_header(doc: fitz.Document) -> Dict[str, Any]:
    header = dict(DEFAULT_HEADER)
    # The SOP page text explicitly contains the fields. Keep defaults if parsing fails.
    text = doc[0].get_text("text")
    patterns = {
        "date_of_release": r"Date of Release\s+([^\n]+)",
        "date_of_applicability": r"Date of Applicability\s+([^\n]+)",
        "process_owner": r"Process Owner\s+([^\n]+)",
        "security_classification": r"Security Classification\s+([^\n]+)",
        "document_name": r"Document Name\s+([^\n]+)",
        "document_no": r"Document No\.\s+([^\n]+)",
        "revision_no": r"Revision No\.\s+([^\n]+)",
        "document_status": r"Document Status\s+([^\n]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.I)
        if m:
            value = _clean_text(m.group(1))
            if value:
                header[key] = value
    return header


def get_image_rects(page: fitz.Page) -> List[fitz.Rect]:
    rects = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") == 1:
            r = fitz.Rect(block.get("bbox"))
            rects.append(r)
    return rects


def get_text_lines(page: fitz.Page, *, skip_image_text: bool = True) -> List[ParsedLine]:
    image_rects = get_image_rects(page)
    groups: Dict[int, List[Tuple[float, float, str]]] = defaultdict(list)
    for w in page.get_text("words"):
        x0, y0, x1, y1, word = w[:5]
        if y0 < TABLE_Y_MIN:
            continue
        if skip_image_text and _is_inside_any_image((x0 + x1) / 2, (y0 + y1) / 2, image_rects):
            continue
        groups[_line_key(y0)].append((x0, y0, str(word)))

    lines: List[ParsedLine] = []
    for items in groups.values():
        items = sorted(items, key=lambda x: x[0])
        y = sum(i[1] for i in items) / len(items)
        desc_tokens, act_tokens, oe_tokens, repl_tokens, scrap_tokens = [], [], [], [], []
        all_tokens = []
        for x, _, word in items:
            all_tokens.append(word)
            if DESC_X0 <= x < DESC_X1:
                desc_tokens.append(word)
            elif ACTION_X0 <= x < ACTION_X1:
                act_tokens.append(word)
            elif OE_X0 <= x < OE_X1:
                oe_tokens.append(word)
            elif REPL_X0 <= x < REPL_X1:
                repl_tokens.append(word)
            elif SCRAP_X0 <= x < SCRAP_X1:
                scrap_tokens.append(word)

        line = ParsedLine(
            y=y,
            text=_clean_text(" ".join(all_tokens)),
            desc=_clean_text(" ".join(desc_tokens)),
            action_code=_action_code_from_tokens(act_tokens),
            oe=_row_has_x(oe_tokens),
            replacement=_row_has_x(repl_tokens),
            scrap=_row_has_x(scrap_tokens),
            raw_words=items,
        )
        if line.text:
            lines.append(line)
    return sorted(lines, key=lambda l: l.y)


def detect_sections_for_page(page: fitz.Page, page_no: int) -> List[SectionBand]:
    lines = get_text_lines(page, skip_image_text=False)
    raw_sections: List[SectionBand] = []
    for line in lines:
        m = SECTION_RE.match(line.text)
        if not m:
            continue
        code = m.group(1)
        name = _clean_text(m.group(2))
        critical = "*CC" in name or name.upper().startswith("CC")
        name = name.replace("*CC", "").strip()
        raw_sections.append(SectionBand(page_no=page_no, catalog_code=code, section_name=name, start_y=line.y, end_y=page.rect.height, critical=critical))

    # Stop the last table before the version-history block on page 23.
    version_history_y = None
    for line in lines:
        if "Version History" in line.text:
            version_history_y = line.y
            break

    for i in range(len(raw_sections) - 1):
        raw_sections[i].end_y = raw_sections[i + 1].start_y
    if raw_sections and version_history_y is not None:
        raw_sections[-1].end_y = min(raw_sections[-1].end_y, version_history_y)
    return raw_sections


def _append_or_new(rows: List[Dict[str, Any]], pending: Optional[Dict[str, Any]], line: ParsedLine) -> Optional[Dict[str, Any]]:
    desc = _clean_text(line.desc)
    has_cells = line.has_cells

    # Skip non-table lines.
    if not desc and not has_cells:
        return pending
    if desc.lower() in {"classification", "action code", "condition description of condition action code"}:
        return pending
    if "FR-GQ" in line.text or "02.09.2021" in line.text or "Page " in line.text:
        return pending
    if "Version History" in line.text:
        return pending

    # Description-only line can be a continuation or a new pending row.
    if desc and not has_cells:
        if pending is not None and not pending.get("_closed"):
            pending["description"] = _clean_text(pending.get("description", "") + " " + desc)
            return pending
        if rows and (desc.startswith("(") or desc.startswith("*") or len(desc.split()) <= 4):
            rows[-1]["description"] = _clean_text(rows[-1].get("description", "") + " " + desc)
            return None
        pending = {"description": desc, "action_code": "", "oe": False, "replacement": False, "scrap": False, "_closed": False}
        rows.append(pending)
        return pending

    # Cells-only line belongs to pending row if available.
    if not desc and has_cells:
        target = pending or (rows[-1] if rows else None)
        if target is not None:
            if line.action_code:
                target["action_code"] = line.action_code
            target["oe"] = bool(target.get("oe") or line.oe)
            target["replacement"] = bool(target.get("replacement") or line.replacement)
            target["scrap"] = bool(target.get("scrap") or line.scrap)
            target["_closed"] = True
        return target

    # Normal row.
    row = {
        "description": desc,
        "action_code": line.action_code,
        "oe": line.oe,
        "replacement": line.replacement,
        "scrap": line.scrap,
        "_closed": True,
    }
    rows.append(row)
    return row


def parse_catalog_tables(pdf_path: Path) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    header = extract_header(doc)
    section_rows: Dict[str, Dict[str, Any]] = {}
    all_bands: List[SectionBand] = []

    for page_index in range(len(doc)):
        page_no = page_index + 1
        page = doc[page_index]
        bands = detect_sections_for_page(page, page_no)
        all_bands.extend(bands)
        if not bands:
            continue
        lines = get_text_lines(page, skip_image_text=False)
        for band in bands:
            rows: List[Dict[str, Any]] = []
            pending: Optional[Dict[str, Any]] = None
            for line in lines:
                if not (band.start_y < line.y < band.end_y):
                    continue
                if SECTION_RE.match(line.text):
                    continue
                pending = _append_or_new(rows, pending, line)

            clean_rows = []
            for idx, r in enumerate(rows, start=1):
                rr = {k: v for k, v in r.items() if not k.startswith("_")}
                if not rr.get("description"):
                    continue
                rr["row_order"] = idx
                rr["condition_code"] = f"{band.catalog_code}.{idx}"
                rr["source_page"] = band.page_no
                clean_rows.append(rr)

            if band.catalog_code not in section_rows:
                section_rows[band.catalog_code] = {
                    "catalog_code": band.catalog_code,
                    "section_name": band.section_name,
                    "side": infer_side_from_catalog_code(band.catalog_code),
                    "critical_characteristic": band.critical,
                    "source_page": band.page_no,
                    "rows": [],
                }
            section_rows[band.catalog_code]["rows"].extend(clean_rows)

    sections = list(section_rows.values())
    sections.sort(key=lambda s: int(s["catalog_code"]))
    for sec_idx, sec in enumerate(sections, start=1):
        sec["section_order"] = sec_idx
        # Renumber condition_code after possible page continuation.
        for row_idx, row in enumerate(sec["rows"], start=1):
            row["condition_code"] = f"{sec['catalog_code']}.{row_idx}"
            row["row_order"] = row_idx

    version_id = build_version_id(str(header.get("revision_no", "03")), "00")
    return {
        "source": f"pdf:{pdf_path.name}",
        "header": header,
        "version_id": version_id,
        "local_version_no": "00",
        "sections": sections,
        "images": [],
        "notes": "Auto-parsed from SOP PDF. Review generated JSON once before production use.",
    }


def crop_section_images(pdf_path: Path, payload: Dict[str, Any], out_dir: Path, dpi: int = 160) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    section_lookup: List[SectionBand] = []
    for page_index in range(len(doc)):
        section_lookup.extend(detect_sections_for_page(doc[page_index], page_index + 1))

    images: List[Dict[str, Any]] = []
    by_section_count: Dict[str, int] = defaultdict(int)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page_no = page_index + 1
        page = doc[page_index]
        rects = get_image_rects(page)
        for rect in rects:
            if rect.height <= LOGO_MAX_HEIGHT:
                continue
            cy = (rect.y0 + rect.y1) / 2
            matching = [b for b in section_lookup if b.page_no == page_no and b.start_y < cy < b.end_y]
            if not matching:
                continue
            band = matching[0]
            by_section_count[band.catalog_code] += 1
            image_order = by_section_count[band.catalog_code]
            sec_dir = out_dir / band.catalog_code
            sec_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{band.catalog_code}_p{page_no:02d}_{image_order:02d}.png"
            path = sec_dir / fname
            pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
            pix.save(path)
            images.append({
                "catalog_code": band.catalog_code,
                "section_name": band.section_name,
                "side": infer_side_from_catalog_code(band.catalog_code),
                "image_order": image_order,
                "page_no": page_no,
                "image_path": str(path),
                "bbox": [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)],
            })
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to SOP-GQ&BE-001 PDF")
    parser.add_argument("--out-dir", default="media/osc_catalog/rev03", help="Where cropped section images/review JSON are saved")
    parser.add_argument("--replace", action="store_true", help="Replace existing version rows/images")
    parser.add_argument("--publish", action="store_true", help="Publish imported version as ACTIVE/current")
    parser.add_argument("--no-db", action="store_true", help="Only parse/extract and write JSON, do not import MongoDB")
    parser.add_argument("--operator", default="system")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = parse_catalog_tables(pdf_path)
    payload["images"] = crop_section_images(pdf_path, payload, out_dir)

    review_json = out_dir / f"{payload['version_id']}_review_payload.json"
    with review_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    total_rows = sum(len(sec.get("rows", [])) for sec in payload.get("sections", []))
    print(f"[PARSED] sections={len(payload['sections'])} rows={total_rows} images={len(payload['images'])}")
    print(f"[REVIEW_JSON] {review_json}")

    if not args.no_db:
        from src.COMMON.action_code_catalog_db import import_catalog_payload  # type: ignore
        result = import_catalog_payload(payload, replace=args.replace, publish=args.publish, operator=args.operator)
        print(f"[DB_IMPORT] {result}")


if __name__ == "__main__":
    main()

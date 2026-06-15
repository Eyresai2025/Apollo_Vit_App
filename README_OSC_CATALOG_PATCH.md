# OSC Action Code Catalog - Production Patch

## Files to copy

Copy these files into your Apollo project root:

```text
src/COMMON/action_code_catalog_db.py
src/PAGES/action_code_plan_page.py
tools/import_osc_catalog_from_pdf.py
```

Keep your existing `src/COMMON/db.py`. This patch only uses your existing `get_collection`, `ensure_collection`, and `get_gridfs` helpers.

## Install dependency

```bash
pip install pymupdf
```

## First-time import from the client SOP PDF

Run from project root:

```bash
python tools/import_osc_catalog_from_pdf.py --pdf "SOP-GQ&BE-001 Global Off Standard Catalogue for PCR Tyres (1) 2.pdf" --out-dir media/osc_catalog/rev03 --replace --publish --operator admin
```

Expected output is like:

```text
[PARSED] sections=76 rows=237 images=154
[REVIEW_JSON] media/osc_catalog/rev03/OSC_REV_03_V00_review_payload.json
[DB_IMPORT] {'ok': True, ...}
```

## Production workflow

1. Import the client SOP once.
2. Review the generated JSON once because PDF text extraction can contain source-text errors.
3. Use the UI only through `Create Editable Draft -> Save Draft -> Publish Draft`.
4. Never edit the active version directly.
5. In live AI inference, save `version_id`, `condition_code`, `action_code`, and `final_decision` with every defect result for traceability.

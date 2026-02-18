# AGENTS.md — pdf-inspector Codebase Guide

## Project Overview

Rust crate (`pdf-inspector`) that extracts text from PDFs and converts it to structured Markdown. Ships a CLI binary `pdf2md`.

- **Crate name:** `pdf-inspector`
- **Binary:** `pdf2md` (`src/bin/pdf2md.rs`)
- **PDF parsing:** `lopdf` crate (v0.39.0, git dependency)
- **Test PDFs:** `/Users/abimaelmartell/Code/pdf-evals/pdfs/`

## Module Map

```
src/
  lib.rs                          — Public API, re-exports
  types.rs                        — Shared types: TextItem, TextLine, PdfRect, ItemType
  text_utils.rs                   — Character/text helpers: CJK, RTL, ligatures, bold/italic
  detector.rs                     — Fast PDF type detection (text vs scanned) without full load
  glyph_names.rs                  — Adobe Glyph List → Unicode mapping
  tounicode.rs                    — ToUnicode CMap parsing for CID-encoded text

  extractor/
    mod.rs                        — Public API: extract_text, extract_text_with_positions
    fonts.rs                      — Font width parsing, encoding, text decoding
    content_stream.rs             — PDF operator state machine (Tm, Td, Tj, TJ, etc.)
    xobjects.rs                   — Form XObject and image XObject extraction
    links.rs                      — Hyperlink and AcroForm field extraction
    layout.rs                     — Column detection, line grouping, reading order

  tables/
    mod.rs                        — Table struct, TableDetectionMode, re-exports
    detect_rects.rs               — Rectangle-based table detection (union-find clustering)
    detect_heuristic.rs           — Heuristic table detection + validation
    financial.rs                  — Financial token splitting for consolidated values
    grid.rs                       — Column/row boundaries, cell assignment
    format.rs                     — Table → Markdown formatting, footnotes

  markdown/
    mod.rs                        — MarkdownOptions, public API (to_markdown, to_markdown_from_items)
    convert.rs                    — Core line-to-markdown loop, table/image interleaving
    analysis.rs                   — Font statistics, heading tiers, paragraph thresholds
    classify.rs                   — Caption, list, code detection
    preprocess.rs                 — Heading merging, drop cap handling
    postprocess.rs                — Cleanup: dot leaders, hyphenation, page numbers, URLs

  bin/
    pdf2md.rs                     — CLI: PDF → Markdown
    detect_pdf.rs                 — CLI: detect PDF type
    debug_spaces.rs               — Debug: dump text items with x/y/width per page
    dump_ops.rs                   — Debug: dump raw PDF content stream operators
    debug_ygaps.rs                — Debug: Y-gap analysis between lines
    debug_fonts.rs                — Debug: font information
    debug_ligatures.rs            — Debug: ligature expansion
    debug_order.rs                — Debug: reading order
    debug_pages.rs                — Debug: page-level info
    detection_report.rs           — Batch detection report on PDF directory
    profile_stages.rs             — Performance profiling of pipeline stages
```

## Data Flow

```
PDF bytes
  │
  ├─► detector.rs          → PdfType (TextBased / Scanned / ImageBased)
  │
  └─► extractor/
        ├─ fonts.rs         → font widths, encodings
        ├─ content_stream.rs → walk operators → Vec<TextItem> + Vec<PdfRect>
        ├─ xobjects.rs      → Form XObject text, image placeholders
        ├─ links.rs         → hyperlinks, AcroForm fields
        └─ layout.rs        → column detection → group_into_lines → Vec<TextLine>
              │
              ├─► tables/
              │     ├─ detect_rects.rs     → rect-based tables (PdfRect clusters)
              │     ├─ detect_heuristic.rs → heuristic tables (font-size + alignment)
              │     ├─ grid.rs             → column/row assignment → cells
              │     └─ format.rs           → Table → Markdown string
              │
              └─► markdown/
                    ├─ analysis.rs    → font stats, heading tiers
                    ├─ preprocess.rs  → merge headings, drop caps
                    ├─ convert.rs     → line loop + table/image insertion
                    ├─ classify.rs    → captions, lists, code
                    └─ postprocess.rs → cleanup → final Markdown string
```

## Critical Implementation Details

### Text Matrix Math
PDF text positioning uses two matrices: `text_matrix` (Tm) and `line_matrix`. The `Td`/`TD` operators provide offsets in **text space**, which must be scaled by `line_matrix`:

```
e += tx * a + ty * c
f += tx * b + ty * d
```

When `Tm` has scaling (e.g., `[12,0,0,12,x,y]`), failing to apply this scaling produces incorrect positions. The `T*` and `'` operators are equivalent to `0 -TL Td` and need the same treatment.

### Font Size
Font size can come from the `Tf` operand **or** the `Tm` matrix scaling. Use `effective_font_size()` from `text_utils.rs` to get the correct value.

### White-Fill Text
Text drawn with white fill (`1 g` before text ops) should be skipped during extraction but the text matrix must still advance to keep positions correct.

### CID Fonts
Fonts named `C2_*` or `C0_*` are CID fonts that emit one word per `Tj` operator. Spaces must be inserted between consecutive `Tj` items.

### lopdf Quirks
- `lopdf::error::ParseError` is private — match by string for `InvalidFileHeader`
- Clippy enforces `-D warnings` — use `is_some_and(...)` instead of `map_or(false, ...)`

## Testing

```bash
cargo test                         # Run all 66 unit tests
cargo clippy -- -D warnings        # Lint (enforced in CI)
cargo fmt --check                  # Format check
cargo run --release --bin pdf2md -- <file.pdf>   # Smoke test
```

## Common Tasks

| Task | Where to Edit |
|------|--------------|
| Fix text positioning bugs | `extractor/content_stream.rs` |
| Add font encoding support | `extractor/fonts.rs`, `tounicode.rs` |
| Fix column/reading order | `extractor/layout.rs` |
| Improve table detection | `tables/detect_heuristic.rs` |
| Fix table formatting | `tables/format.rs`, `tables/grid.rs` |
| Add rectangle-based tables | `tables/detect_rects.rs` |
| Change heading detection | `markdown/analysis.rs` |
| Fix list/code detection | `markdown/classify.rs` |
| Fix paragraph breaks | `markdown/convert.rs`, `markdown/analysis.rs` |
| Fix URL/hyphenation cleanup | `markdown/postprocess.rs` |
| Add new PDF type detection | `detector.rs` |
| Add new text item type | `types.rs`, then update consumers |

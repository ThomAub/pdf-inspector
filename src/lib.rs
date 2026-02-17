//! Smart PDF detection and text extraction using lopdf
//!
//! This module provides:
//! - Fast detection of scanned vs text-based PDFs without full document load
//! - Direct text extraction from text-based PDFs
//! - Markdown conversion with structure detection

pub mod detector;
pub mod extractor;
pub mod glyph_names;
pub mod markdown;
pub mod tables;
pub mod tounicode;

pub use detector::{
    detect_pdf_type, detect_pdf_type_mem, detect_pdf_type_mem_with_config,
    detect_pdf_type_with_config, DetectionConfig, PdfType, PdfTypeResult, ScanStrategy,
};
pub use extractor::{
    extract_text, extract_text_with_positions, extract_text_with_positions_pages, PdfRect, TextItem,
};
pub use markdown::{
    to_markdown, to_markdown_from_items, to_markdown_from_items_with_rects, MarkdownOptions,
};

use std::path::Path;

/// High-level PDF processing result
#[derive(Debug)]
pub struct PdfProcessResult {
    /// The detected PDF type
    pub pdf_type: PdfType,
    /// Extracted text (if text-based PDF)
    pub text: Option<String>,
    /// Markdown output (if text-based PDF)
    pub markdown: Option<String>,
    /// Page count
    pub page_count: u32,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// 1-indexed page numbers that need OCR.
    pub pages_needing_ocr: Vec<u32>,
    /// Title from PDF metadata (if available)
    pub title: Option<String>,
    /// Detection confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Process a PDF file with smart detection and extraction
///
/// This function will:
/// 1. Quickly detect if the PDF is text-based or scanned
/// 2. If text-based, extract text and convert to markdown
/// 3. If scanned, return early indicating OCR is needed
pub fn process_pdf<P: AsRef<Path>>(path: P) -> Result<PdfProcessResult, PdfError> {
    let start = std::time::Instant::now();

    validate_pdf_file(&path)?;

    // Step 1: Smart detection (fast, no full load)
    let detection = detect_pdf_type(&path)?;
    let page_count = detection.page_count;
    let pdf_type = detection.pdf_type;
    let pages_needing_ocr = detection.pages_needing_ocr;
    let title = detection.title;
    let confidence = detection.confidence;

    let result = match pdf_type {
        PdfType::TextBased => {
            // Step 2: Full extraction with position-aware reading order
            let (items, rects) = extractor::extract_text_with_positions_and_rects(&path, None)?;
            let markdown =
                to_markdown_from_items_with_rects(items, MarkdownOptions::default(), &rects);

            PdfProcessResult {
                pdf_type,
                text: None, // We now produce markdown directly
                markdown: Some(markdown),
                page_count,
                processing_time_ms: start.elapsed().as_millis() as u64,
                pages_needing_ocr,
                title,
                confidence,
            }
        }
        PdfType::Scanned | PdfType::ImageBased => {
            // Return early - OCR needed
            PdfProcessResult {
                pdf_type,
                text: None,
                markdown: None,
                page_count,
                processing_time_ms: start.elapsed().as_millis() as u64,
                pages_needing_ocr,
                title,
                confidence,
            }
        }
        PdfType::Mixed => {
            // Try to extract what we can with position-aware reading order
            let result = extractor::extract_text_with_positions_and_rects(&path, None).ok();
            let markdown = result.map(|(items, rects)| {
                to_markdown_from_items_with_rects(items, MarkdownOptions::default(), &rects)
            });

            PdfProcessResult {
                pdf_type,
                text: None,
                markdown,
                page_count,
                processing_time_ms: start.elapsed().as_millis() as u64,
                pages_needing_ocr,
                title,
                confidence,
            }
        }
    };

    Ok(result)
}

/// Process a PDF file with custom detection and markdown configuration
pub fn process_pdf_with_config<P: AsRef<Path>>(
    path: P,
    config: DetectionConfig,
    markdown_options: MarkdownOptions,
) -> Result<PdfProcessResult, PdfError> {
    process_pdf_with_config_pages(path, config, markdown_options, None)
}

/// Process a PDF file with custom configuration and optional page filter.
///
/// `page_filter` limits extraction to the given 1-indexed page numbers.
/// When `None`, all pages are processed.
pub fn process_pdf_with_config_pages<P: AsRef<Path>>(
    path: P,
    config: DetectionConfig,
    markdown_options: MarkdownOptions,
    page_filter: Option<&std::collections::HashSet<u32>>,
) -> Result<PdfProcessResult, PdfError> {
    let start = std::time::Instant::now();

    validate_pdf_file(&path)?;

    let detection = detect_pdf_type_with_config(&path, config)?;
    let page_count = detection.page_count;
    let pdf_type = detection.pdf_type;
    let pages_needing_ocr = detection.pages_needing_ocr;
    let title = detection.title;
    let confidence = detection.confidence;

    let result = match pdf_type {
        PdfType::TextBased => {
            let (items, rects) =
                extractor::extract_text_with_positions_and_rects(&path, page_filter)?;
            let markdown = to_markdown_from_items_with_rects(items, markdown_options, &rects);

            PdfProcessResult {
                pdf_type,
                text: None,
                markdown: Some(markdown),
                page_count,
                processing_time_ms: start.elapsed().as_millis() as u64,
                pages_needing_ocr,
                title,
                confidence,
            }
        }
        PdfType::Scanned | PdfType::ImageBased => PdfProcessResult {
            pdf_type,
            text: None,
            markdown: None,
            page_count,
            processing_time_ms: start.elapsed().as_millis() as u64,
            pages_needing_ocr,
            title,
            confidence,
        },
        PdfType::Mixed => {
            let result = extractor::extract_text_with_positions_and_rects(&path, page_filter).ok();
            let markdown = result.map(|(items, rects)| {
                to_markdown_from_items_with_rects(items, markdown_options.clone(), &rects)
            });

            PdfProcessResult {
                pdf_type,
                text: None,
                markdown,
                page_count,
                processing_time_ms: start.elapsed().as_millis() as u64,
                pages_needing_ocr,
                title,
                confidence,
            }
        }
    };

    Ok(result)
}

/// Process PDF from memory buffer
pub fn process_pdf_mem(buffer: &[u8]) -> Result<PdfProcessResult, PdfError> {
    let start = std::time::Instant::now();

    validate_pdf_bytes(buffer)?;

    // Step 1: Smart detection (fast, no full load)
    let detection = detector::detect_pdf_type_mem(buffer)?;
    let page_count = detection.page_count;
    let pdf_type = detection.pdf_type;
    let pages_needing_ocr = detection.pages_needing_ocr;
    let title = detection.title;
    let confidence = detection.confidence;

    let result = match pdf_type {
        PdfType::TextBased => {
            // Step 2: Full extraction with position-aware reading order
            let (items, rects) =
                extractor::extract_text_with_positions_mem_and_rects(buffer, None)?;
            let markdown =
                to_markdown_from_items_with_rects(items, MarkdownOptions::default(), &rects);

            PdfProcessResult {
                pdf_type,
                text: None,
                markdown: Some(markdown),
                page_count,
                processing_time_ms: start.elapsed().as_millis() as u64,
                pages_needing_ocr,
                title,
                confidence,
            }
        }
        PdfType::Scanned | PdfType::ImageBased => PdfProcessResult {
            pdf_type,
            text: None,
            markdown: None,
            page_count,
            processing_time_ms: start.elapsed().as_millis() as u64,
            pages_needing_ocr,
            title,
            confidence,
        },
        PdfType::Mixed => {
            let result = extractor::extract_text_with_positions_mem_and_rects(buffer, None).ok();
            let markdown = result.map(|(items, rects)| {
                to_markdown_from_items_with_rects(items, MarkdownOptions::default(), &rects)
            });

            PdfProcessResult {
                pdf_type,
                text: None,
                markdown,
                page_count,
                processing_time_ms: start.elapsed().as_millis() as u64,
                pages_needing_ocr,
                title,
                confidence,
            }
        }
    };

    Ok(result)
}

/// Process PDF from memory buffer with custom detection and markdown configuration
pub fn process_pdf_mem_with_config(
    buffer: &[u8],
    config: DetectionConfig,
    markdown_options: MarkdownOptions,
) -> Result<PdfProcessResult, PdfError> {
    let start = std::time::Instant::now();

    validate_pdf_bytes(buffer)?;

    let detection = detector::detect_pdf_type_mem_with_config(buffer, config)?;
    let page_count = detection.page_count;
    let pdf_type = detection.pdf_type;
    let pages_needing_ocr = detection.pages_needing_ocr;
    let title = detection.title;
    let confidence = detection.confidence;

    let result = match pdf_type {
        PdfType::TextBased => {
            let (items, rects) =
                extractor::extract_text_with_positions_mem_and_rects(buffer, None)?;
            let markdown = to_markdown_from_items_with_rects(items, markdown_options, &rects);

            PdfProcessResult {
                pdf_type,
                text: None,
                markdown: Some(markdown),
                page_count,
                processing_time_ms: start.elapsed().as_millis() as u64,
                pages_needing_ocr,
                title,
                confidence,
            }
        }
        PdfType::Scanned | PdfType::ImageBased => PdfProcessResult {
            pdf_type,
            text: None,
            markdown: None,
            page_count,
            processing_time_ms: start.elapsed().as_millis() as u64,
            pages_needing_ocr,
            title,
            confidence,
        },
        PdfType::Mixed => {
            let result = extractor::extract_text_with_positions_mem_and_rects(buffer, None).ok();
            let markdown = result.map(|(items, rects)| {
                to_markdown_from_items_with_rects(items, markdown_options.clone(), &rects)
            });

            PdfProcessResult {
                pdf_type,
                text: None,
                markdown,
                page_count,
                processing_time_ms: start.elapsed().as_millis() as u64,
                pages_needing_ocr,
                title,
                confidence,
            }
        }
    };

    Ok(result)
}

#[derive(Debug, thiserror::Error)]
pub enum PdfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PDF parsing error: {0}")]
    Parse(String),
    #[error("PDF is encrypted")]
    Encrypted,
    #[error("Invalid PDF structure")]
    InvalidStructure,
    #[error("Not a PDF: {0}")]
    NotAPdf(String),
}

impl From<lopdf::Error> for PdfError {
    fn from(e: lopdf::Error) -> Self {
        match e {
            lopdf::Error::IO(io_err) => PdfError::Io(io_err),
            lopdf::Error::Decryption(_)
            | lopdf::Error::InvalidPassword
            | lopdf::Error::AlreadyEncrypted
            | lopdf::Error::UnsupportedSecurityHandler(_) => PdfError::Encrypted,
            lopdf::Error::Parse(ref pe) if pe.to_string().contains("invalid file header") => {
                PdfError::NotAPdf("invalid PDF file header".to_string())
            }
            lopdf::Error::MissingXrefEntry
            | lopdf::Error::Xref(_)
            | lopdf::Error::IndirectObject { .. }
            | lopdf::Error::ObjectIdMismatch
            | lopdf::Error::InvalidObjectStream(_)
            | lopdf::Error::InvalidOffset(_) => PdfError::InvalidStructure,
            other => PdfError::Parse(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// PDF validation helpers
// ---------------------------------------------------------------------------

/// Strip UTF-8 BOM and leading ASCII whitespace from a byte slice.
fn strip_bom_and_whitespace(bytes: &[u8]) -> &[u8] {
    let b = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        bytes
    };
    let start = b
        .iter()
        .position(|&c| !c.is_ascii_whitespace())
        .unwrap_or(b.len());
    &b[start..]
}

/// Case-insensitive prefix check on byte slices.
fn starts_with_ci(haystack: &[u8], needle: &[u8]) -> bool {
    if haystack.len() < needle.len() {
        return false;
    }
    haystack[..needle.len()]
        .iter()
        .zip(needle)
        .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// Try to identify what kind of file the bytes represent.
fn detect_file_type_hint(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "file is empty".to_string();
    }

    let trimmed = strip_bom_and_whitespace(bytes);

    // HTML
    if starts_with_ci(trimmed, b"<!doctype html")
        || starts_with_ci(trimmed, b"<html")
        || starts_with_ci(trimmed, b"<head")
        || starts_with_ci(trimmed, b"<body")
    {
        return "file appears to be HTML".to_string();
    }

    // XML (but not HTML)
    if trimmed.starts_with(b"<?xml") || trimmed.starts_with(b"<") {
        // Distinguish generic XML from HTML-like XML
        if starts_with_ci(trimmed, b"<?xml") {
            return "file appears to be XML".to_string();
        }
        // Other tags that look like XML
        if trimmed.starts_with(b"<") && !trimmed.starts_with(b"<%") {
            return "file appears to be XML".to_string();
        }
    }

    // JSON
    if trimmed.starts_with(b"{") || trimmed.starts_with(b"[") {
        return "file appears to be JSON".to_string();
    }

    // PNG
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return "file appears to be a PNG image".to_string();
    }

    // JPEG
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return "file appears to be a JPEG image".to_string();
    }

    // ZIP / Office documents
    if bytes.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
        return "file appears to be a ZIP archive (possibly an Office document)".to_string();
    }

    // If it looks like mostly printable ASCII/UTF-8, call it plain text
    let sample = &bytes[..bytes.len().min(512)];
    let printable = sample
        .iter()
        .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
        .count();
    if printable > sample.len() * 3 / 4 {
        return "file appears to be plain text".to_string();
    }

    "file is not a PDF".to_string()
}

/// Validate that a byte buffer looks like a PDF (has `%PDF-` magic).
///
/// Scans the first 1024 bytes, allowing for a UTF-8 BOM and leading whitespace.
pub(crate) fn validate_pdf_bytes(buffer: &[u8]) -> Result<(), PdfError> {
    if buffer.is_empty() {
        return Err(PdfError::NotAPdf(detect_file_type_hint(buffer)));
    }

    let header = &buffer[..buffer.len().min(1024)];
    let trimmed = strip_bom_and_whitespace(header);

    if trimmed.starts_with(b"%PDF-") {
        Ok(())
    } else {
        Err(PdfError::NotAPdf(detect_file_type_hint(buffer)))
    }
}

/// Validate that a file on disk looks like a PDF.
///
/// Reads only the first 1024 bytes and delegates to [`validate_pdf_bytes`].
pub(crate) fn validate_pdf_file<P: AsRef<Path>>(path: P) -> Result<(), PdfError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buf = [0u8; 1024];
    let n = file.read(&mut buf)?;
    validate_pdf_bytes(&buf[..n])
}

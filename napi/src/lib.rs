#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Lightweight PDF classification result.
#[napi(object)]
pub struct PdfClassification {
  pub pdf_type: String,
  pub page_count: u32,
  pub pages_needing_ocr: Vec<u32>,
  pub confidence: f64,
}

/// A page's regions for text extraction: (page_index_0based, bboxes).
#[napi(object)]
pub struct PageRegions {
  pub page: u32,
  /// Each bbox is [x1, y1, x2, y2] in PDF points, top-left origin.
  pub regions: Vec<Vec<f64>>,
}

/// Extracted text for a single region.
#[napi(object)]
pub struct RegionText {
  pub text: String,
  /// `true` when the text should not be trusted (empty, GID fonts, garbage, encoding issues).
  pub needs_ocr: bool,
}

/// Extracted text for one page's regions.
#[napi(object)]
pub struct PageRegionTexts {
  pub page: u32,
  pub regions: Vec<RegionText>,
}

/// Classify a PDF: detect type (TextBased/Scanned/Mixed/ImageBased),
/// page count, and which pages need OCR. Takes PDF bytes as Buffer.
#[napi]
pub fn classify_pdf(buffer: Buffer) -> Result<PdfClassification> {
  let result = pdf_inspector::classify_pdf_mem(&buffer).map_err(|e| {
    Error::new(Status::GenericFailure, format!("classify_pdf failed: {e}"))
  })?;

  Ok(PdfClassification {
    pdf_type: match result.pdf_type {
      pdf_inspector::PdfType::TextBased => "TextBased".to_string(),
      pdf_inspector::PdfType::Scanned => "Scanned".to_string(),
      pdf_inspector::PdfType::ImageBased => "ImageBased".to_string(),
      pdf_inspector::PdfType::Mixed => "Mixed".to_string(),
    },
    page_count: result.page_count,
    pages_needing_ocr: result.pages_needing_ocr,
    confidence: result.confidence as f64,
  })
}

/// Extract text within bounding-box regions from a PDF.
///
/// For hybrid OCR: layout model detects regions in rendered images,
/// this extracts PDF text within those regions — skipping GPU OCR
/// for text-based pages.
///
/// Each region result includes `needs_ocr` — set when the extracted text
/// is unreliable (empty, GID-encoded fonts, garbage, encoding issues).
///
/// Coordinates are PDF points with top-left origin.
#[napi]
pub fn extract_text_in_regions(
  buffer: Buffer,
  page_regions: Vec<PageRegions>,
) -> Result<Vec<PageRegionTexts>> {
  // Convert from napi types to the Rust API's expected format
  let regions: Vec<(u32, Vec<[f32; 4]>)> = page_regions
    .iter()
    .map(|pr| {
      let bboxes: Vec<[f32; 4]> = pr
        .regions
        .iter()
        .map(|r| {
          if r.len() != 4 {
            [0.0, 0.0, 0.0, 0.0]
          } else {
            [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]
          }
        })
        .collect();
      (pr.page, bboxes)
    })
    .collect();

  let results = pdf_inspector::extract_text_in_regions_mem(&buffer, &regions).map_err(|e| {
    Error::new(
      Status::GenericFailure,
      format!("extract_text_in_regions failed: {e}"),
    )
  })?;

  Ok(
    results
      .into_iter()
      .map(|page_result| PageRegionTexts {
        page: page_result.page,
        regions: page_result
          .regions
          .into_iter()
          .map(|r| RegionText {
            text: r.text,
            needs_ocr: r.needs_ocr,
          })
          .collect(),
      })
      .collect(),
  )
}

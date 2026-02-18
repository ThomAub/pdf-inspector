//! Font width parsing, encoding, and text decoding.

use crate::glyph_names::glyph_to_char;
use crate::tounicode::FontCMaps;
use crate::types::{FontEncodingMap, FontWidthInfo, PageFontEncodings, PageFontWidths};
use lopdf::{Document, Encoding, Object};
use std::collections::HashMap;

/// Resolve a PDF object reference to an array
pub(crate) fn resolve_array<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Vec<Object>> {
    match obj {
        Object::Array(arr) => Some(arr),
        Object::Reference(r) => {
            if let Ok(Object::Array(arr)) = doc.get_object(*r) {
                Some(arr)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Resolve a PDF object reference to a dictionary
pub(crate) fn resolve_dict<'a>(
    doc: &'a Document,
    obj: &'a Object,
) -> Option<&'a lopdf::Dictionary> {
    match obj {
        Object::Dictionary(d) => Some(d),
        Object::Reference(r) => doc.get_dictionary(*r).ok(),
        _ => None,
    }
}

/// Build font width info for all fonts on a page
pub(crate) fn build_font_widths(
    doc: &Document,
    fonts: &std::collections::BTreeMap<Vec<u8>, &lopdf::Dictionary>,
) -> PageFontWidths {
    let mut widths = PageFontWidths::new();

    for (font_name, font_dict) in fonts {
        let resource_name = String::from_utf8_lossy(font_name).to_string();
        if let Some(info) = parse_font_widths(doc, font_dict) {
            widths.insert(resource_name, info);
        }
    }

    widths
}

/// Parse font widths from a font dictionary, dispatching by Subtype
pub(crate) fn parse_font_widths(
    doc: &Document,
    font_dict: &lopdf::Dictionary,
) -> Option<FontWidthInfo> {
    // Get the font subtype
    let subtype = font_dict.get(b"Subtype").ok()?;
    let subtype_name = subtype.as_name().ok()?;

    match subtype_name {
        b"Type0" => parse_type0_widths(doc, font_dict),
        b"Type1" | b"TrueType" | b"MMType1" | b"Type3" => parse_simple_font_widths(doc, font_dict),
        _ => None,
    }
}

/// Parse widths for simple fonts (Type1, TrueType, MMType1, Type3)
/// Reads FirstChar, LastChar, and Widths array.
/// For Type3 fonts, reads FontMatrix to determine the correct units_scale.
pub(crate) fn parse_simple_font_widths(
    doc: &Document,
    font_dict: &lopdf::Dictionary,
) -> Option<FontWidthInfo> {
    let first_char = font_dict.get(b"FirstChar").ok().and_then(|o| match o {
        Object::Integer(n) => Some(*n as u16),
        Object::Reference(r) => doc.get_object(*r).ok().and_then(|o| {
            if let Object::Integer(n) = o {
                Some(*n as u16)
            } else {
                None
            }
        }),
        _ => None,
    })?;

    let last_char = font_dict.get(b"LastChar").ok().and_then(|o| match o {
        Object::Integer(n) => Some(*n as u16),
        Object::Reference(r) => doc.get_object(*r).ok().and_then(|o| {
            if let Object::Integer(n) = o {
                Some(*n as u16)
            } else {
                None
            }
        }),
        _ => None,
    })?;

    let widths_obj = font_dict.get(b"Widths").ok()?;
    let widths_array = resolve_array(doc, widths_obj)?;

    let mut widths = HashMap::new();
    let mut space_width: u16 = 0;

    for (i, w_obj) in widths_array.iter().enumerate() {
        let code = first_char + i as u16;
        if code > last_char {
            break;
        }
        let w = match w_obj {
            Object::Integer(n) => *n as u16,
            Object::Real(n) => *n as u16,
            Object::Reference(r) => {
                if let Ok(obj) = doc.get_object(*r) {
                    match obj {
                        Object::Integer(n) => *n as u16,
                        Object::Real(n) => *n as u16,
                        _ => continue,
                    }
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        if code == 32 {
            space_width = w;
        }
        widths.insert(code, w);
    }

    // Determine units_scale: for Type3 fonts, use FontMatrix[0]; for others, use 1/1000
    let units_scale = if let Ok(fm) = font_dict.get(b"FontMatrix") {
        if let Some(arr) = resolve_array(doc, fm) {
            if !arr.is_empty() {
                match &arr[0] {
                    Object::Real(r) => r.abs(),
                    Object::Integer(i) => (*i as f32).abs(),
                    _ => 0.001,
                }
            } else {
                0.001
            }
        } else {
            0.001
        }
    } else {
        0.001 // Standard 1000-unit system
    };

    // If space width wasn't found in the table, estimate from font metrics.
    // The default of 250 is calibrated for standard 1000-unit fonts (units_scale=0.001).
    // For Type3 fonts with different coordinate systems, use average glyph width instead.
    if space_width == 0 {
        if !widths.is_empty() && (units_scale - 0.001).abs() > 0.0005 {
            // Non-standard scale: estimate space as ~45% of average glyph width
            let sum: u32 = widths.values().map(|&w| w as u32).sum();
            let avg = sum as f32 / widths.len() as f32;
            space_width = (avg * 0.45).max(1.0) as u16;
        } else {
            space_width = 250;
        }
    }

    Some(FontWidthInfo {
        widths,
        default_width: 0,
        space_width,
        is_cid: false,
        units_scale,
        wmode: 0,
    })
}

/// Parse widths for Type0 (composite/CID) fonts
/// Reads DescendantFonts → CIDFont → W array and DW value
pub(crate) fn parse_type0_widths(
    doc: &Document,
    font_dict: &lopdf::Dictionary,
) -> Option<FontWidthInfo> {
    let desc_fonts_obj = font_dict.get(b"DescendantFonts").ok()?;
    let desc_fonts = resolve_array(doc, desc_fonts_obj)?;

    if desc_fonts.is_empty() {
        return None;
    }

    // Get the first descendant font dictionary
    let cid_font_dict = resolve_dict(doc, &desc_fonts[0])?;

    // Get DW (default width)
    let default_width = cid_font_dict
        .get(b"DW")
        .ok()
        .and_then(|o| match o {
            Object::Integer(n) => Some(*n as u16),
            Object::Real(n) => Some(*n as u16),
            _ => None,
        })
        .unwrap_or(1000);

    let mut widths = HashMap::new();

    // Parse W array if present
    if let Ok(w_obj) = cid_font_dict.get(b"W") {
        if let Some(w_array) = resolve_array(doc, w_obj) {
            parse_cid_w_array(doc, w_array, &mut widths);
        }
    }

    // Try to determine space width (CID 32 or CID 3 are common for space)
    let space_width = widths
        .get(&32)
        .or_else(|| widths.get(&3))
        .copied()
        .unwrap_or(if default_width > 0 {
            default_width / 4
        } else {
            250
        });

    let wmode = font_dict
        .get(b"WMode")
        .ok()
        .and_then(|o| match o {
            Object::Integer(n) => Some(*n as u8),
            _ => None,
        })
        .unwrap_or(0);

    Some(FontWidthInfo {
        widths,
        default_width,
        space_width,
        is_cid: true,
        units_scale: 0.001, // CID fonts use standard 1000-unit system
        wmode,
    })
}

/// Parse a CID W array into widths map
/// Format: [c [w1 w2 ...]] (consecutive from c) or [c_first c_last w] (range with same width)
pub(crate) fn parse_cid_w_array(
    doc: &Document,
    w_array: &[Object],
    widths: &mut HashMap<u16, u16>,
) {
    let mut i = 0;
    while i < w_array.len() {
        let start_cid = match &w_array[i] {
            Object::Integer(n) => *n as u16,
            Object::Real(n) => *n as u16,
            _ => {
                i += 1;
                continue;
            }
        };
        i += 1;
        if i >= w_array.len() {
            break;
        }

        // Check if next element is an array (consecutive widths) or integer (range)
        match &w_array[i] {
            Object::Array(arr) => {
                // [c [w1 w2 ...]] — consecutive widths starting at c
                for (j, w_obj) in arr.iter().enumerate() {
                    let w = match w_obj {
                        Object::Integer(n) => *n as u16,
                        Object::Real(n) => *n as u16,
                        _ => continue,
                    };
                    widths.insert(start_cid + j as u16, w);
                }
                i += 1;
            }
            Object::Reference(r) => {
                // Could be a reference to an array
                if let Ok(Object::Array(arr)) = doc.get_object(*r) {
                    for (j, w_obj) in arr.iter().enumerate() {
                        let w = match w_obj {
                            Object::Integer(n) => *n as u16,
                            Object::Real(n) => *n as u16,
                            _ => continue,
                        };
                        widths.insert(start_cid + j as u16, w);
                    }
                    i += 1;
                } else {
                    // Treat as c_first c_last w
                    i += 1; // skip this
                }
            }
            Object::Integer(end_cid) => {
                // [c_first c_last w] — range with uniform width
                let end = *end_cid as u16;
                i += 1;
                if i >= w_array.len() {
                    break;
                }
                let w = match &w_array[i] {
                    Object::Integer(n) => *n as u16,
                    Object::Real(n) => *n as u16,
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                for cid in start_cid..=end {
                    widths.insert(cid, w);
                }
                i += 1;
            }
            Object::Real(end_cid) => {
                let end = *end_cid as u16;
                i += 1;
                if i >= w_array.len() {
                    break;
                }
                let w = match &w_array[i] {
                    Object::Integer(n) => *n as u16,
                    Object::Real(n) => *n as u16,
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                for cid in start_cid..=end {
                    widths.insert(cid, w);
                }
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
}

/// Compute the width of a string in text space units,
/// given raw bytes and font width info.
/// Returns width in text space units (font_units * units_scale * font_size).
pub(crate) fn compute_string_width_ts(
    bytes: &[u8],
    font_info: &FontWidthInfo,
    font_size: f32,
) -> f32 {
    let mut total: f32 = 0.0;
    if font_info.is_cid {
        // 2-byte (big-endian) character codes
        let mut j = 0;
        while j + 1 < bytes.len() {
            let cid = u16::from_be_bytes([bytes[j], bytes[j + 1]]);
            let w = font_info
                .widths
                .get(&cid)
                .copied()
                .unwrap_or(font_info.default_width);
            total += w as f32;
            j += 2;
        }
    } else {
        // 1-byte character codes
        for &b in bytes {
            let code = b as u16;
            let w = font_info
                .widths
                .get(&code)
                .copied()
                .unwrap_or(font_info.default_width);
            total += w as f32;
        }
    }
    // Convert from font units to text space using the font's scale factor
    total * font_info.units_scale * font_size
}

/// Extract raw bytes from a PDF operand (String object)
pub(crate) fn get_operand_bytes(obj: &Object) -> Option<&[u8]> {
    if let Object::String(bytes, _) = obj {
        Some(bytes)
    } else {
        None
    }
}

/// Build encoding maps for all fonts on a page
pub(crate) fn build_font_encodings(
    doc: &Document,
    fonts: &std::collections::BTreeMap<Vec<u8>, &lopdf::Dictionary>,
) -> PageFontEncodings {
    let mut encodings = PageFontEncodings::new();

    for (font_name, font_dict) in fonts {
        let resource_name = String::from_utf8_lossy(font_name).to_string();

        if let Some(encoding_map) = parse_font_encoding(doc, font_dict) {
            encodings.insert(resource_name, encoding_map);
        }
    }

    encodings
}

/// Parse font encoding from a font dictionary
pub(crate) fn parse_font_encoding(
    doc: &Document,
    font_dict: &lopdf::Dictionary,
) -> Option<FontEncodingMap> {
    let encoding_obj = font_dict.get(b"Encoding").ok()?;

    // Encoding can be a name or a dictionary
    match encoding_obj {
        Object::Name(_name) => {
            // Standard encoding name (e.g., MacRomanEncoding, WinAnsiEncoding)
            // For standard encodings, we can use the standard tables
            // But we still need to check for Differences
            None // Let lopdf handle standard encodings
        }
        Object::Reference(obj_ref) => {
            // Reference to encoding dictionary
            if let Ok(enc_dict) = doc.get_dictionary(*obj_ref) {
                parse_encoding_dictionary(doc, enc_dict)
            } else {
                None
            }
        }
        Object::Dictionary(enc_dict) => parse_encoding_dictionary(doc, enc_dict),
        _ => None,
    }
}

/// Parse an encoding dictionary with Differences array
pub(crate) fn parse_encoding_dictionary(
    doc: &Document,
    enc_dict: &lopdf::Dictionary,
) -> Option<FontEncodingMap> {
    let differences = enc_dict.get(b"Differences").ok()?;

    let diff_array = match differences {
        Object::Array(arr) => arr.clone(),
        Object::Reference(obj_ref) => {
            if let Ok(Object::Array(arr)) = doc.get_object(*obj_ref) {
                arr.clone()
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let mut encoding_map = FontEncodingMap::new();
    let mut current_code: u8 = 0;

    for item in diff_array {
        match item {
            Object::Integer(n) => {
                // This sets the starting code for subsequent glyph names
                current_code = n as u8;
            }
            Object::Name(name) => {
                // Map current code to glyph name -> Unicode
                let glyph_name = String::from_utf8_lossy(&name).to_string();
                if let Some(ch) = glyph_to_char(&glyph_name) {
                    encoding_map.insert(current_code, ch);
                }
                current_code = current_code.wrapping_add(1);
            }
            _ => {}
        }
    }

    if encoding_map.is_empty() {
        None
    } else {
        Some(encoding_map)
    }
}

/// Decode text from a PDF string operand using font CMaps, encodings, and fallbacks.
pub(crate) fn extract_text_from_operand(
    obj: &Object,
    current_font: &str,
    font_cmaps: &FontCMaps,
    font_base_names: &std::collections::HashMap<String, String>,
    font_tounicode_refs: &std::collections::HashMap<String, u32>,
    font_encodings: &PageFontEncodings,
    encoding_cache: &HashMap<String, Encoding<'_>>,
) -> Option<String> {
    if let Object::String(bytes, _) = obj {
        // First, try to look up CMap by ToUnicode object reference (most reliable)
        // This handles cases where multiple fonts have the same BaseFont but different ToUnicode
        if let Some(&obj_num) = font_tounicode_refs.get(current_font) {
            if let Some(cmap) = font_cmaps.get_by_obj(obj_num) {
                let decoded = cmap.decode_cids(bytes);
                if !decoded.is_empty() {
                    return Some(decoded);
                }
            }
        }

        // Fall back to base name lookup with object number
        if let (Some(base_name), Some(&obj_num)) = (
            font_base_names.get(current_font),
            font_tounicode_refs.get(current_font),
        ) {
            if let Some(cmap) = font_cmaps.get_with_obj(base_name, obj_num) {
                let decoded = cmap.decode_cids(bytes);
                if !decoded.is_empty() {
                    return Some(decoded);
                }
            }
        }

        // Try base name only (legacy fallback)
        if let Some(base_name) = font_base_names.get(current_font) {
            if let Some(cmap) = font_cmaps.get(base_name) {
                let decoded = cmap.decode_cids(bytes);
                if !decoded.is_empty() {
                    return Some(decoded);
                }
            }
        }

        // Also try looking up by resource name directly
        if let Some(cmap) = font_cmaps.get(current_font) {
            let decoded = cmap.decode_cids(bytes);
            if !decoded.is_empty() {
                return Some(decoded);
            }
        }

        // Try our custom encoding map from Differences arrays.
        // The Differences array overrides specific codes in a base encoding (typically
        // WinAnsiEncoding). We must combine Differences entries with the base encoding
        // rather than using filter_map which silently drops unmapped bytes.
        if let Some(encoding_map) = font_encodings.get(current_font) {
            let has_diff_match = bytes.iter().any(|b| encoding_map.contains_key(b));
            if has_diff_match {
                let decoded: String = bytes
                    .iter()
                    .filter_map(|&b| {
                        if let Some(&ch) = encoding_map.get(&b) {
                            Some(ch)
                        } else if b >= 0x20 {
                            // Base encoding fallback for printable bytes.
                            // For codes 0x20-0x7E this matches all standard PDF encodings.
                            Some(b as char)
                        } else {
                            None // Skip unmapped control characters
                        }
                    })
                    .collect();
                if !decoded.is_empty() {
                    return Some(decoded);
                }
            }
        }

        // Try to decode using cached font encoding from lopdf
        if let Some(encoding) = encoding_cache.get(current_font) {
            if let Ok(text) = Document::decode_text(encoding, bytes) {
                return Some(text);
            }
        }

        // Fallback: try UTF-16BE then Latin-1
        if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
            let utf16: Vec<u16> = bytes[2..]
                .chunks_exact(2)
                .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
                .collect();
            return Some(String::from_utf16_lossy(&utf16));
        }

        // Latin-1 fallback
        Some(bytes.iter().map(|&b| b as char).collect())
    } else {
        None
    }
}

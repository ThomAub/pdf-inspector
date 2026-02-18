//! Rectangle-based table detection using union-find clustering.

use std::collections::HashMap;

use crate::types::{PdfRect, TextItem};

use super::Table;

/// Disjoint-set (union-find) for clustering indices.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Check if two rects overlap after expanding each by `tol` on all sides.
pub(crate) fn rects_overlap(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32), tol: f32) -> bool {
    // a and b are (x, y, w, h) where (x,y) is bottom-left corner
    let (ax, ay, aw, ah) = *a;
    let (bx, by, bw, bh) = *b;
    // Expand each rect by tol
    let a_left = ax - tol;
    let a_right = ax + aw + tol;
    let a_bottom = ay - tol;
    let a_top = ay + ah + tol;
    let b_left = bx - tol;
    let b_right = bx + bw + tol;
    let b_bottom = by - tol;
    let b_top = by + bh + tol;
    // AABB overlap: NOT (separated)
    !(a_right < b_left || b_right < a_left || a_top < b_bottom || b_top < a_bottom)
}

/// Cluster rects by spatial overlap using union-find.
/// Returns groups of rect indices; only groups with ≥ `min_size` rects are returned.
pub(crate) fn cluster_rects(
    rects: &[(f32, f32, f32, f32)],
    tolerance: f32,
    min_size: usize,
) -> Vec<Vec<usize>> {
    let n = rects.len();
    let mut uf = UnionFind::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            if rects_overlap(&rects[i], &rects[j], tolerance) {
                uf.union(i, j);
            }
        }
    }

    // Group indices by root
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    groups
        .into_values()
        .filter(|g| g.len() >= min_size)
        .collect()
}

/// Detect tables from explicit rectangle (`re`) operators in the PDF.
///
/// Many PDFs draw cell borders using `re` (rectangle) operators.  Table pages
/// typically have 100-200+ rects while non-table pages have < 30.  This function
/// clusters spatially connected rectangles into groups, then identifies grids of
/// cell-sized rectangles within each cluster and assigns text items to cells.
pub fn detect_tables_from_rects(items: &[TextItem], rects: &[PdfRect], page: u32) -> Vec<Table> {
    // Filter rects on this page; normalize negative widths/heights; skip tiny rects.
    let mut page_rects: Vec<(f32, f32, f32, f32)> = Vec::new(); // (x, y, w, h) normalized
    for r in rects {
        if r.page != page {
            continue;
        }
        let (mut x, mut y, mut w, mut h) = (r.x, r.y, r.width, r.height);
        if w < 0.0 {
            x += w;
            w = -w;
        }
        if h < 0.0 {
            y += h;
            h = -h;
        }
        // Skip tiny rects (borders, dots, decorations)
        if w < 5.0 || h < 5.0 {
            continue;
        }
        page_rects.push((x, y, w, h));
    }

    // Need a reasonable number of cell rects to form a table
    if page_rects.len() < 6 {
        return vec![];
    }

    // Cluster spatially connected rects into groups
    let clusters = cluster_rects(&page_rects, 3.0, 6);

    let mut tables = Vec::new();
    for cluster_indices in &clusters {
        let group_rects: Vec<(f32, f32, f32, f32)> =
            cluster_indices.iter().map(|&i| page_rects[i]).collect();
        if let Some(table) = detect_table_from_rect_group(items, &group_rects, page) {
            tables.push(table);
        }
    }

    tables
}

/// Detect a single table from a cluster of spatially connected rects.
///
/// Contains the grid-detection logic: snap edges, fill-ratio check,
/// assign items to grid, content density validation.
pub(crate) fn detect_table_from_rect_group(
    items: &[TextItem],
    group_rects: &[(f32, f32, f32, f32)],
    page: u32,
) -> Option<Table> {
    // Extract unique X and Y edges from all rects
    let mut x_edges: Vec<f32> = Vec::new();
    let mut y_edges: Vec<f32> = Vec::new();
    for &(x, y, w, h) in group_rects {
        x_edges.push(x);
        x_edges.push(x + w);
        y_edges.push(y);
        y_edges.push(y + h);
    }

    let x_edges = snap_edges(&x_edges, 2.0);
    let y_edges = snap_edges(&y_edges, 2.0);

    if x_edges.len() < 3 || y_edges.len() < 4 {
        // Need at least 2 columns (3 edges) and 3 rows (4 edges)
        return None;
    }

    // Sort column edges left-to-right, row edges top-to-bottom (highest Y first for PDF)
    let mut col_edges = x_edges;
    col_edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut row_edges = y_edges;
    row_edges.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let num_cols = col_edges.len() - 1;
    let num_rows = row_edges.len() - 1;

    if num_cols < 2 || num_rows < 2 {
        return None;
    }

    // Reject grids that are too large — real tables rarely exceed 12 columns.
    // Form-style PDFs with scattered field boxes produce huge sparse grids.
    if num_cols > 12 {
        return None;
    }

    // Verify that cell-sized rects actually fill the grid
    // Count how many grid cells have a matching rect
    let mut filled_cells = 0u32;
    for row in 0..num_rows {
        let y_top = row_edges[row];
        let y_bot = row_edges[row + 1];
        for col in 0..num_cols {
            let x_left = col_edges[col];
            let x_right = col_edges[col + 1];
            // Check if any rect approximately covers this cell
            let cell_covered = group_rects.iter().any(|&(rx, ry, rw, rh)| {
                let tol = 3.0;
                rx <= x_left + tol
                    && (rx + rw) >= x_right - tol
                    && ry <= y_top + tol
                    && (ry + rh) >= y_bot - tol
            });
            if cell_covered {
                filled_cells += 1;
            }
        }
    }

    let total_cells = (num_cols * num_rows) as f32;
    let fill_ratio = filled_cells as f32 / total_cells;

    // Require at least 30% of cells to be backed by rects
    if fill_ratio < 0.3 {
        return None;
    }

    // Build table: assign text items to cells
    let (cells, item_indices) = assign_items_to_grid(items, &col_edges, &row_edges, page);

    // Compute column centers and row centers for the Table struct
    let columns: Vec<f32> = (0..num_cols)
        .map(|c| (col_edges[c] + col_edges[c + 1]) / 2.0)
        .collect();
    let rows: Vec<f32> = (0..num_rows)
        .map(|r| (row_edges[r] + row_edges[r + 1]) / 2.0)
        .collect();

    // Skip if no text was assigned
    if item_indices.is_empty() {
        return None;
    }

    // Skip tables with only 1 row of content (header-only)
    let non_empty_rows = cells
        .iter()
        .filter(|row| row.iter().any(|c| !c.trim().is_empty()))
        .count();
    if non_empty_rows < 2 {
        return None;
    }

    // Content density check: reject tables where most cells are empty.
    // Real tables have content in most cells; form layouts produce sparse grids.
    let non_empty_cells = cells
        .iter()
        .flat_map(|row| row.iter())
        .filter(|c| !c.trim().is_empty())
        .count();
    let content_ratio = non_empty_cells as f32 / total_cells;
    if content_ratio < 0.25 {
        return None;
    }

    // Reject tables with any completely empty column — indicates a bad grid.
    for col in 0..num_cols {
        let col_has_content = cells
            .iter()
            .any(|row| row.get(col).is_some_and(|c| !c.trim().is_empty()));
        if !col_has_content {
            return None;
        }
    }

    Some(Table {
        columns,
        rows,
        cells,
        item_indices,
    })
}

/// Deduplicate nearby edge values within a tolerance, returning sorted unique edges.
pub(crate) fn snap_edges(values: &[f32], tolerance: f32) -> Vec<f32> {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut snapped: Vec<f32> = Vec::new();
    for &v in &sorted {
        if let Some(last) = snapped.last() {
            if (v - *last).abs() <= tolerance {
                continue; // Skip — too close to previous edge
            }
        }
        snapped.push(v);
    }
    snapped
}

/// Assign text items to grid cells defined by column/row edges.
///
/// Returns `(cells, item_indices)` where `cells[row][col]` is the cell text
/// and `item_indices` lists the original item indices that were consumed.
pub(crate) fn assign_items_to_grid(
    items: &[TextItem],
    col_edges: &[f32],
    row_edges: &[f32],
    page: u32,
) -> (Vec<Vec<String>>, Vec<usize>) {
    let num_cols = col_edges.len() - 1;
    let num_rows = row_edges.len() - 1;

    // Collect items per cell for proper sorting before joining
    let mut cell_items: Vec<Vec<Vec<(usize, &TextItem)>>> =
        vec![vec![Vec::new(); num_cols]; num_rows];
    let mut indices = Vec::new();

    for (idx, item) in items.iter().enumerate() {
        if item.page != page {
            continue;
        }
        // Use item center for assignment
        let cx = item.x + item.width / 2.0;
        let cy = item.y;

        // Find column: cx must be between col_edges[c] and col_edges[c+1]
        let col = (0..num_cols).find(|&c| cx >= col_edges[c] - 2.0 && cx <= col_edges[c + 1] + 2.0);
        // Find row: cy must be between row_edges[r+1] (bottom) and row_edges[r] (top)
        let row = (0..num_rows).find(|&r| cy >= row_edges[r + 1] - 2.0 && cy <= row_edges[r] + 2.0);

        if let (Some(c), Some(r)) = (col, row) {
            cell_items[r][c].push((idx, item));
            indices.push(idx);
        }
    }

    // Build cell strings: sort items within each cell by Y descending then X ascending
    let mut cells: Vec<Vec<String>> = Vec::with_capacity(num_rows);
    for row_items in &mut cell_items {
        let mut row_cells = Vec::with_capacity(num_cols);
        for col_items in row_items.iter_mut() {
            col_items.sort_by(|a, b| {
                b.1.y
                    .partial_cmp(&a.1.y)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.1.x
                            .partial_cmp(&b.1.x)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
            let text: String = col_items
                .iter()
                .map(|(_, item)| item.text.trim())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            row_cells.push(text);
        }
        cells.push(row_cells);
    }

    (cells, indices)
}

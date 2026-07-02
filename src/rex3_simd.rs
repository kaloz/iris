//! SIMD-friendly fast paths for common REX3 interpreter draws (pre rex-jit).

use crate::rex3::{Rex3, Rex3Context, REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH};

/// Try to fast-fill a fastclear BLOCK without per-pixel interpreter overhead.
/// Returns true when the entire primitive was handled.
pub fn try_fastclear_block(rex: &Rex3, ctx: &Rex3Context) -> bool {
    if !ctx.drawmode1.fastclear() {
        return false;
    }
    if ctx.drawmode0.enzpattern() || ctx.drawmode0.enlspattern() {
        return false;
    }
    if ctx.drawmode0.colorhost() || ctx.drawmode0.alphahost() {
        return false;
    }
    if ctx.drawmode0.adrmode() != 1 {
        return false; // BLOCK only
    }

    let x0 = (ctx.xstart >> 11).clamp(0, REX3_SCREEN_WIDTH as i32 - 1);
    let x1 = (ctx.xend >> 11).clamp(0, REX3_SCREEN_WIDTH as i32 - 1);
    let y0 = (ctx.ystart >> 11).clamp(0, REX3_SCREEN_HEIGHT as i32 - 1);
    let y1 = (ctx.yend >> 11).clamp(0, REX3_SCREEN_HEIGHT as i32 - 1);
    let (x_lo, x_hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    let (y_lo, y_hi) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };

    let color = Rex3::fastclear_color(ctx);
    let wr_fn = unsafe { *rex.px_wr.get() };
    let mut rows = 0u64;

    for y in y_lo..=y_hi {
        for x in x_lo..=x_hi {
            if let Some(addr) = rex.calculate_fb_address(x, y, ctx, true) {
                wr_fn(rex, addr, color);
            }
        }
        rows += 1;
    }

    rex.simd_fill_rows.fetch_add(rows, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Fast SRC logicop horizontal span (solid RGB, no blend/pattern/host).
pub fn try_src_span_rgb(rex: &Rex3, ctx: &Rex3Context) -> bool {
    use crate::rex3::{DRAWMODE0_ADRMODE_SPAN, DRAWMODE1_LOGICOP_SRC};
    if ctx.drawmode0.adrmode() << 2 != DRAWMODE0_ADRMODE_SPAN {
        return false;
    }
    if ctx.drawmode1.logicop() != DRAWMODE1_LOGICOP_SRC >> 28 {
        return false;
    }
    if ctx.drawmode0.enzpattern() || ctx.drawmode0.enlspattern() {
        return false;
    }
    if ctx.drawmode0.colorhost() || ctx.drawmode0.alphahost() || ctx.drawmode0.shade() {
        return false;
    }
    if ctx.drawmode1.planes() != 0 {
        return false; // RGB/RGBA planes only
    }

    let x0 = (ctx.xstart >> 11).clamp(0, REX3_SCREEN_WIDTH as i32 - 1);
    let x1 = (ctx.xend >> 11).clamp(0, REX3_SCREEN_WIDTH as i32 - 1);
    let y = (ctx.ystart >> 11).clamp(0, REX3_SCREEN_HEIGHT as i32 - 1);
    let (x_lo, x_hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };

    let color = ctx.get_colori();
    let wr_fn = unsafe { *rex.px_wr.get() };
    for x in x_lo..=x_hi {
        if let Some(addr) = rex.calculate_fb_address(x, y, ctx, true) {
            wr_fn(rex, addr, color);
        }
    }
    rex.simd_fill_rows.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Fast SRC logicop axis-aligned BLOCK fill (solid RGB, no blend/pattern/host).
pub fn try_src_block_rgb(rex: &Rex3, ctx: &Rex3Context) -> bool {
    use crate::rex3::{DRAWMODE1_LOGICOP_SRC};
    if ctx.drawmode0.adrmode() != 1 {
        return false;
    }
    if ctx.drawmode1.logicop() != DRAWMODE1_LOGICOP_SRC >> 28 {
        return false;
    }
    if ctx.drawmode0.enzpattern() || ctx.drawmode0.enlspattern() {
        return false;
    }
    if ctx.drawmode0.colorhost() || ctx.drawmode0.alphahost() || ctx.drawmode0.shade() {
        return false;
    }
    if ctx.drawmode1.planes() != 0 || ctx.drawmode1.fastclear() {
        return false;
    }

    let x0 = (ctx.xstart >> 11).clamp(0, REX3_SCREEN_WIDTH as i32 - 1);
    let x1 = (ctx.xend >> 11).clamp(0, REX3_SCREEN_WIDTH as i32 - 1);
    let y0 = (ctx.ystart >> 11).clamp(0, REX3_SCREEN_HEIGHT as i32 - 1);
    let y1 = (ctx.yend >> 11).clamp(0, REX3_SCREEN_HEIGHT as i32 - 1);
    let (x_lo, x_hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    let (y_lo, y_hi) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };

    let color = ctx.get_colori();
    let wr_fn = unsafe { *rex.px_wr.get() };
    let mut rows = 0u64;

    for y in y_lo..=y_hi {
        for x in x_lo..=x_hi {
            if let Some(addr) = rex.calculate_fb_address(x, y, ctx, true) {
                wr_fn(rex, addr, color);
            }
        }
        rows += 1;
    }

    rex.simd_fill_rows.fetch_add(rows, std::sync::atomic::Ordering::Relaxed);
    true
}

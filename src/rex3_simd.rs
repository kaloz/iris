//! SIMD-friendly fast paths for common REX3 interpreter draws (pre rex-jit).

use crate::rex3::{Rex3, Rex3Context, REX3_COORD_BIAS, REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH};

// ctx.xstart/ystart/xend/yend (>>11) are screen coordinates biased by
// REX3_COORD_BIAS (4096) — the same convention calculate_fb_address expects
// and subtracts internally. The bounding-box clamps below exist only to cap
// a degenerate/garbage coordinate range before looping, not to reject
// off-screen pixels (calculate_fb_address already does that per-pixel via
// its own bias-aware bounds check, including XYWIN/xymove offsets these
// clamps don't know about) — so the clamp bounds must be shifted by the
// same bias, not the raw [0, SCREEN_WIDTH-1] screen range. Using unbiased
// bounds here previously collapsed every on-screen coordinate (always
// >= REX3_COORD_BIAS) down to REX3_SCREEN_WIDTH-1/REX3_SCREEN_HEIGHT-1,
// making these fast paths silently draw a single degenerate off-screen
// pixel instead of the requested region — confirmed via jit_fastclear_rgb24.
const REX3_BIASED_X_MIN: i32 = REX3_COORD_BIAS;
const REX3_BIASED_X_MAX: i32 = REX3_COORD_BIAS + REX3_SCREEN_WIDTH - 1;
const REX3_BIASED_Y_MIN: i32 = REX3_COORD_BIAS;
const REX3_BIASED_Y_MAX: i32 = REX3_COORD_BIAS + REX3_SCREEN_HEIGHT - 1;

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

    let x0 = (ctx.xstart >> 11).clamp(REX3_BIASED_X_MIN, REX3_BIASED_X_MAX);
    let x1 = (ctx.xend >> 11).clamp(REX3_BIASED_X_MIN, REX3_BIASED_X_MAX);
    let y0 = (ctx.ystart >> 11).clamp(REX3_BIASED_Y_MIN, REX3_BIASED_Y_MAX);
    let y1 = (ctx.yend >> 11).clamp(REX3_BIASED_Y_MIN, REX3_BIASED_Y_MAX);
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

    let x0 = (ctx.xstart >> 11).clamp(REX3_BIASED_X_MIN, REX3_BIASED_X_MAX);
    let x1 = (ctx.xend >> 11).clamp(REX3_BIASED_X_MIN, REX3_BIASED_X_MAX);
    let y = (ctx.ystart >> 11).clamp(REX3_BIASED_Y_MIN, REX3_BIASED_Y_MAX);
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

    let x0 = (ctx.xstart >> 11).clamp(REX3_BIASED_X_MIN, REX3_BIASED_X_MAX);
    let x1 = (ctx.xend >> 11).clamp(REX3_BIASED_X_MIN, REX3_BIASED_X_MAX);
    let y0 = (ctx.ystart >> 11).clamp(REX3_BIASED_Y_MIN, REX3_BIASED_Y_MAX);
    let y1 = (ctx.yend >> 11).clamp(REX3_BIASED_Y_MIN, REX3_BIASED_Y_MAX);
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

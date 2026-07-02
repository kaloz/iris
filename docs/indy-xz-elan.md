# Indy XZ / Elan graphics — research notes

IRIS emulates **Newport (REX3)** by default. The Indy **XZ** and **Elan** options use the
Express **GR3** two-board set (HQ2 + GE7/RE3 + VB2) instead. This document collects public
register-map and bus-layout research for the Wave 4 preview stub in `src/xz.rs`.

**Status:** preview stub only — probe-friendly ID/status registers, no GL/command processing.

## Hardware summary

| Marketing | `hinv` / `gfxinfo` | GE count | Z-buffer | Display path |
|-----------|-------------------|----------|----------|--------------|
| Indy XZ   | GR3-XZ            | 4× GE7   | yes      | VC1, XMAP5, VB2 RAMDAC |
| Indy Elan | GR3-Elan          | 4× GE7   | yes      | same family as XZ |

XZ and Elan share the GR3 base board; Elan is the higher-clock / fully-populated variant.
Both replace Newport — IRIX loads the Express/`gfx` driver stack, not `newport`.

Sources:

- [Indy Technical Report ch.5 (XZ)](https://erikarn.github.io/sgi/indy/IndyReport/Indy_Report.ch5.pdf)
- [sgistuff Express overview](https://archive.irixnet.org/sgistuff/hardware/graphics/express.html)
- [Elan Technical Report](http://www.sgistuff.net/hardware/graphics/documents/ElanTR.html)

## GIO64 placement (IP24 / Indy)

Indy maps the graphics option in the **gfx** GIO64 slot:

| Space | Physical range | Size |
|-------|----------------|------|
| gfx   | `0x1F000000`–`0x1F3FFFFF` | 4 MB |

MAME models Newport the same way: the REX3 register window is at offset `+0x0F0000`
within the slot (`0x1F0F0000`–`0x1F0F1FFF`, 8 KB). See
`src/devices/bus/gio64/newport.cpp` (`mem_map`).

The XZ stub follows that layout: full 4 MB aperture owned by the device; CPU registers
decoded at `XZ_REG_BASE` (`0x1F0F0000`).

IRQ: GIO graphics interrupt line 0 (same as Newport) → IOC `GIO_EXP0` on Indy.

## HQ2 command engine (CPU-facing)

HQ2 is an ~80k-gate control ASIC implementing:

- Graphics input FIFO (CPU → command engine)
- Command sequencer / microcode store interface
- Geometry-engine delegation (4× GE7 on Indy XZ)
- Interrupt aggregation

Public documentation does **not** publish a complete HQ2 register spec comparable to
`newport.h`. IRIX and the Express kernel driver talk to HQ2 through a small MMIO window;
Linux/NetBSD/OpenBSD Newport headers (`include/video/newport.h`) document REX3, not HQ2.

### Stub register map (`src/xz.rs`)

Offsets are **research placeholders** grouped like typical SGI graphics engines until a
primary source (IRIX `gfx` driver disassembly or leaked HQ2 spec) confirms them:

| Offset | Name | R/W | Reset / read behaviour |
|--------|------|-----|------------------------|
| `0x0000` | `BOARD_ID` | R | `0x00030001` (GR3 + XZ class tag) |
| `0x0004` | `REVISION` | R | `0x00000021` (HQ2.1 placeholder) |
| `0x0008` | `STATUS` | R | FIFO empty, GE idle, RE idle |
| `0x000C` | `INTR_STATUS` | RW | write-1-to-clear (stored) |
| `0x0010` | `INTR_ENABLE` | RW | stored |
| `0x0018` | `FIFO_WRITE` | W | accepted, counted, not executed |
| `0x001C` | `FIFO_READ` | R | `0` |
| `0x0020` | `RESET` | W | soft-reset stub state |

Unlisted offsets: read `0`, writes logged and ignored.

## Related chips (not emulated)

| Chip | Role |
|------|------|
| GE7  | Geometry engine (4× SIMD) |
| RE3  | Raster engine |
| VC1  | Video timing / cursor (Express VC1, not Newport VC2) |
| XMAP5| Display mode / pixel mapping |
| ZRB1 | Z-buffer ASIC |
| VB2  | RAMDAC + video I/O connectors |

## Configuration

```toml
[machine]
profile = "indy_ip24"

[graphics]
board = "xz"
heads = 1
```

`board = "xz"` is **Indy-only**, disables the Newport compositor, and maps `src/xz.rs` at
the gfx slot. IRIX may probe the board but will not get a working framebuffer from this stub.

## References

- MAME: `src/devices/bus/gio64/newport.{h,cpp}` — GIO slot layout, REX3 window offset
- MAME: `src/mame/sgi/indy_indigo2.cpp` — IP24 GIO64 slot wiring
- Linux: `include/video/newport.h` (element_kernel mirror) — Newport/REX3 only; useful DCB naming
- SGI GIO64 bus: [datasheet mirror](https://erikarn.github.io/sgi/indy/datasheets/sgi_indy_gio64.pdf)
- IRIX GIO driver chapter: [DevDriver PG ch.18](https://techpubs.jurassic.nl/library/manuals/0000/007-0911-060/sgi_html/ch18.html)

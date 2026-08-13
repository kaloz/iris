//! Single source of truth for "does jitv2 have a real emitter for this raw
//! instruction word" — shared by `analyzer.rs` (deciding `Excluded` vs
//! `Sequential` at classify time) and `codegen.rs` (the actual emitter
//! lookup). Kept Cranelift-free like `analyzer.rs` itself: this module only
//! answers `bool`, it never touches `cranelift_*` types or emits IR.
//!
//! Before this module existed, `analyzer::classify()` had its own
//! independent opinion of what's compilable (anything not explicitly
//! excluded fell through to `Sequential`), separate from `codegen.rs`'s
//! `lookup_semantics`/`lookup_cp1_semantics`/`lookup_branch_or_jump`/
//! `lookup_regjump` tables — the two could (and did) drift: several
//! opcodes classify() called `Sequential` had no codegen emitter at all,
//! silently poisoning (declining, not just excluding-in-place) every
//! region that happened to walk through them (`compile_region`'s upfront
//! rejection loop declines the *whole* region if any visited instruction
//! lacks an emitter — see `rules/jitv2/unsupported-instructions.md` for the
//! inventory that motivated this fix). Routing `classify()` through
//! `has_emitter()` instead makes that drift structurally impossible: an
//! opcode without an emitter is `Excluded` (the analyzer never walks past
//! it as a head, so it's a clean region boundary instead of poisoning
//! everything downstream of it) until an emitter actually exists, at which
//! point it becomes `Sequential` automatically — no `analyzer.rs` edit
//! needed when a new emitter lands.
//!
//! Every `match` arm here must mirror the corresponding arm in
//! `codegen.rs`'s lookup functions exactly (same opcode/funct/rs
//! partitioning) — this is intentionally the *coverage* question only
//! ("is there an emitter"), not the emitter itself.

use crate::mips_isa::*;

/// True if `codegen.rs` has a real emitter for `raw` — a plain data-flow
/// instruction (`lookup_semantics`/`lookup_cp1_semantics`), a branch/jump
/// (`lookup_branch_or_jump`), or a register-indirect jump
/// (`lookup_regjump`). Callers that already know `raw` is a branch/jump/
/// regjump via `analyzer::classify`'s own dispatch (`Classify::Branch`/
/// `Jump`/`RegJump`) don't need to call this at all — those are handled by
/// construction, this is purely for the `Sequential`-or-`Excluded`
/// question on everything else.
pub fn has_emitter(raw: u32) -> bool {
    has_integer_emitter(raw) || has_cp1_emitter(raw)
}

/// Mirrors `codegen::lookup_semantics` exactly (plain integer/load/store
/// data-flow ops — everything reachable through `OP_SPECIAL`'s funct field
/// or a top-level opcode outside COP0/COP1/COP1X/COP2).
fn has_integer_emitter(raw: u32) -> bool {
    let op = (raw >> 26) & 0x3F;
    let funct = raw & 0x3F;
    match op {
        OP_SPECIAL => matches!(funct,
            FUNCT_ADD | FUNCT_ADDU | FUNCT_SUB | FUNCT_SUBU
            | FUNCT_AND | FUNCT_OR | FUNCT_XOR | FUNCT_NOR
            | FUNCT_SLT | FUNCT_SLTU
            | FUNCT_SLL | FUNCT_SRL | FUNCT_SRA | FUNCT_SLLV | FUNCT_SRLV | FUNCT_SRAV
            | FUNCT_MFHI | FUNCT_MTHI | FUNCT_MFLO | FUNCT_MTLO
            | FUNCT_MULT | FUNCT_MULTU | FUNCT_DIV | FUNCT_DIVU
            | FUNCT_DMULT | FUNCT_DMULTU | FUNCT_DDIV | FUNCT_DDIVU
            | FUNCT_DADD | FUNCT_DADDU | FUNCT_DSUB | FUNCT_DSUBU
            | FUNCT_DSLL | FUNCT_DSRL | FUNCT_DSRA
            | FUNCT_DSLL32 | FUNCT_DSRL32 | FUNCT_DSRA32
            | FUNCT_DSLLV | FUNCT_DSRLV | FUNCT_DSRAV
            | FUNCT_MOVZ | FUNCT_MOVN | FUNCT_MOVCI
            | FUNCT_TGE | FUNCT_TGEU | FUNCT_TLT | FUNCT_TLTU | FUNCT_TEQ | FUNCT_TNE
            | FUNCT_SYNC
        ),
        OP_REGIMM => matches!((raw >> 16) & 0x1F,
            RT_TGEI | RT_TGEIU | RT_TLTI | RT_TLTIU | RT_TEQI | RT_TNEI
        ),
        OP_ADDI | OP_ADDIU | OP_DADDI | OP_SLTI | OP_SLTIU | OP_ANDI | OP_ORI | OP_XORI | OP_LUI
        | OP_LB | OP_LBU | OP_LH | OP_LHU | OP_LW | OP_LWU | OP_LD
        | OP_LWL | OP_LWR | OP_LDL | OP_LDR
        | OP_SB | OP_SH | OP_SW | OP_SD
        | OP_SWL | OP_SWR | OP_SDL | OP_SDR
        | OP_PREF => true,
        _ => false,
    }
}

/// Mirrors `codegen::lookup_cp1_semantics` exactly — `OP_COP1`, plus
/// `OP_LWC1`/`OP_LDC1`/`OP_SWC1`/`OP_SDC1` (FPU load/store: not
/// `OP_COP1`-encoded, but routed through the same lookup table so the
/// region-wide CU1/FR guard's single trigger check catches them too — see
/// `codegen::lookup_cp1_semantics`'s doc comment). `OP_COP1X` has no
/// emitters at all yet, see `rules/jitv2/unsupported-instructions.md`.
fn has_cp1_emitter(raw: u32) -> bool {
    let op = (raw >> 26) & 0x3F;
    if matches!(op, OP_LWC1 | OP_LDC1 | OP_SWC1 | OP_SDC1) {
        return true;
    }
    if op != OP_COP1 {
        return false;
    }
    let rs = (raw >> 21) & 0x1F;
    let funct = raw & 0x3F;
    match rs {
        RS_S | RS_D => matches!(funct,
            FUNCT_FADD | FUNCT_FSUB | FUNCT_FMUL | FUNCT_FDIV | FUNCT_FSQRT
            | FUNCT_FABS | FUNCT_FNEG | FUNCT_FMOV | FUNCT_FMOVCF
            | FUNCT_FCVT_D | FUNCT_FCVT_S | FUNCT_FCVT_W | FUNCT_FCVT_L
            | FUNCT_FROUND_W | FUNCT_FTRUNC_W | FUNCT_FCEIL_W | FUNCT_FFLOOR_W
            | FUNCT_FROUND_L | FUNCT_FTRUNC_L | FUNCT_FCEIL_L | FUNCT_FFLOOR_L
        ) || (FUNCT_FC_F..=FUNCT_FC_NGT).contains(&funct),
        RS_W => matches!(funct, FUNCT_FCVT_S | FUNCT_FCVT_D),
        RS_L => matches!(funct, FUNCT_FCVT_S | FUNCT_FCVT_D),
        RS_MFC1 | RS_DMFC1 | RS_CFC1 | RS_MTC1 | RS_DMTC1 | RS_CTC1 => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r_type(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct
    }
    fn i_type(op: u32, rs: u32, rt: u32, imm: u16) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | imm as u32
    }

    #[test]
    fn plain_addu_has_emitter() {
        assert!(has_emitter(r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADDU)));
    }

    #[test]
    fn addiu_has_emitter() {
        assert!(has_emitter(i_type(OP_ADDIU, 1, 2, 5)));
    }

    #[test]
    fn movz_has_emitter() {
        assert!(has_emitter(r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_MOVZ)));
    }

    #[test]
    fn teq_has_emitter() {
        assert!(has_emitter(r_type(OP_SPECIAL, 1, 2, 0, 0, FUNCT_TEQ)));
    }

    #[test]
    fn tgei_has_emitter() {
        assert!(has_emitter(i_type(OP_REGIMM, 1, RT_TGEI, 5)));
    }

    #[test]
    fn prefx_has_no_emitter_yet() {
        // PREFX is COP1X-encoded and, unlike plain PREF, exec_prefx checks
        // STATUS_CU1 and raises cpu_unusable if it's clear — a genuine
        // COP0-adjacent side effect this codebase's hard-no on
        // privilege/COP0-touching instructions excludes from jitv2 for now.
        assert!(!has_emitter(r_type(OP_COP1X, 1, 2, 3, 4, FUNCT_PREFX)));
    }

    #[test]
    fn daddi_has_emitter() {
        assert!(has_emitter(i_type(OP_DADDI, 1, 2, 5)));
    }

    #[test]
    fn lwl_has_emitter() {
        assert!(has_emitter(i_type(OP_LWL, 1, 2, 0)));
    }

    #[test]
    fn swl_has_emitter() {
        assert!(has_emitter(i_type(OP_SWL, 1, 2, 0)));
    }

    #[test]
    fn cop1x_has_no_emitter_yet() {
        assert!(!has_emitter(r_type(OP_COP1X, 1, 2, 3, 4, FUNCT_MADD_S)));
    }

    #[test]
    fn cp1_add_s_has_emitter() {
        assert!(has_emitter(r_type(OP_COP1, RS_S, 2, 3, 0, FUNCT_FADD)));
    }

    #[test]
    fn cp1_movz_fmt_has_no_emitter_yet() {
        assert!(!has_emitter(r_type(OP_COP1, RS_S, 2, 3, 0, FUNCT_FMOVZ)));
    }

    #[test]
    fn mfc1_has_emitter() {
        assert!(has_emitter(r_type(OP_COP1, RS_MFC1, 2, 3, 0, 0)));
    }

    #[test]
    fn cop0_has_no_emitter() {
        // OP_COP0 is excluded at the analyzer level entirely (not routed
        // through has_emitter at all in practice), but has_emitter must
        // still correctly say "no" for it — nothing in either lookup table
        // matches OP_COP0.
        assert!(!has_emitter(r_type(OP_COP0, RS_MFC0, 2, 3, 0, 0)));
    }

    #[test]
    fn branch_and_regjump_report_false_here_by_design() {
        // BEQ/JR are handled by analyzer::classify's own Branch/RegJump
        // dispatch before has_emitter would ever be consulted for them —
        // has_emitter only answers the Sequential-vs-Excluded question for
        // everything else, so it correctly says "no" for these (codegen's
        // lookup_branch_or_jump/lookup_regjump are the real answer for
        // that different question).
        assert!(!has_emitter(i_type(OP_BEQ, 1, 2, 0)));
        assert!(!has_emitter(r_type(OP_SPECIAL, 1, 0, 0, 0, FUNCT_JR)));
    }
}

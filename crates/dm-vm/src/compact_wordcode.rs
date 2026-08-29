//! Compact numeric dispatch metadata for fully linked VM modules.
//!
//! The first format is deliberately an execution sidecar rather than a second
//! semantic bytecode. Every logical instruction owns exactly one 32-bit word:
//! the upper byte is a stable selector and the lower 24 bits are its common
//! numeric operand. Instructions whose complete operand contract is not yet
//! represented use selector zero and continue through the reference
//! [`Instruction`] path.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::{Instruction, Module, ProcedureId};

const MAGIC: &[u8; 8] = b"DM64CWC\0";
const VERSION: u16 = 1;
const HEADER_BYTES: usize = 24;
const PROCEDURE_BYTES: usize = 24;
const MAX_STRINGS: usize = 16_777_216;
const MAX_PROCEDURES: usize = 1_000_000;
const MAX_WORDS: usize = 500_000_000;
const MAX_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_TOTAL_STRING_BYTES: usize = 1024 * 1024 * 1024;
const PAYLOAD_MASK: u32 = 0x00ff_ffff;

const OP_ESCAPE: u8 = 0;
const OP_PUSH_NULL: u8 = 1;
const OP_ADDRESS_LOCAL: u8 = 2;
const OP_LOAD_LOCAL_RAW: u8 = 3;
const OP_LOAD_LOCAL: u8 = 4;
const OP_STORE_LOCAL: u8 = 5;
const OP_INITIALIZE_STATIC_LOCAL: u8 = 6;
const OP_LOAD_SRC: u8 = 7;
const OP_STORE_SRC: u8 = 8;
const OP_LOAD_USR: u8 = 9;
const OP_STORE_USR: u8 = 10;
const OP_LOAD_CALLER: u8 = 11;
const OP_LOAD_RESULT: u8 = 12;
const OP_STORE_RESULT: u8 = 13;
const OP_POP: u8 = 14;
const OP_DUPLICATE: u8 = 15;
const OP_LOAD_FIELD: u8 = 16;
const OP_LOAD_DECLARED_FIELD: u8 = 17;
const OP_STORE_FIELD: u8 = 18;
const OP_STORE_FIELD_KEEP: u8 = 19;
const OP_LOAD_GLOBAL: u8 = 20;
const OP_LOAD_INITIAL_GLOBAL: u8 = 21;
const OP_STORE_GLOBAL: u8 = 22;
const OP_INITIAL_FIELD: u8 = 23;
const OP_LOGICAL_OR_GLOBAL: u8 = 24;
const OP_LOGICAL_OR_FIELD: u8 = 25;
const OP_TYPE_INSTANCES: u8 = 26;
const OP_ITERATION_TYPE_FILTER: u8 = 27;
const OP_MAKE_LIST: u8 = 28;
const OP_MAKE_ARRAY: u8 = 29;
const OP_MAKE_ARGS: u8 = 30;
const OP_INDEX_LIST: u8 = 31;
const OP_SET_LIST_INDEX: u8 = 32;
const OP_SET_LIST_INDEX_KEEP: u8 = 33;
const OP_LIST_LENGTH: u8 = 34;
const OP_PREPARE_ITERATION: u8 = 35;
const OP_INDEX_LOCAL_LIST: u8 = 36;
const OP_LIST_LENGTH_LOCAL: u8 = 37;
const OP_LOGICAL_OR_LOCAL: u8 = 38;
const OP_LOGICAL_OR_INDEX: u8 = 39;
const OP_PREPARE_RHS_INDEX: u8 = 40;
const OP_LOAD_DYNAMIC_FIELD: u8 = 41;
const OP_STORE_DYNAMIC_FIELD: u8 = 42;
const OP_LOAD_GLOBAL_VARS: u8 = 43;
const OP_LOAD_DATUM_VARS: u8 = 44;
const OP_ADD: u8 = 45;
const OP_SUBTRACT: u8 = 46;
const OP_MULTIPLY: u8 = 47;
const OP_POWER: u8 = 48;
const OP_DIVIDE: u8 = 49;
const OP_REMAINDER: u8 = 50;
const OP_FRACTIONAL_REMAINDER: u8 = 51;
const OP_BIT_AND: u8 = 52;
const OP_BIT_OR: u8 = 53;
const OP_BIT_XOR: u8 = 54;
const OP_SHIFT_LEFT: u8 = 55;
const OP_SHIFT_RIGHT: u8 = 56;
const OP_BIT_NOT: u8 = 57;
const OP_NEGATE: u8 = 58;
const OP_NOT: u8 = 59;
const OP_EQUAL: u8 = 60;
const OP_NOT_EQUAL: u8 = 61;
const OP_EQUIVALENT: u8 = 62;
const OP_NOT_EQUIVALENT: u8 = 63;
const OP_COMPARE: u8 = 64;
const OP_CONTAINS: u8 = 65;
const OP_LESS: u8 = 66;
const OP_LESS_EQUAL: u8 = 67;
const OP_GREATER: u8 = 68;
const OP_GREATER_EQUAL: u8 = 69;
const OP_AND: u8 = 70;
const OP_OR: u8 = 71;
const OP_JUMP_IF_NULL: u8 = 72;
const OP_JUMP_IF_FALSE: u8 = 73;
const OP_JUMP: u8 = 74;
const OP_CALL: u8 = 75;
const OP_RETURN: u8 = 76;
const OP_SPAWN: u8 = 77;
const OP_SLEEP: u8 = 78;
const OP_END_TRY: u8 = 79;
const OP_THROW: u8 = 80;
const OP_CRASH: u8 = 81;
const OP_OUTPUT: u8 = 82;
const OP_INPUT: u8 = 83;
const OP_LENGTH: u8 = 84;
const OP_REF: u8 = 85;
const OP_PROB: u8 = 86;

/// A compact wordcode codec or validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactWordcodeError {
    message: String,
}

impl CompactWordcodeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CompactWordcodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for CompactWordcodeError {}

/// Fixed metadata for one procedure's range in a compact wordcode image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactProcedureRecord {
    path_id: u32,
    word_offset: u32,
    instruction_count: u32,
    local_count: u32,
    parameter_count: u32,
    flags: u32,
}

impl CompactProcedureRecord {
    /// Stable string-pool identity of the canonical procedure path.
    #[must_use]
    pub const fn path_id(self) -> u32 {
        self.path_id
    }

    /// First word in the image's shared word array.
    #[must_use]
    pub const fn word_offset(self) -> u32 {
        self.word_offset
    }

    /// Number of logical instructions and dispatch words.
    #[must_use]
    pub const fn instruction_count(self) -> u32 {
        self.instruction_count
    }

    /// Number of local slots required by this procedure.
    #[must_use]
    pub const fn local_count(self) -> u32 {
        self.local_count
    }

    /// Declared positional parameter count.
    #[must_use]
    pub const fn parameter_count(self) -> u32 {
        self.parameter_count
    }

    /// Stable procedure flags. Bit zero is `waitfor`.
    #[must_use]
    pub const fn flags(self) -> u32 {
        self.flags
    }
}

/// Immutable ID-linked numeric dispatch image for one fully eager module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactWordcodeImage {
    strings: Vec<String>,
    procedures: Vec<CompactProcedureRecord>,
    words: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompactFastInstruction {
    PushNull,
    LoadSrc,
    StoreSrc,
    LoadUsr,
    StoreUsr,
    LoadResult,
    StoreResult,
    Pop,
    Duplicate,
}

impl CompactWordcodeImage {
    /// Builds an image from a fully eager linked module.
    ///
    /// # Errors
    ///
    /// Returns an error for deferred bodies or values outside the bounded
    /// 24-bit dispatch operand and 32-bit directory ranges.
    pub fn build(module: &Module) -> Result<Self, CompactWordcodeError> {
        if module.deferred_procedure_count() != 0 {
            return Err(CompactWordcodeError::new(
                "compact wordcode requires a fully eager module",
            ));
        }
        if module.procedures.len() > MAX_PROCEDURES {
            return Err(CompactWordcodeError::new(
                "compact wordcode procedure count exceeds limit",
            ));
        }

        let mut strings = Vec::new();
        let mut string_ids = BTreeMap::new();
        let mut procedures = Vec::with_capacity(module.procedures.len());
        let total_words = module
            .procedures
            .iter()
            .try_fold(0usize, |total, program| {
                total
                    .checked_add(program.instructions.len())
                    .ok_or_else(|| CompactWordcodeError::new("compact word count overflow"))
            })?;
        if total_words > MAX_WORDS {
            return Err(CompactWordcodeError::new(
                "compact wordcode instruction count exceeds limit",
            ));
        }
        let mut words = Vec::with_capacity(total_words);

        for (index, program) in module.procedures.iter().enumerate() {
            let path = module
                .paths
                .get(index)
                .ok_or_else(|| CompactWordcodeError::new("missing procedure path"))?;
            let path_id = intern_string(path, &mut strings, &mut string_ids)?;
            let word_offset = u32::try_from(words.len())
                .map_err(|_| CompactWordcodeError::new("compact word offset exceeds u32"))?;
            let instruction_count = u32::try_from(program.instructions.len()).map_err(|_| {
                CompactWordcodeError::new("procedure instruction count exceeds u32")
            })?;
            let local_count = u32::try_from(program.local_count)
                .map_err(|_| CompactWordcodeError::new("local count exceeds u32"))?;
            let parameter_count = u32::try_from(program.parameter_count)
                .map_err(|_| CompactWordcodeError::new("parameter count exceeds u32"))?;
            procedures.push(CompactProcedureRecord {
                path_id,
                word_offset,
                instruction_count,
                local_count,
                parameter_count,
                flags: u32::from(program.wait_for),
            });
            for instruction in &program.instructions {
                words.push(selector_word(instruction, &mut strings, &mut string_ids)?);
            }
        }

        let image = Self {
            strings,
            procedures,
            words,
        };
        image.validate_against(module)?;
        Ok(image)
    }

    /// Encodes this image with explicit little-endian fields.
    ///
    /// # Errors
    ///
    /// Returns an error if an in-memory image violates codec bounds.
    pub fn encode(&self) -> Result<Vec<u8>, CompactWordcodeError> {
        validate_image(self)?;
        let string_bytes = self.strings.iter().try_fold(0usize, |total, value| {
            total
                .checked_add(4)
                .and_then(|next| next.checked_add(value.len()))
                .ok_or_else(|| CompactWordcodeError::new("string table length overflow"))
        })?;
        let procedure_bytes = self
            .procedures
            .len()
            .checked_mul(PROCEDURE_BYTES)
            .ok_or_else(|| CompactWordcodeError::new("procedure table length overflow"))?;
        let word_bytes = self
            .words
            .len()
            .checked_mul(4)
            .ok_or_else(|| CompactWordcodeError::new("word table length overflow"))?;
        let capacity = HEADER_BYTES
            .checked_add(string_bytes)
            .and_then(|length| length.checked_add(procedure_bytes))
            .and_then(|length| length.checked_add(word_bytes))
            .ok_or_else(|| CompactWordcodeError::new("compact image length overflow"))?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(MAGIC);
        write_u16(&mut bytes, VERSION);
        write_u16(&mut bytes, 0);
        write_u32(&mut bytes, self.strings.len() as u32);
        write_u32(&mut bytes, self.procedures.len() as u32);
        write_u32(&mut bytes, self.words.len() as u32);
        for value in &self.strings {
            write_u32(&mut bytes, value.len() as u32);
            bytes.extend_from_slice(value.as_bytes());
        }
        for record in &self.procedures {
            write_u32(&mut bytes, record.path_id);
            write_u32(&mut bytes, record.word_offset);
            write_u32(&mut bytes, record.instruction_count);
            write_u32(&mut bytes, record.local_count);
            write_u32(&mut bytes, record.parameter_count);
            write_u32(&mut bytes, record.flags);
        }
        for word in &self.words {
            write_u32(&mut bytes, *word);
        }
        debug_assert_eq!(bytes.len(), capacity);
        Ok(bytes)
    }

    /// Decodes a bounded compact wordcode image.
    ///
    /// # Errors
    ///
    /// Returns an error for corrupt, truncated, oversized, or unsupported
    /// input. Use [`Self::validate_against`] before attaching it to a module.
    pub fn decode(bytes: &[u8]) -> Result<Self, CompactWordcodeError> {
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len(), "header")? != MAGIC {
            return Err(CompactWordcodeError::new("invalid compact wordcode header"));
        }
        let version = reader.u16("version")?;
        if version != VERSION {
            return Err(CompactWordcodeError::new(format!(
                "unsupported compact wordcode version {version}",
            )));
        }
        if reader.u16("flags")? != 0 {
            return Err(CompactWordcodeError::new("unknown compact wordcode flags"));
        }
        let string_count = reader.count(MAX_STRINGS, "string count")?;
        let procedure_count = reader.count(MAX_PROCEDURES, "procedure count")?;
        let word_count = reader.count(MAX_WORDS, "word count")?;
        let mut strings = Vec::with_capacity(string_count);
        let mut total_string_bytes = 0usize;
        for _ in 0..string_count {
            let length = reader.count(MAX_STRING_BYTES, "string length")?;
            total_string_bytes = total_string_bytes
                .checked_add(length)
                .ok_or_else(|| CompactWordcodeError::new("string byte count overflow"))?;
            if total_string_bytes > MAX_TOTAL_STRING_BYTES {
                return Err(CompactWordcodeError::new(
                    "compact string table exceeds byte limit",
                ));
            }
            let value = std::str::from_utf8(reader.take(length, "string bytes")?)
                .map_err(|_| CompactWordcodeError::new("compact string is not UTF-8"))?;
            strings.push(value.to_owned());
        }
        let mut procedures = Vec::with_capacity(procedure_count);
        for _ in 0..procedure_count {
            procedures.push(CompactProcedureRecord {
                path_id: reader.u32("path id")?,
                word_offset: reader.u32("word offset")?,
                instruction_count: reader.u32("instruction count")?,
                local_count: reader.u32("local count")?,
                parameter_count: reader.u32("parameter count")?,
                flags: reader.u32("procedure flags")?,
            });
        }
        let mut words = Vec::with_capacity(word_count);
        for _ in 0..word_count {
            words.push(reader.u32("dispatch word")?);
        }
        if !reader.is_empty() {
            return Err(CompactWordcodeError::new(
                "trailing bytes after compact wordcode image",
            ));
        }
        let image = Self {
            strings,
            procedures,
            words,
        };
        validate_image(&image)?;
        Ok(image)
    }

    /// Validates directory identities and every dispatch selector against the
    /// authoritative rich module.
    ///
    /// # Errors
    ///
    /// Returns an error on any procedure, metadata, path, range, or selector
    /// mismatch.
    pub fn validate_against(&self, module: &Module) -> Result<(), CompactWordcodeError> {
        validate_image(self)?;
        if module.deferred_procedure_count() != 0 {
            return Err(CompactWordcodeError::new(
                "cannot validate compact wordcode against a deferred module",
            ));
        }
        if self.procedures.len() != module.procedures.len() {
            return Err(CompactWordcodeError::new(
                "compact and rich procedure counts differ",
            ));
        }
        let ids = self
            .strings
            .iter()
            .enumerate()
            .map(|(index, value)| (value.as_str(), index as u32))
            .collect::<BTreeMap<_, _>>();
        for (index, (record, program)) in self.procedures.iter().zip(&module.procedures).enumerate()
        {
            let path = module
                .paths
                .get(index)
                .ok_or_else(|| CompactWordcodeError::new("missing rich procedure path"))?;
            let compact_path = self
                .strings
                .get(record.path_id as usize)
                .ok_or_else(|| CompactWordcodeError::new("compact path id is invalid"))?;
            if compact_path != path {
                return Err(CompactWordcodeError::new(format!(
                    "compact procedure {index} path differs",
                )));
            }
            if record.instruction_count as usize != program.instructions.len()
                || record.local_count as usize != program.local_count
                || record.parameter_count as usize != program.parameter_count
                || record.flags != u32::from(program.wait_for)
            {
                return Err(CompactWordcodeError::new(format!(
                    "compact procedure {index} metadata differs",
                )));
            }
            let start = record.word_offset as usize;
            let end = start
                .checked_add(record.instruction_count as usize)
                .ok_or_else(|| CompactWordcodeError::new("compact procedure range overflow"))?;
            let words = self
                .words
                .get(start..end)
                .ok_or_else(|| CompactWordcodeError::new("compact procedure range is invalid"))?;
            for (pc, (instruction, actual)) in program.instructions.iter().zip(words).enumerate() {
                let expected = selector_word_from_ids(instruction, &ids)?;
                if *actual != expected {
                    return Err(CompactWordcodeError::new(format!(
                        "compact procedure {index} selector differs at instruction {pc}",
                    )));
                }
            }
        }
        Ok(())
    }

    /// Returns one fixed procedure directory record.
    #[must_use]
    pub fn procedure(&self, procedure: ProcedureId) -> Option<CompactProcedureRecord> {
        self.procedures.get(procedure.index()).copied()
    }

    /// Returns one 32-bit selector word for a logical instruction.
    #[must_use]
    pub fn word(&self, procedure: ProcedureId, instruction: usize) -> Option<u32> {
        let record = self.procedure(procedure)?;
        if instruction >= record.instruction_count as usize {
            return None;
        }
        let index = (record.word_offset as usize).checked_add(instruction)?;
        self.words.get(index).copied()
    }

    /// Returns the stable selector byte stored in a dispatch word.
    #[must_use]
    pub const fn selector(word: u32) -> u8 {
        (word >> 24) as u8
    }

    /// Returns the unsigned 24-bit operand stored in a dispatch word.
    #[must_use]
    pub const fn operand(word: u32) -> u32 {
        word & PAYLOAD_MASK
    }

    #[inline(always)]
    pub(crate) const fn fast_instruction(word: u32) -> Option<CompactFastInstruction> {
        match Self::selector(word) {
            OP_PUSH_NULL => Some(CompactFastInstruction::PushNull),
            OP_LOAD_SRC => Some(CompactFastInstruction::LoadSrc),
            OP_STORE_SRC => Some(CompactFastInstruction::StoreSrc),
            OP_LOAD_USR => Some(CompactFastInstruction::LoadUsr),
            OP_STORE_USR => Some(CompactFastInstruction::StoreUsr),
            OP_LOAD_RESULT => Some(CompactFastInstruction::LoadResult),
            OP_STORE_RESULT => Some(CompactFastInstruction::StoreResult),
            OP_POP => Some(CompactFastInstruction::Pop),
            OP_DUPLICATE => Some(CompactFastInstruction::Duplicate),
            _ => None,
        }
    }

    /// Returns the number of interned strings.
    #[must_use]
    pub const fn string_count(&self) -> usize {
        self.strings.len()
    }

    /// Returns the number of procedure records.
    #[must_use]
    pub const fn procedure_count(&self) -> usize {
        self.procedures.len()
    }

    /// Returns the number of 32-bit dispatch words.
    #[must_use]
    pub const fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Returns how many words have a specialized numeric selector.
    #[must_use]
    pub fn specialized_word_count(&self) -> usize {
        self.words
            .iter()
            .filter(|word| Self::selector(**word) != OP_ESCAPE)
            .count()
    }
}

fn validate_image(image: &CompactWordcodeImage) -> Result<(), CompactWordcodeError> {
    if image.strings.len() > MAX_STRINGS
        || image.procedures.len() > MAX_PROCEDURES
        || image.words.len() > MAX_WORDS
    {
        return Err(CompactWordcodeError::new(
            "compact wordcode image exceeds item limits",
        ));
    }
    let mut total_string_bytes = 0usize;
    let mut unique = BTreeMap::new();
    for (index, value) in image.strings.iter().enumerate() {
        if value.len() > MAX_STRING_BYTES {
            return Err(CompactWordcodeError::new(
                "compact string exceeds byte limit",
            ));
        }
        total_string_bytes = total_string_bytes
            .checked_add(value.len())
            .ok_or_else(|| CompactWordcodeError::new("string byte count overflow"))?;
        if total_string_bytes > MAX_TOTAL_STRING_BYTES {
            return Err(CompactWordcodeError::new(
                "compact string table exceeds byte limit",
            ));
        }
        if unique.insert(value, index).is_some() {
            return Err(CompactWordcodeError::new(
                "compact string table contains a duplicate",
            ));
        }
    }
    let mut expected_offset = 0usize;
    for record in &image.procedures {
        if record.path_id as usize >= image.strings.len() {
            return Err(CompactWordcodeError::new("compact path id is invalid"));
        }
        if record.flags & !1 != 0 {
            return Err(CompactWordcodeError::new(
                "compact procedure has unknown flags",
            ));
        }
        if record.parameter_count > record.local_count {
            return Err(CompactWordcodeError::new(
                "compact parameter count exceeds local count",
            ));
        }
        if record.word_offset as usize != expected_offset {
            return Err(CompactWordcodeError::new(
                "compact procedure ranges are not contiguous",
            ));
        }
        expected_offset = expected_offset
            .checked_add(record.instruction_count as usize)
            .ok_or_else(|| CompactWordcodeError::new("compact procedure range overflow"))?;
        if expected_offset > image.words.len() {
            return Err(CompactWordcodeError::new(
                "compact procedure range exceeds word table",
            ));
        }
    }
    if expected_offset != image.words.len() {
        return Err(CompactWordcodeError::new(
            "compact word table has unclaimed words",
        ));
    }
    Ok(())
}

fn intern_string(
    value: &str,
    strings: &mut Vec<String>,
    ids: &mut BTreeMap<String, u32>,
) -> Result<u32, CompactWordcodeError> {
    if let Some(id) = ids.get(value) {
        return Ok(*id);
    }
    if value.len() > MAX_STRING_BYTES || strings.len() >= MAX_STRINGS {
        return Err(CompactWordcodeError::new(
            "compact string pool exceeds limit",
        ));
    }
    let id = u32::try_from(strings.len())
        .map_err(|_| CompactWordcodeError::new("compact string id exceeds u32"))?;
    strings.push(value.to_owned());
    ids.insert(value.to_owned(), id);
    Ok(id)
}

fn selector_word(
    instruction: &Instruction,
    strings: &mut Vec<String>,
    ids: &mut BTreeMap<String, u32>,
) -> Result<u32, CompactWordcodeError> {
    let mut intern = |value: &str| intern_string(value, strings, ids);
    selector_word_with(instruction, &mut intern)
}

fn selector_word_from_ids(
    instruction: &Instruction,
    ids: &BTreeMap<&str, u32>,
) -> Result<u32, CompactWordcodeError> {
    let mut lookup = |value: &str| {
        ids.get(value)
            .copied()
            .ok_or_else(|| CompactWordcodeError::new("compact operand string is missing"))
    };
    selector_word_with(instruction, &mut lookup)
}

#[allow(clippy::too_many_lines)]
fn selector_word_with(
    instruction: &Instruction,
    string_id: &mut impl FnMut(&str) -> Result<u32, CompactWordcodeError>,
) -> Result<u32, CompactWordcodeError> {
    let unit = |opcode| Ok(pack(opcode, 0));
    let numeric = |opcode, value: usize| pack_checked(opcode, value);
    let mut field = |opcode, value: &str| pack_checked_u32(opcode, string_id(value)?);
    match instruction {
        Instruction::PushNull => unit(OP_PUSH_NULL),
        Instruction::AddressLocal(slot) => numeric(OP_ADDRESS_LOCAL, usize::from(*slot)),
        Instruction::LoadLocalRaw(slot) => numeric(OP_LOAD_LOCAL_RAW, usize::from(*slot)),
        Instruction::LoadLocal(slot) => numeric(OP_LOAD_LOCAL, usize::from(*slot)),
        Instruction::StoreLocal(slot) => numeric(OP_STORE_LOCAL, usize::from(*slot)),
        Instruction::InitializeStaticLocal(slot) => {
            numeric(OP_INITIALIZE_STATIC_LOCAL, usize::from(*slot))
        }
        Instruction::LoadSrc => unit(OP_LOAD_SRC),
        Instruction::StoreSrc => unit(OP_STORE_SRC),
        Instruction::LoadUsr => unit(OP_LOAD_USR),
        Instruction::StoreUsr => unit(OP_STORE_USR),
        Instruction::LoadCaller => unit(OP_LOAD_CALLER),
        Instruction::LoadResult => unit(OP_LOAD_RESULT),
        Instruction::StoreResult => unit(OP_STORE_RESULT),
        Instruction::Pop => unit(OP_POP),
        Instruction::Duplicate => unit(OP_DUPLICATE),
        Instruction::LoadField(name) => field(OP_LOAD_FIELD, name.as_str()),
        Instruction::LoadDeclaredField(name) => field(OP_LOAD_DECLARED_FIELD, name.as_str()),
        Instruction::StoreField(name) => field(OP_STORE_FIELD, name.as_str()),
        Instruction::StoreFieldKeep(name) => field(OP_STORE_FIELD_KEEP, name.as_str()),
        Instruction::LoadGlobal(name) => field(OP_LOAD_GLOBAL, name.as_str()),
        Instruction::LoadInitialGlobal(name) => field(OP_LOAD_INITIAL_GLOBAL, name.as_str()),
        Instruction::StoreGlobal(name) => field(OP_STORE_GLOBAL, name.as_str()),
        Instruction::InitialField(name) => field(OP_INITIAL_FIELD, name.as_str()),
        Instruction::LogicalOrEmptyListGlobal(name) => field(OP_LOGICAL_OR_GLOBAL, name.as_str()),
        Instruction::LogicalOrEmptyListField(name) => field(OP_LOGICAL_OR_FIELD, name.as_str()),
        Instruction::TypeInstances(path) => field(OP_TYPE_INSTANCES, path.as_str()),
        Instruction::IterationTypeFilter(path) => field(OP_ITERATION_TYPE_FILTER, path.as_str()),
        Instruction::MakeList(length) => numeric(OP_MAKE_LIST, usize::from(*length)),
        Instruction::MakeArray(dimensions) => numeric(OP_MAKE_ARRAY, usize::from(*dimensions)),
        Instruction::MakeArgs => unit(OP_MAKE_ARGS),
        Instruction::IndexList => unit(OP_INDEX_LIST),
        Instruction::SetListIndex => unit(OP_SET_LIST_INDEX),
        Instruction::SetListIndexKeep => unit(OP_SET_LIST_INDEX_KEEP),
        Instruction::ListLength => unit(OP_LIST_LENGTH),
        Instruction::PrepareIteration => unit(OP_PREPARE_ITERATION),
        Instruction::IndexLocalList(slot) => numeric(OP_INDEX_LOCAL_LIST, usize::from(*slot)),
        Instruction::ListLengthLocal(slot) => numeric(OP_LIST_LENGTH_LOCAL, usize::from(*slot)),
        Instruction::LogicalOrEmptyListLocal(slot) => {
            numeric(OP_LOGICAL_OR_LOCAL, usize::from(*slot))
        }
        Instruction::LogicalOrEmptyListIndex => unit(OP_LOGICAL_OR_INDEX),
        Instruction::PrepareRhsFirstIndexAssignment => unit(OP_PREPARE_RHS_INDEX),
        Instruction::LoadDynamicField => unit(OP_LOAD_DYNAMIC_FIELD),
        Instruction::StoreDynamicField => unit(OP_STORE_DYNAMIC_FIELD),
        Instruction::LoadGlobalVars => unit(OP_LOAD_GLOBAL_VARS),
        Instruction::LoadDatumVars => unit(OP_LOAD_DATUM_VARS),
        Instruction::Add => unit(OP_ADD),
        Instruction::Subtract => unit(OP_SUBTRACT),
        Instruction::Multiply => unit(OP_MULTIPLY),
        Instruction::Power => unit(OP_POWER),
        Instruction::Divide => unit(OP_DIVIDE),
        Instruction::Remainder => unit(OP_REMAINDER),
        Instruction::FractionalRemainder => unit(OP_FRACTIONAL_REMAINDER),
        Instruction::BitAnd => unit(OP_BIT_AND),
        Instruction::BitOr => unit(OP_BIT_OR),
        Instruction::BitXor => unit(OP_BIT_XOR),
        Instruction::ShiftLeft => unit(OP_SHIFT_LEFT),
        Instruction::ShiftRight => unit(OP_SHIFT_RIGHT),
        Instruction::BitNot => unit(OP_BIT_NOT),
        Instruction::Negate => unit(OP_NEGATE),
        Instruction::Not => unit(OP_NOT),
        Instruction::Equal => unit(OP_EQUAL),
        Instruction::NotEqual => unit(OP_NOT_EQUAL),
        Instruction::Equivalent => unit(OP_EQUIVALENT),
        Instruction::NotEquivalent => unit(OP_NOT_EQUIVALENT),
        Instruction::Compare => unit(OP_COMPARE),
        Instruction::Contains => unit(OP_CONTAINS),
        Instruction::Less => unit(OP_LESS),
        Instruction::LessEqual => unit(OP_LESS_EQUAL),
        Instruction::Greater => unit(OP_GREATER),
        Instruction::GreaterEqual => unit(OP_GREATER_EQUAL),
        Instruction::And => unit(OP_AND),
        Instruction::Or => unit(OP_OR),
        Instruction::JumpIfNull(target) => numeric(OP_JUMP_IF_NULL, *target),
        Instruction::JumpIfFalse(target) => numeric(OP_JUMP_IF_FALSE, *target),
        Instruction::Jump(target) => numeric(OP_JUMP, *target),
        Instruction::Call { procedure, .. } => numeric(OP_CALL, procedure.index()),
        Instruction::Return => unit(OP_RETURN),
        Instruction::Spawn { entry } => numeric(OP_SPAWN, *entry),
        Instruction::Sleep => unit(OP_SLEEP),
        Instruction::EndTry => unit(OP_END_TRY),
        Instruction::Throw => unit(OP_THROW),
        Instruction::Crash => unit(OP_CRASH),
        Instruction::Output => unit(OP_OUTPUT),
        Instruction::Input => unit(OP_INPUT),
        Instruction::Length => unit(OP_LENGTH),
        Instruction::Ref => unit(OP_REF),
        Instruction::Prob => unit(OP_PROB),
        _ => unit(OP_ESCAPE),
    }
}

const fn pack(opcode: u8, operand: u32) -> u32 {
    (opcode as u32) << 24 | (operand & PAYLOAD_MASK)
}

fn pack_checked(opcode: u8, operand: usize) -> Result<u32, CompactWordcodeError> {
    let operand = u32::try_from(operand)
        .map_err(|_| CompactWordcodeError::new("compact operand exceeds u32"))?;
    pack_checked_u32(opcode, operand)
}

fn pack_checked_u32(opcode: u8, operand: u32) -> Result<u32, CompactWordcodeError> {
    if operand > PAYLOAD_MASK {
        Ok(pack(OP_ESCAPE, 0))
    } else {
        Ok(pack(opcode, operand))
    }
}

fn write_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize, label: &str) -> Result<&'a [u8], CompactWordcodeError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| CompactWordcodeError::new(format!("{label} length overflow")))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| CompactWordcodeError::new(format!("truncated {label}")))?;
        self.position = end;
        Ok(value)
    }

    fn u16(&mut self, label: &str) -> Result<u16, CompactWordcodeError> {
        Ok(u16::from_le_bytes(
            self.take(2, label)?.try_into().expect("exact width"),
        ))
    }

    fn u32(&mut self, label: &str) -> Result<u32, CompactWordcodeError> {
        Ok(u32::from_le_bytes(
            self.take(4, label)?.try_into().expect("exact width"),
        ))
    }

    fn count(&mut self, maximum: usize, label: &str) -> Result<usize, CompactWordcodeError> {
        let count = self.u32(label)? as usize;
        if count > maximum {
            return Err(CompactWordcodeError::new(format!("{label} exceeds limit")));
        }
        Ok(count)
    }

    const fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{CompactWordcodeImage, OP_ESCAPE};
    use std::collections::HashMap;
    use std::sync::Arc;

    use dm_core::{DmNumberBits, SourceSpan};

    use crate::{Instruction, Module, ProcedureId, Program, next_module_identity};

    fn module_with(instructions: Vec<Instruction>) -> Module {
        let instruction_count = instructions.len();
        Module {
            identity: next_module_identity(),
            procedures: vec![Arc::new(Program {
                wait_for: true,
                parameter_count: 1,
                parameter_names: vec!["a".to_owned()],
                verb_parameter_types: vec![crate::VerbParameterType::Unsupported],
                verb_name: None,
                local_count: 2,
                instructions,
                source_spans: vec![SourceSpan::new(0, 1); instruction_count],
            })],
            paths: vec!["/proc/main".to_owned()],
            names: HashMap::from([("/proc/main".to_owned(), ProcedureId(0))]),
            dynamic_names: HashMap::from([("/proc/main".to_owned(), ProcedureId(0))]),
            deferred: Arc::new(HashMap::new()),
            procedure_types: Vec::new(),
            initializer_call_names: None,
            compact_wordcode: Default::default(),
            semantic_digests: Default::default(),
        }
    }

    #[test]
    fn compact_wordcode_round_trips_and_validates() {
        let module = module_with(vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Add,
            Instruction::StoreLocal(1),
            Instruction::LoadLocal(1),
            Instruction::Return,
        ]);
        let image = CompactWordcodeImage::build(&module).expect("image should build");
        let bytes = image.encode().expect("image should encode");
        let decoded = CompactWordcodeImage::decode(&bytes).expect("image should decode");
        decoded
            .validate_against(&module)
            .expect("decoded image should match module");
        assert_eq!(decoded, image);
        assert_eq!(decoded.procedure_count(), 1);
        assert_eq!(
            decoded.word_count(),
            module
                .procedure(module.procedure_id("/proc/main").unwrap())
                .unwrap()
                .instructions
                .len()
        );
        assert!(decoded.specialized_word_count() > 0);
    }

    #[test]
    fn complex_instructions_retain_an_explicit_escape() {
        let module = module_with(vec![
            Instruction::PushText(Arc::from("hello")),
            Instruction::Return,
        ]);
        let image = CompactWordcodeImage::build(&module).expect("image should build");
        let procedure = module.procedure_id("/proc/main").unwrap();
        let rich = module.procedure(procedure).unwrap();
        let push_text = rich
            .instructions
            .iter()
            .position(|instruction| matches!(instruction, Instruction::PushText(_)))
            .expect("fixture should contain PushText");
        let word = image.word(procedure, push_text).expect("word should exist");
        assert_eq!(CompactWordcodeImage::selector(word), OP_ESCAPE);
    }

    #[test]
    fn corrupt_or_mismatched_images_are_rejected() {
        let module = module_with(vec![Instruction::PushNull, Instruction::Return]);
        let image = CompactWordcodeImage::build(&module).expect("image should build");
        let mut bytes = image.encode().expect("image should encode");
        bytes.pop();
        assert!(CompactWordcodeImage::decode(&bytes).is_err());

        let other = module_with(vec![
            Instruction::PushNumber(DmNumberBits::from_f32(3.0)),
            Instruction::Return,
        ]);
        assert!(image.validate_against(&other).is_err());
    }

    #[test]
    fn attached_compact_execution_matches_reference_result() {
        let module = module_with(vec![
            Instruction::PushNumber(DmNumberBits::from_f32(7.0)),
            Instruction::StoreResult,
            Instruction::LoadResult,
            Instruction::Duplicate,
            Instruction::Pop,
            Instruction::Return,
        ]);
        let entry = module.procedure_id("/proc/main").unwrap();
        let reference = crate::execute_module(&module, entry, &[])
            .expect("reference execution should complete");
        let mut compact = module.clone();
        compact
            .install_compact_wordcode()
            .expect("compact wordcode should install");
        let candidate =
            crate::execute_module(&compact, entry, &[]).expect("compact execution should complete");
        assert_eq!(candidate, reference);
        assert_eq!(candidate, crate::Value::number(7.0));
    }

    #[test]
    #[ignore = "bounded compact dispatch microbenchmark; run explicitly"]
    fn compact_dispatch_microbenchmark() {
        use std::time::Instant;

        let mut instructions = vec![Instruction::PushNumber(DmNumberBits::from_f32(7.0))];
        for _ in 0..256 {
            instructions.extend([
                Instruction::StoreResult,
                Instruction::LoadResult,
                Instruction::Duplicate,
                Instruction::Pop,
            ]);
        }
        instructions.extend([Instruction::LoadResult, Instruction::Return]);
        let mut module = module_with(instructions);
        module
            .install_compact_wordcode()
            .expect("compact wordcode should install");
        let entry = module.procedure_id("/proc/main").unwrap();
        let compact = module.compact_wordcode().unwrap();
        let encoded = compact.encode().unwrap();
        let iterations = 2_000;
        let started = Instant::now();
        for _ in 0..iterations {
            let result = crate::execute_module(&module, entry, &[])
                .expect("benchmark execution should complete");
            std::hint::black_box(result);
        }
        let elapsed = started.elapsed();
        eprintln!(
            "compact-dispatch-gate enabled={} iterations={iterations} instructions_per_iteration={} elapsed_ms={} image_bytes={} words={} specialized={} rich_instruction_bytes={}",
            std::env::var_os("DREAM64_DISABLE_COMPACT_WORDCODE").is_none(),
            compact.word_count(),
            elapsed.as_millis(),
            encoded.len(),
            compact.word_count(),
            compact.specialized_word_count(),
            compact.word_count() * std::mem::size_of::<Instruction>(),
        );
    }
}

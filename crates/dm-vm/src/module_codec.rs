//! Versioned, portable binary storage for fully eager VM modules.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use dm_core::{DmNumberBits, SourceSpan};
use dm_value::{FieldName, TypePath};

use crate::{
    CompoundAssignmentOperator, CompoundListIndexOperator, Instruction, ListEntryKind, Module,
    ProcedureId, Program, TypePredicateKind, VerbParameterType, dynamic_name_index,
    next_module_identity,
};

const MAGIC: &[u8; 8] = b"DM64MOD\0";
const VERSION: u16 = 7;
#[cfg(test)]
const INSTRUCTION_TAG_COUNT: u8 = 139;
const MAX_ARTIFACT_BYTES: usize = 8 * 1024 * 1024 * 1024;
const MAX_PROCEDURES: usize = 1_000_000;
const MAX_PROCEDURE_TYPES: usize = 1_000_000;
const MAX_INSTRUCTIONS_PER_PROGRAM: usize = 32_000_000;
const MAX_TOTAL_INSTRUCTIONS: usize = 500_000_000;
const MAX_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_VECTOR_ELEMENTS: usize = 1_000_000;

/// A structural or representation error in a portable VM module artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleCodecError {
    message: String,
}

impl ModuleCodecError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ModuleCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ModuleCodecError {}

impl Module {
    /// Encodes this fully eager module into the versioned Dream64 module format.
    ///
    /// The format uses explicit little-endian scalars and tagged instruction
    /// payloads. It never persists Rust pointers, native enum layouts, source
    /// syntax, deferred lowering state, runtime cache indexes, or module identity.
    ///
    /// # Errors
    ///
    /// Returns [`ModuleCodecError`] when the module contains deferred bodies,
    /// violates an artifact bound, or contains an invalid structural reference.
    pub fn encode_portable(&self) -> Result<Vec<u8>, ModuleCodecError> {
        validate_module(self)?;
        let mut writer = Writer::new();
        writer.bytes.extend_from_slice(MAGIC);
        writer.u16(VERSION);
        writer.u16(0);
        writer.len(self.procedures.len(), MAX_PROCEDURES, "procedure count")?;
        for path in &self.paths {
            writer.string(path)?;
        }
        writer.len(
            self.procedure_types.len(),
            MAX_PROCEDURE_TYPES,
            "procedure type count",
        )?;
        for path in &self.procedure_types {
            writer.string(path.as_str())?;
        }
        for program in &self.procedures {
            encode_program(&mut writer, program)?;
        }
        if writer.bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(ModuleCodecError::new(
                "encoded module exceeds artifact byte limit",
            ));
        }
        Ok(writer.bytes)
    }

    /// Decodes one versioned Dream64 portable module artifact.
    ///
    /// The decoded module receives a fresh runtime cache identity. Canonical
    /// and dynamic procedure-name indexes are reconstructed from its stable
    /// path table rather than trusted from the artifact.
    ///
    /// # Errors
    ///
    /// Returns [`ModuleCodecError`] for corrupt, truncated, oversized, or
    /// unsupported input and for every invalid structural reference.
    pub fn decode_portable(bytes: &[u8]) -> Result<Self, ModuleCodecError> {
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(ModuleCodecError::new("module artifact exceeds byte limit"));
        }
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len(), "module header")? != MAGIC {
            return Err(ModuleCodecError::new("invalid portable module header"));
        }
        let version = reader.u16("module version")?;
        if version != VERSION {
            return Err(ModuleCodecError::new(format!(
                "unsupported portable module version {version}"
            )));
        }
        if reader.u16("reserved header flags")? != 0 {
            return Err(ModuleCodecError::new(
                "unknown portable module header flags",
            ));
        }

        let procedure_count = reader.len(MAX_PROCEDURES, "procedure count")?;
        if procedure_count > reader.remaining() / 4 {
            return Err(ModuleCodecError::new("truncated procedure path table"));
        }
        let mut paths = Vec::with_capacity(procedure_count);
        for _ in 0..procedure_count {
            paths.push(reader.string("procedure path")?);
        }
        let procedure_type_count = reader.len(MAX_PROCEDURE_TYPES, "procedure type count")?;
        if procedure_type_count > reader.remaining() / 4 {
            return Err(ModuleCodecError::new("truncated procedure type table"));
        }
        let mut procedure_types = Vec::with_capacity(procedure_type_count);
        for _ in 0..procedure_type_count {
            let raw = reader.string("procedure type path")?;
            procedure_types.push(reader.intern_type_path(raw, "procedure type path")?);
        }

        if procedure_count > reader.remaining() / 17 {
            return Err(ModuleCodecError::new("truncated program table"));
        }
        let mut procedures = Vec::with_capacity(procedure_count);
        let mut total_instructions = 0usize;
        for procedure_index in 0..procedure_count {
            let program = decode_program(&mut reader, procedure_count).map_err(|error| {
                ModuleCodecError::new(format!(
                    "procedure {procedure_index} {}: {error}",
                    paths[procedure_index]
                ))
            })?;
            total_instructions = total_instructions
                .checked_add(program.instructions.len())
                .ok_or_else(|| ModuleCodecError::new("total instruction count overflow"))?;
            if total_instructions > MAX_TOTAL_INSTRUCTIONS {
                return Err(ModuleCodecError::new(
                    "module exceeds total instruction limit",
                ));
            }
            procedures.push(Arc::new(program));
        }
        if !reader.is_empty() {
            return Err(ModuleCodecError::new(
                "trailing bytes after portable module",
            ));
        }

        let mut names = HashMap::with_capacity(paths.len());
        for (index, path) in paths.iter().enumerate() {
            if !path.starts_with('/') {
                continue;
            }
            let id = ProcedureId(index as u32);
            if names.insert(path.clone(), id).is_some() {
                return Err(ModuleCodecError::new(format!(
                    "duplicate procedure path {path:?}"
                )));
            }
        }
        let dynamic_names =
            dynamic_name_index(&paths).map_err(|error| ModuleCodecError::new(error.message))?;
        let module = Self {
            identity: next_module_identity(),
            procedures,
            paths,
            names,
            dynamic_names,
            deferred: Arc::new(HashMap::new()),
            procedure_types,
            initializer_call_names: None,
        };
        validate_module(&module)?;
        Ok(module)
    }
}

fn validate_module(module: &Module) -> Result<(), ModuleCodecError> {
    if !module.deferred.is_empty() {
        return Err(ModuleCodecError::new(
            "portable module encoding requires a fully eager module",
        ));
    }
    if module.procedures.len() > MAX_PROCEDURES {
        return Err(ModuleCodecError::new("module exceeds procedure limit"));
    }
    if module.paths.len() != module.procedures.len() {
        return Err(ModuleCodecError::new(
            "procedure path and program table lengths differ",
        ));
    }
    if module.procedure_types.len() > MAX_PROCEDURE_TYPES {
        return Err(ModuleCodecError::new("module exceeds procedure type limit"));
    }
    let mut total = 0usize;
    for (procedure_index, program) in module.procedures.iter().enumerate() {
        validate_program(program, module.procedures.len()).map_err(|error| {
            ModuleCodecError::new(format!(
                "procedure {procedure_index} {}: {error}",
                module.paths[procedure_index]
            ))
        })?;
        total = total
            .checked_add(program.instructions.len())
            .ok_or_else(|| ModuleCodecError::new("total instruction count overflow"))?;
        if total > MAX_TOTAL_INSTRUCTIONS {
            return Err(ModuleCodecError::new(
                "module exceeds total instruction limit",
            ));
        }
    }
    Ok(())
}

fn validate_program(program: &Program, procedure_count: usize) -> Result<(), ModuleCodecError> {
    if program.instructions.len() > MAX_INSTRUCTIONS_PER_PROGRAM {
        return Err(ModuleCodecError::new("program exceeds instruction limit"));
    }
    if program.source_spans.len() != program.instructions.len() {
        return Err(ModuleCodecError::new(
            "instruction and source-span table lengths differ",
        ));
    }
    if program.parameter_names.len() != program.parameter_count {
        return Err(ModuleCodecError::new(
            "parameter name and parameter count differ",
        ));
    }
    if program.verb_parameter_types.len() != program.parameter_count {
        return Err(ModuleCodecError::new(
            "verb parameter type and parameter count differ",
        ));
    }
    if program.parameter_count > program.local_count {
        return Err(ModuleCodecError::new(
            "parameter count exceeds local slot count",
        ));
    }
    if program.parameter_count > u32::MAX as usize || program.local_count > u32::MAX as usize {
        return Err(ModuleCodecError::new(
            "program slot count exceeds format limit",
        ));
    }
    for span in &program.source_spans {
        if span.start > span.end {
            return Err(ModuleCodecError::new("source span has inverted bounds"));
        }
        if u64::try_from(span.start).is_err() || u64::try_from(span.end).is_err() {
            return Err(ModuleCodecError::new("source span exceeds format limit"));
        }
    }
    let instruction_count = program.instructions.len();
    for instruction in &program.instructions {
        let valid_jump = |target: usize| {
            if target >= instruction_count {
                Err(ModuleCodecError::new(format!(
                    "jump target {target} is outside {instruction_count} instructions"
                )))
            } else {
                Ok(())
            }
        };
        match instruction {
            Instruction::LoadStaticLocalOrJump { target, .. }
            | Instruction::JumpIfNull(target)
            | Instruction::JumpIfFalse(target)
            | Instruction::Jump(target)
            | Instruction::JumpIfArgumentSupplied { target, .. }
            | Instruction::Spawn { entry: target }
            | Instruction::CheckNumericLoop { exit: target, .. }
            | Instruction::AdvanceNumericLoop { target, .. } => valid_jump(*target)?,
            Instruction::BeginTry { catch, end, .. } => {
                valid_jump(*catch)?;
                valid_jump(*end)?;
            }
            Instruction::Call { procedure, .. } => validate_procedure(*procedure, procedure_count)?,
            Instruction::CallParent {
                procedure: Some(procedure),
                ..
            } => validate_procedure(*procedure, procedure_count)?,
            _ => {}
        }
    }
    Ok(())
}

fn validate_procedure(
    procedure: ProcedureId,
    procedure_count: usize,
) -> Result<(), ModuleCodecError> {
    if procedure.index() >= procedure_count {
        Err(ModuleCodecError::new(format!(
            "invalid procedure id {} for {procedure_count} procedures",
            procedure.index()
        )))
    } else {
        Ok(())
    }
}

fn encode_program(writer: &mut Writer, program: &Program) -> Result<(), ModuleCodecError> {
    writer.boolean(program.wait_for);
    writer.u32(program.parameter_count as u32);
    writer.len(
        program.parameter_names.len(),
        u32::MAX as usize,
        "parameter name count",
    )?;
    for name in &program.parameter_names {
        writer.string(name)?;
    }
    writer.len(
        program.verb_parameter_types.len(),
        u32::MAX as usize,
        "verb parameter type count",
    )?;
    for parameter_type in &program.verb_parameter_types {
        writer.u8(match parameter_type {
            VerbParameterType::Unsupported => 0,
            VerbParameterType::Text => 1,
            VerbParameterType::Number => 2,
            VerbParameterType::Message => 3,
            VerbParameterType::Color => 4,
            VerbParameterType::File => 5,
            VerbParameterType::Anything => 6,
            VerbParameterType::Atom(mask) => 0x80 | (mask & 0x0f),
        });
    }
    writer.boolean(program.verb_name.is_some());
    if let Some(name) = &program.verb_name {
        writer.string(name)?;
    }
    writer.u32(program.local_count as u32);
    writer.len(
        program.instructions.len(),
        MAX_INSTRUCTIONS_PER_PROGRAM,
        "instruction count",
    )?;
    for instruction in &program.instructions {
        encode_instruction(writer, instruction)?;
    }
    writer.len(
        program.source_spans.len(),
        MAX_INSTRUCTIONS_PER_PROGRAM,
        "source span count",
    )?;
    for span in &program.source_spans {
        writer.u64(span.start as u64);
        writer.u64(span.end as u64);
    }
    Ok(())
}

fn decode_program(
    reader: &mut Reader<'_>,
    procedure_count: usize,
) -> Result<Program, ModuleCodecError> {
    let wait_for = reader.boolean("wait-for flag")?;
    let parameter_count = reader.u32("parameter count")? as usize;
    if parameter_count > MAX_VECTOR_ELEMENTS {
        return Err(ModuleCodecError::new("parameter count exceeds limit"));
    }
    let parameter_name_count =
        reader.len_with_min(MAX_VECTOR_ELEMENTS, "parameter name count", 4)?;
    if parameter_name_count != parameter_count {
        return Err(ModuleCodecError::new(
            "parameter name and parameter count differ",
        ));
    }
    let mut parameter_names = Vec::with_capacity(parameter_name_count);
    for _ in 0..parameter_name_count {
        parameter_names.push(reader.string("parameter name")?);
    }
    let verb_parameter_type_count =
        reader.len_with_min(MAX_VECTOR_ELEMENTS, "verb parameter type count", 1)?;
    if verb_parameter_type_count != parameter_count {
        return Err(ModuleCodecError::new(
            "verb parameter type and parameter count differ",
        ));
    }
    let mut verb_parameter_types = Vec::with_capacity(verb_parameter_type_count);
    for _ in 0..verb_parameter_type_count {
        verb_parameter_types.push(match reader.u8("verb parameter type")? {
            0 => VerbParameterType::Unsupported,
            1 => VerbParameterType::Text,
            2 => VerbParameterType::Number,
            3 => VerbParameterType::Message,
            4 => VerbParameterType::Color,
            5 => VerbParameterType::File,
            6 => VerbParameterType::Anything,
            tag if tag & 0xf0 == 0x80 && tag & 0x0f != 0 => VerbParameterType::Atom(tag & 0x0f),
            tag => {
                return Err(ModuleCodecError::new(format!(
                    "unknown verb parameter type tag {tag}",
                )));
            }
        });
    }
    let verb_name = reader
        .boolean("verb name presence")?
        .then(|| reader.string("verb name"))
        .transpose()?;
    let local_count = reader.u32("local count")? as usize;
    let instruction_count =
        reader.len_with_min(MAX_INSTRUCTIONS_PER_PROGRAM, "instruction count", 1)?;
    let mut instructions = Vec::with_capacity(instruction_count);
    for _ in 0..instruction_count {
        instructions.push(decode_instruction(reader, procedure_count)?);
    }
    let source_span_count =
        reader.len_with_min(MAX_INSTRUCTIONS_PER_PROGRAM, "source span count", 16)?;
    if source_span_count != instruction_count {
        return Err(ModuleCodecError::new(
            "instruction and source-span table lengths differ",
        ));
    }
    let mut source_spans = Vec::with_capacity(source_span_count);
    for _ in 0..source_span_count {
        let start = reader.usize64("source span start")?;
        let end = reader.usize64("source span end")?;
        if start > end {
            return Err(ModuleCodecError::new("source span has inverted bounds"));
        }
        source_spans.push(SourceSpan::new(start, end));
    }
    let program = Program {
        wait_for,
        parameter_count,
        parameter_names,
        verb_parameter_types,
        verb_name,
        local_count,
        instructions,
        source_spans,
    };
    validate_program(&program, procedure_count)?;
    Ok(program)
}

fn encode_instruction(
    writer: &mut Writer,
    instruction: &Instruction,
) -> Result<(), ModuleCodecError> {
    macro_rules! unit {
        ($tag:expr) => {{
            writer.u8($tag);
        }};
    }
    macro_rules! one_u8 {
        ($tag:expr, $value:expr) => {{
            writer.u8($tag);
            writer.u8(*$value);
        }};
    }
    macro_rules! one_u16 {
        ($tag:expr, $value:expr) => {{
            writer.u8($tag);
            writer.u16(*$value);
        }};
    }
    macro_rules! one_usize {
        ($tag:expr, $value:expr) => {{
            writer.u8($tag);
            writer.u32(
                u32::try_from(*$value).map_err(|_| {
                    ModuleCodecError::new("instruction target exceeds format limit")
                })?,
            );
        }};
    }
    match instruction {
        Instruction::PushNull => unit!(0),
        Instruction::PushNumber(value) => {
            writer.u8(1);
            writer.u32(value.bits());
        }
        Instruction::PushText(value) => {
            writer.u8(2);
            writer.string(value)?;
        }
        Instruction::PushFile(value) => {
            writer.u8(3);
            writer.string(value)?;
        }
        Instruction::PushTypePath(value) => {
            writer.u8(4);
            writer.string(value.as_str())?;
        }
        Instruction::AddressLocal(value) => one_u16!(5, value),
        Instruction::LoadLocalRaw(value) => one_u16!(6, value),
        Instruction::MakeModifiedTypePath { fields } => {
            writer.u8(7);
            writer.fields(fields)?;
        }
        Instruction::AllocateDatum {
            argument_count,
            argument_names,
        } => {
            writer.u8(8);
            writer.u16(*argument_count);
            writer.optional_strings(argument_names)?;
        }
        Instruction::AllocateCurrentDatum { argument_count } => one_u16!(9, argument_count),
        Instruction::MakeRegex { argument_count } => one_u8!(10, argument_count),
        Instruction::MakeMutableAppearance { argument_count } => one_u16!(11, argument_count),
        Instruction::MakeMatrix { argument_count } => one_u8!(12, argument_count),
        Instruction::MakeVector { argument_count } => one_u8!(13, argument_count),
        Instruction::ReplaceText {
            argument_count,
            exact,
            character_indices,
        } => {
            writer.u8(14);
            writer.u8(*argument_count);
            writer.boolean(*exact);
            writer.boolean(*character_indices);
        }
        Instruction::CopyText {
            argument_count,
            character_indices,
        } => {
            writer.u8(15);
            writer.u8(*argument_count);
            writer.boolean(*character_indices);
        }
        Instruction::StandardBuiltin {
            name,
            argument_count,
            argument_names,
        } => {
            writer.u8(16);
            writer.string(name)?;
            writer.u16(*argument_count);
            writer.optional_strings(argument_names)?;
        }
        Instruction::NativeSrcMethod {
            name,
            argument_count,
        } => {
            writer.u8(17);
            writer.string(name)?;
            writer.u16(*argument_count);
        }
        Instruction::Output => unit!(18),
        Instruction::Input => unit!(19),
        Instruction::ExternalCall { argument_count } => one_u16!(20, argument_count),
        Instruction::Animate {
            argument_names,
            expanded_indices,
        } => {
            writer.u8(21);
            writer.optional_strings(argument_names)?;
            writer.u16s(expanded_indices)?;
        }
        Instruction::MakeFilter {
            argument_names,
            expanded_indices,
        } => {
            writer.u8(22);
            writer.optional_strings(argument_names)?;
            writer.u16s(expanded_indices)?;
        }
        Instruction::InitialField(value) => {
            writer.u8(23);
            writer.string(value.as_str())?;
        }
        Instruction::InitialDynamicField => unit!(24),
        Instruction::Block { argument_count } => one_u8!(25, argument_count),
        Instruction::Rand { argument_count } => one_u8!(26, argument_count),
        Instruction::Roll { argument_count } => one_u8!(27, argument_count),
        Instruction::Pick { weighted } => {
            writer.u8(28);
            writer.booleans(weighted)?;
        }
        Instruction::Prob => unit!(29),
        Instruction::Round { argument_count } => one_u8!(30, argument_count),
        Instruction::Length => unit!(31),
        Instruction::Ref => unit!(32),
        Instruction::GetStep => unit!(33),
        Instruction::GetStepTowards => unit!(34),
        Instruction::Range { argument_count } => one_u8!(35, argument_count),
        Instruction::TypesOf { argument_count } => one_u8!(36, argument_count),
        Instruction::HasCall => unit!(37),
        Instruction::TypeInstances(value) => {
            writer.u8(38);
            writer.string(value.as_str())?;
        }
        Instruction::TypePredicate {
            kind,
            argument_count,
        } => {
            writer.u8(39);
            writer.u8(type_predicate_tag(*kind));
            writer.u8(*argument_count);
        }
        Instruction::MakeList(value) => one_u16!(40, value),
        Instruction::MakeArray(value) => one_u8!(41, value),
        Instruction::MakeArgs => unit!(42),
        Instruction::MakeListEntries(values) => {
            writer.u8(43);
            writer.list_entry_kinds(values)?;
        }
        Instruction::MakeAssociativeListEntries(values) => {
            writer.u8(44);
            writer.list_entry_kinds(values)?;
        }
        Instruction::IndexList => unit!(45),
        Instruction::SetListIndex => unit!(46),
        Instruction::SetListIndexKeep => unit!(47),
        Instruction::CompoundListIndex(value) => {
            writer.u8(48);
            writer.u8(compound_list_tag(*value));
        }
        Instruction::CompoundListIndexKeep(value) => {
            writer.u8(49);
            writer.u8(compound_list_tag(*value));
        }
        Instruction::ListLength => unit!(50),
        Instruction::PrepareIteration => unit!(51),
        Instruction::IterationTypeFilter(value) => {
            writer.u8(52);
            writer.string(value.as_str())?;
        }
        Instruction::LoadLocal(value) => one_u16!(53, value),
        Instruction::StoreLocal(value) => one_u16!(54, value),
        Instruction::LoadStaticLocalOrJump { slot, target } => {
            writer.u8(55);
            writer.u16(*slot);
            writer.target(*target)?;
        }
        Instruction::InitializeStaticLocal(value) => one_u16!(56, value),
        Instruction::LoadSrc => unit!(57),
        Instruction::StoreSrc => unit!(58),
        Instruction::LoadUsr => unit!(59),
        Instruction::StoreUsr => unit!(60),
        Instruction::LoadCaller => unit!(61),
        Instruction::LoadField(value) => {
            writer.u8(62);
            writer.string(value.as_str())?;
        }
        Instruction::StoreField(value) => {
            writer.u8(63);
            writer.string(value.as_str())?;
        }
        Instruction::StoreFieldKeep(value) => {
            writer.u8(64);
            writer.string(value.as_str())?;
        }
        Instruction::LoadGlobal(value) => {
            writer.u8(65);
            writer.string(value.as_str())?;
        }
        Instruction::LoadGlobalVars => unit!(66),
        Instruction::LoadDatumVars => unit!(67),
        Instruction::LoadInitialGlobal(value) => {
            writer.u8(68);
            writer.string(value.as_str())?;
        }
        Instruction::StoreGlobal(value) => {
            writer.u8(69);
            writer.string(value.as_str())?;
        }
        Instruction::MutateLocal {
            slot,
            delta,
            prefix,
        } => {
            writer.u8(70);
            writer.u16(*slot);
            writer.i8(*delta);
            writer.boolean(*prefix);
        }
        Instruction::MutateField {
            name,
            delta,
            prefix,
        } => {
            writer.u8(71);
            writer.string(name.as_str())?;
            writer.i8(*delta);
            writer.boolean(*prefix);
        }
        Instruction::MutateGlobal {
            name,
            delta,
            prefix,
        } => {
            writer.u8(72);
            writer.string(name.as_str())?;
            writer.i8(*delta);
            writer.boolean(*prefix);
        }
        Instruction::MutateListIndex { delta, prefix } => {
            writer.u8(73);
            writer.i8(*delta);
            writer.boolean(*prefix);
        }
        Instruction::MutateResult { delta, prefix } => {
            writer.u8(74);
            writer.i8(*delta);
            writer.boolean(*prefix);
        }
        Instruction::Duplicate => unit!(75),
        Instruction::LoadResult => unit!(76),
        Instruction::StoreResult => unit!(77),
        Instruction::Pop => unit!(78),
        Instruction::Crash => unit!(79),
        Instruction::BeginTry { catch, end, local } => {
            writer.u8(80);
            writer.target(*catch)?;
            writer.target(*end)?;
            writer.optional_u16(*local);
        }
        Instruction::EndTry => unit!(81),
        Instruction::Throw => unit!(82),
        Instruction::Locate { argument_count } => one_u16!(83, argument_count),
        Instruction::LocateIn { argument_count } => one_u16!(84, argument_count),
        Instruction::CompoundAssignment(value) => {
            writer.u8(85);
            writer.u8(compound_assignment_tag(*value));
        }
        Instruction::Add => unit!(86),
        Instruction::Subtract => unit!(87),
        Instruction::Multiply => unit!(88),
        Instruction::Power => unit!(89),
        Instruction::Divide => unit!(90),
        Instruction::Remainder => unit!(91),
        Instruction::FractionalRemainder => unit!(92),
        Instruction::BitAnd => unit!(93),
        Instruction::BitOr => unit!(94),
        Instruction::BitXor => unit!(95),
        Instruction::ShiftLeft => unit!(96),
        Instruction::ShiftRight => unit!(97),
        Instruction::BitNot => unit!(98),
        Instruction::Negate => unit!(99),
        Instruction::Not => unit!(100),
        Instruction::Equal => unit!(101),
        Instruction::NotEqual => unit!(102),
        Instruction::Equivalent => unit!(103),
        Instruction::NotEquivalent => unit!(104),
        Instruction::Compare => unit!(105),
        Instruction::Contains => unit!(106),
        Instruction::Less => unit!(107),
        Instruction::LessEqual => unit!(108),
        Instruction::Greater => unit!(109),
        Instruction::GreaterEqual => unit!(110),
        Instruction::And => unit!(111),
        Instruction::Or => unit!(112),
        Instruction::JumpIfNull(value) => one_usize!(113, value),
        Instruction::JumpIfFalse(value) => one_usize!(114, value),
        Instruction::Jump(value) => one_usize!(115, value),
        Instruction::JumpIfArgumentSupplied { parameter, target } => {
            writer.u8(116);
            writer.u16(*parameter);
            writer.target(*target)?;
        }
        Instruction::Call {
            procedure,
            argument_count,
            argument_names,
        } => {
            writer.u8(117);
            writer.u32(procedure.0);
            writer.u16(*argument_count);
            writer.optional_strings(argument_names)?;
        }
        Instruction::CallCurrent { argument_count } => {
            writer.u8(118);
            writer.optional_u16(*argument_count);
        }
        Instruction::CallParent {
            procedure,
            argument_count,
        } => {
            writer.u8(119);
            writer.optional_procedure(*procedure);
            writer.optional_u16(*argument_count);
        }
        Instruction::CallDynamic {
            static_selector,
            argument_count,
            argument_names,
            null_receiver_is_global,
        } => {
            writer.u8(120);
            writer.optional_string(static_selector.as_deref())?;
            writer.u16(*argument_count);
            writer.optional_strings(argument_names)?;
            writer.boolean(*null_receiver_is_global);
        }
        Instruction::ExpandArgumentLists {
            argument_count,
            argument_names,
            expanded_indices,
        } => {
            writer.u8(121);
            writer.u16(*argument_count);
            writer.optional_strings(argument_names)?;
            writer.u16s(expanded_indices)?;
        }
        Instruction::Return => unit!(122),
        Instruction::Spawn { entry } => one_usize!(123, entry),
        Instruction::Sleep => unit!(124),
        Instruction::LogicalOrEmptyListLocal(value) => one_u16!(125, value),
        Instruction::LogicalOrEmptyListGlobal(value) => {
            writer.u8(126);
            writer.string(value.as_str())?;
        }
        Instruction::LogicalOrEmptyListField(value) => {
            writer.u8(127);
            writer.string(value.as_str())?;
        }
        Instruction::LogicalOrEmptyListIndex => unit!(128),
        Instruction::PickExpandedArguments => unit!(129),
        Instruction::PrepareRhsFirstIndexAssignment => unit!(130),
        Instruction::LoadDeclaredField(value) => {
            writer.u8(131);
            writer.string(value.as_str())?;
        }
        Instruction::LoadDynamicField => unit!(132),
        Instruction::StoreDynamicField => unit!(133),
        Instruction::IndexLocalList(value) => one_u16!(134, value),
        Instruction::ListLengthLocal(value) => one_u16!(135, value),
        Instruction::NextLocalListIteration {
            list_slot,
            index_slot,
            item_slot,
            exit,
        } => {
            writer.u8(136);
            writer.u16(*list_slot);
            writer.u16(*index_slot);
            writer.u16(*item_slot);
            writer.target(*exit)?;
        }
        Instruction::CheckNumericLoop {
            current_slot,
            end_slot,
            exit,
        } => {
            writer.u8(137);
            writer.u16(*current_slot);
            writer.u16(*end_slot);
            writer.target(*exit)?;
        }
        Instruction::AdvanceNumericLoop { slot, step, target } => {
            writer.u8(138);
            writer.u16(*slot);
            writer.u32(step.bits());
            writer.target(*target)?;
        }
    }
    Ok(())
}

fn decode_instruction(
    reader: &mut Reader<'_>,
    procedure_count: usize,
) -> Result<Instruction, ModuleCodecError> {
    let tag = reader.u8("instruction tag")?;
    let instruction = match tag {
        0 => Instruction::PushNull,
        1 => Instruction::PushNumber(DmNumberBits::from_f32(f32::from_bits(
            reader.u32("number bits")?,
        ))),
        2 => Instruction::PushText(reader.text()?),
        3 => Instruction::PushFile(reader.string("file constant")?),
        4 => Instruction::PushTypePath(reader.type_path()?),
        5 => Instruction::AddressLocal(reader.u16("local slot")?),
        6 => Instruction::LoadLocalRaw(reader.u16("local slot")?),
        7 => Instruction::MakeModifiedTypePath {
            fields: Arc::from(reader.fields()?.into_boxed_slice()),
        },
        8 => Instruction::AllocateDatum {
            argument_count: reader.u16("argument count")?,
            argument_names: reader.optional_strings()?,
        },
        9 => Instruction::AllocateCurrentDatum {
            argument_count: reader.u16("argument count")?,
        },
        10 => Instruction::MakeRegex {
            argument_count: reader.u8("argument count")?,
        },
        11 => Instruction::MakeMutableAppearance {
            argument_count: reader.u16("argument count")?,
        },
        12 => Instruction::MakeMatrix {
            argument_count: reader.u8("argument count")?,
        },
        13 => Instruction::MakeVector {
            argument_count: reader.u8("argument count")?,
        },
        14 => Instruction::ReplaceText {
            argument_count: reader.u8("argument count")?,
            exact: reader.boolean("exact flag")?,
            character_indices: reader.boolean("character-index flag")?,
        },
        15 => Instruction::CopyText {
            argument_count: reader.u8("argument count")?,
            character_indices: reader.boolean("character-index flag")?,
        },
        16 => Instruction::StandardBuiltin {
            name: reader.string("builtin name")?,
            argument_count: reader.u16("argument count")?,
            argument_names: reader.optional_strings()?,
        },
        17 => Instruction::NativeSrcMethod {
            name: reader.string("method name")?,
            argument_count: reader.u16("argument count")?,
        },
        18 => Instruction::Output,
        19 => Instruction::Input,
        20 => Instruction::ExternalCall {
            argument_count: reader.u16("argument count")?,
        },
        21 => Instruction::Animate {
            argument_names: reader.optional_strings()?,
            expanded_indices: reader.u16s()?,
        },
        22 => Instruction::MakeFilter {
            argument_names: reader.optional_strings()?,
            expanded_indices: reader.u16s()?,
        },
        23 => Instruction::InitialField(reader.field()?),
        24 => Instruction::InitialDynamicField,
        25 => Instruction::Block {
            argument_count: reader.u8("argument count")?,
        },
        26 => Instruction::Rand {
            argument_count: reader.u8("argument count")?,
        },
        27 => Instruction::Roll {
            argument_count: reader.u8("argument count")?,
        },
        28 => Instruction::Pick {
            weighted: reader.booleans()?,
        },
        29 => Instruction::Prob,
        30 => Instruction::Round {
            argument_count: reader.u8("argument count")?,
        },
        31 => Instruction::Length,
        32 => Instruction::Ref,
        33 => Instruction::GetStep,
        34 => Instruction::GetStepTowards,
        35 => Instruction::Range {
            argument_count: reader.u8("argument count")?,
        },
        36 => Instruction::TypesOf {
            argument_count: reader.u8("argument count")?,
        },
        37 => Instruction::HasCall,
        38 => Instruction::TypeInstances(reader.type_path()?),
        39 => Instruction::TypePredicate {
            kind: decode_type_predicate(reader.u8("type predicate tag")?)?,
            argument_count: reader.u8("argument count")?,
        },
        40 => Instruction::MakeList(reader.u16("list length")?),
        41 => Instruction::MakeArray(reader.u8("array dimension count")?),
        42 => Instruction::MakeArgs,
        43 => Instruction::MakeListEntries(reader.list_entry_kinds()?),
        44 => Instruction::MakeAssociativeListEntries(reader.list_entry_kinds()?),
        45 => Instruction::IndexList,
        46 => Instruction::SetListIndex,
        47 => Instruction::SetListIndexKeep,
        48 => {
            Instruction::CompoundListIndex(decode_compound_list(reader.u8("list operator tag")?)?)
        }
        49 => Instruction::CompoundListIndexKeep(decode_compound_list(
            reader.u8("list operator tag")?,
        )?),
        50 => Instruction::ListLength,
        51 => Instruction::PrepareIteration,
        52 => Instruction::IterationTypeFilter(reader.type_path()?),
        53 => Instruction::LoadLocal(reader.u16("local slot")?),
        54 => Instruction::StoreLocal(reader.u16("local slot")?),
        55 => Instruction::LoadStaticLocalOrJump {
            slot: reader.u16("local slot")?,
            target: reader.target()?,
        },
        56 => Instruction::InitializeStaticLocal(reader.u16("local slot")?),
        57 => Instruction::LoadSrc,
        58 => Instruction::StoreSrc,
        59 => Instruction::LoadUsr,
        60 => Instruction::StoreUsr,
        61 => Instruction::LoadCaller,
        62 => Instruction::LoadField(reader.field()?),
        63 => Instruction::StoreField(reader.field()?),
        64 => Instruction::StoreFieldKeep(reader.field()?),
        65 => Instruction::LoadGlobal(reader.field()?),
        66 => Instruction::LoadGlobalVars,
        67 => Instruction::LoadDatumVars,
        68 => Instruction::LoadInitialGlobal(reader.field()?),
        69 => Instruction::StoreGlobal(reader.field()?),
        70 => Instruction::MutateLocal {
            slot: reader.u16("local slot")?,
            delta: reader.i8("mutation delta")?,
            prefix: reader.boolean("prefix flag")?,
        },
        71 => Instruction::MutateField {
            name: reader.field()?,
            delta: reader.i8("mutation delta")?,
            prefix: reader.boolean("prefix flag")?,
        },
        72 => Instruction::MutateGlobal {
            name: reader.field()?,
            delta: reader.i8("mutation delta")?,
            prefix: reader.boolean("prefix flag")?,
        },
        73 => Instruction::MutateListIndex {
            delta: reader.i8("mutation delta")?,
            prefix: reader.boolean("prefix flag")?,
        },
        74 => Instruction::MutateResult {
            delta: reader.i8("mutation delta")?,
            prefix: reader.boolean("prefix flag")?,
        },
        75 => Instruction::Duplicate,
        76 => Instruction::LoadResult,
        77 => Instruction::StoreResult,
        78 => Instruction::Pop,
        79 => Instruction::Crash,
        80 => Instruction::BeginTry {
            catch: reader.target()?,
            end: reader.target()?,
            local: reader.optional_u16()?,
        },
        81 => Instruction::EndTry,
        82 => Instruction::Throw,
        83 => Instruction::Locate {
            argument_count: reader.u16("argument count")?,
        },
        84 => Instruction::LocateIn {
            argument_count: reader.u16("argument count")?,
        },
        85 => Instruction::CompoundAssignment(decode_compound_assignment(
            reader.u8("assignment operator tag")?,
        )?),
        86 => Instruction::Add,
        87 => Instruction::Subtract,
        88 => Instruction::Multiply,
        89 => Instruction::Power,
        90 => Instruction::Divide,
        91 => Instruction::Remainder,
        92 => Instruction::FractionalRemainder,
        93 => Instruction::BitAnd,
        94 => Instruction::BitOr,
        95 => Instruction::BitXor,
        96 => Instruction::ShiftLeft,
        97 => Instruction::ShiftRight,
        98 => Instruction::BitNot,
        99 => Instruction::Negate,
        100 => Instruction::Not,
        101 => Instruction::Equal,
        102 => Instruction::NotEqual,
        103 => Instruction::Equivalent,
        104 => Instruction::NotEquivalent,
        105 => Instruction::Compare,
        106 => Instruction::Contains,
        107 => Instruction::Less,
        108 => Instruction::LessEqual,
        109 => Instruction::Greater,
        110 => Instruction::GreaterEqual,
        111 => Instruction::And,
        112 => Instruction::Or,
        113 => Instruction::JumpIfNull(reader.target()?),
        114 => Instruction::JumpIfFalse(reader.target()?),
        115 => Instruction::Jump(reader.target()?),
        116 => Instruction::JumpIfArgumentSupplied {
            parameter: reader.u16("parameter index")?,
            target: reader.target()?,
        },
        117 => {
            let procedure = reader.procedure(procedure_count)?;
            Instruction::Call {
                procedure,
                argument_count: reader.u16("argument count")?,
                argument_names: reader.optional_strings()?,
            }
        }
        118 => Instruction::CallCurrent {
            argument_count: reader.optional_u16()?,
        },
        119 => Instruction::CallParent {
            procedure: reader.optional_procedure(procedure_count)?,
            argument_count: reader.optional_u16()?,
        },
        120 => Instruction::CallDynamic {
            static_selector: reader.optional_string()?,
            argument_count: reader.u16("argument count")?,
            argument_names: reader.optional_strings()?,
            null_receiver_is_global: reader.boolean("global receiver flag")?,
        },
        121 => Instruction::ExpandArgumentLists {
            argument_count: reader.u16("argument count")?,
            argument_names: reader.optional_strings()?,
            expanded_indices: reader.u16s()?,
        },
        122 => Instruction::Return,
        123 => Instruction::Spawn {
            entry: reader.target()?,
        },
        124 => Instruction::Sleep,
        125 => Instruction::LogicalOrEmptyListLocal(reader.u16("local slot")?),
        126 => Instruction::LogicalOrEmptyListGlobal(reader.field()?),
        127 => Instruction::LogicalOrEmptyListField(reader.field()?),
        128 => Instruction::LogicalOrEmptyListIndex,
        129 => Instruction::PickExpandedArguments,
        130 => Instruction::PrepareRhsFirstIndexAssignment,
        131 => Instruction::LoadDeclaredField(reader.field()?),
        132 => Instruction::LoadDynamicField,
        133 => Instruction::StoreDynamicField,
        134 => Instruction::IndexLocalList(reader.u16("local slot")?),
        135 => Instruction::ListLengthLocal(reader.u16("local slot")?),
        136 => Instruction::NextLocalListIteration {
            list_slot: reader.u16("list local slot")?,
            index_slot: reader.u16("index local slot")?,
            item_slot: reader.u16("item local slot")?,
            exit: reader.target()?,
        },
        137 => Instruction::CheckNumericLoop {
            current_slot: reader.u16("current local slot")?,
            end_slot: reader.u16("end local slot")?,
            exit: reader.target()?,
        },
        138 => Instruction::AdvanceNumericLoop {
            slot: reader.u16("numeric loop local slot")?,
            step: DmNumberBits::from_f32(f32::from_bits(reader.u32("numeric loop step")?)),
            target: reader.target()?,
        },
        unknown => {
            return Err(ModuleCodecError::new(format!(
                "unknown instruction tag {unknown}"
            )));
        }
    };
    Ok(instruction)
}

fn type_predicate_tag(value: TypePredicateKind) -> u8 {
    match value {
        TypePredicateKind::IsNull => 0,
        TypePredicateKind::IsNum => 1,
        TypePredicateKind::IsPath => 2,
        TypePredicateKind::IsList => 3,
        TypePredicateKind::IsMovable => 4,
        TypePredicateKind::IsTurf => 5,
        TypePredicateKind::IsLoc => 6,
        TypePredicateKind::IsIcon => 7,
        TypePredicateKind::IsType => 8,
    }
}

fn decode_type_predicate(tag: u8) -> Result<TypePredicateKind, ModuleCodecError> {
    match tag {
        0 => Ok(TypePredicateKind::IsNull),
        1 => Ok(TypePredicateKind::IsNum),
        2 => Ok(TypePredicateKind::IsPath),
        3 => Ok(TypePredicateKind::IsList),
        4 => Ok(TypePredicateKind::IsMovable),
        5 => Ok(TypePredicateKind::IsTurf),
        6 => Ok(TypePredicateKind::IsLoc),
        7 => Ok(TypePredicateKind::IsIcon),
        8 => Ok(TypePredicateKind::IsType),
        _ => Err(ModuleCodecError::new(format!(
            "unknown type predicate tag {tag}"
        ))),
    }
}

fn compound_list_tag(value: CompoundListIndexOperator) -> u8 {
    match value {
        CompoundListIndexOperator::Add => 0,
        CompoundListIndexOperator::Subtract => 1,
        CompoundListIndexOperator::Multiply => 2,
        CompoundListIndexOperator::Divide => 3,
        CompoundListIndexOperator::Remainder => 4,
        CompoundListIndexOperator::FractionalRemainder => 5,
        CompoundListIndexOperator::BitAnd => 6,
        CompoundListIndexOperator::BitOr => 7,
        CompoundListIndexOperator::BitXor => 8,
        CompoundListIndexOperator::ShiftLeft => 9,
        CompoundListIndexOperator::ShiftRight => 10,
    }
}

fn decode_compound_list(tag: u8) -> Result<CompoundListIndexOperator, ModuleCodecError> {
    match tag {
        0 => Ok(CompoundListIndexOperator::Add),
        1 => Ok(CompoundListIndexOperator::Subtract),
        2 => Ok(CompoundListIndexOperator::Multiply),
        3 => Ok(CompoundListIndexOperator::Divide),
        4 => Ok(CompoundListIndexOperator::Remainder),
        5 => Ok(CompoundListIndexOperator::FractionalRemainder),
        6 => Ok(CompoundListIndexOperator::BitAnd),
        7 => Ok(CompoundListIndexOperator::BitOr),
        8 => Ok(CompoundListIndexOperator::BitXor),
        9 => Ok(CompoundListIndexOperator::ShiftLeft),
        10 => Ok(CompoundListIndexOperator::ShiftRight),
        _ => Err(ModuleCodecError::new(format!(
            "unknown list operator tag {tag}"
        ))),
    }
}

fn compound_assignment_tag(value: CompoundAssignmentOperator) -> u8 {
    match value {
        CompoundAssignmentOperator::Add => 0,
        CompoundAssignmentOperator::Subtract => 1,
        CompoundAssignmentOperator::Multiply => 2,
        CompoundAssignmentOperator::Divide => 3,
        CompoundAssignmentOperator::Remainder => 4,
        CompoundAssignmentOperator::FractionalRemainder => 5,
        CompoundAssignmentOperator::BitAnd => 6,
        CompoundAssignmentOperator::BitOr => 7,
        CompoundAssignmentOperator::BitXor => 8,
        CompoundAssignmentOperator::ShiftLeft => 9,
        CompoundAssignmentOperator::ShiftRight => 10,
    }
}

fn decode_compound_assignment(tag: u8) -> Result<CompoundAssignmentOperator, ModuleCodecError> {
    match tag {
        0 => Ok(CompoundAssignmentOperator::Add),
        1 => Ok(CompoundAssignmentOperator::Subtract),
        2 => Ok(CompoundAssignmentOperator::Multiply),
        3 => Ok(CompoundAssignmentOperator::Divide),
        4 => Ok(CompoundAssignmentOperator::Remainder),
        5 => Ok(CompoundAssignmentOperator::FractionalRemainder),
        6 => Ok(CompoundAssignmentOperator::BitAnd),
        7 => Ok(CompoundAssignmentOperator::BitOr),
        8 => Ok(CompoundAssignmentOperator::BitXor),
        9 => Ok(CompoundAssignmentOperator::ShiftLeft),
        10 => Ok(CompoundAssignmentOperator::ShiftRight),
        _ => Err(ModuleCodecError::new(format!(
            "unknown assignment operator tag {tag}"
        ))),
    }
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    fn i8(&mut self, value: i8) {
        self.u8(value as u8);
    }
    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn boolean(&mut self, value: bool) {
        self.u8(u8::from(value));
    }
    fn len(&mut self, value: usize, max: usize, label: &str) -> Result<(), ModuleCodecError> {
        if value > max || value > u32::MAX as usize {
            return Err(ModuleCodecError::new(format!(
                "{label} exceeds format limit"
            )));
        }
        self.u32(value as u32);
        Ok(())
    }
    fn string(&mut self, value: &str) -> Result<(), ModuleCodecError> {
        self.len(value.len(), MAX_STRING_BYTES, "string length")?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }
    fn optional_string(&mut self, value: Option<&str>) -> Result<(), ModuleCodecError> {
        self.boolean(value.is_some());
        if let Some(value) = value {
            self.string(value)?;
        }
        Ok(())
    }
    fn optional_strings(&mut self, values: &[Option<String>]) -> Result<(), ModuleCodecError> {
        self.len(
            values.len(),
            MAX_VECTOR_ELEMENTS,
            "optional string vector length",
        )?;
        for value in values {
            self.optional_string(value.as_deref())?;
        }
        Ok(())
    }
    fn target(&mut self, value: usize) -> Result<(), ModuleCodecError> {
        self.u32(
            u32::try_from(value)
                .map_err(|_| ModuleCodecError::new("instruction target exceeds format limit"))?,
        );
        Ok(())
    }
    fn optional_u16(&mut self, value: Option<u16>) {
        self.boolean(value.is_some());
        if let Some(value) = value {
            self.u16(value);
        }
    }
    fn optional_procedure(&mut self, value: Option<ProcedureId>) {
        self.boolean(value.is_some());
        if let Some(value) = value {
            self.u32(value.0);
        }
    }
    fn fields(&mut self, values: &[FieldName]) -> Result<(), ModuleCodecError> {
        self.len(values.len(), MAX_VECTOR_ELEMENTS, "field vector length")?;
        for value in values {
            self.string(value.as_str())?;
        }
        Ok(())
    }
    fn u16s(&mut self, values: &[u16]) -> Result<(), ModuleCodecError> {
        self.len(values.len(), MAX_VECTOR_ELEMENTS, "u16 vector length")?;
        for value in values {
            self.u16(*value);
        }
        Ok(())
    }
    fn booleans(&mut self, values: &[bool]) -> Result<(), ModuleCodecError> {
        self.len(values.len(), MAX_VECTOR_ELEMENTS, "boolean vector length")?;
        for value in values {
            self.boolean(*value);
        }
        Ok(())
    }
    fn list_entry_kinds(&mut self, values: &[ListEntryKind]) -> Result<(), ModuleCodecError> {
        self.len(
            values.len(),
            MAX_VECTOR_ELEMENTS,
            "list-entry vector length",
        )?;
        for value in values {
            self.u8(match value {
                ListEntryKind::Positional => 0,
                ListEntryKind::Associative => 1,
            });
        }
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
    field_names: HashMap<String, FieldName>,
    type_paths: HashMap<String, TypePath>,
    texts: HashMap<Arc<str>, Arc<str>>,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            field_names: HashMap::new(),
            type_paths: HashMap::new(),
            texts: HashMap::new(),
        }
    }
    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }
    fn take(&mut self, length: usize, label: &str) -> Result<&'a [u8], ModuleCodecError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| ModuleCodecError::new(format!("{label} length overflow")))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| ModuleCodecError::new(format!("truncated {label}")))?;
        self.position = end;
        Ok(value)
    }
    fn u8(&mut self, label: &str) -> Result<u8, ModuleCodecError> {
        Ok(self.take(1, label)?[0])
    }
    fn i8(&mut self, label: &str) -> Result<i8, ModuleCodecError> {
        Ok(self.u8(label)? as i8)
    }
    fn u16(&mut self, label: &str) -> Result<u16, ModuleCodecError> {
        Ok(u16::from_le_bytes(
            self.take(2, label)?.try_into().expect("exact width"),
        ))
    }
    fn u32(&mut self, label: &str) -> Result<u32, ModuleCodecError> {
        Ok(u32::from_le_bytes(
            self.take(4, label)?.try_into().expect("exact width"),
        ))
    }
    fn u64(&mut self, label: &str) -> Result<u64, ModuleCodecError> {
        Ok(u64::from_le_bytes(
            self.take(8, label)?.try_into().expect("exact width"),
        ))
    }
    fn usize64(&mut self, label: &str) -> Result<usize, ModuleCodecError> {
        usize::try_from(self.u64(label)?)
            .map_err(|_| ModuleCodecError::new(format!("{label} exceeds host limit")))
    }
    fn boolean(&mut self, label: &str) -> Result<bool, ModuleCodecError> {
        match self.u8(label)? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(ModuleCodecError::new(format!("invalid {label} {value}"))),
        }
    }
    fn len(&mut self, max: usize, label: &str) -> Result<usize, ModuleCodecError> {
        let value = self.u32(label)? as usize;
        if value > max {
            Err(ModuleCodecError::new(format!("{label} exceeds limit")))
        } else {
            Ok(value)
        }
    }
    fn len_with_min(
        &mut self,
        max: usize,
        label: &str,
        minimum_item_bytes: usize,
    ) -> Result<usize, ModuleCodecError> {
        let value = self.len(max, label)?;
        if value > self.remaining() / minimum_item_bytes {
            Err(ModuleCodecError::new(format!("truncated {label}")))
        } else {
            Ok(value)
        }
    }
    fn string(&mut self, label: &str) -> Result<String, ModuleCodecError> {
        let length = self.len(MAX_STRING_BYTES, &format!("{label} length"))?;
        let raw = self.take(length, label)?;
        String::from_utf8(raw.to_vec())
            .map_err(|_| ModuleCodecError::new(format!("{label} is not UTF-8")))
    }
    fn text(&mut self) -> Result<Arc<str>, ModuleCodecError> {
        let raw = self.string("text constant")?;
        if let Some(existing) = self.texts.get(raw.as_str()) {
            return Ok(Arc::clone(existing));
        }
        let text = Arc::<str>::from(raw);
        self.texts.insert(Arc::clone(&text), Arc::clone(&text));
        Ok(text)
    }
    fn optional_string(&mut self) -> Result<Option<String>, ModuleCodecError> {
        if self.boolean("optional string flag")? {
            Ok(Some(self.string("optional string")?))
        } else {
            Ok(None)
        }
    }
    fn optional_strings(&mut self) -> Result<Vec<Option<String>>, ModuleCodecError> {
        let length = self.len_with_min(MAX_VECTOR_ELEMENTS, "optional string vector length", 1)?;
        (0..length).map(|_| self.optional_string()).collect()
    }
    fn target(&mut self) -> Result<usize, ModuleCodecError> {
        Ok(self.u32("instruction target")? as usize)
    }
    fn optional_u16(&mut self) -> Result<Option<u16>, ModuleCodecError> {
        if self.boolean("optional u16 flag")? {
            Ok(Some(self.u16("optional u16")?))
        } else {
            Ok(None)
        }
    }
    fn procedure(&mut self, procedure_count: usize) -> Result<ProcedureId, ModuleCodecError> {
        let value = ProcedureId(self.u32("procedure id")?);
        validate_procedure(value, procedure_count)?;
        Ok(value)
    }
    fn optional_procedure(
        &mut self,
        procedure_count: usize,
    ) -> Result<Option<ProcedureId>, ModuleCodecError> {
        if self.boolean("optional procedure flag")? {
            Ok(Some(self.procedure(procedure_count)?))
        } else {
            Ok(None)
        }
    }
    fn field(&mut self) -> Result<FieldName, ModuleCodecError> {
        let raw = self.string("field name")?;
        if let Some(existing) = self.field_names.get(&raw) {
            return Ok(existing.clone());
        }
        let field = FieldName::parse(&raw)
            .map_err(|error| ModuleCodecError::new(format!("invalid field name: {error}")))?;
        self.field_names.insert(raw, field.clone());
        Ok(field)
    }
    fn fields(&mut self) -> Result<Vec<FieldName>, ModuleCodecError> {
        let length = self.len_with_min(MAX_VECTOR_ELEMENTS, "field vector length", 4)?;
        (0..length).map(|_| self.field()).collect()
    }
    fn type_path(&mut self) -> Result<TypePath, ModuleCodecError> {
        let raw = self.string("type path")?;
        self.intern_type_path(raw, "type path")
    }
    fn intern_type_path(&mut self, raw: String, label: &str) -> Result<TypePath, ModuleCodecError> {
        if let Some(existing) = self.type_paths.get(&raw) {
            return Ok(existing.clone());
        }
        let path = TypePath::parse(&raw)
            .map_err(|error| ModuleCodecError::new(format!("invalid {label}: {error}")))?;
        self.type_paths.insert(raw, path.clone());
        Ok(path)
    }
    fn u16s(&mut self) -> Result<Vec<u16>, ModuleCodecError> {
        let length = self.len_with_min(MAX_VECTOR_ELEMENTS, "u16 vector length", 2)?;
        (0..length).map(|_| self.u16("u16 vector item")).collect()
    }
    fn booleans(&mut self) -> Result<Vec<bool>, ModuleCodecError> {
        let length = self.len_with_min(MAX_VECTOR_ELEMENTS, "boolean vector length", 1)?;
        (0..length)
            .map(|_| self.boolean("boolean vector item"))
            .collect()
    }
    fn list_entry_kinds(&mut self) -> Result<Vec<ListEntryKind>, ModuleCodecError> {
        let length = self.len_with_min(MAX_VECTOR_ELEMENTS, "list-entry vector length", 1)?;
        (0..length)
            .map(|_| match self.u8("list-entry kind")? {
                0 => Ok(ListEntryKind::Positional),
                1 => Ok(ListEntryKind::Associative),
                tag => Err(ModuleCodecError::new(format!(
                    "unknown list-entry kind {tag}"
                ))),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use dm_syntax::parse;

    use super::*;
    use crate::{
        ProcedureSpec, Value, compile_module, compile_module_specs_selective, execute_module,
    };

    fn field(name: &str) -> FieldName {
        FieldName::parse(name).unwrap()
    }

    fn path(name: &str) -> TypePath {
        TypePath::parse(name).unwrap()
    }

    fn all_instruction_variants() -> Vec<Instruction> {
        let names = vec![Some("named".to_owned()), None];
        let fields: Arc<[FieldName]> = Arc::from([field("one"), field("two")]);
        vec![
            Instruction::PushNull,
            Instruction::PushNumber(DmNumberBits::from_f32(-12.5)),
            Instruction::PushText(Arc::from("text")),
            Instruction::PushFile("asset.dmi".to_owned()),
            Instruction::PushTypePath(path("/datum/example")),
            Instruction::AddressLocal(1),
            Instruction::LoadLocalRaw(2),
            Instruction::MakeModifiedTypePath { fields },
            Instruction::AllocateDatum {
                argument_count: 2,
                argument_names: names.clone(),
            },
            Instruction::AllocateCurrentDatum { argument_count: 3 },
            Instruction::MakeRegex { argument_count: 2 },
            Instruction::MakeMutableAppearance { argument_count: 4 },
            Instruction::MakeMatrix { argument_count: 6 },
            Instruction::MakeVector { argument_count: 3 },
            Instruction::ReplaceText {
                argument_count: 5,
                exact: true,
                character_indices: false,
            },
            Instruction::CopyText {
                argument_count: 3,
                character_indices: true,
            },
            Instruction::StandardBuiltin {
                name: "rgb".to_owned(),
                argument_count: 2,
                argument_names: names.clone(),
            },
            Instruction::NativeSrcMethod {
                name: "Cut".to_owned(),
                argument_count: 2,
            },
            Instruction::Output,
            Instruction::Input,
            Instruction::ExternalCall { argument_count: 2 },
            Instruction::Animate {
                argument_names: names.clone(),
                expanded_indices: vec![1],
            },
            Instruction::MakeFilter {
                argument_names: names.clone(),
                expanded_indices: vec![0],
            },
            Instruction::InitialField(field("value")),
            Instruction::InitialDynamicField,
            Instruction::Block { argument_count: 2 },
            Instruction::Rand { argument_count: 2 },
            Instruction::Roll { argument_count: 1 },
            Instruction::Pick {
                weighted: vec![false, true],
            },
            Instruction::PickExpandedArguments,
            Instruction::PrepareRhsFirstIndexAssignment,
            Instruction::LoadDeclaredField(field("cell")),
            Instruction::LoadDynamicField,
            Instruction::StoreDynamicField,
            Instruction::Prob,
            Instruction::Round { argument_count: 2 },
            Instruction::Length,
            Instruction::Ref,
            Instruction::GetStep,
            Instruction::GetStepTowards,
            Instruction::Range { argument_count: 2 },
            Instruction::TypesOf { argument_count: 1 },
            Instruction::HasCall,
            Instruction::TypeInstances(path("/datum/example")),
            Instruction::TypePredicate {
                kind: TypePredicateKind::IsType,
                argument_count: 2,
            },
            Instruction::MakeList(2),
            Instruction::MakeArray(3),
            Instruction::MakeArgs,
            Instruction::MakeListEntries(vec![
                ListEntryKind::Positional,
                ListEntryKind::Associative,
            ]),
            Instruction::MakeAssociativeListEntries(vec![ListEntryKind::Associative]),
            Instruction::IndexList,
            Instruction::IndexLocalList(4),
            Instruction::ListLengthLocal(4),
            Instruction::NextLocalListIteration {
                list_slot: 4,
                index_slot: 5,
                item_slot: 6,
                exit: 9,
            },
            Instruction::CheckNumericLoop {
                current_slot: 5,
                end_slot: 6,
                exit: 9,
            },
            Instruction::AdvanceNumericLoop {
                slot: 5,
                step: DmNumberBits::from_f32(8.0),
                target: 9,
            },
            Instruction::SetListIndex,
            Instruction::SetListIndexKeep,
            Instruction::CompoundListIndex(CompoundListIndexOperator::Add),
            Instruction::CompoundListIndexKeep(CompoundListIndexOperator::ShiftRight),
            Instruction::ListLength,
            Instruction::PrepareIteration,
            Instruction::IterationTypeFilter(path("/atom/movable")),
            Instruction::LoadLocal(3),
            Instruction::StoreLocal(3),
            Instruction::LoadStaticLocalOrJump { slot: 4, target: 0 },
            Instruction::InitializeStaticLocal(4),
            Instruction::LoadSrc,
            Instruction::StoreSrc,
            Instruction::LoadUsr,
            Instruction::StoreUsr,
            Instruction::LoadCaller,
            Instruction::LoadField(field("alpha")),
            Instruction::StoreField(field("alpha")),
            Instruction::StoreFieldKeep(field("alpha")),
            Instruction::LoadGlobal(field("world")),
            Instruction::LoadGlobalVars,
            Instruction::LoadDatumVars,
            Instruction::LoadInitialGlobal(field("world")),
            Instruction::StoreGlobal(field("world")),
            Instruction::MutateLocal {
                slot: 1,
                delta: -1,
                prefix: true,
            },
            Instruction::MutateField {
                name: field("alpha"),
                delta: 1,
                prefix: false,
            },
            Instruction::MutateGlobal {
                name: field("counter"),
                delta: -1,
                prefix: true,
            },
            Instruction::MutateListIndex {
                delta: 1,
                prefix: false,
            },
            Instruction::MutateResult {
                delta: -1,
                prefix: true,
            },
            Instruction::Duplicate,
            Instruction::LoadResult,
            Instruction::StoreResult,
            Instruction::Pop,
            Instruction::Crash,
            Instruction::BeginTry {
                catch: 0,
                end: 0,
                local: Some(2),
            },
            Instruction::EndTry,
            Instruction::Throw,
            Instruction::Locate { argument_count: 1 },
            Instruction::LocateIn { argument_count: 2 },
            Instruction::CompoundAssignment(CompoundAssignmentOperator::BitOr),
            Instruction::Add,
            Instruction::Subtract,
            Instruction::Multiply,
            Instruction::Power,
            Instruction::Divide,
            Instruction::Remainder,
            Instruction::FractionalRemainder,
            Instruction::BitAnd,
            Instruction::BitOr,
            Instruction::BitXor,
            Instruction::ShiftLeft,
            Instruction::ShiftRight,
            Instruction::BitNot,
            Instruction::Negate,
            Instruction::Not,
            Instruction::Equal,
            Instruction::NotEqual,
            Instruction::Equivalent,
            Instruction::NotEquivalent,
            Instruction::Compare,
            Instruction::Contains,
            Instruction::Less,
            Instruction::LessEqual,
            Instruction::Greater,
            Instruction::GreaterEqual,
            Instruction::And,
            Instruction::Or,
            Instruction::JumpIfNull(0),
            Instruction::JumpIfFalse(0),
            Instruction::Jump(0),
            Instruction::JumpIfArgumentSupplied {
                parameter: 1,
                target: 0,
            },
            Instruction::Call {
                procedure: ProcedureId(0),
                argument_count: 2,
                argument_names: names.clone(),
            },
            Instruction::CallCurrent {
                argument_count: Some(2),
            },
            Instruction::CallParent {
                procedure: Some(ProcedureId(0)),
                argument_count: None,
            },
            Instruction::CallDynamic {
                static_selector: Some("run".to_owned()),
                argument_count: 2,
                argument_names: names.clone(),
                null_receiver_is_global: false,
            },
            Instruction::ExpandArgumentLists {
                argument_count: 2,
                argument_names: names,
                expanded_indices: vec![1],
            },
            Instruction::Return,
            Instruction::Spawn { entry: 0 },
            Instruction::Sleep,
            Instruction::LogicalOrEmptyListLocal(3),
            Instruction::LogicalOrEmptyListGlobal(field("world")),
            Instruction::LogicalOrEmptyListField(field("alpha")),
            Instruction::LogicalOrEmptyListIndex,
        ]
    }

    fn module_with(instructions: Vec<Instruction>) -> Module {
        let instruction_count = instructions.len();
        Module {
            identity: next_module_identity(),
            procedures: vec![Arc::new(Program {
                wait_for: false,
                parameter_count: 2,
                parameter_names: vec!["first".to_owned(), "second".to_owned()],
                verb_parameter_types: vec![VerbParameterType::Text, VerbParameterType::Number],
                verb_name: None,
                local_count: 5,
                instructions,
                source_spans: vec![SourceSpan::new(10, 20); instruction_count],
            })],
            paths: vec!["/datum/example/proc/run".to_owned()],
            names: HashMap::from([("/datum/example/proc/run".to_owned(), ProcedureId(0))]),
            dynamic_names: HashMap::from([("/datum/example/proc/run".to_owned(), ProcedureId(0))]),
            deferred: Arc::new(HashMap::new()),
            procedure_types: vec![path("/datum/example/proc/run")],
            initializer_call_names: None,
        }
    }

    fn first_instruction_offset(bytes: &[u8]) -> usize {
        let mut reader = Reader::new(bytes);
        reader.take(12, "header").unwrap();
        assert_eq!(reader.u32("procedures").unwrap(), 1);
        reader.string("path").unwrap();
        let type_count = reader.u32("types").unwrap();
        for _ in 0..type_count {
            reader.string("type").unwrap();
        }
        reader.boolean("wait").unwrap();
        reader.u32("parameters").unwrap();
        let names = reader.u32("names").unwrap();
        for _ in 0..names {
            reader.string("name").unwrap();
        }
        let parameter_types = reader.u32("verb parameter types").unwrap();
        reader
            .take(parameter_types as usize, "verb parameter type entries")
            .unwrap();
        let has_verb_name = reader.boolean("verb name presence").unwrap();
        if has_verb_name {
            reader.string("verb name").unwrap();
        }
        reader.u32("locals").unwrap();
        reader.u32("instructions").unwrap();
        reader.position
    }

    #[test]
    fn every_instruction_variant_round_trips_deterministically() {
        let instructions = all_instruction_variants();
        assert_eq!(instructions.len(), INSTRUCTION_TAG_COUNT as usize);
        let module = module_with(instructions);

        let first = module.encode_portable().unwrap();
        let second = module.encode_portable().unwrap();
        assert_eq!(first, second);
        let mut cache_polluted = module.clone();
        assert!(Arc::ptr_eq(
            &module.procedures[0],
            &cache_polluted.procedures[0]
        ));
        cache_polluted.names.clear();
        cache_polluted.dynamic_names.clear();
        cache_polluted.initializer_call_names = Some(crate::InitializerCallNameIndex {
            names: Arc::new(HashMap::from([("stale".to_owned(), ProcedureId(0))])),
            module_names_scanned: 99,
        });
        assert_eq!(cache_polluted.encode_portable().unwrap(), first);
        let decoded = Module::decode_portable(&first).unwrap();

        assert_eq!(decoded, module);
        assert_ne!(decoded.identity.0, module.identity.0);
        assert!(decoded.initializer_call_names.is_none());
        assert_eq!(
            decoded.procedure_id("/datum/example/proc/run"),
            Some(ProcedureId(0))
        );
        assert_eq!(
            decoded.effective_procedure_id("/datum/example/proc/run"),
            Some(ProcedureId(0))
        );
    }

    #[test]
    fn reopened_name_indexes_are_reconstructed_with_latest_dynamic_target() {
        let mut module = module_with(vec![Instruction::Return]);
        module.procedures.push(module.procedures[0].clone());
        module.paths = vec![
            "/datum/example/proc/run@1".to_owned(),
            "/datum/example/proc/run@2".to_owned(),
        ];
        module.names = HashMap::from([
            (module.paths[0].clone(), ProcedureId(0)),
            (module.paths[1].clone(), ProcedureId(1)),
        ]);
        module.dynamic_names =
            HashMap::from([("/datum/example/proc/run".to_owned(), ProcedureId(1))]);

        let decoded = Module::decode_portable(&module.encode_portable().unwrap()).unwrap();
        assert_eq!(decoded.procedure_id(&module.paths[0]), Some(ProcedureId(0)));
        assert_eq!(decoded.procedure_id(&module.paths[1]), Some(ProcedureId(1)));
        assert_eq!(
            decoded.effective_procedure_id("/datum/example/proc/run"),
            Some(ProcedureId(1))
        );
    }

    #[test]
    fn decoded_module_preserves_real_dynamic_dispatch() {
        let syntax = parse(
            "/datum/base/proc/value()\n\treturn 1\n/datum/child/proc/value()\n\treturn 2\n/proc/main()\n\tvar/datum/child/item = new\n\treturn item.value()\n",
        )
        .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();
        let bytes = module.encode_portable().unwrap();
        let decoded = Module::decode_portable(&bytes).unwrap();
        let main = decoded.procedure_id("/proc/main").unwrap();

        assert_eq!(execute_module(&decoded, main, &[]), Ok(Value::number(2.0)));
    }

    #[test]
    fn compiled_control_flow_targets_are_valid_portable_references() {
        let syntax = parse(
            "/proc/run(should_throw)\n\tvar/static/result = 1\n\ttry\n\t\tif (should_throw)\n\t\t\tthrow 5\n\t\tresult = 2\n\tcatch(var/error)\n\t\tresult = error + 10\n\tspawn(1)\n\t\tresult = 20\n\treturn result\n",
        )
        .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();

        let bytes = module.encode_portable().unwrap();
        assert_eq!(Module::decode_portable(&bytes).unwrap(), module);
    }

    #[test]
    fn both_returning_if_else_omits_dead_program_end_jump_and_round_trips() {
        let syntax =
            parse("/proc/run(condition)\n\tif (condition)\n\t\treturn 7\n\telse\n\t\treturn 9\n")
                .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();
        let run = module.procedure_id("/proc/run").unwrap();
        let program = module.procedure(run).unwrap();
        assert!(!program.instructions.iter().any(
            |instruction| matches!(instruction, Instruction::Jump(target) if *target >= program.instructions.len())
        ));

        let bytes = module
            .encode_portable()
            .expect("compiled control flow is valid");
        let decoded = Module::decode_portable(&bytes).expect("valid control flow should decode");
        assert_eq!(
            execute_module(&decoded, run, &[Value::number(1.0)]),
            Ok(Value::number(7.0)),
        );
        assert_eq!(
            execute_module(&decoded, run, &[Value::number(0.0)]),
            Ok(Value::number(9.0)),
        );
    }

    #[test]
    fn production_safe_json_try_catch_omits_dead_end_jump_and_round_trips() {
        let syntax = parse(
            "/proc/safe_json_decode(data)\n\ttry\n\t\treturn json_decode(data)\n\tcatch\n\t\treturn null\n",
        )
        .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();
        let run = module.procedure_id("/proc/safe_json_decode").unwrap();
        let program = module.procedure(run).unwrap();
        assert!(!program.instructions.iter().any(
            |instruction| matches!(instruction, Instruction::Jump(target) if *target >= program.instructions.len())
        ));

        let bytes = module
            .encode_portable()
            .expect("compiled control flow is valid");
        assert_eq!(Module::decode_portable(&bytes).unwrap(), module);
    }

    #[test]
    fn jump_to_or_past_program_end_is_rejected_by_encoder_and_decoder() {
        let invalid_end = module_with(vec![Instruction::Jump(1)]);
        let error = invalid_end.encode_portable().unwrap_err().to_string();
        assert!(error.contains("procedure 0 /datum/example/proc/run"));
        assert!(error.contains("jump target 1 is outside 1 instructions"));

        let invalid = module_with(vec![Instruction::Jump(2)]);
        assert!(
            invalid
                .encode_portable()
                .unwrap_err()
                .to_string()
                .contains("jump target 2 is outside 1 instructions")
        );

        let valid = module_with(vec![Instruction::Jump(0)]);
        let mut bytes = valid.encode_portable().unwrap();
        let jump_tag = first_instruction_offset(&bytes);
        assert_eq!(bytes[jump_tag], 115);
        bytes[jump_tag + 1..jump_tag + 5].copy_from_slice(&1u32.to_le_bytes());
        let error = Module::decode_portable(&bytes).unwrap_err().to_string();
        assert!(error.contains("procedure 0 /datum/example/proc/run"));
        assert!(error.contains("jump target 1 is outside 1 instructions"));
    }

    #[test]
    fn corrupt_header_tag_length_id_jump_and_span_are_rejected() {
        let minimal = module_with(vec![Instruction::Return]);
        let bytes = minimal.encode_portable().unwrap();

        let mut corrupt = bytes.clone();
        corrupt[0] ^= 0xff;
        assert!(
            Module::decode_portable(&corrupt)
                .unwrap_err()
                .to_string()
                .contains("header")
        );
        assert!(Module::decode_portable(&bytes[..bytes.len() - 1]).is_err());

        let instruction_tag_offset = first_instruction_offset(&bytes);
        let mut unknown = bytes.clone();
        unknown[instruction_tag_offset] = INSTRUCTION_TAG_COUNT;
        assert!(
            Module::decode_portable(&unknown)
                .unwrap_err()
                .to_string()
                .contains("unknown instruction")
        );

        let mut oversized = bytes.clone();
        oversized[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(
            Module::decode_portable(&oversized)
                .unwrap_err()
                .to_string()
                .contains("procedure count")
        );

        let invalid_id = module_with(vec![Instruction::Call {
            procedure: ProcedureId(0),
            argument_count: 0,
            argument_names: Vec::new(),
        }]);
        let mut invalid_id_bytes = invalid_id.encode_portable().unwrap();
        let call_tag = first_instruction_offset(&invalid_id_bytes);
        assert_eq!(invalid_id_bytes[call_tag], 117);
        invalid_id_bytes[call_tag + 1..call_tag + 5].copy_from_slice(&9u32.to_le_bytes());
        assert!(
            Module::decode_portable(&invalid_id_bytes)
                .unwrap_err()
                .to_string()
                .contains("invalid procedure id")
        );

        let invalid_jump = module_with(vec![Instruction::Jump(0)]);
        let mut invalid_jump_bytes = invalid_jump.encode_portable().unwrap();
        let jump_tag = first_instruction_offset(&invalid_jump_bytes);
        assert_eq!(invalid_jump_bytes[jump_tag], 115);
        invalid_jump_bytes[jump_tag + 1..jump_tag + 5].copy_from_slice(&9u32.to_le_bytes());
        assert!(
            Module::decode_portable(&invalid_jump_bytes)
                .unwrap_err()
                .to_string()
                .contains("jump target")
        );

        let mut inverted_span = bytes.clone();
        let span_start = inverted_span.len() - 16;
        inverted_span[span_start..span_start + 8].copy_from_slice(&30u64.to_le_bytes());
        inverted_span[span_start + 8..].copy_from_slice(&20u64.to_le_bytes());
        assert!(
            Module::decode_portable(&inverted_span)
                .unwrap_err()
                .to_string()
                .contains("inverted")
        );

        let mut bad_span_count = bytes.clone();
        let span_count = instruction_tag_offset + 1;
        bad_span_count[span_count..span_count + 4].copy_from_slice(&2u32.to_le_bytes());
        assert!(
            Module::decode_portable(&bad_span_count)
                .unwrap_err()
                .to_string()
                .contains("source span")
        );
    }

    #[test]
    fn deferred_module_must_be_made_fully_eager_before_encoding() {
        let syntax = parse("/proc/lazy()\n\treturn 1\n").unwrap();
        let specs = [ProcedureSpec {
            path: "/proc/lazy".to_owned(),
            definition: &syntax.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        }];
        let module =
            compile_module_specs_selective(&specs, &[BTreeMap::new()], &BTreeSet::new()).unwrap();
        assert_eq!(module.deferred_procedure_count(), 1);
        assert!(
            module
                .encode_portable()
                .unwrap_err()
                .to_string()
                .contains("fully eager")
        );
    }
}

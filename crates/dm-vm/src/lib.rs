//! Portable stack bytecode and the deterministic reference interpreter.

#![cfg_attr(not(test), deny(missing_docs))]

mod builtins;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use builtins::{
    execute_list_binary_operator, execute_list_compound_operator, execute_list_method,
    execute_standard_builtin, is_subtype, standard_builtin_arity,
};

use dm_core::{DmNumberBits, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};
use dm_syntax::{Definition, DefinitionKind, SourceLine};
pub use dm_value::Value;
use dm_value::{FieldName, ListId, TypePath, ValueError, ValueHeap};

/// One instruction in the portable reference bytecode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Instruction {
    /// Pushes `null`.
    PushNull,
    /// Pushes a numeric constant.
    PushNumber(DmNumberBits),
    /// Pushes a text constant.
    PushText(String),
    /// Pushes a canonical absolute DM type path.
    PushTypePath(TypePath),
    /// Pops constructor arguments followed by a type path, allocates a datum,
    /// and pushes its stable handle. Constructor dispatch is intentionally
    /// deferred to the lifecycle layer; this establishes allocation identity.
    AllocateDatum {
        /// Number of already-evaluated constructor arguments to discard.
        argument_count: u16,
    },
    /// Allocates another datum of the current `src` datum's runtime type.
    ///
    /// This is the headless interpretation of an unqualified `new(...)`.
    AllocateCurrentDatum {
        /// Number of already-evaluated constructor arguments to discard.
        argument_count: u16,
    },
    /// Constructs BYOND's built-in `/regex` datum from a pattern and optional
    /// flags value. The resulting datum preserves its source pattern in
    /// `text` and its flags in `flags`, allowing later regex-aware builtins to
    /// consume the same object identity rather than a fabricated text value.
    MakeRegex {
        /// Number of already-evaluated constructor arguments (one or two).
        argument_count: u8,
    },
    /// Constructs BYOND's built-in `/mutable_appearance` datum.
    ///
    /// Constructor arguments are discarded after evaluation; rendering and
    /// appearance inheritance are outside this headless VM's scope.
    MakeMutableAppearance {
        /// Number of already-evaluated constructor arguments to discard.
        argument_count: u16,
    },
    /// Replaces every matching text fragment in a bounded 1-based region.
    ///
    /// This implements the `replacetext` builtin family. `exact` selects
    /// BYOND's `Ex` case-sensitive form; `character_indices` selects the
    /// `_char` family rather than the legacy byte-indexed spelling.
    ReplaceText {
        /// Number of supplied arguments (three through five).
        argument_count: u8,
        /// Whether matches are case-sensitive.
        exact: bool,
        /// Whether optional bounds count Unicode scalar values.
        character_indices: bool,
    },
    /// Copies a bounded section of text using BYOND's 1-based positions.
    CopyText {
        /// Number of supplied arguments (one through three).
        argument_count: u8,
        /// Whether positions count Unicode scalar values rather than bytes.
        character_indices: bool,
    },
    /// Executes a documented BYOND global procedure handled by the native runtime.
    StandardBuiltin {
        /// Canonical global procedure name.
        name: String,
        /// Number of already-evaluated arguments.
        argument_count: u16,
    },
    /// Reads a field's compile-time initial value from a datum or type path.
    InitialField(FieldName),
    /// Enumerates every materialized turf in an inclusive 3D rectangular block.
    Block {
        /// Number of supplied arguments: two turfs, or three through six coordinates.
        argument_count: u8,
    },
    /// Produces a deterministic pseudo-random integer in an inclusive range.
    Rand {
        /// Number of supplied bounds (one or two).
        argument_count: u8,
    },
    /// Selects one deterministic pseudo-random candidate, optionally using
    /// BYOND's `weight; candidate` spelling.
    Pick {
        /// Whether each candidate has a preceding numeric weight on the stack.
        weighted: Vec<bool>,
    },
    /// Evaluates BYOND's percentage chance predicate deterministically.
    Prob,
    /// Rounds a number using BYOND's legacy one-argument floor form or its
    /// two-argument nearest-multiple form.
    Round {
        /// Number of supplied arguments (one or two).
        argument_count: u8,
    },
    /// Returns the legacy BYOND length of text or a list.
    ///
    /// Text length uses the same UTF-8 byte indexing as ordinary DM text
    /// operations; lists return their number of entries.
    Length,
    /// Returns a stable BYOND-style text reference for a heap datum or list.
    Ref,
    /// Returns the turf one BYOND direction away from an atom or turf.
    ///
    /// The headless world model identifies turfs by their materialized `x`,
    /// `y`, and `z` fields, so this instruction searches the live turf set
    /// deterministically rather than requiring a renderer or map facade.
    GetStep,
    /// Returns every materialized atom in BYOND tile range around a center.
    ///
    /// Range uses Chebyshev distance and includes the center tile.  The
    /// headless VM derives membership from live atom `x`, `y`, and `z`
    /// fields, which keeps it useful during map lifecycle execution without a
    /// renderer or separate spatial index.
    Range {
        /// Number of explicitly supplied arguments (one or two).
        argument_count: u8,
    },
    /// Returns a list containing a type path and every registered descendant.
    ///
    /// The catalog belongs to [`ExecutionState`], allowing the runtime image
    /// to provide the complete object tree without coupling bytecode to its
    /// materialization implementation.
    TypesOf,
    /// Classifies a value using BYOND's simple predicate builtins.
    ///
    /// `istype` additionally accepts an optional target type path and treats
    /// descendants of that path as matches.
    TypePredicate {
        /// Predicate to apply.
        kind: TypePredicateKind,
        /// Number of already-evaluated arguments.
        argument_count: u8,
    },
    /// Pops `count` values, allocates a list, and pushes its stable handle.
    ///
    /// Values retain their original source order in 1-based list positions.
    MakeList(u16),
    /// Allocates the current procedure's implicit `args` list.
    ///
    /// The list contains every value supplied to this call in positional
    /// order, including values beyond the declared parameter list.
    MakeArgs,
    /// Builds a list whose positional values and associative keys may intermix.
    MakeListEntries(Vec<ListEntryKind>),
    /// Pops a numeric 1-based index and a list handle, then pushes the entry.
    IndexList,
    /// Pops a value, index/key, and list handle and updates that list.
    SetListIndex,
    /// Like [`Self::SetListIndex`], but leaves the stored value on the stack.
    SetListIndexKeep,
    /// Pops a value, index/key, and list handle, applies a numeric operation to
    /// the current indexed value and the supplied value, then updates that
    /// list.
    CompoundListIndex(CompoundListIndexOperator),
    /// Pops a list handle and pushes its deterministic iteration length.
    ListLength,
    /// Pushes a local value.
    LoadLocal(u16),
    /// Pops into a local slot.
    StoreLocal(u16),
    /// Pushes the current frame's `src` value.
    LoadSrc,
    /// Pushes the current frame's `usr` value.
    LoadUsr,
    /// Pops a datum receiver and pushes one named field.
    LoadField(FieldName),
    /// Pops a value and datum receiver, then writes one named field.
    StoreField(FieldName),
    /// Stores one datum field while preserving the assigned value on the stack.
    StoreFieldKeep(FieldName),
    /// Pushes one persistent runtime global.
    LoadGlobal(FieldName),
    /// Pops and stores one persistent runtime global.
    StoreGlobal(FieldName),
    /// Clones the top stack value.
    Duplicate,
    /// Pushes the current procedure's special `.` return value.
    LoadResult,
    /// Pops into the current procedure's special `.` return value.
    StoreResult,
    /// Discards the top stack value.
    Pop,
    /// Pops a diagnostic value and terminates execution with a runtime error.
    Crash,
    /// Discards `locate` arguments and yields null when no world locator is
    /// available to the headless interpreter.
    Locate {
        /// Number of already-evaluated locator arguments to discard.
        argument_count: u16,
    },
    /// Discards `locate` arguments plus the search container, then yields null
    /// when no world locator is available to the headless interpreter.
    LocateIn {
        /// Number of already-evaluated locator arguments to discard.
        argument_count: u16,
    },
    /// Executes a compound assignment while preserving type-specific mutation semantics.
    CompoundAssignment(CompoundAssignmentOperator),
    /// Numeric/list/text addition.
    Add,
    /// Numeric subtraction.
    Subtract,
    /// Numeric multiplication.
    Multiply,
    /// Numeric exponentiation (`**`).
    Power,
    /// Numeric division.
    Divide,
    /// Numeric remainder.
    Remainder,
    /// 32-bit integer bitwise conjunction.
    BitAnd,
    /// 32-bit integer bitwise disjunction.
    BitOr,
    /// 32-bit integer bitwise exclusive disjunction.
    BitXor,
    /// 32-bit integer left shift.
    ShiftLeft,
    /// 32-bit arithmetic right shift.
    ShiftRight,
    /// 32-bit integer bitwise complement.
    BitNot,
    /// Numeric negation.
    Negate,
    /// DM truth-value negation.
    Not,
    /// Equality comparison.
    Equal,
    /// Inequality comparison.
    NotEqual,
    /// List membership comparison.
    ///
    /// The left value is compared against the list's iteration entries, which
    /// includes positional values and associative keys.
    Contains,
    /// Less-than comparison.
    Less,
    /// Less-than-or-equal comparison.
    LessEqual,
    /// Greater-than comparison.
    Greater,
    /// Greater-than-or-equal comparison.
    GreaterEqual,
    /// Boolean conjunction.
    And,
    /// Boolean disjunction.
    Or,
    /// Pops a value and jumps when it is exactly DM `null`.
    ///
    /// Null-conditional member/index/call lowering duplicates the receiver
    /// before this instruction, leaving the original receiver as the result
    /// on the skipped path while evaluating it only once.
    JumpIfNull(usize),
    /// Pops a condition and jumps to an absolute instruction when it is false.
    JumpIfFalse(usize),
    /// Jumps to an absolute instruction.
    Jump(usize),
    /// Skips a parameter default when that argument was explicitly supplied.
    JumpIfArgumentSupplied {
        /// Zero-based declared parameter index.
        parameter: u16,
        /// Absolute instruction after the parameter's default initializer.
        target: usize,
    },
    /// Calls a procedure with positional values popped from the stack.
    Call {
        /// Stable module-local procedure identity.
        procedure: ProcedureId,
        /// Number of positional values supplied by the caller.
        argument_count: u16,
    },
    /// Calls the currently executing procedure.
    CallCurrent {
        /// Explicit argument count, or `None` to reuse the frame's complete
        /// originally supplied argument vector.
        argument_count: Option<u16>,
    },
    /// Calls the semantically resolved parent implementation.
    CallParent {
        /// Resolved module-local target, or `None` when no parent exists.
        procedure: Option<ProcedureId>,
        /// Explicit argument count, or `None` to reuse the complete original
        /// argument vector of the current frame.
        argument_count: Option<u16>,
    },
    /// Calls a procedure selected at runtime by `call(target, procedure)`.
    ///
    /// The stack holds the receiver (or null for a global procedure), the
    /// procedure name, then the positional arguments.
    CallDynamic {
        /// Number of positional values supplied by the caller.
        argument_count: u16,
    },
    /// Expands `arglist(list)` entries in a pending call argument vector.
    ///
    /// The instruction replaces the selected source positions with the
    /// positional entries from their list values and leaves the resulting
    /// runtime argument count on the stack for a following sentinel call.
    ExpandArgumentLists {
        /// Number of source argument expressions currently on the stack.
        argument_count: u16,
        /// Zero-based source positions whose values are `arglist(...)`.
        expanded_indices: Vec<u16>,
    },
    /// Returns the top stack value.
    Return,
}

/// Calculate-and-assign operator used by [`Instruction::CompoundAssignment`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompoundAssignmentOperator {
    /// Addition assignment (`+=`).
    Add,
    /// Subtraction assignment (`-=`).
    Subtract,
    /// Multiplication assignment (`*=`).
    Multiply,
    /// Division assignment (`/=`).
    Divide,
    /// Remainder assignment (`%=`).
    Remainder,
    /// Bitwise/list-mask assignment (`&=`).
    BitAnd,
    /// Bitwise/list-union assignment (`|=`).
    BitOr,
    /// Bitwise/list-symmetric-difference assignment (`^=`).
    BitXor,
    /// Left-shift assignment (`<<=`).
    ShiftLeft,
    /// Right-shift assignment (`>>=`).
    ShiftRight,
}

/// Numeric operation used by [`Instruction::CompoundListIndex`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompoundListIndexOperator {
    /// Addition assignment (`+=`).
    Add,
    /// Subtraction assignment (`-=`).
    Subtract,
    /// Multiplication assignment (`*=`).
    Multiply,
    /// Division assignment (`/=`).
    Divide,
    /// Remainder assignment (`%=`).
    Remainder,
    /// Bitwise conjunction assignment (`&=`).
    BitAnd,
    /// Bitwise disjunction assignment (`|=`).
    BitOr,
    /// Bitwise exclusive disjunction assignment (`^=`).
    BitXor,
    /// Left-shift assignment (`<<=`).
    ShiftLeft,
    /// Right-shift assignment (`>>=`).
    ShiftRight,
}

/// Built-in value classifier used by [`Instruction::TypePredicate`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypePredicateKind {
    /// Whether the value is DM `null`.
    IsNull,
    /// Whether the value is numeric.
    IsNum,
    /// Whether the value is a canonical type path.
    IsPath,
    /// Whether the value is a DM list.
    IsList,
    /// Whether the value is an `/atom/movable` datum or one of its subtypes.
    IsMovable,
    /// Whether the value is a turf datum or one of its subtypes.
    IsTurf,
    /// Whether every value is a valid DM location (an atom).
    IsLoc,
    /// Whether the value is an icon datum or a headless icon resource.
    IsIcon,
    /// Whether a datum or type path belongs to an optional type hierarchy.
    IsType,
}

/// Stack shape of one entry consumed by [`Instruction::MakeListEntries`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ListEntryKind {
    /// One value is consumed and appended.
    Positional,
    /// A key followed by its associated value are consumed.
    Associative,
}

/// Runtime storage selected for one bare identifier in an initializer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InitializerBinding {
    /// Read a persistent runtime global by its DM identifier.
    Global(FieldName),
    /// Read a field from the initializer's `src` datum.
    SrcField(FieldName),
}

/// Stable procedure identity within one compiled module.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProcedureId(u32);

impl ProcedureId {
    fn from_index(index: usize) -> Result<Self, CompileError> {
        u32::try_from(index)
            .map(Self)
            .map_err(|_| compile_error("module has more than u32::MAX procedures"))
    }

    const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A compiled procedure body.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    /// Declared positional parameter count.
    pub parameter_count: usize,
    /// Number of local slots, including parameters.
    pub local_count: usize,
    /// Portable instructions in execution order.
    pub instructions: Vec<Instruction>,
    /// Source line associated with each instruction for diagnostics/debugging.
    pub source_spans: Vec<SourceSpan>,
}

/// A deterministic table of compiled procedures and their canonical paths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Module {
    procedures: Vec<Program>,
    paths: Vec<String>,
    names: HashMap<String, ProcedureId>,
}

/// An initializer expression linked as an entry point in a VM module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializerProgram {
    module: Module,
    entry: ProcedureId,
}

impl InitializerProgram {
    /// Returns the linked module containing the initializer entry point.
    #[must_use]
    pub const fn module(&self) -> &Module {
        &self.module
    }

    /// Returns the initializer's module-local entry point.
    #[must_use]
    pub const fn entry(&self) -> ProcedureId {
        self.entry
    }
}

impl Module {
    /// Looks up a procedure by canonical path, such as `/proc/main`.
    #[must_use]
    pub fn procedure_id(&self, path: &str) -> Option<ProcedureId> {
        self.names.get(path).copied()
    }

    /// Returns a compiled procedure by module-local identity.
    #[must_use]
    pub fn procedure(&self, procedure: ProcedureId) -> Option<&Program> {
        self.procedures.get(procedure.index())
    }

    /// Returns the canonical path associated with a procedure.
    #[must_use]
    pub fn procedure_path(&self, procedure: ProcedureId) -> Option<&str> {
        self.paths.get(procedure.index()).map(String::as_str)
    }

    /// Returns the stable identity at a procedure-spec index.
    #[must_use]
    pub fn procedure_id_at(&self, index: usize) -> Option<ProcedureId> {
        self.procedures.get(index)?;
        u32::try_from(index).ok().map(ProcedureId)
    }
}

/// One independently identified procedure body supplied by a semantic layer.
#[derive(Clone, Debug)]
pub struct ProcedureSpec<'definition> {
    /// Unique diagnostic path for stack traces and lookup.
    pub path: String,
    /// Parsed procedure definition to compile.
    pub definition: &'definition Definition,
    /// Index of the exact parent implementation in the same spec slice.
    pub parent: Option<usize>,
    /// Semantically resolved bare-call targets, keyed by selector.
    ///
    /// This preserves object-tree inheritance when it differs from lexical
    /// path ancestry, such as `/area` inheriting `/datum`.
    pub static_calls: BTreeMap<String, usize>,
    /// Bare identifiers that resolve to fields on the executing procedure's
    /// `src` datum when they do not name a parameter or local.
    pub src_fields: BTreeMap<String, FieldName>,
    /// Bare identifiers that resolve to persistent runtime globals when they
    /// do not name a parameter, local, or `src` field.
    pub global_fields: BTreeMap<String, FieldName>,
}

/// Failure while compiling the initial executable subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileError {
    /// Human-readable diagnostic.
    pub message: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CompileError {}

/// Failure while executing portable bytecode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeError {
    /// Human-readable runtime diagnostic.
    pub message: String,
    /// Instruction index at which execution failed.
    pub instruction: usize,
    /// Source span associated with the failing instruction, when available.
    pub source_span: Option<SourceSpan>,
    /// Active procedures from the entry point through the failing frame.
    pub call_stack: Vec<CallTrace>,
}

/// One source-mapped procedure in a runtime error's call stack.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallTrace {
    /// Canonical procedure path.
    pub procedure: String,
    /// Instruction active in this frame.
    pub instruction: usize,
    /// Source span associated with the active instruction, when available.
    pub source_span: Option<SourceSpan>,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at instruction {}",
            self.message, self.instruction
        )?;
        if let Some(span) = self.source_span {
            write!(formatter, " (source {}..{})", span.start, span.end)?;
        }
        Ok(())
    }
}

impl std::error::Error for RuntimeError {}

/// Compiles one procedure definition to portable stack bytecode.
///
/// The current vertical slice supports positional parameters and safe default
/// expressions, local `var` declarations, assignment, structured control flow,
/// numeric and text literals, local reads, procedure calls, unary operators,
/// and common binary operators.
///
/// # Errors
///
/// Returns [`CompileError`] for unsupported statements, malformed expressions,
/// unknown locals, or non-procedure definitions.
pub fn compile_procedure(definition: &Definition) -> Result<Program, CompileError> {
    compile_procedure_with_resolver(definition, &HashMap::new())
}

/// Lowers one variable initializer expression to existing VM bytecode.
///
/// Bare names are resolved only through `bindings`; there are no implicit or
/// fabricated built-ins. When `procedures` is supplied, unqualified calls may
/// resolve to global `/proc/name` entries already present in that module.
/// Initializer tokens retain their expanded-source span on every instruction.
///
/// # Errors
///
/// Returns [`CompileError`] for malformed syntax, unresolved identifiers or
/// calls, and expression forms that have no initializer execution context.
pub fn compile_initializer(
    tokens: &[SpannedToken],
    bindings: &BTreeMap<String, InitializerBinding>,
    procedures: Option<&Module>,
) -> Result<InitializerProgram, CompileError> {
    let mut expression = ExpressionParser::new(tokens).parse()?;
    bind_initializer_expression(&mut expression, bindings)?;

    let mut module = procedures.cloned().unwrap_or_else(|| Module {
        procedures: Vec::new(),
        paths: Vec::new(),
        names: HashMap::new(),
    });
    let mut call_names = HashMap::new();
    for (path, procedure) in &module.names {
        if let Some(name) = path.strip_prefix("/proc/")
            && !name.contains('/')
        {
            call_names.insert(name.to_owned(), *procedure);
        }
    }
    let mut instructions = Vec::new();
    emit_expression(
        &expression,
        &LocalTable::default(),
        &mut instructions,
        &call_names,
    )?;
    instructions.push(Instruction::Return);
    let source_span = match (tokens.first(), tokens.last()) {
        (Some(first), Some(last)) => SourceSpan::new(first.span.start, last.span.end),
        _ => return Err(compile_error("expected an initializer expression")),
    };
    let program = Program {
        parameter_count: 0,
        local_count: 0,
        source_spans: vec![source_span; instructions.len()],
        instructions,
    };
    let entry = ProcedureId::from_index(module.procedures.len())?;
    module.procedures.push(program);
    module.paths.push("<initializer>".to_owned());
    Ok(InitializerProgram { module, entry })
}

/// Compiles a deterministic module from procedure definitions in source order.
///
/// This initial call-resolution slice exposes global `/proc/name` procedures to
/// unqualified `name(...)` expressions. Object dispatch and overloads belong to
/// the later object-tree semantic pass.
///
/// # Errors
///
/// Returns [`CompileError`] when a definition is not executable, a canonical
/// procedure path is duplicated, or any procedure body cannot be compiled.
pub fn compile_module(definitions: &[Definition]) -> Result<Module, CompileError> {
    let mut names = HashMap::new();
    let mut call_names = HashMap::new();
    let mut paths = Vec::with_capacity(definitions.len());
    for (index, definition) in definitions.iter().enumerate() {
        if !matches!(
            definition.kind,
            DefinitionKind::Procedure | DefinitionKind::ProcedureOverride | DefinitionKind::Verb
        ) {
            return Err(compile_error(format!(
                "definition {} is not executable",
                definition.path
            )));
        }
        let procedure = ProcedureId::from_index(index)?;
        let path = definition.path.to_string();
        if names.insert(path.clone(), procedure).is_some() {
            return Err(compile_error(format!("duplicate procedure path {path:?}")));
        }
        let segments = definition.path.segments();
        if segments.len() == 2
            && matches!(segments[0].as_str(), "proc" | "verb")
            && call_names.insert(segments[1].clone(), procedure).is_some()
        {
            return Err(compile_error(format!(
                "ambiguous global procedure name {:?}",
                segments[1]
            )));
        }
        paths.push(path);
    }

    let procedures = definitions
        .iter()
        .map(|definition| compile_procedure_with_resolver(definition, &call_names))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Module {
        procedures,
        paths,
        names,
    })
}

/// Compiles procedure bodies whose exact parent implementations were resolved
/// by an independent semantic layer.
///
/// Spec order defines stable module-local identities. Parent indices may point
/// forward or backward, but must refer to this same slice. Diagnostic paths
/// must be unique. Unqualified global call resolution remains the concern of
/// [`compile_module`]; this API focuses on already-resolved implementation
/// chains.
///
/// # Errors
///
/// Returns [`CompileError`] for duplicate paths, invalid parent indices, or
/// procedure bodies outside the supported executable subset.
pub fn compile_module_specs(specs: &[ProcedureSpec<'_>]) -> Result<Module, CompileError> {
    let mut names = HashMap::new();
    let mut paths = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let procedure = ProcedureId::from_index(index)?;
        if names.insert(spec.path.clone(), procedure).is_some() {
            return Err(compile_error(format!(
                "duplicate procedure spec path {:?}",
                spec.path
            )));
        }
        if spec.parent.is_some_and(|parent| parent >= specs.len()) {
            return Err(compile_error(format!(
                "procedure spec {:?} has invalid parent index {:?}",
                spec.path, spec.parent
            )));
        }
        paths.push(spec.path.clone());
    }

    let procedures = specs
        .iter()
        .map(|spec| {
            let mut targets = HashMap::new();
            if let Some(parent) = spec.parent {
                targets.insert("..".to_owned(), ProcedureId::from_index(parent)?);
            }
            targets.extend(static_call_targets(&spec.path, &paths));
            for (selector, target) in &spec.static_calls {
                targets.insert(selector.clone(), ProcedureId::from_index(*target)?);
            }
            compile_procedure_with_resolver_and_fields(
                spec.definition,
                &targets,
                &spec.src_fields,
                &spec.global_fields,
            )
            .map_err(|error| compile_error(format!("{}: {}", spec.path, error.message)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Module {
        procedures,
        paths,
        names,
    })
}

fn static_call_targets(path: &str, paths: &[String]) -> HashMap<String, ProcedureId> {
    let Some((owner, _)) = path.rsplit_once("/proc/") else {
        return HashMap::new();
    };
    let mut targets = HashMap::new();
    for candidate in paths {
        let Some((_, name)) = candidate.rsplit_once("/proc/") else {
            continue;
        };
        let name = name.split('@').next().unwrap_or(name);
        if targets.contains_key(name) {
            continue;
        }
        let mut current_owner = owner;
        loop {
            let expected = if current_owner.is_empty() {
                format!("/proc/{name}")
            } else {
                format!("{current_owner}/proc/{name}")
            };
            if let Some((index, _)) = paths.iter().enumerate().rev().find(|(_, candidate)| {
                *candidate == &expected || candidate.starts_with(&format!("{expected}@"))
            }) {
                if let Ok(procedure) = ProcedureId::from_index(index) {
                    targets.insert(name.to_owned(), procedure);
                }
                break;
            }
            let Some((parent, _)) = current_owner.rsplit_once('/') else {
                break;
            };
            current_owner = parent;
        }
    }
    targets
}

fn compile_procedure_with_resolver(
    definition: &Definition,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<Program, CompileError> {
    compile_procedure_with_resolver_and_fields(
        definition,
        procedures,
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
}

fn compile_procedure_with_resolver_and_fields(
    definition: &Definition,
    procedures: &HashMap<String, ProcedureId>,
    src_fields: &BTreeMap<String, FieldName>,
    global_fields: &BTreeMap<String, FieldName>,
) -> Result<Program, CompileError> {
    if !matches!(
        definition.kind,
        DefinitionKind::Procedure | DefinitionKind::ProcedureOverride | DefinitionKind::Verb
    ) {
        return Err(compile_error("definition is not executable"));
    }

    let mut locals = LocalTable::with_fields(src_fields.clone(), global_fields.clone());
    for (index, parameter) in definition.parameters.iter().enumerate() {
        // BYOND permits an unnamed trailing `...` parameter.  It still
        // occupies an argument position, but cannot be referenced by name.
        if let Some(name) = parameter_name(&parameter.tokens) {
            locals.insert_parameter(name.to_owned(), to_local_index(index)?);
        }
    }
    locals.reserve_parameter_slots(definition.parameters.len())?;
    // `args` is an implicit, per-call list in every DM procedure.  It must
    // be a local (rather than a fabricated global) so recursive and nested
    // calls retain their own complete supplied argument vectors.
    let args_slot = locals.declare("args".to_owned())?;

    let mut instructions = Vec::new();
    let mut source_spans = Vec::new();
    let mut loops = Vec::new();
    push_instruction(
        &mut instructions,
        &mut source_spans,
        Instruction::MakeArgs,
        definition.span,
    );
    push_instruction(
        &mut instructions,
        &mut source_spans,
        Instruction::StoreLocal(args_slot),
        definition.span,
    );
    compile_parameter_defaults(
        definition,
        &locals,
        &mut instructions,
        &mut source_spans,
        procedures,
    )?;
    // The DM preprocessor is allowed to expand a macro into several
    // statements on one logical source line.  The common `QDEL_NULL(x)`
    // helper, for example, becomes `qdel(x); x = null`.  Keep statement
    // separators out of the expression parser by turning only top-level
    // semicolons into ordinary logical lines before lowering the body.
    let body = split_top_level_semicolon_statements(&definition.body);
    let falls_through = if let Some(first_line) = body.first() {
        let block_indentation = indentation(first_line);
        let (next_line, falls_through) = compile_block(
            &body,
            0,
            block_indentation,
            &mut locals,
            &mut instructions,
            &mut source_spans,
            procedures,
            &mut loops,
        )?;
        if next_line != body.len() {
            return Err(compile_error("procedure body contains invalid indentation"));
        }
        falls_through
    } else {
        true
    };
    if falls_through {
        push_instruction(
            &mut instructions,
            &mut source_spans,
            Instruction::LoadResult,
            definition.span,
        );
        push_instruction(
            &mut instructions,
            &mut source_spans,
            Instruction::Return,
            definition.span,
        );
    }

    Ok(Program {
        parameter_count: definition.parameters.len(),
        local_count: locals.slot_count,
        instructions,
        source_spans,
    })
}

/// Expands DM's macro-style statement separators and compact brace bodies.
///
/// BYOND macros commonly use C-style compact bodies even though ordinary DM
/// source is indentation based: `if (!value) { value = list(); } value += x;`.
/// The preprocessor leaves that expansion on the invocation's logical line.
/// Re-present its braces and semicolons as indentation-based logical lines
/// before statement lowering.  Parenthesized and indexed expressions retain
/// their punctuation unchanged. Empty statements (including a physical line
/// containing only `}`) are legal and discarded.
fn split_top_level_semicolon_statements(lines: &[SourceLine]) -> Vec<SourceLine> {
    let mut result = Vec::with_capacity(lines.len());
    for line in lines {
        let mut statement = Vec::new();
        let mut grouping_depth = 0usize;
        let mut brace_depth = 0usize;
        let base_indentation = indentation(line);
        let mut emit = |tokens: &mut Vec<SpannedToken>, brace_depth: usize| {
            if tokens.is_empty() {
                return;
            }
            let mut logical_line = line.clone();
            logical_line.indentation.tabs = 0;
            logical_line.indentation.spaces = base_indentation.saturating_add(brace_depth);
            logical_line.tokens = std::mem::take(tokens);
            result.push(logical_line);
        };
        for token in &line.tokens {
            match token.kind {
                TokenKind::Punctuation('(' | '[') => {
                    grouping_depth += 1;
                    statement.push(token.clone());
                }
                TokenKind::Punctuation(')' | ']') => {
                    grouping_depth = grouping_depth.saturating_sub(1);
                    statement.push(token.clone());
                }
                TokenKind::Punctuation('{') if grouping_depth == 0 => {
                    emit(&mut statement, brace_depth);
                    brace_depth += 1;
                }
                TokenKind::Punctuation('}') if grouping_depth == 0 => {
                    emit(&mut statement, brace_depth);
                    brace_depth = brace_depth.saturating_sub(1);
                }
                TokenKind::Punctuation(';') if grouping_depth == 0 => {
                    emit(&mut statement, brace_depth);
                }
                _ => statement.push(token.clone()),
            }
        }
        emit(&mut statement, brace_depth);
    }
    result
}

#[derive(Default)]
struct LocalTable {
    names: HashMap<String, u16>,
    src_fields: BTreeMap<String, FieldName>,
    global_fields: BTreeMap<String, FieldName>,
    slot_count: usize,
}

impl LocalTable {
    fn with_fields(
        src_fields: BTreeMap<String, FieldName>,
        global_fields: BTreeMap<String, FieldName>,
    ) -> Self {
        Self {
            names: HashMap::new(),
            src_fields,
            global_fields,
            slot_count: 0,
        }
    }
    fn insert_parameter(&mut self, name: String, slot: u16) {
        self.names.insert(name, slot);
        self.slot_count = self.slot_count.max(usize::from(slot) + 1);
    }

    fn reserve_parameter_slots(&mut self, count: usize) -> Result<(), CompileError> {
        // Keep unnamed varargs positions available to the frame binder and
        // ensure subsequent locals are allocated after every parameter.
        let count = usize::from(to_local_index(count)?);
        self.slot_count = self.slot_count.max(count);
        Ok(())
    }

    fn declare(&mut self, name: String) -> Result<u16, CompileError> {
        if self.names.contains_key(&name) {
            return Err(compile_error(format!("local {name:?} is already declared")));
        }
        let slot = to_local_index(self.slot_count)?;
        self.slot_count += 1;
        self.names.insert(name, slot);
        Ok(slot)
    }

    fn declare_hidden(&mut self) -> Result<u16, CompileError> {
        let slot = to_local_index(self.slot_count)?;
        self.slot_count += 1;
        Ok(slot)
    }

    fn get(&self, name: &str) -> Option<u16> {
        self.names.get(name).copied()
    }

    fn src_field(&self, name: &str) -> Option<&FieldName> {
        self.src_fields.get(name)
    }

    fn global_field(&self, name: &str) -> Option<&FieldName> {
        self.global_fields.get(name)
    }

    fn remove(&mut self, name: &str) {
        self.names.remove(name);
    }
}

fn compile_parameter_defaults(
    definition: &Definition,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    for (parameter_index, parameter) in definition.parameters.iter().enumerate() {
        let Some(assignment) = parameter.tokens.iter().position(
            |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
        ) else {
            continue;
        };
        let default_tokens = &parameter.tokens[assignment + 1..];
        if default_tokens.is_empty() {
            return Err(compile_error("procedure parameter default is empty"));
        }
        let parameter_slot = to_local_index(parameter_index)?;
        let default_jump = instructions.len();
        push_instruction(
            instructions,
            source_spans,
            Instruction::JumpIfArgumentSupplied {
                parameter: parameter_slot,
                target: usize::MAX,
            },
            parameter.span,
        );
        let expression = ExpressionParser::new(default_tokens).parse()?;
        let first_default_instruction = instructions.len();
        emit_expression(&expression, locals, instructions, procedures)?;
        instructions.push(Instruction::StoreLocal(parameter_slot));
        source_spans.extend(std::iter::repeat_n(
            parameter.span,
            instructions.len() - first_default_instruction,
        ));
        let end_target = instructions.len();
        patch_jump(instructions, default_jump, end_target)?;
    }
    Ok(())
}

struct LoopContext {
    continue_target: Option<usize>,
    continue_jumps: Vec<usize>,
    break_jumps: Vec<usize>,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_block(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    // DM locals are lexical to their block. Macro helpers routinely expand
    // repeated `do { var/_L = ... } while(0)` scopes; retaining those names
    // after the child block makes unrelated invocations collide.
    let saved_names = locals.names.clone();
    let result = compile_block_inner(
        lines,
        line_index,
        block_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    locals.names = saved_names;
    result
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_block_inner(
    lines: &[SourceLine],
    mut line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    let mut falls_through = true;
    while let Some(line) = lines.get(line_index) {
        let line_indentation = indentation(line);
        if line_indentation < block_indentation {
            break;
        }
        if line_indentation > block_indentation {
            return Err(compile_error("unexpected indentation in procedure body"));
        }
        let first = line
            .tokens
            .first()
            .expect("syntax source lines always contain tokens");
        match &first.kind {
            TokenKind::Identifier(keyword) if keyword == "if" => {
                let (next_line, statement_falls_through) = compile_if(
                    lines,
                    line_index,
                    block_indentation,
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?;
                falls_through &= statement_falls_through;
                line_index = next_line;
                continue;
            }
            // `switch` is a statement in DM, not a procedure call.  Each
            // indented `if` arm is a case list (with comma-separated values
            // and `low to high` ranges), while `else` is the default arm.
            // Keep this distinct from ordinary `if`: a switch selector is
            // evaluated exactly once and every case compares against it.
            TokenKind::Identifier(keyword) if keyword == "switch" => {
                let (next_line, statement_falls_through) = compile_switch(
                    lines,
                    line_index,
                    block_indentation,
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?;
                falls_through &= statement_falls_through;
                line_index = next_line;
                continue;
            }
            TokenKind::Identifier(keyword) if keyword == "while" => {
                let next_line = compile_while(
                    lines,
                    line_index,
                    block_indentation,
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?;
                line_index = next_line;
                continue;
            }
            TokenKind::Identifier(keyword) if keyword == "do" => {
                let next_line = compile_do_while(
                    lines,
                    line_index,
                    block_indentation,
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?;
                line_index = next_line;
                continue;
            }
            TokenKind::Identifier(keyword) if keyword == "for" => {
                let next_line = compile_for(
                    lines,
                    line_index,
                    block_indentation,
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?;
                line_index = next_line;
                continue;
            }
            TokenKind::Identifier(keyword) if keyword == "else" => {
                return Err(compile_error("else without a matching if"));
            }
            TokenKind::Identifier(keyword) if keyword == "break" => {
                if line.tokens.len() != 1 {
                    return Err(compile_error("break does not accept an expression"));
                }
                let Some(loop_context) = loops.last_mut() else {
                    return Err(compile_error("break outside a loop"));
                };
                let jump = instructions.len();
                push_instruction(
                    instructions,
                    source_spans,
                    Instruction::Jump(usize::MAX),
                    line.span,
                );
                loop_context.break_jumps.push(jump);
                falls_through = false;
            }
            TokenKind::Identifier(keyword) if keyword == "continue" => {
                if line.tokens.len() != 1 {
                    return Err(compile_error("continue does not accept an expression"));
                }
                let Some(loop_context) = loops.last_mut() else {
                    return Err(compile_error("continue outside a loop"));
                };
                let target = loop_context.continue_target.unwrap_or(usize::MAX);
                let jump = instructions.len();
                push_instruction(
                    instructions,
                    source_spans,
                    Instruction::Jump(target),
                    line.span,
                );
                if loop_context.continue_target.is_none() {
                    loop_context.continue_jumps.push(jump);
                }
                falls_through = false;
            }
            TokenKind::Identifier(keyword) if keyword == "return" => {
                let first_instruction = instructions.len();
                if line.tokens.len() == 1 {
                    instructions.push(Instruction::PushNull);
                } else {
                    compile_expression(&line.tokens[1..], locals, instructions, procedures)?;
                }
                instructions.push(Instruction::Return);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
                falls_through = false;
            }
            TokenKind::Identifier(keyword) if keyword == "CRASH" => {
                let first_instruction = instructions.len();
                compile_crash_statement(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
                falls_through = false;
            }
            // `waitfor` controls BYOND's cooperative scheduling of a procedure.
            // Dream64's current headless executor is synchronous and has no
            // sleeping instructions, so the declaration is intentionally a
            // compile-time no-op rather than an executable assignment.
            TokenKind::Identifier(keyword)
                if keyword == "set" && is_waitfor_directive(&line.tokens) => {}
            TokenKind::Identifier(keyword) if keyword == "var" => {
                let first_instruction = instructions.len();
                let _ = compile_local(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(_) if top_level_assignment(&line.tokens).is_some() => {
                let first_instruction = instructions.len();
                compile_assignment_statement(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            // Postfix/prefix increments are valid standalone statements as
            // well as for-loop clauses.  In particular, bare datum fields
            // such as `areasize++` resolve through `src` rather than a local
            // binding, so they must take the same lowering path as compound
            // assignments.
            TokenKind::Identifier(_) | TokenKind::Operator(_)
                if local_increment(&line.tokens).is_some() =>
            {
                let first_instruction = instructions.len();
                // `compile_for_clause` owns the shared prefix/postfix
                // increment lowering (including bare `src` fields).  The
                // standalone statement form has identical semantics.
                compile_for_clause(&line.tokens, false, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Operator(operator) if operator == "." => {
                let first_instruction = instructions.len();
                compile_result_assignment(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(keyword) if keyword == "call" => {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            // `new /type(...)` is also commonly written as a pure
            // side-effect statement, especially for controller singletons.
            TokenKind::Identifier(keyword) if keyword == "new" => {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            // A parent call is also a valid side-effect-only statement.  It
            // starts with the `..` operator rather than an identifier, so it
            // cannot share the ordinary static-call statement arm below.
            TokenKind::Operator(operator)
                if operator == ".."
                    && matches!(
                        line.tokens.get(1).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) =>
            {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            // Parenthesized expressions are valid as discarded-result
            // statements too.  Macro expansions commonly wrap an assignment
            // or a side-effecting call in parentheses, which means these
            // lines begin with punctuation rather than an identifier.
            TokenKind::Punctuation('(') => {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            // Calls may be used purely for their side effects.  `call(...)`
            // has its own syntax above, but ordinary static calls (including
            // datum helper calls such as `RegisterSignals(...)`) and dotted
            // datum calls such as `atom_storage.set_holdable(...)` both begin
            // with an identifier.  The latter have the opening parenthesis
            // after the receiver and selector rather than immediately after
            // the first identifier, so recognize any call-shaped expression
            // on the source line and lower its discarded result uniformly.
            TokenKind::Identifier(_)
                if line
                    .tokens
                    .iter()
                    .any(|token| token.kind == TokenKind::Punctuation('(')) =>
            {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            _ => {
                return Err(compile_error(format!(
                    "unsupported statement beginning with {:?}",
                    first.kind
                )));
            }
        }
        line_index += 1;
    }
    Ok((line_index, falls_through))
}

fn is_waitfor_directive(tokens: &[SpannedToken]) -> bool {
    matches!(
        tokens,
        [
            SpannedToken {
                kind: TokenKind::Identifier(set),
                ..
            },
            SpannedToken {
                kind: TokenKind::Identifier(name),
                ..
            },
            SpannedToken {
                kind: TokenKind::Operator(operator),
                ..
            },
            SpannedToken {
                kind: TokenKind::Identifier(value),
                ..
            }
        ] if set == "set"
            && name == "waitfor"
            && operator == "="
            && matches!(value.as_str(), "TRUE" | "FALSE")
    ) || matches!(
        tokens,
        [
            SpannedToken {
                kind: TokenKind::Identifier(set),
                ..
            },
            SpannedToken {
                kind: TokenKind::Identifier(name),
                ..
            },
            SpannedToken {
                kind: TokenKind::Operator(operator),
                ..
            },
            SpannedToken {
                kind: TokenKind::Number(value),
                ..
            }
        ] if set == "set"
            && name == "waitfor"
            && operator == "="
            && matches!(value.as_str(), "0" | "1")
    )
}

fn compile_crash_statement(
    tokens: &[SpannedToken],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let Some((first, rest)) = tokens.split_first() else {
        return Err(compile_error("CRASH requires a message expression"));
    };
    if !matches!(&first.kind, TokenKind::Identifier(keyword) if keyword == "CRASH")
        || rest.len() < 2
        || !matches!(
            rest.first().map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        )
        || !matches!(
            rest.last().map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        )
    {
        return Err(compile_error(
            "CRASH requires one parenthesized message expression",
        ));
    }
    let expression = &rest[1..rest.len() - 1];
    if expression.is_empty() {
        instructions.push(Instruction::PushText("CRASH".to_owned()));
    } else {
        compile_expression(expression, locals, instructions, procedures)?;
    }
    instructions.push(Instruction::Crash);
    Ok(())
}

fn top_level_assignment(tokens: &[SpannedToken]) -> Option<(usize, &str)> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator)
                if matches!(
                    operator.as_str(),
                    "=" | "+="
                        | "-="
                        | "*="
                        | "/="
                        | "%="
                        | "&="
                        | "|="
                        | "^="
                        | "<<="
                        | ">>="
                        | "&&="
                        | "||="
                ) && depth == 0 =>
            {
                return Some((index, operator));
            }
            _ => {}
        }
    }
    None
}

fn compile_assignment_statement(
    tokens: &[SpannedToken],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let (assignment, operator) = top_level_assignment(tokens)
        .ok_or_else(|| compile_error("assignment statement requires '='"))?;
    if matches!(operator, "||=" | "&&=") {
        compile_expression(tokens, locals, instructions, procedures)?;
        instructions.push(Instruction::Pop);
        return Ok(());
    }
    if assignment == 0 || assignment + 1 == tokens.len() {
        return Err(compile_error("assignment requires a target and value"));
    }
    let target = ExpressionParser::new(&tokens[..assignment]).parse()?;
    match target {
        Expression::Local(name) => {
            let local = locals.get(&name);
            let field = locals.src_field(&name).cloned();
            let global = locals.global_field(&name).cloned();
            let Some(slot) = local else {
                if field.is_none() && global.is_none() {
                    return Err(compile_error(format!("unknown local {name:?}")));
                }
                if let Some(global) = global {
                    if operator != "=" {
                        instructions.push(Instruction::LoadGlobal(global.clone()));
                    }
                    compile_expression(
                        &tokens[assignment + 1..],
                        locals,
                        instructions,
                        procedures,
                    )?;
                    if operator != "=" {
                        instructions.push(compound_instruction(operator)?);
                    }
                    instructions.push(Instruction::StoreGlobal(global));
                    return Ok(());
                }
                if operator == "=" {
                    instructions.push(Instruction::LoadSrc);
                } else {
                    instructions.push(Instruction::LoadSrc);
                    instructions.push(Instruction::Duplicate);
                    instructions.push(Instruction::LoadField(
                        field.clone().expect("field was checked"),
                    ));
                }
                compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::StoreField(field.expect("field was checked")));
                return Ok(());
            };
            if operator != "=" {
                instructions.push(Instruction::LoadLocal(slot));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreLocal(slot));
        }
        Expression::Index { list, index } => {
            emit_expression(&list, locals, instructions, procedures)?;
            emit_expression(&index, locals, instructions, procedures)?;
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator == "=" {
                instructions.push(Instruction::SetListIndex);
            } else {
                instructions.push(Instruction::CompoundListIndex(
                    compound_list_index_operator(operator)?,
                ));
            }
        }
        Expression::SafeIndex { list, index } => {
            emit_expression(&list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(&index, locals, instructions, procedures)?;
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator == "=" {
                instructions.push(Instruction::SetListIndex);
            } else {
                instructions.push(Instruction::CompoundListIndex(
                    compound_list_index_operator(operator)?,
                ));
            }
            let end_jump = instructions.len();
            instructions.push(Instruction::Jump(usize::MAX));
            let null_target = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(null_target);
            instructions.push(Instruction::Pop);
            let end = instructions.len();
            instructions[end_jump] = Instruction::Jump(end);
        }
        Expression::Field { receiver, name } => {
            emit_expression(&receiver, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreField(name));
        }
        Expression::SafeField { receiver, name } => {
            emit_expression(&receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreField(name));
            let end_jump = instructions.len();
            instructions.push(Instruction::Jump(usize::MAX));
            let null_target = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(null_target);
            instructions.push(Instruction::Pop);
            let end = instructions.len();
            instructions[end_jump] = Instruction::Jump(end);
        }
        Expression::GlobalField(name) => {
            if operator != "=" {
                instructions.push(Instruction::LoadGlobal(name.clone()));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreGlobal(name));
        }
        _ => return Err(compile_error("assignment target is not writable")),
    }
    Ok(())
}

fn compound_instruction(operator: &str) -> Result<Instruction, CompileError> {
    let operator = match operator {
        "+=" => CompoundAssignmentOperator::Add,
        "-=" => CompoundAssignmentOperator::Subtract,
        "*=" => CompoundAssignmentOperator::Multiply,
        "/=" => CompoundAssignmentOperator::Divide,
        "%=" => CompoundAssignmentOperator::Remainder,
        "&=" => CompoundAssignmentOperator::BitAnd,
        "|=" => CompoundAssignmentOperator::BitOr,
        "^=" => CompoundAssignmentOperator::BitXor,
        "<<=" => CompoundAssignmentOperator::ShiftLeft,
        ">>=" => CompoundAssignmentOperator::ShiftRight,
        _ => {
            return Err(compile_error(format!(
                "unsupported compound operator {operator}"
            )));
        }
    };
    Ok(Instruction::CompoundAssignment(operator))
}

fn compound_list_index_operator(operator: &str) -> Result<CompoundListIndexOperator, CompileError> {
    Ok(match operator {
        "+=" => CompoundListIndexOperator::Add,
        "-=" => CompoundListIndexOperator::Subtract,
        "*=" => CompoundListIndexOperator::Multiply,
        "/=" => CompoundListIndexOperator::Divide,
        "%=" => CompoundListIndexOperator::Remainder,
        "&=" => CompoundListIndexOperator::BitAnd,
        "|=" => CompoundListIndexOperator::BitOr,
        "^=" => CompoundListIndexOperator::BitXor,
        "<<=" => CompoundListIndexOperator::ShiftLeft,
        ">>=" => CompoundListIndexOperator::ShiftRight,
        _ => {
            return Err(compile_error(format!(
                "unsupported compound operator {operator:?}"
            )));
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn compile_while(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let condition_target = instructions.len();
    let condition = condition_tokens(&line.tokens, "while")?;
    compile_expression(condition, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - condition_target,
    ));
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );

    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("while statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("while statement requires an indented body"));
    }

    loops.push(LoopContext {
        continue_target: Some(condition_target),
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    let loop_context = loops.pop().expect("the active while context was pushed");
    let (after_body, _) = body?;
    push_instruction(
        instructions,
        source_spans,
        Instruction::Jump(condition_target),
        line.span,
    );
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    Ok(after_body)
}

/// Compiles BYOND's post-test `do`/`while` loop form.  The trailing `while`
/// belongs to the `do` statement, at its original indentation, rather than
/// beginning a second statement after the body.
#[allow(clippy::too_many_arguments)]
fn compile_do_while(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<usize, CompileError> {
    let do_line = &lines[line_index];
    if do_line.tokens.len() != 1 {
        return Err(compile_error("do statement does not accept a condition"));
    }
    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("do statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("do statement requires an indented body"));
    }

    let body_target = instructions.len();
    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    let loop_context = loops.pop().expect("the active do context was pushed");
    let (while_index, _) = body?;
    let while_line = lines
        .get(while_index)
        .ok_or_else(|| compile_error("do statement requires a trailing while condition"))?;
    if indentation(while_line) != block_indentation
        || !matches!(
            while_line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "while"
        )
    {
        return Err(compile_error(
            "do statement requires a trailing while condition",
        ));
    }

    let condition_target = instructions.len();
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, condition_target)?;
    }
    let condition = condition_tokens(&while_line.tokens, "while")?;
    let condition_start = instructions.len();
    compile_expression(condition, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        while_line.span,
        instructions.len() - condition_start,
    ));
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        while_line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::Jump(body_target),
        while_line.span,
    );
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    Ok(while_index + 1)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_for(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    if let Some((local_name, start, end, step)) = for_to_parts(&line.tokens)? {
        return compile_for_to(
            lines,
            line_index,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
            &local_name,
            start,
            end,
            step,
        );
    }
    if let Some((local_name, iterable)) = for_in_parts(&line.tokens)? {
        return compile_for_in(
            lines,
            line_index,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
            &local_name,
            iterable,
        );
    }
    let [initializer, condition, increment] = for_clauses(&line.tokens)?;
    let initializer_start = instructions.len();
    let scoped_local = compile_for_clause(initializer, true, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - initializer_start,
    ));

    let condition_target = instructions.len();
    if condition.is_empty() {
        instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
    } else {
        compile_expression(condition, locals, instructions, procedures)?;
    }
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - condition_target,
    ));
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );

    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("for statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("for statement requires an indented body"));
    }
    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    let loop_context = loops.pop().expect("the active for context was pushed");
    let (after_body, _) = body?;

    let increment_target = instructions.len();
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, increment_target)?;
    }
    let increment_start = instructions.len();
    compile_for_clause(increment, false, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - increment_start,
    ));
    push_instruction(
        instructions,
        source_spans,
        Instruction::Jump(condition_target),
        line.span,
    );
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    if let Some(scoped_local) = scoped_local {
        locals.remove(&scoped_local);
    }
    Ok(after_body)
}

/// Compiles DM's inclusive numeric range loop, `for(var/i in first to last)`.
/// The end expression is evaluated once, matching the normal DM range-loop
/// header semantics and avoiding re-evaluating a mutable field on each turn.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_for_to(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
    local_name: &str,
    start: &[SpannedToken],
    end: &[SpannedToken],
    step: Option<&[SpannedToken]>,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let item_slot = locals.declare(local_name.to_owned())?;
    let end_slot = locals.declare_hidden()?;
    let step_slot = locals.declare_hidden()?;

    let initialization_start = instructions.len();
    compile_expression(start, locals, instructions, procedures)?;
    instructions.push(Instruction::StoreLocal(item_slot));
    compile_expression(end, locals, instructions, procedures)?;
    instructions.push(Instruction::StoreLocal(end_slot));
    if let Some(step) = step {
        compile_expression(step, locals, instructions, procedures)?;
    } else {
        instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
    }
    instructions.push(Instruction::StoreLocal(step_slot));
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - initialization_start,
    ));

    let condition_target = instructions.len();
    // `step` controls both the increment and direction.  Keep the bounds
    // inclusive, just like BYOND: positive steps run while `i <= end` and
    // negative steps run while `i >= end`.  The step expression is evaluated
    // once at loop entry, rather than once per iteration.
    for instruction in [
        Instruction::LoadLocal(step_slot),
        Instruction::PushNumber(DmNumberBits::from_f32(0.0)),
        Instruction::GreaterEqual,
        Instruction::LoadLocal(item_slot),
        Instruction::LoadLocal(end_slot),
        Instruction::LessEqual,
        Instruction::And,
        Instruction::LoadLocal(step_slot),
        Instruction::PushNumber(DmNumberBits::from_f32(0.0)),
        Instruction::Less,
        Instruction::LoadLocal(item_slot),
        Instruction::LoadLocal(end_slot),
        Instruction::GreaterEqual,
        Instruction::And,
        Instruction::Or,
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );

    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("for-to statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("for-to statement requires an indented body"));
    }
    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    let loop_context = loops.pop().expect("the active for-to context was pushed");
    let (after_body, _) = body?;

    let increment_target = instructions.len();
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, increment_target)?;
    }
    for instruction in [
        Instruction::LoadLocal(item_slot),
        Instruction::LoadLocal(step_slot),
        Instruction::Add,
        Instruction::StoreLocal(item_slot),
        Instruction::Jump(condition_target),
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    locals.remove(local_name);
    Ok(after_body)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_for_in(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
    local_name: &str,
    iterable: &[SpannedToken],
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let item_slot = locals.declare(local_name.to_owned())?;
    let list_slot = locals.declare_hidden()?;
    let index_slot = locals.declare_hidden()?;

    let initialization_start = instructions.len();
    compile_expression(iterable, locals, instructions, procedures)?;
    instructions.push(Instruction::StoreLocal(list_slot));
    instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
    instructions.push(Instruction::StoreLocal(index_slot));
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - initialization_start,
    ));

    let condition_target = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::LoadLocal(index_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::LoadLocal(list_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::ListLength,
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::LessEqual,
        line.span,
    );
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );

    push_instruction(
        instructions,
        source_spans,
        Instruction::LoadLocal(list_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::LoadLocal(index_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::IndexList,
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::StoreLocal(item_slot),
        line.span,
    );

    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("for-in statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("for-in statement requires an indented body"));
    }
    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    let loop_context = loops.pop().expect("the active for-in context was pushed");
    let (after_body, _) = body?;

    let increment_target = instructions.len();
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, increment_target)?;
    }
    for instruction in [
        Instruction::LoadLocal(index_slot),
        Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
        Instruction::Add,
        Instruction::StoreLocal(index_slot),
        Instruction::Jump(condition_target),
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    locals.remove(local_name);
    Ok(after_body)
}

fn for_in_parts(
    tokens: &[SpannedToken],
) -> Result<Option<(String, &[SpannedToken])>, CompileError> {
    let header = &tokens[1..];
    if !matches!(
        header.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) || !matches!(
        header.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(')'))
    ) {
        return Ok(None);
    }
    let clauses = &header[1..header.len() - 1];
    if clauses
        .iter()
        .any(|token| token.kind == TokenKind::Punctuation(';'))
    {
        return Ok(None);
    }
    let Some(separator) = clauses.iter().position(
        |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "in"),
    ) else {
        return Ok(None);
    };
    let declaration = &clauses[..separator];
    let iterable = &clauses[separator + 1..];
    if iterable.is_empty() {
        return Err(compile_error("for-in requires an iterable expression"));
    }
    if !matches!(
        declaration.first().map(|token| &token.kind),
        Some(TokenKind::Identifier(identifier)) if identifier == "var"
    ) {
        return Err(compile_error("for-in currently requires a declared var"));
    }
    // A typed loop declaration may carry a cast qualifier after the local,
    // e.g. `var/turf/area_turf as anything`.  The qualifier describes the
    // iteration mode, not a second local name.  Restrict the name search to
    // the declaration portion before `as`, otherwise the old reverse scan
    // incorrectly registered `anything` and left `area_turf` unresolved in
    // the loop body.
    let declaration_end = declaration
        .iter()
        .position(
            |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "as"),
        )
        .unwrap_or(declaration.len());
    let local_name = declaration[..declaration_end]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) if identifier != "var" => Some(identifier.clone()),
            _ => None,
        })
        .ok_or_else(|| compile_error("for-in variable declaration has no name"))?;
    Ok(Some((local_name, iterable)))
}

/// Recognizes `for(var/name in first to last [step increment])`, rather than treating the
/// range's `to` keyword as the beginning of a normal iterable expression.
#[allow(clippy::type_complexity)]
fn for_to_parts(
    tokens: &[SpannedToken],
) -> Result<
    Option<(
        String,
        &[SpannedToken],
        &[SpannedToken],
        Option<&[SpannedToken]>,
    )>,
    CompileError,
> {
    let Some((local_name, iterable)) = for_in_parts(tokens)? else {
        return Ok(None);
    };
    let separators = top_level_keyword_positions(iterable, "to");
    let [separator] = separators.as_slice() else {
        return Ok(None);
    };
    let start = &iterable[..*separator];
    let after_to = &iterable[*separator + 1..];
    // The first top-level `step` begins the increment expression. Subsequent
    // occurrences are ordinary identifiers inside that expression (for
    // example, `step step` when the increment is held in a local named
    // `step`).
    let step_separator = top_level_keyword_positions(after_to, "step")
        .into_iter()
        .next();
    let (end, step) = match step_separator {
        None => (after_to, None),
        Some(separator) => (&after_to[..separator], Some(&after_to[separator + 1..])),
    };
    if start.is_empty() || end.is_empty() {
        return Err(compile_error("for-to range requires both bounds"));
    }
    if step.is_some_and(<[SpannedToken]>::is_empty) {
        return Err(compile_error("for-to range step requires an increment"));
    }
    Ok(Some((local_name, start, end, step)))
}

/// Finds DM header keywords outside nested calls, indexes, and list literals.
/// Range bounds may legally refer to locals named `to` or `step` inside a
/// nested expression, so only a top-level occurrence can delimit a `for`
/// range clause.
fn top_level_keyword_positions(tokens: &[SpannedToken], keyword: &str) -> Vec<usize> {
    let mut depth = 0usize;
    let mut positions = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Identifier(identifier) if depth == 0 && identifier == keyword => {
                positions.push(index);
            }
            _ => {}
        }
    }
    positions
}

fn for_clauses(tokens: &[SpannedToken]) -> Result<[&[SpannedToken]; 3], CompileError> {
    let header = &tokens[1..];
    if !matches!(
        header.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) || !matches!(
        header.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(')'))
    ) {
        return Err(compile_error("C-style for requires a parenthesized header"));
    }
    let clauses = &header[1..header.len() - 1];
    let mut separators = Vec::new();
    let mut depth = 0_usize;
    for (index, token) in clauses.iter().enumerate() {
        match token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Punctuation(';') if depth == 0 => separators.push(index),
            _ => {}
        }
    }
    if separators.len() != 2 {
        if clauses.iter().any(
            |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "in"),
        ) {
            return Err(compile_error("for-in list iteration is not implemented"));
        }
        return Err(compile_error(
            "C-style for requires initializer, condition, and increment clauses separated by ';'",
        ));
    }
    Ok([
        &clauses[..separators[0]],
        &clauses[separators[0] + 1..separators[1]],
        &clauses[separators[1] + 1..],
    ])
}

fn compile_for_clause(
    tokens: &[SpannedToken],
    allow_declaration: bool,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<Option<String>, CompileError> {
    if tokens.is_empty() {
        return Ok(None);
    }
    if matches!(
        tokens.first().map(|token| &token.kind),
        Some(TokenKind::Identifier(identifier)) if identifier == "var"
    ) {
        if !allow_declaration {
            return Err(compile_error(
                "for increment clause cannot declare a local variable",
            ));
        }
        return compile_local(tokens, locals, instructions, procedures).map(Some);
    }
    if let [first, operator, expression @ ..] = tokens
        && let (TokenKind::Identifier(name), TokenKind::Operator(operator)) =
            (&first.kind, &operator.kind)
        && operator == "="
    {
        if let Some(slot) = locals.get(name) {
            compile_expression(expression, locals, instructions, procedures)?;
            instructions.push(Instruction::StoreLocal(slot));
        } else if let Some(field) = locals.src_field(name) {
            instructions.push(Instruction::LoadSrc);
            compile_expression(expression, locals, instructions, procedures)?;
            instructions.push(Instruction::StoreField(field.clone()));
        } else if let Some(global) = locals.global_field(name) {
            if operator != "=" {
                instructions.push(Instruction::LoadGlobal(global.clone()));
            }
            compile_expression(expression, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreGlobal(global.clone()));
        } else {
            return Err(compile_error(format!("unknown local {name:?}")));
        }
        return Ok(None);
    }
    if let Some((name, increment)) = local_increment(tokens) {
        let local = locals.get(name);
        let field = locals.src_field(name).cloned();
        if let Some(slot) = local {
            instructions.push(Instruction::LoadLocal(slot));
        } else if let Some(field) = &field {
            instructions.push(Instruction::LoadSrc);
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::LoadField(field.clone()));
        } else {
            return Err(compile_error(format!("unknown local {name:?}")));
        }
        instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
        instructions.push(if increment {
            Instruction::Add
        } else {
            Instruction::Subtract
        });
        if let Some(slot) = local {
            instructions.push(Instruction::StoreLocal(slot));
        } else {
            instructions.push(Instruction::StoreField(field.expect("field was checked")));
        }
        return Ok(None);
    }
    compile_expression(tokens, locals, instructions, procedures)?;
    instructions.push(Instruction::Pop);
    Ok(None)
}

fn local_increment(tokens: &[SpannedToken]) -> Option<(&str, bool)> {
    let [first, second] = tokens else {
        return None;
    };
    match (&first.kind, &second.kind) {
        (TokenKind::Identifier(name), TokenKind::Operator(operator))
        | (TokenKind::Operator(operator), TokenKind::Identifier(name))
            if matches!(operator.as_str(), "++" | "--") =>
        {
            Some((name, operator == "++"))
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_if(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    let line = &lines[line_index];
    let first_instruction = instructions.len();
    let condition = condition_tokens(&line.tokens, "if")?;
    compile_expression(condition, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - first_instruction,
    ));
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );
    // DM permits a single statement after the closing condition delimiter,
    // e.g. `if (ready) continue` and `if (missing) return`.  SourceLine
    // keeps that statement on the same physical line, so compile it through
    // the ordinary block machinery using a synthetic one-line block.  This
    // deliberately also preserves `break`/`continue` loop context and all
    // ordinary statement lowering instead of special-casing return here.
    let (after_then, then_falls_through) = if let Some(body) = inline_conditional_body(&line.tokens)
    {
        let mut inline_line = line.clone();
        inline_line.tokens = body.to_vec();
        let (_, falls_through) = compile_block(
            std::slice::from_ref(&inline_line),
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        (line_index + 1, falls_through)
    } else {
        let child_index = line_index + 1;
        let child = lines
            .get(child_index)
            .ok_or_else(|| compile_error("if statement requires an indented body"))?;
        let child_indentation = indentation(child);
        if child_indentation <= block_indentation {
            return Err(compile_error("if statement requires an indented body"));
        }
        compile_block(
            lines,
            child_index,
            child_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?
    };
    if !lines
        .get(after_then)
        .is_some_and(|candidate| is_else(candidate, block_indentation))
    {
        let end_target = instructions.len();
        patch_jump(instructions, false_jump, end_target)?;
        return Ok((after_then, true));
    }

    let else_line = &lines[after_then];
    let end_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::Jump(usize::MAX),
        else_line.span,
    );
    let else_target = instructions.len();
    patch_jump(instructions, false_jump, else_target)?;
    let (after_else, else_falls_through) = if is_else_if(else_line) {
        // `else if` is a nested conditional in DM.  Re-present the tail of
        // the source as an `if` block so its condition and any inline body
        // take the same lowering path as a top-level conditional.
        let mut nested_lines = lines[after_then..].to_vec();
        nested_lines[0].tokens = nested_lines[0].tokens[1..].to_vec();
        let (after_nested, falls_through) = compile_if(
            &nested_lines,
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        (after_then + after_nested, falls_through)
    } else if let Some(body) = inline_else_body(&else_line.tokens) {
        let mut inline_line = else_line.clone();
        inline_line.tokens = body.to_vec();
        let (_, falls_through) = compile_block(
            std::slice::from_ref(&inline_line),
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        (after_then + 1, falls_through)
    } else {
        let else_child_index = after_then + 1;
        let else_child = lines
            .get(else_child_index)
            .ok_or_else(|| compile_error("else statement requires an indented body"))?;
        let else_indentation = indentation(else_child);
        if else_indentation <= block_indentation {
            return Err(compile_error("else statement requires an indented body"));
        }
        compile_block(
            lines,
            else_child_index,
            else_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?
    };
    let end_target = instructions.len();
    patch_jump(instructions, end_jump, end_target)?;
    Ok((after_else, then_falls_through || else_falls_through))
}

/// Compiles DM's selector-based `switch` statement.
///
/// Unlike C, DM switch arms do not fall through.  Case arms are written as
/// `if(value)` (or `if(first to last)`) below the selector and are therefore
/// not ordinary conditional statements despite sharing their spelling.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_switch(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    let switch_line = &lines[line_index];
    let selector = condition_tokens(&switch_line.tokens, "switch")?;
    let selector_start = instructions.len();
    compile_expression(selector, locals, instructions, procedures)?;
    let selector_slot = locals.declare_hidden()?;
    push_instruction(
        instructions,
        source_spans,
        Instruction::StoreLocal(selector_slot),
        switch_line.span,
    );
    source_spans.extend(std::iter::repeat_n(
        switch_line.span,
        instructions.len() - selector_start - 1,
    ));

    let first_case_index = line_index + 1;
    let first_case = lines
        .get(first_case_index)
        .ok_or_else(|| compile_error("switch statement requires an indented case body"))?;
    let case_indentation = indentation(first_case);
    if case_indentation <= block_indentation {
        return Err(compile_error(
            "switch statement requires an indented case body",
        ));
    }

    let mut next_case_index = first_case_index;
    let mut end_jumps = Vec::new();
    let mut saw_default = false;
    while let Some(case_line) = lines.get(next_case_index) {
        let current_indentation = indentation(case_line);
        if current_indentation < case_indentation {
            break;
        }
        if current_indentation > case_indentation {
            return Err(compile_error("unexpected indentation in switch statement"));
        }
        let is_case = matches!(
            case_line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "if"
        );
        let is_default = matches!(
            case_line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "else"
        );
        if !is_case && !is_default {
            return Err(compile_error(
                "switch statement requires if cases or an else default",
            ));
        }
        if saw_default {
            return Err(compile_error("switch case cannot follow an else default"));
        }
        if is_default {
            if case_line.tokens.len() != 1 {
                return Err(compile_error(
                    "switch else default does not accept a condition",
                ));
            }
            saw_default = true;
        } else {
            let condition_start = instructions.len();
            emit_switch_case_condition(
                condition_tokens(&case_line.tokens, "switch case")?,
                selector_slot,
                locals,
                instructions,
                procedures,
            )?;
            source_spans.extend(std::iter::repeat_n(
                case_line.span,
                instructions.len() - condition_start,
            ));
        }
        let false_jump = if is_case {
            let jump = instructions.len();
            push_instruction(
                instructions,
                source_spans,
                Instruction::JumpIfFalse(usize::MAX),
                case_line.span,
            );
            Some(jump)
        } else {
            None
        };
        let body_index = next_case_index + 1;
        let body_line = lines
            .get(body_index)
            .ok_or_else(|| compile_error("switch case requires an indented body"))?;
        let body_indentation = indentation(body_line);
        if body_indentation <= case_indentation {
            return Err(compile_error("switch case requires an indented body"));
        }
        let (after_body, _) = compile_block(
            lines,
            body_index,
            body_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        if !saw_default {
            let end_jump = instructions.len();
            push_instruction(
                instructions,
                source_spans,
                Instruction::Jump(usize::MAX),
                case_line.span,
            );
            end_jumps.push(end_jump);
        }
        if let Some(jump) = false_jump {
            let next_case_target = instructions.len();
            patch_jump(instructions, jump, next_case_target)?;
        }
        next_case_index = after_body;
        if saw_default {
            if lines
                .get(next_case_index)
                .is_some_and(|next| indentation(next) == case_indentation)
            {
                return Err(compile_error("switch case cannot follow an else default"));
            }
            break;
        }
    }
    let end_target = instructions.len();
    for jump in end_jumps {
        patch_jump(instructions, jump, end_target)?;
    }
    Ok((next_case_index, true))
}

fn emit_switch_case_condition(
    tokens: &[SpannedToken],
    selector_slot: u16,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let alternatives = split_switch_tokens(tokens, ',')?;
    if alternatives.is_empty() {
        return Err(compile_error("switch case requires at least one value"));
    }
    for (alternative_index, alternative) in alternatives.iter().enumerate() {
        if alternative.is_empty() {
            return Err(compile_error("switch case contains an empty value"));
        }
        let range = split_switch_keyword(alternative, "to")?;
        if let Some((lower, upper)) = range {
            if lower.is_empty() || upper.is_empty() {
                return Err(compile_error("switch range requires both bounds"));
            }
            instructions.push(Instruction::LoadLocal(selector_slot));
            compile_expression(lower, locals, instructions, procedures)?;
            instructions.push(Instruction::GreaterEqual);
            instructions.push(Instruction::LoadLocal(selector_slot));
            compile_expression(upper, locals, instructions, procedures)?;
            instructions.push(Instruction::LessEqual);
            instructions.push(Instruction::And);
        } else {
            instructions.push(Instruction::LoadLocal(selector_slot));
            compile_expression(alternative, locals, instructions, procedures)?;
            instructions.push(Instruction::Equal);
        }
        if alternative_index > 0 {
            instructions.push(Instruction::Or);
        }
    }
    Ok(())
}

fn split_switch_tokens(
    tokens: &[SpannedToken],
    separator: char,
) -> Result<Vec<&[SpannedToken]>, CompileError> {
    let mut groups = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        match token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    compile_error("switch case contains unmatched closing punctuation")
                })?;
            }
            TokenKind::Punctuation(punctuation) if punctuation == separator && depth == 0 => {
                groups.push(&tokens[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(compile_error(
            "switch case contains unmatched opening punctuation",
        ));
    }
    groups.push(&tokens[start..]);
    Ok(groups)
}

#[allow(clippy::type_complexity)]
fn split_switch_keyword<'a>(
    tokens: &'a [SpannedToken],
    keyword: &str,
) -> Result<Option<(&'a [SpannedToken], &'a [SpannedToken])>, CompileError> {
    let mut depth = 0usize;
    let mut found = None;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    compile_error("switch range contains unmatched closing punctuation")
                })?;
            }
            TokenKind::Identifier(name)
                if name == keyword && depth == 0 && found.replace(index).is_some() =>
            {
                return Err(compile_error(
                    "switch range contains multiple 'to' keywords",
                ));
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(compile_error(
            "switch range contains unmatched opening punctuation",
        ));
    }
    Ok(found.map(|index| (&tokens[..index], &tokens[index + 1..])))
}

fn condition_tokens<'a>(
    tokens: &'a [SpannedToken],
    keyword: &str,
) -> Result<&'a [SpannedToken], CompileError> {
    let mut expression = &tokens[1..];
    // The preprocessor can retain the opening brace from a compact C-style
    // conditional such as `if (condition) {`.  Block structure remains
    // indentation-based in the lowered syntax, so it is not expression input.
    if matches!(
        expression.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation('{'))
    ) {
        expression = &expression[..expression.len() - 1];
    }
    if matches!(
        expression.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) {
        let mut depth = 0usize;
        for (index, token) in expression.iter().enumerate() {
            match &token.kind {
                TokenKind::Punctuation('(') => depth += 1,
                TokenKind::Punctuation(')') => {
                    depth = depth.checked_sub(1).ok_or_else(|| {
                        compile_error(format!("{keyword} condition is missing '('"))
                    })?;
                    if depth == 0 {
                        return Ok(&expression[1..index]);
                    }
                }
                _ => {}
            }
        }
        return Err(compile_error(format!("{keyword} condition is missing ')'")));
    }
    if expression.is_empty() {
        return Err(compile_error(format!("{keyword} requires a condition")));
    }
    Ok(expression)
}

/// Returns the statement written after a parenthesized conditional on the
/// same physical source line.  A trailing `{` belongs to the preprocessor's
/// compact brace form and is not an inline DM statement.
fn inline_conditional_body(tokens: &[SpannedToken]) -> Option<&[SpannedToken]> {
    let expression = tokens.get(1..)?;
    if !matches!(
        expression.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) {
        return None;
    }
    let mut depth = 0usize;
    for (index, token) in expression.iter().enumerate() {
        match token.kind {
            TokenKind::Punctuation('(') => depth += 1,
            TokenKind::Punctuation(')') => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let body = &expression[index + 1..];
                    return (!body.is_empty()
                        && !matches!(
                            body.first().map(|token| &token.kind),
                            Some(TokenKind::Punctuation('{'))
                        ))
                    .then_some(body);
                }
            }
            _ => {}
        }
    }
    None
}

/// Returns a body written directly after `else`, such as `else return`.
/// `else if` deliberately remains a nested conditional form and is handled
/// by the regular indented parser path.
fn inline_else_body(tokens: &[SpannedToken]) -> Option<&[SpannedToken]> {
    let body = tokens.get(1..)?;
    (!body.is_empty()
        && !matches!(
            body.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "if"
        )
        && !matches!(
            body.first().map(|token| &token.kind),
            Some(TokenKind::Punctuation('{'))
        ))
    .then_some(body)
}

fn is_else_if(line: &SourceLine) -> bool {
    matches!(
        line.tokens.as_slice(),
        [
            SpannedToken {
                kind: TokenKind::Identifier(else_keyword),
                ..
            },
            SpannedToken {
                kind: TokenKind::Identifier(if_keyword),
                ..
            },
            ..
        ] if else_keyword == "else" && if_keyword == "if"
    )
}

fn indentation(line: &SourceLine) -> usize {
    line.indentation
        .tabs
        .saturating_mul(8)
        .saturating_add(line.indentation.spaces)
}

fn is_else(line: &SourceLine, expected_indentation: usize) -> bool {
    indentation(line) == expected_indentation
        && matches!(
            line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "else"
        )
}

fn push_instruction(
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    instruction: Instruction,
    span: SourceSpan,
) {
    instructions.push(instruction);
    source_spans.push(span);
}

fn patch_jump(
    instructions: &mut [Instruction],
    instruction_index: usize,
    target: usize,
) -> Result<(), CompileError> {
    match instructions.get_mut(instruction_index) {
        Some(
            Instruction::JumpIfFalse(destination)
            | Instruction::Jump(destination)
            | Instruction::JumpIfArgumentSupplied {
                target: destination,
                ..
            },
        ) => {
            *destination = target;
            Ok(())
        }
        _ => Err(compile_error("compiler attempted to patch a non-jump")),
    }
}

fn compile_result_assignment(
    tokens: &[SpannedToken],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let Some(TokenKind::Operator(assignment)) = tokens.get(1).map(|token| &token.kind) else {
        return Err(compile_error(
            "special return value '.' requires an assignment",
        ));
    };
    if tokens.len() < 3 {
        return Err(compile_error(
            "special return value assignment requires an expression",
        ));
    }
    if assignment != "=" {
        instructions.push(Instruction::LoadResult);
    }
    compile_expression(&tokens[2..], locals, instructions, procedures)?;
    if assignment != "=" {
        instructions.push(match assignment.as_str() {
            "+=" => Instruction::Add,
            "-=" => Instruction::Subtract,
            "*=" => Instruction::Multiply,
            "/=" => Instruction::Divide,
            "%=" => Instruction::Remainder,
            "&=" => Instruction::BitAnd,
            "|=" => Instruction::BitOr,
            "^=" => Instruction::BitXor,
            "<<=" => Instruction::ShiftLeft,
            ">>=" => Instruction::ShiftRight,
            _ => {
                return Err(compile_error(format!(
                    "unsupported special return value assignment operator {assignment:?}"
                )));
            }
        });
    }
    instructions.push(Instruction::StoreResult);
    Ok(())
}

fn compile_local(
    tokens: &[SpannedToken],
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<String, CompileError> {
    let assignment = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="));
    let declaration_end = assignment.unwrap_or(tokens.len());
    let name = tokens[1..declaration_end]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.clone()),
            _ => None,
        })
        .ok_or_else(|| compile_error("local declaration has no name"))?;
    let slot = locals.declare(name.clone())?;
    if let Some(assignment) = assignment {
        compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
    } else {
        // Typed and untyped local declarations without an initializer begin
        // as null in DM.
        instructions.push(Instruction::PushNull);
    }
    instructions.push(Instruction::StoreLocal(slot));
    Ok(name)
}

fn parameter_name(tokens: &[SpannedToken]) -> Option<&str> {
    let end = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="))
        .unwrap_or(tokens.len());
    tokens[..end]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.as_str()),
            _ => None,
        })
}

fn to_local_index(index: usize) -> Result<u16, CompileError> {
    u16::try_from(index).map_err(|_| compile_error("procedure has more than 65536 locals"))
}

fn compile_expression(
    tokens: &[SpannedToken],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let expression = ExpressionParser::new(tokens).parse()?;
    emit_expression(&expression, locals, instructions, procedures)
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Expression {
    Null,
    Number(DmNumberBits),
    Text(String),
    TypePath(TypePath),
    New {
        type_path: Option<Box<Self>>,
        arguments: Vec<Self>,
    },
    Regex {
        arguments: Vec<Self>,
    },
    MutableAppearance {
        arguments: Vec<Self>,
    },
    ReplaceText {
        arguments: Vec<Self>,
        exact: bool,
        character_indices: bool,
    },
    CopyText {
        arguments: Vec<Self>,
        character_indices: bool,
    },
    StandardBuiltin {
        name: String,
        arguments: Vec<Self>,
    },
    Initial(Box<Self>),
    Block {
        arguments: Vec<Self>,
    },
    Rand {
        arguments: Vec<Self>,
    },
    Pick {
        entries: Vec<(Option<Self>, Self)>,
    },
    Prob(Box<Self>),
    Round {
        arguments: Vec<Self>,
    },
    Length {
        value: Box<Self>,
    },
    Ref {
        value: Box<Self>,
    },
    GetStep {
        source: Box<Self>,
        direction: Box<Self>,
    },
    Range {
        arguments: Vec<Self>,
    },
    TypesOf {
        value: Box<Self>,
    },
    TypePredicate {
        kind: TypePredicateKind,
        arguments: Vec<Self>,
    },
    Local(String),
    Src,
    Usr,
    GlobalNamespace,
    Field {
        receiver: Box<Self>,
        name: FieldName,
    },
    SafeField {
        receiver: Box<Self>,
        name: FieldName,
    },
    GlobalField(FieldName),
    Result,
    Call {
        procedure: String,
        arguments: Vec<Self>,
    },
    /// A list expansion used only in an enclosing call or constructor
    /// argument list (`target(arglist(values))`).
    ArgList(Box<Self>),
    Locate {
        arguments: Vec<Self>,
    },
    LocateIn {
        arguments: Vec<Self>,
        container: Box<Self>,
    },
    CurrentCall {
        arguments: Option<Vec<Self>>,
    },
    ParentCall {
        arguments: Option<Vec<Self>>,
    },
    DynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
    },
    SafeDynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
    },
    List(Vec<ListExpressionEntry>),
    Index {
        list: Box<Self>,
        index: Box<Self>,
    },
    SafeIndex {
        list: Box<Self>,
        index: Box<Self>,
    },
    Unary {
        operator: String,
        operand: Box<Self>,
    },
    Binary {
        operator: String,
        left: Box<Self>,
        right: Box<Self>,
    },
    Conditional {
        condition: Box<Self>,
        when_true: Box<Self>,
        when_false: Box<Self>,
    },
    Assignment {
        target: Box<Self>,
        operator: String,
        value: Box<Self>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ListExpressionEntry {
    Positional(Expression),
    Associative { key: Expression, value: Expression },
}

/// Numeric constants supplied by the BYOND language rather than by project
/// source. Keep this deliberately finite: an unrecognised identifier must
/// continue through ordinary local/field resolution and retain its useful
/// diagnostic instead of silently becoming a number.
fn dm_builtin_text_constant(identifier: &str) -> Option<&'static str> {
    match identifier {
        "UNIX" => Some("UNIX"),
        "MS_WINDOWS" => Some("MS Windows"),
        "MALE" => Some("male"),
        "FEMALE" => Some("female"),
        "NEUTER" => Some("neuter"),
        "PLURAL" => Some("plural"),
        _ => None,
    }
}

fn dm_builtin_numeric_constant(identifier: &str) -> Option<f32> {
    match identifier {
        "FALSE" | "BLEND_DEFAULT" => Some(0.0),
        "TRUE" | "BLEND_OVERLAY" | "KEEP_TOGETHER" | "NORTH" => Some(1.0),
        "BLEND_ADD" | "KEEP_APART" | "SOUTH" => Some(2.0),
        "BLEND_SUBTRACT" => Some(3.0),
        "BLEND_MULTIPLY" | "LONG_GLIDE" | "EAST" => Some(4.0),
        "BLEND_INSET_OVERLAY" | "NORTHEAST" => Some(5.0),
        "SOUTHEAST" => Some(6.0),
        "WEST" | "RESET_TRANSFORM" => Some(8.0),
        "NORTHWEST" => Some(9.0),
        "SOUTHWEST" => Some(10.0),
        "UP" | "RESET_COLOR" => Some(16.0),
        "DOWN" | "RESET_ALPHA" => Some(32.0),
        // Appearance flags are BYOND bitflags. Keep the complete contiguous
        // built-in flag family here rather than teaching project code about
        // individual flags as each one is encountered.
        // These make an overlay/image ignore the corresponding value
        // inherited from its parent.
        "PIXEL_SCALE" => Some(64.0),
        "TILE_BOUND" => Some(128.0),
        "INHERIT_ID" => Some(256.0),
        "NO_CLIENT_COLOR" => Some(512.0),
        "RESET_CONTENTS" => Some(1024.0),
        "PLANE_MASTER" => Some(2048.0),
        "PASS_MOUSE" => Some(4096.0),
        "TILE_MOVER" => Some(8192.0),
        _ => None,
    }
}

struct ExpressionParser<'a> {
    tokens: &'a [SpannedToken],
    index: usize,
}

impl<'a> ExpressionParser<'a> {
    const fn new(tokens: &'a [SpannedToken]) -> Self {
        Self { tokens, index: 0 }
    }

    fn parse(mut self) -> Result<Expression, CompileError> {
        let expression = self.parse_assignment()?;
        if self.index != self.tokens.len() {
            return Err(compile_error(format!(
                "unexpected token {:?} in expression",
                self.tokens[self.index].kind
            )));
        }
        Ok(expression)
    }

    /// Parses right-associative assignment expressions. DM permits an
    /// assignment anywhere an expression is accepted, for example
    /// `(GLOB.initialized = TRUE)` in a macro expansion.
    fn parse_assignment(&mut self) -> Result<Expression, CompileError> {
        let target = self.parse_conditional()?;
        let Some(TokenKind::Operator(operator)) =
            self.tokens.get(self.index).map(|token| &token.kind)
        else {
            return Ok(target);
        };
        if !matches!(
            operator.as_str(),
            "=" | "+="
                | "-="
                | "*="
                | "/="
                | "%="
                | "&="
                | "|="
                | "^="
                | "<<="
                | ">>="
                | "&&="
                | "||="
        ) {
            return Ok(target);
        }
        let operator = operator.clone();
        self.index += 1;
        let value = self.parse_assignment()?;
        if operator == "||=" {
            let assignment = Expression::Assignment {
                target: Box::new(target.clone()),
                operator: "=".to_owned(),
                value: Box::new(value),
            };
            return Ok(Expression::Conditional {
                condition: Box::new(target.clone()),
                when_true: Box::new(target),
                when_false: Box::new(assignment),
            });
        }
        if operator == "&&=" {
            let assignment = Expression::Assignment {
                target: Box::new(target.clone()),
                operator: "=".to_owned(),
                value: Box::new(value),
            };
            return Ok(Expression::Conditional {
                condition: Box::new(target.clone()),
                when_true: Box::new(assignment),
                when_false: Box::new(target),
            });
        }
        Ok(Expression::Assignment {
            target: Box::new(target),
            operator,
            value: Box::new(value),
        })
    }

    /// Parses DM's right-associative `condition ? when_true : when_false`
    /// expression.  It deliberately sits below every binary operator, so a
    /// condition such as `a || b ? c : d` is parsed as expected.
    fn parse_conditional(&mut self) -> Result<Expression, CompileError> {
        let condition = self.parse_binary(1)?;
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Operator(operator)) if operator == "?"
        ) {
            return Ok(condition);
        }
        self.index += 1;
        let when_true = self.parse_assignment()?;
        match self.tokens.get(self.index).map(|token| &token.kind) {
            Some(TokenKind::Operator(operator)) if operator == ":" => self.index += 1,
            _ => return Err(compile_error("expected ':' in conditional expression")),
        }
        let when_false = self.parse_assignment()?;
        Ok(Expression::Conditional {
            condition: Box::new(condition),
            when_true: Box::new(when_true),
            when_false: Box::new(when_false),
        })
    }

    fn parse_binary(&mut self, minimum_precedence: u8) -> Result<Expression, CompileError> {
        let mut left = self.parse_unary()?;
        while let Some(operator) = self.current_operator() {
            let Some(precedence) = binary_precedence(operator) else {
                break;
            };
            if precedence < minimum_precedence {
                break;
            }
            let operator = operator.to_owned();
            self.index += 1;
            let right_precedence = if operator == "**" {
                precedence
            } else {
                precedence + 1
            };
            let right = self.parse_binary(right_precedence)?;
            left = if operator == "in" {
                match left {
                    Expression::Locate { arguments } => Expression::LocateIn {
                        arguments,
                        container: Box::new(right),
                    },
                    left => Expression::Binary {
                        operator,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                }
            } else {
                Expression::Binary {
                    operator,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression, CompileError> {
        if let Some(operator @ ("!" | "+" | "-" | "~")) = self.current_operator() {
            let operator = operator.to_owned();
            self.index += 1;
            return Ok(Expression::Unary {
                operator,
                operand: Box::new(self.parse_unary()?),
            });
        }
        let mut expression = self.parse_primary()?;
        loop {
            let safe_list_index = matches!(self.current_operator(), Some("?["));
            let starts_list_index = matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('['))
            ) || safe_list_index;
            if starts_list_index {
                self.index += 1;
                let index = self.parse_binary(1)?;
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation(']'))
                ) {
                    return Err(compile_error("expected ']' after list index"));
                }
                self.index += 1;
                expression = if safe_list_index {
                    Expression::SafeIndex {
                        list: Box::new(expression),
                        index: Box::new(index),
                    }
                } else {
                    Expression::Index {
                        list: Box::new(expression),
                        index: Box::new(index),
                    }
                };
                continue;
            }
            if matches!(self.current_operator(), Some("." | "?." | "?:")) {
                let safe_member = matches!(self.current_operator(), Some("?." | "?:"));
                self.index += 1;
                let Some(TokenKind::Identifier(name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected a field name after member access"));
                };
                let name =
                    FieldName::parse(name).map_err(|error| compile_error(error.to_string()))?;
                self.index += 1;
                expression = if matches!(expression, Expression::GlobalNamespace) {
                    Expression::GlobalField(name)
                } else if safe_member {
                    Expression::SafeField {
                        receiver: Box::new(expression),
                        name,
                    }
                } else {
                    Expression::Field {
                        receiver: Box::new(expression),
                        name,
                    }
                };
                continue;
            }
            // A datum procedure call is a postfix operation in DM.  The
            // regular `name(...)` arm in `parse_primary` handles static
            // calls, while `receiver.name(...)` must retain both the datum
            // receiver and its dynamically-selected procedure name.  This
            // occurs extensively in lifecycle code after macro expansion
            // (for example signal dispatch helpers).
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('('))
            ) {
                expression = match expression {
                    Expression::Field { receiver, name } => {
                        let arguments = self.parse_call_arguments()?;
                        Expression::DynamicCall {
                            target: receiver,
                            procedure: Box::new(Expression::Text(name.as_str().to_owned())),
                            arguments,
                        }
                    }
                    Expression::SafeField { receiver, name } => {
                        let arguments = self.parse_call_arguments()?;
                        Expression::SafeDynamicCall {
                            target: receiver,
                            procedure: Box::new(Expression::Text(name.as_str().to_owned())),
                            arguments,
                        }
                    }
                    other => {
                        expression = other;
                        break;
                    }
                };
                continue;
            }
            break;
        }
        Ok(expression)
    }

    #[allow(clippy::too_many_lines)]
    fn parse_primary(&mut self) -> Result<Expression, CompileError> {
        let token = self
            .tokens
            .get(self.index)
            .ok_or_else(|| compile_error("expected an expression"))?;
        self.index += 1;
        match &token.kind {
            // Type paths are expression values in DM: `/obj/item/tool` is
            // distinct from text and is accepted by builtins such as
            // `istype`, `ispath`, and `new`. The lexer exposes every slash as
            // an operator, so consume the complete slash-delimited sequence
            // here before ordinary binary division is considered.
            TokenKind::Operator(operator) if operator == "/" => {
                let mut path = String::new();
                loop {
                    let Some(TokenKind::Identifier(segment)) =
                        self.tokens.get(self.index).map(|token| &token.kind)
                    else {
                        return Err(compile_error("expected a type path segment after '/'"));
                    };
                    path.push('/');
                    path.push_str(segment);
                    self.index += 1;
                    if !matches!(self.current_operator(), Some("/")) {
                        break;
                    }
                    self.index += 1;
                }
                TypePath::parse(&path)
                    .map(Expression::TypePath)
                    .map_err(|error| compile_error(error.to_string()))
            }
            TokenKind::Operator(operator)
                if operator == ".."
                    && matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) =>
            {
                let arguments = self.parse_call_arguments()?;
                Ok(Expression::ParentCall {
                    arguments: if arguments.is_empty() {
                        None
                    } else {
                        Some(arguments)
                    },
                })
            }
            TokenKind::Operator(operator)
                if operator == "."
                    && matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) =>
            {
                let arguments = self.parse_call_arguments()?;
                Ok(Expression::CurrentCall {
                    arguments: if arguments.is_empty() {
                        None
                    } else {
                        Some(arguments)
                    },
                })
            }
            TokenKind::Operator(operator) if operator == "." => Ok(Expression::Result),
            TokenKind::Number(spelling) => parse_number(spelling).map(Expression::Number),
            // A resource literal is a first-class DM value.  The headless VM
            // has no asset loader, so preserve its canonical path as text;
            // this allows resource-valued arguments and field assignments to
            // compile while retaining a deterministic value for inspection.
            TokenKind::String(text)
            | TokenKind::RawString(text)
            | TokenKind::TextBlock(text)
            | TokenKind::Resource(text) => Ok(Expression::Text(text.clone())),
            TokenKind::Identifier(identifier) if identifier == "null" => Ok(Expression::Null),
            TokenKind::Identifier(identifier)
                if let Some(value) = dm_builtin_numeric_constant(identifier) =>
            {
                Ok(Expression::Number(DmNumberBits::from_f32(value)))
            }
            TokenKind::Identifier(identifier)
                if let Some(value) = dm_builtin_text_constant(identifier) =>
            {
                Ok(Expression::Text(value.to_owned()))
            }
            TokenKind::Identifier(identifier) if identifier == "src" => Ok(Expression::Src),
            TokenKind::Identifier(identifier) if identifier == "usr" => Ok(Expression::Usr),
            // `GLOB` is the conventional SS13 alias for DM's built-in
            // `global` namespace. It is not a local datum and must lower to
            // the same persistent-global operations as the spelling BYOND
            // exposes directly.
            TokenKind::Identifier(identifier) if identifier == "global" || identifier == "GLOB" => {
                Ok(Expression::GlobalNamespace)
            }
            TokenKind::Identifier(identifier) if identifier == "new" => {
                // `new /path(args)` is the common explicit form.  An
                // unqualified `new(args)` constructs the current datum type.
                // Keep the constructor arguments in the AST even though the
                // headless VM currently only establishes object identity.
                if matches!(self.current_operator(), Some("/")) {
                    let type_path = self.parse_primary()?;
                    let arguments = if matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) {
                        self.parse_call_arguments()?
                    } else {
                        Vec::new()
                    };
                    Ok(Expression::New {
                        type_path: Some(Box::new(type_path)),
                        arguments,
                    })
                } else if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    Ok(Expression::New {
                        type_path: None,
                        arguments: self.parse_call_arguments()?,
                    })
                } else if let Some(TokenKind::Identifier(type_name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                {
                    // DM also permits a runtime type expression, for example
                    // `new starting_organ(src)`.  This is distinct from
                    // unqualified `new(...)`: the identifier is the type to
                    // instantiate, not a constructor argument.
                    // Do not delegate this to `parse_unary`: its ordinary
                    // identifier rule interprets the following `(` as a
                    // static procedure call. Here it belongs to `new`.
                    let type_path = Expression::Local(type_name.clone());
                    self.index += 1;
                    let arguments = if matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) {
                        self.parse_call_arguments()?
                    } else {
                        Vec::new()
                    };
                    Ok(Expression::New {
                        type_path: Some(Box::new(type_path)),
                        arguments,
                    })
                } else {
                    Ok(Expression::New {
                        type_path: None,
                        arguments: Vec::new(),
                    })
                }
            }
            TokenKind::Identifier(identifier) if identifier == "call" => {
                let selectors = self.parse_call_arguments()?;
                let (target, procedure) = match selectors.as_slice() {
                    [procedure] => (Expression::Null, procedure.clone()),
                    [target, procedure] => (target.clone(), procedure.clone()),
                    _ => {
                        return Err(compile_error(
                            "call requires a procedure or a receiver and procedure",
                        ));
                    }
                };
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    return Err(compile_error("call selector requires an argument list"));
                }
                Ok(Expression::DynamicCall {
                    target: Box::new(target),
                    procedure: Box::new(procedure),
                    arguments: self.parse_call_arguments()?,
                })
            }
            TokenKind::Identifier(identifier)
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) =>
            {
                if identifier == "list" {
                    Ok(Expression::List(self.parse_list_arguments()?))
                } else if identifier == "arglist" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "arglist requires exactly one list, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::ArgList(Box::new(
                        arguments.pop().expect("argument count was validated"),
                    )))
                } else if let Some(kind) = type_predicate_kind(identifier) {
                    let arguments = self.parse_call_arguments()?;
                    let valid_count = match kind {
                        TypePredicateKind::IsType | TypePredicateKind::IsPath => {
                            (1..=2).contains(&arguments.len())
                        }
                        // BYOND's location classifiers accept multiple values
                        // and succeed only when every supplied value matches.
                        TypePredicateKind::IsLoc
                        | TypePredicateKind::IsMovable
                        | TypePredicateKind::IsTurf => !arguments.is_empty(),
                        _ => arguments.len() == 1,
                    };
                    if !valid_count {
                        return Err(compile_error(format!(
                            "{identifier} received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::TypePredicate { kind, arguments })
                } else if identifier == "initial" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "initial requires exactly one variable reference, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Initial(Box::new(
                        arguments.pop().expect("validated initial argument"),
                    )))
                } else if identifier == "regex" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "regex requires a pattern and optional flags, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Regex { arguments })
                } else if identifier == "mutable_appearance" {
                    Ok(Expression::MutableAppearance {
                        arguments: self.parse_call_arguments()?,
                    })
                } else if let Some((exact, character_indices)) = replacetext_kind(identifier) {
                    let arguments = self.parse_call_arguments()?;
                    if !(3..=5).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "{identifier} requires text, needle, replacement, and optional start/end; received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::ReplaceText {
                        arguments,
                        exact,
                        character_indices,
                    })
                } else if matches!(identifier.as_str(), "copytext" | "copytext_char") {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=3).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "{identifier} requires text and optional start/end; received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::CopyText {
                        arguments,
                        character_indices: identifier == "copytext_char",
                    })
                } else if identifier == "length" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "length requires exactly one argument, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Length {
                        value: Box::new(
                            arguments
                                .pop()
                                .expect("length argument count was validated"),
                        ),
                    })
                } else if identifier == "ref" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "ref requires exactly one argument, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Ref {
                        value: Box::new(arguments.pop().expect("ref argument count was validated")),
                    })
                } else if identifier == "get_step" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 2 {
                        return Err(compile_error(format!(
                            "get_step requires exactly an atom/turf and direction, received {} arguments",
                            arguments.len()
                        )));
                    }
                    let direction = arguments
                        .pop()
                        .expect("get_step argument count was validated");
                    let source = arguments
                        .pop()
                        .expect("get_step argument count was validated");
                    Ok(Expression::GetStep {
                        source: Box::new(source),
                        direction: Box::new(direction),
                    })
                } else if identifier == "range" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "range requires a distance and optional center, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Range { arguments })
                } else if identifier == "block" {
                    let arguments = self.parse_call_arguments()?;
                    if !(arguments.len() == 2 || (3..=6).contains(&arguments.len())) {
                        return Err(compile_error(format!(
                            "block requires two turfs or three through six coordinates, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Block { arguments })
                } else if identifier == "typesof" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "typesof requires exactly one type argument, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::TypesOf {
                        value: Box::new(
                            arguments
                                .pop()
                                .expect("typesof argument count was validated"),
                        ),
                    })
                } else if identifier == "rand" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "rand requires one or two numeric bounds, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Rand { arguments })
                } else if identifier == "pick" {
                    Ok(Expression::Pick {
                        entries: self.parse_pick_arguments()?,
                    })
                } else if identifier == "prob" {
                    let arguments = self.parse_call_arguments()?;
                    let [chance] = arguments.as_slice() else {
                        return Err(compile_error(format!(
                            "prob requires exactly one percentage, received {} arguments",
                            arguments.len()
                        )));
                    };
                    Ok(Expression::Prob(Box::new(chance.clone())))
                } else if identifier == "round" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "round requires a number and optional multiple, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Round { arguments })
                } else if identifier == "locate" {
                    Ok(Expression::Locate {
                        arguments: self.parse_call_arguments()?,
                    })
                } else if identifier == "nameof" {
                    self.parse_nameof_expression()
                } else if let Some((minimum, maximum)) = standard_builtin_arity(identifier) {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() < minimum || arguments.len() > maximum {
                        return Err(compile_error(format!(
                            "{identifier} received {} arguments; expected {minimum} through {maximum}",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::StandardBuiltin {
                        name: identifier.clone(),
                        arguments,
                    })
                } else {
                    let arguments = self.parse_call_arguments()?;
                    Ok(Expression::Call {
                        procedure: identifier.clone(),
                        arguments,
                    })
                }
            }
            TokenKind::Identifier(identifier) => Ok(Expression::Local(identifier.clone())),
            TokenKind::Punctuation('(') => {
                let expression = self.parse_assignment()?;
                match self.tokens.get(self.index).map(|token| &token.kind) {
                    Some(TokenKind::Punctuation(')')) => {
                        self.index += 1;
                        Ok(expression)
                    }
                    found => Err(compile_error(format!(
                        "expected ')' after expression; found {found:?}; next {:?}",
                        self.tokens.get(self.index + 1).map(|token| &token.kind),
                    ))),
                }
            }
            _ => Err(compile_error(format!(
                "unexpected token {:?} in expression",
                token.kind
            ))),
        }
    }

    /// Parses BYOND's compile-time `nameof(reference)` form.
    ///
    /// The argument is a reference grammar rather than an ordinary runtime
    /// expression.  In particular, tgstation uses all of these shapes:
    /// `nameof(.proc/name)`, `nameof(/datum/example.proc/name)`, and
    /// `nameof(type::field)`.  Each evaluates to the referenced member's
    /// final textual component.  Retaining that component is sufficient for
    /// headless callback and signal registration and also supports
    /// `NAMEOF_STATIC` without pretending its compile-time reference is a
    /// datum field read.
    fn parse_nameof_expression(&mut self) -> Result<Expression, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut nesting = 0_usize;
        let mut final_name = None;
        loop {
            let token = self
                .tokens
                .get(self.index)
                .ok_or_else(|| compile_error("expected ')' after nameof reference"))?;
            match &token.kind {
                TokenKind::Punctuation('(') => nesting += 1,
                TokenKind::Punctuation(')') if nesting == 0 => {
                    self.index += 1;
                    break;
                }
                TokenKind::Punctuation(')') => nesting -= 1,
                TokenKind::Identifier(name) => final_name = Some(name.clone()),
                _ => {}
            }
            self.index += 1;
        }
        final_name
            .map(Expression::Text)
            .ok_or_else(|| compile_error("nameof requires a named reference"))
    }

    fn parse_call_arguments(&mut self) -> Result<Vec<Expression>, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut arguments = Vec::new();
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            loop {
                // BYOND permits keyword-style call arguments, e.g.
                // `do_after(user, 4 SECONDS, target = src)`.  The current
                // execution ABI is positional, but retaining the source
                // order here is still the correct lowering for its existing
                // subset and, importantly, lets the compiler continue on to
                // report the next unsupported construct instead of rejecting
                // the call syntax itself.
                if matches!(
                    (
                        self.tokens.get(self.index).map(|token| &token.kind),
                        self.tokens.get(self.index + 1).map(|token| &token.kind),
                    ),
                    (
                        Some(TokenKind::Identifier(_)),
                        Some(TokenKind::Operator(operator)),
                    ) if operator == "="
                ) {
                    self.index += 2;
                }
                arguments.push(self.parse_assignment()?);
                match self.tokens.get(self.index).map(|token| &token.kind) {
                    // DM's weighted `pick()` syntax separates a weight from
                    // its candidate with `;`, e.g. `pick(10; red, 1; blue)`.
                    // The headless call ABI is positional, so retaining both
                    // expressions is the most faithful representation it can
                    // currently carry.
                    Some(TokenKind::Punctuation(',' | ';')) => {
                        self.index += 1;
                        // DM accepts a trailing separator in a parenthesized
                        // argument list, including multiline calls.  Do not
                        // attempt to parse the closing parenthesis as the
                        // next argument expression.
                        if matches!(
                            self.tokens.get(self.index).map(|token| &token.kind),
                            Some(TokenKind::Punctuation(')'))
                        ) {
                            break;
                        }
                    }
                    Some(TokenKind::Punctuation(')')) => break,
                    _ => {
                        return Err(compile_error(
                            "expected ',' or ')' after procedure argument",
                        ));
                    }
                }
            }
        }
        self.index += 1;
        Ok(arguments)
    }

    /// Parses `pick()` entries while retaining its `weight; candidate` form.
    fn parse_pick_arguments(
        &mut self,
    ) -> Result<Vec<(Option<Expression>, Expression)>, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut entries = Vec::new();
        while !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            let first = self.parse_assignment()?;
            let entry = if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation(';'))
            ) {
                self.index += 1;
                (Some(first), self.parse_assignment()?)
            } else {
                (None, first)
            };
            entries.push(entry);
            match self.tokens.get(self.index).map(|token| &token.kind) {
                Some(TokenKind::Punctuation(',')) => self.index += 1,
                Some(TokenKind::Punctuation(')')) => break,
                _ => return Err(compile_error("expected ',' or ')' after pick entry")),
            }
        }
        if entries.is_empty() {
            return Err(compile_error("pick requires at least one candidate"));
        }
        self.index += 1;
        Ok(entries)
    }

    fn parse_list_arguments(&mut self) -> Result<Vec<ListExpressionEntry>, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut entries = Vec::new();
        while !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            // The unparenthesized `=` in a list literal introduces an
            // associative entry rather than an assignment expression. A
            // parenthesized assignment still reaches `parse_assignment` via
            // primary-expression parsing.
            let key_or_value = self.parse_conditional()?;
            if matches!(self.current_operator(), Some("=")) {
                self.index += 1;
                let value = self.parse_conditional()?;
                entries.push(ListExpressionEntry::Associative {
                    key: key_or_value,
                    value,
                });
            } else {
                entries.push(ListExpressionEntry::Positional(key_or_value));
            }
            match self.tokens.get(self.index).map(|token| &token.kind) {
                Some(TokenKind::Punctuation(',')) => self.index += 1,
                Some(TokenKind::Punctuation(')')) => break,
                _ => return Err(compile_error("expected ',' or ')' after list entry")),
            }
        }
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            return Err(compile_error("expected ')' after list entries"));
        }
        self.index += 1;
        Ok(entries)
    }

    fn current_operator(&self) -> Option<&str> {
        match self.tokens.get(self.index).map(|token| &token.kind) {
            Some(TokenKind::Operator(operator)) => Some(operator),
            Some(TokenKind::Identifier(identifier)) if identifier == "in" => Some(identifier),
            _ => None,
        }
    }
}

/// Classifies BYOND's four `replacetext` builtin spellings without treating
/// them as project procedures.  The `_char` variants use character positions,
/// while `Ex` means exact (case-sensitive) matching.
fn replacetext_kind(identifier: &str) -> Option<(bool, bool)> {
    match identifier {
        "replacetext" => Some((false, false)),
        "replacetextEx" => Some((true, false)),
        "replacetext_char" => Some((false, true)),
        "replacetextEx_char" => Some((true, true)),
        _ => None,
    }
}

/// Identifies the compiler-handled BYOND value predicates.
fn type_predicate_kind(identifier: &str) -> Option<TypePredicateKind> {
    match identifier {
        "isnull" => Some(TypePredicateKind::IsNull),
        "isnum" => Some(TypePredicateKind::IsNum),
        "ispath" => Some(TypePredicateKind::IsPath),
        "islist" => Some(TypePredicateKind::IsList),
        "ismovable" => Some(TypePredicateKind::IsMovable),
        "isturf" => Some(TypePredicateKind::IsTurf),
        "isloc" => Some(TypePredicateKind::IsLoc),
        "isicon" => Some(TypePredicateKind::IsIcon),
        "istype" => Some(TypePredicateKind::IsType),
        _ => None,
    }
}

fn parse_number(spelling: &str) -> Result<DmNumberBits, CompileError> {
    let normalized = spelling.replace('_', "");
    let value = if let Some(hexadecimal) = normalized
        .strip_prefix("0x")
        .or_else(|| normalized.strip_prefix("0X"))
    {
        let integer = u32::from_str_radix(hexadecimal, 16)
            .map_err(|error| compile_error(format!("invalid number {spelling:?}: {error}")))?;
        integer
            .to_string()
            .parse::<f32>()
            .expect("every u32 decimal spelling is a valid f32")
    } else {
        normalized
            .parse::<f32>()
            .map_err(|error| compile_error(format!("invalid number {spelling:?}: {error}")))?
    };
    Ok(DmNumberBits::from_f32(value))
}

const fn binary_precedence(operator: &str) -> Option<u8> {
    match operator.as_bytes() {
        b"||" => Some(1),
        b"&&" => Some(2),
        b"|" => Some(3),
        b"^" => Some(4),
        b"&" => Some(5),
        b"==" | b"!=" => Some(6),
        b"<<" | b">>" | b"<" | b"<=" | b">" | b">=" | b"in" => Some(7),
        b"+" | b"-" => Some(8),
        b"*" | b"/" | b"%" => Some(9),
        b"**" => Some(10),
        _ => None,
    }
}

#[allow(clippy::too_many_lines)]
/// Emits an associative-list key, preserving macro-expanded named arguments.
///
/// Macro wrappers such as `AddComponent(...)` expand named arguments into
/// `list(name = value)`. The original call grammar is no longer visible, so
/// an unbound bare name here is a textual associative key, not an assignment
/// target. Bound locals and fields retain their ordinary expression meaning.
fn emit_associative_list_key(
    key: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    if let Expression::Local(name) = key
        && locals.get(name).is_none()
        && locals.src_field(name).is_none()
        && locals.global_field(name).is_none()
    {
        instructions.push(Instruction::PushText(name.clone()));
        return Ok(());
    }
    emit_expression(key, locals, instructions, procedures)
}

/// Marker used by call-like instructions to consume the count produced by
/// [`Instruction::ExpandArgumentLists`].  A source procedure cannot have
/// this many arguments, so it is unambiguous in the compact bytecode ABI.
const EXPANDED_ARGUMENT_COUNT: u16 = u16::MAX;

/// Emits a call argument vector, retaining BYOND's runtime `arglist()`
/// expansion semantics.  Ordinary expressions preserve the compact static
/// count; an expansion emits a small preparation instruction and returns the
/// sentinel consumed by the following call-like instruction.
fn emit_call_arguments(
    arguments: &[Expression],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<u16, CompileError> {
    let argument_count = u16::try_from(arguments.len())
        .map_err(|_| compile_error("call has more than 65535 positional arguments"))?;
    let mut expanded_indices = Vec::new();
    for (index, argument) in arguments.iter().enumerate() {
        if let Expression::ArgList(value) = argument {
            expanded_indices.push(to_local_index(index)?);
            emit_expression(value, locals, instructions, procedures)?;
        } else {
            emit_expression(argument, locals, instructions, procedures)?;
        }
    }
    if expanded_indices.is_empty() {
        Ok(argument_count)
    } else {
        instructions.push(Instruction::ExpandArgumentLists {
            argument_count,
            expanded_indices,
        });
        Ok(EXPANDED_ARGUMENT_COUNT)
    }
}

#[allow(clippy::too_many_lines)]
fn emit_expression(
    expression: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match expression {
        Expression::Null => instructions.push(Instruction::PushNull),
        Expression::Number(number) => instructions.push(Instruction::PushNumber(*number)),
        Expression::Text(text) => instructions.push(Instruction::PushText(text.clone())),
        Expression::TypePath(path) => instructions.push(Instruction::PushTypePath(path.clone())),
        Expression::New {
            type_path,
            arguments,
        } => {
            if let Some(type_path) = type_path {
                emit_expression(type_path, locals, instructions, procedures)?;
            }
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(if type_path.is_some() {
                Instruction::AllocateDatum { argument_count }
            } else {
                Instruction::AllocateCurrentDatum { argument_count }
            });
        }
        Expression::Regex { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeRegex {
                argument_count: u8::try_from(arguments.len())
                    .expect("regex argument count was validated by the parser"),
            });
        }
        Expression::MutableAppearance { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeMutableAppearance {
                argument_count: u16::try_from(arguments.len()).map_err(|_| {
                    compile_error("mutable_appearance has more than 65535 constructor arguments")
                })?,
            });
        }
        Expression::ReplaceText {
            arguments,
            exact,
            character_indices,
        } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::ReplaceText {
                argument_count: u8::try_from(arguments.len())
                    .expect("replacetext argument count was validated by the parser"),
                exact: *exact,
                character_indices: *character_indices,
            });
        }
        Expression::CopyText {
            arguments,
            character_indices,
        } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::CopyText {
                argument_count: u8::try_from(arguments.len())
                    .expect("copytext argument count was validated by the parser"),
                character_indices: *character_indices,
            });
        }
        Expression::Block { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Block {
                argument_count: u8::try_from(arguments.len())
                    .expect("block argument count was validated by the parser"),
            });
        }
        Expression::Length { value } => {
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Length);
        }
        Expression::Ref { value } => {
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Ref);
        }
        Expression::GetStep { source, direction } => {
            emit_expression(source, locals, instructions, procedures)?;
            emit_expression(direction, locals, instructions, procedures)?;
            instructions.push(Instruction::GetStep);
        }
        Expression::Range { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Range {
                argument_count: u8::try_from(arguments.len())
                    .expect("range argument count was validated by the parser"),
            });
        }
        Expression::TypesOf { value } => {
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::TypesOf);
        }
        Expression::Rand { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Rand {
                argument_count: u8::try_from(arguments.len())
                    .expect("rand argument count was validated by the parser"),
            });
        }
        Expression::Pick { entries } => {
            let mut weighted = Vec::with_capacity(entries.len());
            for (weight, candidate) in entries {
                weighted.push(weight.is_some());
                if let Some(weight) = weight {
                    emit_expression(weight, locals, instructions, procedures)?;
                }
                emit_expression(candidate, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Pick { weighted });
        }
        Expression::Prob(chance) => {
            emit_expression(chance, locals, instructions, procedures)?;
            instructions.push(Instruction::Prob);
        }
        Expression::Round { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Round {
                argument_count: u8::try_from(arguments.len())
                    .expect("round argument count was validated by the parser"),
            });
        }
        Expression::TypePredicate { kind, arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::TypePredicate {
                kind: *kind,
                argument_count: u8::try_from(arguments.len())
                    .expect("predicate argument count was validated by the parser"),
            });
        }
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                instructions.push(Instruction::LoadLocal(slot));
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::LoadField(field.clone()));
            } else if let Some(global) = locals.global_field(name) {
                instructions.push(Instruction::LoadGlobal(global.clone()));
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::Src => instructions.push(Instruction::LoadSrc),
        Expression::Usr => instructions.push(Instruction::LoadUsr),
        Expression::GlobalNamespace => {
            return Err(compile_error("global namespace requires a field name"));
        }
        Expression::Field { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::LoadField(name.clone()));
        }
        Expression::SafeField { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            instructions.push(Instruction::LoadField(name.clone()));
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::GlobalField(name) => {
            instructions.push(Instruction::LoadGlobal(name.clone()));
        }
        Expression::Result => instructions.push(Instruction::LoadResult),
        Expression::ArgList(_) => {
            return Err(compile_error(
                "arglist may only appear in a call or constructor argument list",
            ));
        }
        Expression::StandardBuiltin { name, arguments } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("native builtin has more than 65535 arguments"))?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::StandardBuiltin {
                name: name.clone(),
                argument_count,
            });
        }
        Expression::Initial(reference) => match reference.as_ref() {
            Expression::Field { receiver, name } => {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::InitialField(name.clone()));
            }
            Expression::Local(name) => {
                let field = locals.src_field(name).ok_or_else(|| {
                    compile_error(format!("initial target {name:?} is not an instance field"))
                })?;
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::InitialField(field.clone()));
            }
            _ => return Err(compile_error("initial requires a field reference")),
        },
        Expression::Call {
            procedure,
            arguments,
        } => {
            let target = procedures
                .get(procedure)
                .copied()
                .ok_or_else(|| compile_error(format!("unknown procedure {procedure:?}")))?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::Call {
                procedure: target,
                argument_count,
            });
        }
        Expression::Locate { arguments } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("locate has more than 65535 positional arguments"))?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Locate { argument_count });
        }
        Expression::LocateIn {
            arguments,
            container,
        } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("locate has more than 65535 positional arguments"))?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            emit_expression(container, locals, instructions, procedures)?;
            instructions.push(Instruction::LocateIn { argument_count });
        }
        Expression::CurrentCall { arguments } => {
            let argument_count = if let Some(arguments) = arguments {
                Some(emit_call_arguments(
                    arguments,
                    locals,
                    instructions,
                    procedures,
                )?)
            } else {
                None
            };
            instructions.push(Instruction::CallCurrent { argument_count });
        }
        Expression::ParentCall { arguments } => {
            let argument_count = if let Some(arguments) = arguments {
                Some(emit_call_arguments(
                    arguments,
                    locals,
                    instructions,
                    procedures,
                )?)
            } else {
                None
            };
            instructions.push(Instruction::CallParent {
                procedure: procedures.get("..").copied(),
                argument_count,
            });
        }
        Expression::DynamicCall {
            target,
            procedure,
            arguments,
        } => {
            emit_expression(target, locals, instructions, procedures)?;
            emit_expression(procedure, locals, instructions, procedures)?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic { argument_count });
        }
        Expression::SafeDynamicCall {
            target,
            procedure,
            arguments,
        } => {
            emit_expression(target, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(procedure, locals, instructions, procedures)?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic { argument_count });
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::List(entries) => {
            let mut kinds = Vec::with_capacity(entries.len());
            for entry in entries {
                match entry {
                    ListExpressionEntry::Positional(value) => {
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Positional);
                    }
                    ListExpressionEntry::Associative { key, value } => {
                        emit_associative_list_key(key, locals, instructions, procedures)?;
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Associative);
                    }
                }
            }
            instructions.push(Instruction::MakeListEntries(kinds));
        }
        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::IndexList);
        }
        Expression::SafeIndex { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::IndexList);
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::Unary { operator, operand } => {
            emit_expression(operand, locals, instructions, procedures)?;
            match operator.as_str() {
                "+" => {}
                "-" => instructions.push(Instruction::Negate),
                "!" => instructions.push(Instruction::Not),
                "~" => instructions.push(Instruction::BitNot),
                _ => {
                    return Err(compile_error(format!(
                        "unsupported unary operator {operator}"
                    )));
                }
            }
        }
        Expression::Binary {
            operator,
            left,
            right,
        } => {
            emit_expression(left, locals, instructions, procedures)?;
            emit_expression(right, locals, instructions, procedures)?;
            instructions.push(match operator.as_str() {
                "+" => Instruction::Add,
                "-" => Instruction::Subtract,
                "*" => Instruction::Multiply,
                "**" => Instruction::Power,
                "/" => Instruction::Divide,
                "%" => Instruction::Remainder,
                "&" => Instruction::BitAnd,
                "|" => Instruction::BitOr,
                "^" => Instruction::BitXor,
                "<<" => Instruction::ShiftLeft,
                ">>" => Instruction::ShiftRight,
                "==" => Instruction::Equal,
                "!=" => Instruction::NotEqual,
                "in" => Instruction::Contains,
                "<" => Instruction::Less,
                "<=" => Instruction::LessEqual,
                ">" => Instruction::Greater,
                ">=" => Instruction::GreaterEqual,
                "&&" => Instruction::And,
                "||" => Instruction::Or,
                _ => {
                    return Err(compile_error(format!(
                        "unsupported binary operator {operator}"
                    )));
                }
            });
        }
        Expression::Conditional {
            condition,
            when_true,
            when_false,
        } => {
            emit_expression(condition, locals, instructions, procedures)?;
            let false_jump = instructions.len();
            instructions.push(Instruction::JumpIfFalse(usize::MAX));
            emit_expression(when_true, locals, instructions, procedures)?;
            let end_jump = instructions.len();
            instructions.push(Instruction::Jump(usize::MAX));
            let false_target = instructions.len();
            patch_jump(instructions, false_jump, false_target)?;
            emit_expression(when_false, locals, instructions, procedures)?;
            let end_target = instructions.len();
            patch_jump(instructions, end_jump, end_target)?;
        }
        Expression::Assignment {
            target,
            operator,
            value,
        } => emit_assignment_expression(target, operator, value, locals, instructions, procedures)?,
    }
    Ok(())
}

fn emit_assignment_expression(
    target: &Expression,
    operator: &str,
    value: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match target {
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                if operator != "=" {
                    instructions.push(Instruction::LoadLocal(slot));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::StoreLocal(slot));
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                if operator != "=" {
                    instructions.push(Instruction::Duplicate);
                    instructions.push(Instruction::LoadField(field.clone()));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::StoreField(field.clone()));
            } else if let Some(global) = locals.global_field(name) {
                if operator != "=" {
                    instructions.push(Instruction::LoadGlobal(global.clone()));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::StoreGlobal(global.clone()));
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::GlobalField(name) => {
            if operator != "=" {
                instructions.push(Instruction::LoadGlobal(name.clone()));
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreGlobal(name.clone()));
        }
        Expression::Field { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreFieldKeep(name.clone()));
        }
        Expression::SafeField { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreFieldKeep(name.clone()));
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            if operator == "=" {
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::SetListIndexKeep);
            } else {
                // CompoundListIndex consumes the list, key, and right operand
                // and leaves no value, so retain an independent copy of the
                // computed result is not possible without a temporary. Keep
                // compound assignment expressions explicit until the VM has a
                // value-preserving variant.
                return Err(compile_error(
                    "compound list assignment is not supported as an expression",
                ));
            }
        }
        Expression::SafeIndex { list, index } => {
            if operator != "=" {
                return Err(compile_error(
                    "compound null-conditional list assignment is not supported as an expression",
                ));
            }
            emit_expression(list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(index, locals, instructions, procedures)?;
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::SetListIndexKeep);
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        _ => return Err(compile_error("assignment target is not writable")),
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn bind_initializer_expression(
    expression: &mut Expression,
    bindings: &BTreeMap<String, InitializerBinding>,
) -> Result<(), CompileError> {
    match expression {
        Expression::Local(name) => {
            let binding = bindings
                .get(name)
                .ok_or_else(|| compile_error(format!("unresolved initializer name {name:?}")))?;
            *expression = match binding {
                InitializerBinding::Global(field) => Expression::GlobalField(field.clone()),
                InitializerBinding::SrcField(field) => Expression::Field {
                    receiver: Box::new(Expression::Src),
                    name: field.clone(),
                },
            };
        }
        Expression::Field { receiver, .. } | Expression::SafeField { receiver, .. } => {
            bind_initializer_expression(receiver, bindings)?;
        }
        Expression::Call { arguments, .. }
        | Expression::StandardBuiltin { arguments, .. }
        | Expression::Regex { arguments }
        | Expression::MutableAppearance { arguments }
        | Expression::ReplaceText { arguments, .. }
        | Expression::CopyText { arguments, .. }
        | Expression::Block { arguments }
        | Expression::Rand { arguments }
        | Expression::Round { arguments }
        | Expression::Range { arguments }
        | Expression::TypePredicate { arguments, .. }
        | Expression::Locate { arguments } => {
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::Length { value }
        | Expression::Ref { value }
        | Expression::TypesOf { value }
        | Expression::Initial(value) => {
            bind_initializer_expression(value, bindings)?;
        }
        Expression::ArgList(value) => bind_initializer_expression(value, bindings)?,
        Expression::GetStep { source, direction } => {
            bind_initializer_expression(source, bindings)?;
            bind_initializer_expression(direction, bindings)?;
        }
        Expression::Prob(chance) => bind_initializer_expression(chance, bindings)?,
        Expression::Pick { entries } => {
            for (weight, candidate) in entries {
                if let Some(weight) = weight {
                    bind_initializer_expression(weight, bindings)?;
                }
                bind_initializer_expression(candidate, bindings)?;
            }
        }
        Expression::New {
            type_path,
            arguments,
        } => {
            if let Some(type_path) = type_path {
                bind_initializer_expression(type_path, bindings)?;
            }
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::LocateIn {
            arguments,
            container,
        } => {
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
            bind_initializer_expression(container, bindings)?;
        }
        Expression::DynamicCall {
            target,
            procedure,
            arguments,
        }
        | Expression::SafeDynamicCall {
            target,
            procedure,
            arguments,
        } => {
            bind_initializer_expression(target, bindings)?;
            bind_initializer_expression(procedure, bindings)?;
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::List(entries) => {
            for entry in entries {
                match entry {
                    ListExpressionEntry::Positional(value) => {
                        bind_initializer_expression(value, bindings)?;
                    }
                    ListExpressionEntry::Associative { key, value } => {
                        bind_initializer_expression(key, bindings)?;
                        bind_initializer_expression(value, bindings)?;
                    }
                }
            }
        }
        Expression::Index { list, index } | Expression::SafeIndex { list, index } => {
            bind_initializer_expression(list, bindings)?;
            bind_initializer_expression(index, bindings)?;
        }
        Expression::Unary { operand, .. } => {
            bind_initializer_expression(operand, bindings)?;
        }
        Expression::Binary { left, right, .. } => {
            bind_initializer_expression(left, bindings)?;
            bind_initializer_expression(right, bindings)?;
        }
        Expression::Conditional {
            condition,
            when_true,
            when_false,
        } => {
            bind_initializer_expression(condition, bindings)?;
            bind_initializer_expression(when_true, bindings)?;
            bind_initializer_expression(when_false, bindings)?;
        }
        Expression::Assignment { target, value, .. } => {
            bind_initializer_expression(target, bindings)?;
            bind_initializer_expression(value, bindings)?;
        }
        Expression::CurrentCall { .. } | Expression::ParentCall { .. } | Expression::Result => {
            return Err(compile_error(
                "current-procedure state is unavailable in a variable initializer",
            ));
        }
        Expression::Null
        | Expression::Number(_)
        | Expression::Text(_)
        | Expression::TypePath(_)
        | Expression::Src
        | Expression::Usr
        | Expression::GlobalNamespace
        | Expression::GlobalField(_) => {}
    }
    Ok(())
}

fn compile_error(message: impl Into<String>) -> CompileError {
    CompileError {
        message: message.into(),
    }
}

/// Limits applied by the deterministic reference interpreter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionLimits {
    /// Maximum number of simultaneously active procedure frames.
    pub max_call_depth: usize,
    /// Maximum total bytecode instructions executed across all call frames.
    pub max_steps: u64,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_call_depth: 1_024,
            max_steps: 10_000_000,
        }
    }
}

/// Mutable heap state shared by executions in one runtime world.
///
/// Values contain only stable logical handles. All mutable list and datum
/// storage remains here so aliases across calls resolve to one identity.
#[derive(Default)]
pub struct ExecutionState {
    heap: ValueHeap,
    globals: BTreeMap<FieldName, Value>,
    type_paths: Arc<std::collections::BTreeSet<TypePath>>,
    type_parents: Arc<BTreeMap<TypePath, Option<TypePath>>>,
    initial_values: Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>>,
    project_root: Option<Arc<PathBuf>>,
    random_state: u64,
}

impl ExecutionState {
    /// Creates an execution state with an empty value heap.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates execution state around an existing runtime heap.
    #[must_use]
    pub fn from_heap(heap: ValueHeap) -> Self {
        Self {
            heap,
            globals: BTreeMap::new(),
            type_paths: Arc::new(std::collections::BTreeSet::new()),
            type_parents: Arc::new(BTreeMap::new()),
            initial_values: Arc::new(BTreeMap::new()),
            project_root: None,
            random_state: 0,
        }
    }

    /// Returns ownership of the runtime heap after execution.
    #[must_use]
    pub fn into_heap(self) -> ValueHeap {
        self.heap
    }

    /// Returns the shared value heap.
    #[must_use]
    pub const fn heap(&self) -> &ValueHeap {
        &self.heap
    }

    /// Returns the shared mutable value heap.
    #[must_use]
    pub const fn heap_mut(&mut self) -> &mut ValueHeap {
        &mut self.heap
    }

    /// Reads a persistent runtime global.
    #[must_use]
    pub fn global(&self, name: &FieldName) -> Option<&Value> {
        self.globals.get(name)
    }

    /// Inserts or replaces a persistent runtime global.
    pub fn set_global(&mut self, name: FieldName, value: Value) -> Option<Value> {
        self.globals.insert(name, value)
    }

    /// Deletes a persistent runtime global.
    pub fn delete_global(&mut self, name: &FieldName) -> Option<Value> {
        self.globals.remove(name)
    }

    /// Replaces the canonical type catalog used by `typesof()`.
    pub fn set_type_paths(&mut self, paths: impl IntoIterator<Item = TypePath>) {
        self.type_paths = Arc::new(paths.into_iter().collect());
    }

    /// Replaces the canonical type catalog used by `typesof()` with a shared
    /// immutable catalog.
    ///
    /// Runtime images use this to avoid cloning a project's complete object
    /// tree for every dynamically evaluated initializer.
    pub fn set_shared_type_paths(&mut self, paths: Arc<std::collections::BTreeSet<TypePath>>) {
        self.type_paths = paths;
    }

    /// Iterates the canonical type catalog in lexical path order.
    pub fn type_paths(&self) -> impl Iterator<Item = &TypePath> {
        self.type_paths.iter()
    }

    /// Replaces the runtime type-parent catalog used by subtype and `parent_type` lookups.
    pub fn set_type_parents(&mut self, parents: BTreeMap<TypePath, Option<TypePath>>) {
        self.type_parents = Arc::new(parents);
    }

    /// Replaces effective compile-time initial field values for every runtime type.
    pub fn set_initial_values(&mut self, values: BTreeMap<TypePath, BTreeMap<FieldName, Value>>) {
        self.initial_values = Arc::new(values);
    }

    /// Sets the project root used by BYOND filesystem procedures such as `fexists()`.
    pub fn set_project_root(&mut self, root: PathBuf) {
        self.project_root = Some(Arc::new(root));
    }

    /// Returns a type's runtime parent when the catalog contains that type.
    #[must_use]
    pub fn type_parent(&self, path: &TypePath) -> Option<&TypePath> {
        self.type_parents.get(path).and_then(Option::as_ref)
    }

    /// Returns one effective compile-time initial value when available.
    #[must_use]
    pub fn initial_value(&self, path: &TypePath, field: &FieldName) -> Option<&Value> {
        self.initial_values
            .get(path)
            .and_then(|fields| fields.get(field))
    }

    /// Returns the project root used for relative filesystem paths.
    #[must_use]
    pub fn project_root(&self) -> Option<&std::path::Path> {
        self.project_root.as_deref().map(PathBuf::as_path)
    }

    /// Iterates globals in canonical field-name order for snapshots.
    pub fn globals(&self) -> impl Iterator<Item = (&FieldName, &Value)> {
        self.globals.iter()
    }
}

/// Entry-frame object context retained across a procedure call chain.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionContext {
    src: Value,
    usr: Value,
}

impl ExecutionContext {
    /// Creates a context with explicit `src` and `usr` values.
    #[must_use]
    pub const fn new(src: Value, usr: Value) -> Self {
        Self { src, usr }
    }

    /// Returns the current source object.
    #[must_use]
    pub const fn src(&self) -> &Value {
        &self.src
    }

    /// Returns the current user object.
    #[must_use]
    pub const fn usr(&self) -> &Value {
        &self.usr
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            src: Value::Null,
            usr: Value::Null,
        }
    }
}

#[derive(Debug)]
struct CallFrame {
    procedure: ProcedureId,
    instruction: usize,
    locals: Vec<Value>,
    stack: Vec<Value>,
    result: Value,
    src: Value,
    usr: Value,
    // Retain all supplied values for the future DM `args` list, including
    // extras beyond the declared parameter slots.
    arguments: Vec<Value>,
}

/// Executes one standalone program to completion on the reference interpreter.
///
/// Calls cannot occur in a standalone program; use [`execute_module`] for
/// programs produced by [`compile_module`].
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode stack/local access or
/// operations on values of unsupported types.
pub fn execute(program: &Program, arguments: &[Value]) -> Result<Value, RuntimeError> {
    execute_with_limits(program, arguments, ExecutionLimits::default())
}

/// Executes one standalone program against persistent runtime state.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode or value operations.
pub fn execute_in_state(
    program: &Program,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    execute_with_limits_in_state(program, arguments, ExecutionLimits::default(), state)
}

/// Executes one standalone program with persistent state and object context.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode or value operations.
pub fn execute_in_context(
    program: &Program,
    arguments: &[Value],
    state: &mut ExecutionState,
    context: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    let entry = ProcedureId(0);
    let module = Module {
        procedures: vec![program.clone()],
        paths: vec!["<standalone>".to_owned()],
        names: HashMap::new(),
    };
    execute_module_with_limits_in_context(
        &module,
        entry,
        arguments,
        ExecutionLimits::default(),
        state,
        context,
    )
}

/// Executes one standalone program with explicit deterministic safety limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for the same failures as [`execute`], including
/// call-depth or total-instruction budget exhaustion.
pub fn execute_with_limits(
    program: &Program,
    arguments: &[Value],
    limits: ExecutionLimits,
) -> Result<Value, RuntimeError> {
    let mut state = ExecutionState::new();
    execute_with_limits_in_state(program, arguments, limits, &mut state)
}

/// Executes one standalone program with persistent state and explicit limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode, unsupported value operations,
/// stale heap references, or safety-limit exhaustion.
pub fn execute_with_limits_in_state(
    program: &Program,
    arguments: &[Value],
    limits: ExecutionLimits,
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    let entry = ProcedureId(0);
    let module = Module {
        procedures: vec![program.clone()],
        paths: vec!["<standalone>".to_owned()],
        names: HashMap::new(),
    };
    execute_module_with_limits_in_state(&module, entry, arguments, limits, state)
}

/// Executes a procedure from a compiled module with default safety limits.
///
/// Declared parameters are bound positionally. Missing parameters are `null`,
/// and extra supplied values are retained in the frame for future `args`
/// support, matching DM's permissive call arity.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid procedure identities or bytecode,
/// unsupported value operations, and call-depth exhaustion.
pub fn execute_module(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
) -> Result<Value, RuntimeError> {
    execute_module_with_limits(module, entry, arguments, ExecutionLimits::default())
}

/// Executes a module procedure against persistent runtime state.
///
/// # Errors
///
/// Returns [`RuntimeError`] for the same failures as [`execute_module`].
pub fn execute_module_in_state(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    execute_module_with_limits_in_state(module, entry, arguments, ExecutionLimits::default(), state)
}

/// Executes a module procedure with persistent state and object context.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode or value operations.
pub fn execute_module_in_context(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    state: &mut ExecutionState,
    context: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    execute_module_with_limits_in_context(
        module,
        entry,
        arguments,
        ExecutionLimits::default(),
        state,
        context,
    )
}

/// Executes a module procedure with explicit deterministic safety limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for the same failures as [`execute_module`].
pub fn execute_module_with_limits(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    limits: ExecutionLimits,
) -> Result<Value, RuntimeError> {
    let mut state = ExecutionState::new();
    execute_module_with_limits_in_state(module, entry, arguments, limits, &mut state)
}

/// Executes a module procedure against persistent state with explicit limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode, unsupported value operations,
/// stale heap references, or safety-limit exhaustion.
pub fn execute_module_with_limits_in_state(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    limits: ExecutionLimits,
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    execute_module_with_limits_in_context(
        module,
        entry,
        arguments,
        limits,
        state,
        &ExecutionContext::default(),
    )
}

/// Executes a module procedure with persistent state, context, and limits.
///
/// Current, parent, and resolved procedure calls inherit both `src` and `usr`
/// unchanged from their caller frame.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode, value operations, stale
/// handles, missing fields/globals, or safety-limit exhaustion.
pub fn execute_module_with_limits_in_context(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    limits: ExecutionLimits,
    state: &mut ExecutionState,
    context: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    let Some(program) = module.procedure(entry) else {
        return Err(RuntimeError {
            message: format!("invalid entry procedure {}", entry.index()),
            instruction: 0,
            source_span: None,
            call_stack: Vec::new(),
        });
    };
    if limits.max_call_depth == 0 {
        return Err(RuntimeError {
            message: "maximum call depth must be at least one".to_owned(),
            instruction: 0,
            source_span: program.source_spans.first().copied(),
            call_stack: vec![trace(module, entry, 0)],
        });
    }

    let frames = vec![make_frame(entry, program, arguments, context)];
    run_frames(module, frames, limits, state)
}

#[allow(clippy::too_many_lines)]
fn run_frames(
    module: &Module,
    mut frames: Vec<CallFrame>,
    limits: ExecutionLimits,
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    let mut remaining_steps = limits.max_steps;
    loop {
        let frame_index = frames.len() - 1;
        let procedure = frames[frame_index].procedure;
        let instruction_index = frames[frame_index].instruction;
        let Some(program) = module.procedure(procedure) else {
            return Err(execution_error(
                module,
                &frames,
                format!("invalid procedure {}", procedure.index()),
            ));
        };
        let Some(instruction) = program.instructions.get(instruction_index).cloned() else {
            return Err(execution_error(
                module,
                &frames,
                "program ended without Return",
            ));
        };
        if remaining_steps == 0 {
            return Err(execution_error(
                module,
                &frames,
                format!("instruction budget of {} exhausted", limits.max_steps),
            ));
        }
        remaining_steps -= 1;

        match instruction {
            Instruction::PushNull => frames[frame_index].stack.push(Value::Null),
            Instruction::PushNumber(number) => {
                frames[frame_index].stack.push(Value::Number(number));
            }
            Instruction::PushText(text) => frames[frame_index].stack.push(Value::text(text)),
            Instruction::PushTypePath(path) => {
                frames[frame_index].stack.push(Value::TypePath(path));
            }
            Instruction::ExpandArgumentLists {
                argument_count,
                expanded_indices,
            } => {
                let count = usize::from(argument_count);
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let source = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let mut expanded = Vec::new();
                for (index, value) in source.into_iter().enumerate() {
                    let index = u16::try_from(index).expect("source argument count is u16");
                    if expanded_indices.binary_search(&index).is_ok() {
                        let Value::List(list) = value else {
                            return Err(execution_error(
                                module,
                                &frames,
                                "arglist requires a list value",
                            ));
                        };
                        let list = state
                            .heap
                            .list(list)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                        expanded.extend(list.positions().map(|(_, value)| value.clone()));
                    } else {
                        expanded.push(value);
                    }
                }
                let expanded_count = u16::try_from(expanded.len()).map_err(|_| {
                    execution_error(
                        module,
                        &frames,
                        "expanded call has more than 65535 arguments",
                    )
                })?;
                let stack = &mut frames[frame_index].stack;
                stack.extend(expanded);
                stack.push(Value::number(f32::from(expanded_count)));
            }
            Instruction::AllocateDatum { argument_count } => {
                let count_result =
                    runtime_argument_count(&mut frames[frame_index].stack, argument_count);
                let count =
                    count_result.map_err(|message| execution_error(module, &frames, message))?;
                let stack = &mut frames[frame_index].stack;
                if stack.len() < count + 1 {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments_start = stack.len() - count;
                let type_path_index = arguments_start - 1;
                let type_path = match stack[type_path_index].clone() {
                    Value::TypePath(path) => path,
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("new requires a type path, received {value}"),
                        ));
                    }
                };
                stack.truncate(type_path_index);
                let datum = state.heap.allocate_datum(type_path);
                stack.push(Value::Datum(datum));
            }
            Instruction::AllocateCurrentDatum { argument_count } => {
                let count = runtime_argument_count(&mut frames[frame_index].stack, argument_count)
                    .map_err(|message| execution_error(module, &frames, message))?;
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let type_path = match frames[frame_index].src.clone() {
                    Value::Datum(datum) => state
                        .heap
                        .datum(datum)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .type_path()
                        .clone(),
                    Value::TypePath(path) => path,
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("unqualified new requires datum src, received {value}"),
                        ));
                    }
                };
                let stack = &mut frames[frame_index].stack;
                stack.truncate(stack.len() - count);
                let datum = state.heap.allocate_datum(type_path);
                stack.push(Value::Datum(datum));
            }
            Instruction::MakeRegex { argument_count } => {
                let count = usize::from(argument_count);
                if !(1..=2).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid regex constructor stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let pattern = arguments[0].clone();
                let flags = arguments.get(1).cloned().unwrap_or(Value::Null);
                let type_path =
                    TypePath::parse("/regex").expect("the built-in regex type path is valid");
                let text =
                    FieldName::parse("text").expect("the built-in regex text field name is valid");
                let flags_name = FieldName::parse("flags")
                    .expect("the built-in regex flags field name is valid");
                let datum = state.heap.allocate_datum(type_path);
                state
                    .heap
                    .set_datum_field(datum, text, pattern)
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                state
                    .heap
                    .set_datum_field(datum, flags_name, flags)
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                frames[frame_index].stack.push(Value::Datum(datum));
            }
            Instruction::MakeMutableAppearance { argument_count } => {
                let count = usize::from(argument_count);
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid mutable_appearance constructor stack",
                    ));
                }
                let stack = &mut frames[frame_index].stack;
                stack.truncate(stack.len() - count);
                let type_path = TypePath::parse("/mutable_appearance")
                    .expect("the built-in mutable_appearance type path is valid");
                let datum = state.heap.allocate_datum(type_path);
                stack.push(Value::Datum(datum));
            }
            Instruction::ReplaceText {
                argument_count,
                exact,
                character_indices,
            } => {
                let count = usize::from(argument_count);
                if !(3..=5).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid replacetext builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = replace_text_builtin(&arguments, exact, character_indices, &state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::text(value));
            }
            Instruction::CopyText {
                argument_count,
                character_indices,
            } => {
                let count = usize::from(argument_count);
                if !(1..=3).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid copytext builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = copy_text_builtin(&arguments, character_indices, &state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::text(value));
            }
            Instruction::StandardBuiltin {
                name,
                argument_count,
            } => {
                let count = usize::from(argument_count);
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = execute_standard_builtin(&name, &arguments, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::Length => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let length = match builtin_length(&value, &state.heap) {
                    Ok(length) => length,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index].stack.push(Value::number(length));
            }
            Instruction::Ref => {
                let value = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(ref_builtin(&value));
            }
            Instruction::GetStep => {
                let direction = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let source = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = get_step_builtin(&source, &direction, &state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::Range { argument_count } => {
                let count = usize::from(argument_count);
                if !(1..=2).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid range builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = range_builtin(&arguments, &frames[frame_index].src, &mut state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::Block { argument_count } => {
                let count = usize::from(argument_count);
                if !(count == 2 || (3..=6).contains(&count))
                    || frames[frame_index].stack.len() < count
                {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid block builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = block_builtin(&arguments, &mut state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::TypesOf => {
                let selector = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let paths = typesof_builtin(&selector, &state.heap, &state.type_paths)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let list = state.heap.allocate_list();
                for path in paths {
                    state
                        .heap
                        .list_mut(list)
                        .expect("a newly allocated list handle must be live")
                        .add(Value::TypePath(path));
                }
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::Rand { argument_count } => {
                let count = usize::from(argument_count);
                if !(1..=2).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid rand builtin stack",
                    ));
                }
                let stack_length = frames[frame_index].stack.len();
                let arguments = frames[frame_index].stack.split_off(stack_length - count);
                let value = random_integer(&arguments, &mut state.random_state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::number(value));
            }
            Instruction::Pick { weighted } => {
                let value_count = weighted
                    .iter()
                    .map(|is_weighted| 1 + usize::from(*is_weighted))
                    .sum::<usize>();
                if value_count > frames[frame_index].stack.len() {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid pick builtin stack",
                    ));
                }
                let stack_length = frames[frame_index].stack.len();
                let values = frames[frame_index]
                    .stack
                    .split_off(stack_length - value_count);
                let value = pick_value(&values, &weighted, &state.heap, &mut state.random_state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::Prob => {
                let chance = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let chance = chance.as_number().ok_or_else(|| {
                    execution_error(
                        module,
                        &frames,
                        format!("prob requires a number, received {chance}"),
                    )
                })?;
                let result =
                    deterministic_unit(&mut state.random_state) * 100.0 < chance.clamp(0.0, 100.0);
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::Round { argument_count } => {
                let count = usize::from(argument_count);
                if !(1..=2).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid round builtin stack",
                    ));
                }
                let stack = &mut frames[frame_index].stack;
                let arguments = stack.split_off(stack.len() - count);
                let value = round_builtin(&arguments)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::number(value));
            }
            Instruction::TypePredicate {
                kind,
                argument_count,
            } => {
                let count = usize::from(argument_count);
                let valid_count = match kind {
                    TypePredicateKind::IsType | TypePredicateKind::IsPath => {
                        (1..=2).contains(&count)
                    }
                    TypePredicateKind::IsLoc
                    | TypePredicateKind::IsMovable
                    | TypePredicateKind::IsTurf => count >= 1,
                    _ => count == 1,
                };
                if !valid_count || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid type predicate builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let result = type_predicate_builtin(kind, &arguments, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::MakeList(item_count) => {
                let count = usize::from(item_count);
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let items = frames[frame_index].stack.split_off(stack_length - count);
                let list = state.heap.allocate_list();
                for item in items {
                    state
                        .heap
                        .list_mut(list)
                        .expect("a newly allocated list handle must be live")
                        .add(item);
                }
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::MakeArgs => {
                let list = state.heap.allocate_list();
                for value in &frames[frame_index].arguments {
                    state
                        .heap
                        .list_mut(list)
                        .expect("a newly allocated list handle must be live")
                        .add(value.clone());
                }
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::MakeListEntries(kinds) => {
                let value_count = kinds.iter().try_fold(0_usize, |count, kind| {
                    count.checked_add(match kind {
                        ListEntryKind::Positional => 1,
                        ListEntryKind::Associative => 2,
                    })
                });
                let Some(value_count) = value_count else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "list literal is too large",
                    ));
                };
                let stack_length = frames[frame_index].stack.len();
                if value_count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let values = frames[frame_index]
                    .stack
                    .split_off(stack_length - value_count);
                let list = state.heap.allocate_list();
                let entries = state
                    .heap
                    .list_mut(list)
                    .expect("a newly allocated list handle must be live");
                let mut values = values.into_iter();
                for kind in kinds {
                    match kind {
                        ListEntryKind::Positional => {
                            entries.add(values.next().expect("validated literal stack shape"));
                        }
                        ListEntryKind::Associative => {
                            let key = values.next().expect("validated literal stack shape");
                            let value = values.next().expect("validated literal stack shape");
                            entries.set_key(key, value);
                        }
                    }
                }
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::IndexList => {
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let list = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::List(list)) => list,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list index operation received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let value = match read_list_value(&state.heap, list, &key) {
                    Ok(value) => value.clone(),
                    // BYOND associative lookup returns null for an absent key.
                    // Lazy-list idioms such as `lists[target] ||= list()` rely
                    // on this before inserting the new association.
                    Err(ValueError::MissingKey) => Value::Null,
                    Err(error) => {
                        return Err(execution_error(module, &frames, error.to_string()));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::SetListIndex => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let list = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::List(list)) => list,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list assignment received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if let Err(error) = write_list_value(&mut state.heap, list, key, value) {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
            }
            Instruction::SetListIndexKeep => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let list = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::List(list)) => list,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list assignment received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if let Err(error) = write_list_value(&mut state.heap, list, key, value.clone()) {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
                frames[frame_index].stack.push(value);
            }
            Instruction::CompoundListIndex(operator) => {
                let right = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let list = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::List(list)) => list,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list assignment received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let current = match read_list_value(&state.heap, list, &key) {
                    Ok(value) => value.clone(),
                    Err(error) => {
                        return Err(execution_error(module, &frames, error.to_string()));
                    }
                };
                let Some(left) = current.as_number() else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("numeric operation received {current}"),
                    ));
                };
                let value =
                    Value::number(execute_compound_list_index_operation(operator, left, right));
                if let Err(error) = write_list_value(&mut state.heap, list, key, value) {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
            }
            Instruction::ListLength => {
                let list = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::List(list)) => list,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list length operation received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let length = match state.heap.list(list) {
                    Ok(values) => values.len(),
                    Err(error) => {
                        return Err(execution_error(module, &frames, error.to_string()));
                    }
                };
                let length = length.to_string().parse::<f32>().map_err(|error| {
                    execution_error(
                        module,
                        &frames,
                        format!("list length cannot be represented as binary32: {error}"),
                    )
                })?;
                frames[frame_index].stack.push(Value::number(length));
            }
            Instruction::LoadSrc => {
                let src = frames[frame_index].src.clone();
                frames[frame_index].stack.push(src);
            }
            Instruction::LoadUsr => {
                let usr = frames[frame_index].usr.clone();
                frames[frame_index].stack.push(usr);
            }
            Instruction::LoadField(name) => {
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let value = match receiver {
                    Value::TypePath(path) if name.as_str() == "parent_type" => state
                        .type_parent(&path)
                        .cloned()
                        .map_or(Value::Null, Value::TypePath),
                    Value::List(list) if name.as_str() == "len" => {
                        let len = state
                            .heap
                            .list(list)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?
                            .len();
                        Value::number(len.to_string().parse::<f32>().map_err(|error| {
                            execution_error(
                                module,
                                &frames,
                                format!("list length cannot be represented as binary32: {error}"),
                            )
                        })?)
                    }
                    Value::Datum(datum) => {
                        let runtime_type = match state.heap.datum(datum) {
                            Ok(datum) => datum.type_path().clone(),
                            Err(error) => {
                                return Err(execution_error(module, &frames, error.to_string()));
                            }
                        };
                        if name.as_str() == "type" {
                            Value::TypePath(runtime_type)
                        } else if name.as_str() == "parent_type" {
                            state
                                .type_parent(&runtime_type)
                                .cloned()
                                .map_or(Value::Null, Value::TypePath)
                        } else {
                            match state.heap.datum_field(datum, &name) {
                                Ok(value) => value.clone(),
                                Err(error) => {
                                    return Err(execution_error(
                                        module,
                                        &frames,
                                        error.to_string(),
                                    ));
                                }
                            }
                        }
                    }
                    Value::Null => {
                        return Err(execution_error(module, &frames, "field read received null"));
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("field read requires a datum, received {value}"),
                        ));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::InitialField(name) => {
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let runtime_type = match receiver {
                    Value::TypePath(path) => path,
                    Value::Datum(datum) => match state.heap.datum(datum) {
                        Ok(datum) => datum.type_path().clone(),
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    },
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!(
                                "initial requires a datum or type path receiver, received {value}"
                            ),
                        ));
                    }
                };
                let value = state
                    .initial_value(&runtime_type, &name)
                    .cloned()
                    .unwrap_or(Value::Null);
                frames[frame_index].stack.push(value);
            }
            Instruction::StoreField(ref name) | Instruction::StoreFieldKeep(ref name) => {
                let keep = matches!(instruction, Instruction::StoreFieldKeep(_));
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                match receiver {
                    Value::Datum(datum) => {
                        if let Err(error) =
                            state
                                .heap
                                .set_datum_field(datum, name.clone(), value.clone())
                        {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    }
                    Value::List(list) if name.as_str() == "len" => {
                        let new_len = match &value {
                            Value::Number(number) if number.to_f32().is_finite() => number
                                .to_f32()
                                .trunc()
                                .max(0.0)
                                .to_string()
                                .parse::<usize>()
                                .unwrap_or(usize::MAX),
                            _ => 0,
                        };
                        if let Err(error) = state
                            .heap
                            .list_mut(list)
                            .and_then(|values| values.resize(new_len))
                        {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    }
                    Value::Null => {
                        return Err(execution_error(
                            module,
                            &frames,
                            "field write received null",
                        ));
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("field write requires a datum or list.len, received {value}"),
                        ));
                    }
                }
                if keep {
                    frames[frame_index].stack.push(value);
                }
            }
            Instruction::LoadGlobal(name) => {
                let Some(value) = state.global(&name).cloned() else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("runtime global {name:?} is absent"),
                    ));
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::StoreGlobal(name) => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                state.set_global(name, value);
            }
            Instruction::Duplicate => {
                let Some(value) = frames[frame_index].stack.last().cloned() else {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::LoadLocal(slot) => {
                let Some(value) = frames[frame_index].locals.get(usize::from(slot)).cloned() else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::StoreLocal(slot) => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let Some(local) = frames[frame_index].locals.get_mut(usize::from(slot)) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                *local = value;
            }
            Instruction::LoadResult => {
                let result = frames[frame_index].result.clone();
                frames[frame_index].stack.push(result);
            }
            Instruction::StoreResult => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index].result = value;
            }
            Instruction::Pop => {
                if let Err(message) = pop(&mut frames[frame_index].stack) {
                    return Err(execution_error(module, &frames, message));
                }
            }
            Instruction::Crash => {
                let message = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                return Err(execution_error(
                    module,
                    &frames,
                    format!("CRASH: {message}"),
                ));
            }
            Instruction::Locate { argument_count } => {
                let count = usize::from(argument_count);
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                frames[frame_index].stack.truncate(stack_length - count);
                frames[frame_index].stack.push(Value::Null);
            }
            Instruction::LocateIn { argument_count } => {
                let count = usize::from(argument_count)
                    .checked_add(1)
                    .expect("u16 argument count plus container fits usize");
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                frames[frame_index].stack.truncate(stack_length - count);
                frames[frame_index].stack.push(Value::Null);
            }
            Instruction::Negate => {
                let value = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index].stack.push(Value::number(-value));
            }
            Instruction::BitNot => {
                let value = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(bitwise_not(value)));
            }
            Instruction::Not => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let is_truthy = runtime_truthy(&state.heap, &value)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(!is_truthy)));
            }
            Instruction::CompoundAssignment(operator) => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = if let Value::List(list) = left {
                    execute_list_compound_operator(operator, list, &right, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    execute_scalar_compound_assignment(operator, left, right)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Add => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = if let Value::List(list) = left {
                    execute_list_binary_operator("+", list, &right, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    execute_scalar_add(left, right)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Subtract
            | Instruction::BitAnd
            | Instruction::BitOr
            | Instruction::BitXor => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let operator = match instruction {
                    Instruction::Subtract => "-",
                    Instruction::BitAnd => "&",
                    Instruction::BitOr => "|",
                    Instruction::BitXor => "^",
                    _ => unreachable!(),
                };
                let value = if let Value::List(list) = left {
                    execute_list_binary_operator(operator, list, &right, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    let left = scalar_number_string(left)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let right = scalar_number_string(right)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    Value::number(execute_numeric_binary(&instruction, left, right))
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Multiply
            | Instruction::Power
            | Instruction::Divide
            | Instruction::Remainder
            | Instruction::ShiftLeft
            | Instruction::ShiftRight
            | Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual => {
                let right = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let left = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(execute_numeric_binary(
                        &instruction,
                        left,
                        right,
                    )));
            }
            Instruction::Equal | Instruction::NotEqual => {
                let right = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let left = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let equal = values_equal(&left, &right);
                let result = if matches!(instruction, Instruction::NotEqual) {
                    !equal
                } else {
                    equal
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::Contains => {
                let list = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::List(list)) => list,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("right operand of 'in' must be a list, received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let needle = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let contains = state
                    .heap
                    .list(list)
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?
                    .positions()
                    .any(|(_, value)| values_equal(&needle, value));
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(contains)));
            }
            Instruction::And | Instruction::Or => {
                let right = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => runtime_truthy(&state.heap, &value)
                        .map_err(|message| execution_error(module, &frames, message))?,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let left = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => runtime_truthy(&state.heap, &value)
                        .map_err(|message| execution_error(module, &frames, message))?,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let result = if matches!(instruction, Instruction::And) {
                    left && right
                } else {
                    left || right
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::JumpIfNull(target) => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if matches!(value, Value::Null) {
                    if let Err(message) = validate_jump(target, program.instructions.len()) {
                        return Err(execution_error(module, &frames, message));
                    }
                    frames[frame_index].instruction = target;
                    continue;
                }
            }
            Instruction::JumpIfFalse(target) => {
                let condition = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if !runtime_truthy(&state.heap, &condition)
                    .map_err(|message| execution_error(module, &frames, message))?
                {
                    if let Err(message) = validate_jump(target, program.instructions.len()) {
                        return Err(execution_error(module, &frames, message));
                    }
                    frames[frame_index].instruction = target;
                    continue;
                }
            }
            Instruction::Jump(target) => {
                if let Err(message) = validate_jump(target, program.instructions.len()) {
                    return Err(execution_error(module, &frames, message));
                }
                frames[frame_index].instruction = target;
                continue;
            }
            Instruction::JumpIfArgumentSupplied { parameter, target } => {
                if frames[frame_index].arguments.len() > usize::from(parameter) {
                    if let Err(message) = validate_jump(target, program.instructions.len()) {
                        return Err(execution_error(module, &frames, message));
                    }
                    frames[frame_index].instruction = target;
                    continue;
                }
            }
            Instruction::Call {
                procedure: target,
                argument_count,
            } => {
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let count = runtime_argument_count(&mut frames[frame_index].stack, argument_count)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments = frames[frame_index].stack.split_off(stack_length - count);
                let Some(target_program) = module.procedure(target) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid call target {}", target.index()),
                    ));
                };
                let context = frame_context(&frames[frame_index]);
                frames.push(make_frame(target, target_program, &arguments, &context));
                continue;
            }
            Instruction::CallCurrent { argument_count } => {
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let arguments = if let Some(argument_count) = argument_count {
                    let count =
                        runtime_argument_count(&mut frames[frame_index].stack, argument_count)
                            .map_err(|message| execution_error(module, &frames, message))?;
                    let stack_length = frames[frame_index].stack.len();
                    if count > stack_length {
                        return Err(execution_error(module, &frames, "bytecode stack underflow"));
                    }
                    frames[frame_index].stack.split_off(stack_length - count)
                } else {
                    frames[frame_index].arguments.clone()
                };
                let context = frame_context(&frames[frame_index]);
                frames.push(make_frame(procedure, program, &arguments, &context));
                continue;
            }
            Instruction::CallParent {
                procedure: target,
                argument_count,
            } => {
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let Some(target) = target else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "parent procedure call has no resolved target",
                    ));
                };
                let arguments = if let Some(argument_count) = argument_count {
                    let count =
                        runtime_argument_count(&mut frames[frame_index].stack, argument_count)
                            .map_err(|message| execution_error(module, &frames, message))?;
                    let stack_length = frames[frame_index].stack.len();
                    if count > stack_length {
                        return Err(execution_error(module, &frames, "bytecode stack underflow"));
                    }
                    frames[frame_index].stack.split_off(stack_length - count)
                } else {
                    frames[frame_index].arguments.clone()
                };
                let Some(target_program) = module.procedure(target) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid parent call target {}", target.index()),
                    ));
                };
                let context = frame_context(&frames[frame_index]);
                frames.push(make_frame(target, target_program, &arguments, &context));
                continue;
            }
            Instruction::CallDynamic { argument_count } => {
                let count = runtime_argument_count(&mut frames[frame_index].stack, argument_count)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let stack_length = frames[frame_index].stack.len();
                // Receiver and selector precede all explicit arguments.
                if stack_length < count + 2 {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments = frames[frame_index].stack.split_off(stack_length - count);
                let selector = frames[frame_index]
                    .stack
                    .pop()
                    .expect("stack length was checked");
                let receiver = frames[frame_index]
                    .stack
                    .pop()
                    .expect("stack length was checked");
                if let Value::List(list) = receiver {
                    let Value::Text(method) = selector else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list procedure selector must be text, received {selector}"),
                        ));
                    };
                    let Some(result) = execute_list_method(&method, list, &arguments, state) else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("unknown /list procedure {method:?}"),
                        ));
                    };
                    let result =
                        result.map_err(|message| execution_error(module, &frames, message))?;
                    frames[frame_index].stack.push(result);
                } else {
                    if frames.len() >= limits.max_call_depth {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("maximum call depth {} exceeded", limits.max_call_depth),
                        ));
                    }
                    let caller_context = frame_context(&frames[frame_index]);
                    let (target, context) =
                        dynamic_call_target(module, state, &receiver, &selector, &caller_context)
                            .map_err(|message| execution_error(module, &frames, message))?;
                    let Some(target_program) = module.procedure(target) else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("invalid dynamic call target {}", target.index()),
                        ));
                    };
                    frames.push(make_frame(target, target_program, &arguments, &context));
                    continue;
                }
            }
            Instruction::Return => {
                let result = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames.pop();
                let Some(caller) = frames.last_mut() else {
                    return Ok(result);
                };
                caller.stack.push(result);
                caller.instruction += 1;
                continue;
            }
        }
        frames[frame_index].instruction += 1;
    }
}

fn make_frame(
    procedure: ProcedureId,
    program: &Program,
    arguments: &[Value],
    context: &ExecutionContext,
) -> CallFrame {
    let mut locals = vec![Value::Null; program.local_count];
    let bound_count = arguments
        .len()
        .min(program.parameter_count)
        .min(locals.len());
    locals[..bound_count].clone_from_slice(&arguments[..bound_count]);
    CallFrame {
        procedure,
        instruction: 0,
        locals,
        stack: Vec::new(),
        result: Value::Null,
        src: context.src.clone(),
        usr: context.usr.clone(),
        arguments: arguments.to_vec(),
    }
}

fn frame_context(frame: &CallFrame) -> ExecutionContext {
    ExecutionContext::new(frame.src.clone(), frame.usr.clone())
}

fn execution_error(
    module: &Module,
    frames: &[CallFrame],
    message: impl Into<String>,
) -> RuntimeError {
    let instruction = frames.last().map_or(0, |frame| frame.instruction);
    let source_span = frames.last().and_then(|frame| {
        module
            .procedure(frame.procedure)
            .and_then(|program| program.source_spans.get(frame.instruction))
            .copied()
    });
    RuntimeError {
        message: message.into(),
        instruction,
        source_span,
        call_stack: frames
            .iter()
            .map(|frame| trace(module, frame.procedure, frame.instruction))
            .collect(),
    }
}

fn trace(module: &Module, procedure: ProcedureId, instruction: usize) -> CallTrace {
    CallTrace {
        procedure: module
            .procedure_path(procedure)
            .unwrap_or("<invalid procedure>")
            .to_owned(),
        instruction,
        source_span: module
            .procedure(procedure)
            .and_then(|program| program.source_spans.get(instruction))
            .copied(),
    }
}

fn execute_numeric_binary(instruction: &Instruction, left: f32, right: f32) -> f32 {
    match instruction {
        Instruction::Add => left + right,
        Instruction::Subtract => left - right,
        Instruction::Multiply => left * right,
        Instruction::Power => left.powf(right),
        Instruction::Divide => left / right,
        Instruction::Remainder => left % right,
        Instruction::BitAnd => bitwise_binary(left, right, |left, right| left & right),
        Instruction::BitOr => bitwise_binary(left, right, |left, right| left | right),
        Instruction::BitXor => bitwise_binary(left, right, |left, right| left ^ right),
        Instruction::ShiftLeft => bitwise_shift(left, right, |left, right| left << right),
        Instruction::ShiftRight => bitwise_shift(left, right, |left, right| left >> right),
        Instruction::Less => f32::from(left < right),
        Instruction::LessEqual => f32::from(left <= right),
        Instruction::Greater => f32::from(left > right),
        Instruction::GreaterEqual => f32::from(left >= right),
        _ => unreachable!("instruction came from the numeric operation group"),
    }
}

fn execute_compound_list_index_operation(
    operator: CompoundListIndexOperator,
    left: f32,
    right: f32,
) -> f32 {
    match operator {
        CompoundListIndexOperator::Add => left + right,
        CompoundListIndexOperator::Subtract => left - right,
        CompoundListIndexOperator::Multiply => left * right,
        CompoundListIndexOperator::Divide => left / right,
        CompoundListIndexOperator::Remainder => left % right,
        CompoundListIndexOperator::BitAnd => {
            bitwise_binary(left, right, |left, right| left & right)
        }
        CompoundListIndexOperator::BitOr => bitwise_binary(left, right, |left, right| left | right),
        CompoundListIndexOperator::BitXor => {
            bitwise_binary(left, right, |left, right| left ^ right)
        }
        CompoundListIndexOperator::ShiftLeft => {
            bitwise_shift(left, right, |left, right| left << right)
        }
        CompoundListIndexOperator::ShiftRight => {
            bitwise_shift(left, right, |left, right| left >> right)
        }
    }
}

/// DM bitwise operations coerce their binary32 numeric operands to signed
/// 32-bit integers by truncation and return the resulting integer as a DM
/// number. Rust's float-to-int conversion also gives deterministic saturation
/// for values outside the integer range.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "DM bitwise operators intentionally bridge binary32 numbers and signed 32-bit integers"
)]
fn bitwise_binary(left: f32, right: f32, operation: impl FnOnce(i32, i32) -> i32) -> f32 {
    operation(left as i32, right as i32) as f32
}

/// Executes a DM bitwise complement after integer coercion.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "DM bitwise operators intentionally bridge binary32 numbers and signed 32-bit integers"
)]
fn bitwise_not(value: f32) -> f32 {
    (!(value as i32)) as f32
}

/// Executes a DM shift after integer coercion. Shift counts are masked to the
/// low five bits, matching the fixed-width 32-bit integer representation.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "DM bitwise operators intentionally bridge binary32 numbers and signed 32-bit integers"
)]
fn bitwise_shift(left: f32, right: f32, operation: impl FnOnce(i32, u32) -> i32) -> f32 {
    let count = u32::from_ne_bytes((right as i32).to_ne_bytes()) & 31;
    operation(left as i32, count) as f32
}

fn validate_jump(target: usize, instruction_count: usize) -> Result<(), String> {
    if target > instruction_count {
        return Err(format!("invalid jump target {target}"));
    }
    Ok(())
}

fn values_equal(left: &Value, right: &Value) -> bool {
    left.semantic_eq(right)
}

fn replace_text_builtin(
    arguments: &[Value],
    exact: bool,
    character_indices: bool,
    heap: &ValueHeap,
) -> Result<String, String> {
    let source = builtin_text(&arguments[0], heap, "replacetext text")?;
    let needle = builtin_text(&arguments[1], heap, "replacetext needle")?;
    let replacement = builtin_text(&arguments[2], heap, "replacetext replacement")?;
    if needle.is_empty() {
        return Ok(source);
    }

    let (start, end) = replacement_bounds(&source, arguments, character_indices)?;
    let prefix = &source[..start];
    let target = &source[start..end];
    let suffix = &source[end..];
    let replaced = if exact {
        target.replace(&needle, &replacement)
    } else {
        replace_text_ascii_insensitive(target, &needle, &replacement)
    };
    Ok(format!("{prefix}{replaced}{suffix}"))
}

fn copy_text_builtin(
    arguments: &[Value],
    character_indices: bool,
    heap: &ValueHeap,
) -> Result<String, String> {
    let source = builtin_text(&arguments[0], heap, "copytext text")?;
    let logical_length = if character_indices {
        source.chars().count()
    } else {
        source.len()
    };
    let start = signed_text_index(arguments.get(1), 1)?;
    let end = signed_text_index(arguments.get(2), 0)?;
    let start = resolve_text_position(start, logical_length);
    let end = if end == 0 {
        logical_length.saturating_add(1)
    } else {
        resolve_text_position(end, logical_length)
    };
    if end <= start {
        return Ok(String::new());
    }
    let start = start.saturating_sub(1);
    let end = end.saturating_sub(1);
    let (start, end) = if character_indices {
        (
            character_offset(&source, start),
            character_offset(&source, end),
        )
    } else {
        (
            previous_char_boundary(&source, start),
            previous_char_boundary(&source, end),
        )
    };
    Ok(source[start..end].to_owned())
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "DM text positions are integralized from binary32 at the language boundary"
)]
fn signed_text_index(value: Option<&Value>, default: i64) -> Result<i64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => {
            let number = number.to_f32();
            if !number.is_finite() {
                return Ok(default);
            }
            Ok(number.trunc() as i64)
        }
        Some(value) => Err(format!(
            "copytext bounds require a number, received {value}"
        )),
    }
}

fn resolve_text_position(position: i64, logical_length: usize) -> usize {
    let limit = i64::try_from(logical_length)
        .unwrap_or(i64::MAX - 1)
        .saturating_add(1);
    let position = if position < 0 {
        limit.saturating_add(position)
    } else {
        position
    };
    usize::try_from(position.clamp(1, limit)).unwrap_or(usize::MAX)
}

fn builtin_text(value: &Value, heap: &ValueHeap, context: &str) -> Result<String, String> {
    match value {
        Value::Text(text) => Ok(String::from(text.as_ref())),
        Value::Datum(datum)
            if heap
                .datum(*datum)
                .map_err(|error| error.to_string())?
                .type_path()
                .to_string()
                == "/regex" =>
        {
            Err(format!("{context} regex matching is not yet supported"))
        }
        _ => Err(format!("{context} requires text, received {value}")),
    }
}

/// Advances the fixed-seed headless random stream and returns a unit interval
/// sample. Keeping it in [`ExecutionState`] makes repeated calls vary while
/// fresh headless worlds remain reproducible.
#[allow(
    clippy::cast_precision_loss,
    reason = "the upper 24 generator bits deliberately map onto the binary32 unit interval"
)]
fn deterministic_unit(state: &mut u64) -> f32 {
    if *state == 0 {
        *state = 0x9e37_79b9_7f4a_7c15;
    }
    *state ^= *state << 7;
    *state ^= *state >> 9;
    *state ^= *state << 8;
    let high = (*state >> 40) as u32;
    high as f32 / 16_777_216.0
}

/// Implements DM's `round(A)` and `round(A, B)` forms for scalar numbers.
///
/// The single-argument form is the historical floor operation.  With a
/// non-zero multiple, BYOND chooses the nearest multiple; an exact halfway
/// value goes toward positive infinity, as in `floor(A / B + 0.5)`.  A zero
/// multiple follows the legacy floor form rather than dividing by zero.
fn round_builtin(arguments: &[Value]) -> Result<f32, String> {
    let value = arguments[0]
        .as_number()
        .ok_or_else(|| format!("round requires a number, received {}", arguments[0]))?;
    if arguments.len() == 1 {
        return Ok(value.floor());
    }
    let multiple = arguments[1].as_number().ok_or_else(|| {
        format!(
            "round multiple requires a number, received {}",
            arguments[1]
        )
    })?;
    if multiple == 0.0 {
        return Ok(value.floor());
    }
    // The sign of a multiple does not alter its set of multiples.  Using its
    // magnitude also preserves BYOND's increasing-number-line tie rule.
    Ok((value / multiple.abs() + 0.5).floor() * multiple.abs())
}

fn random_integer(arguments: &[Value], state: &mut u64) -> Result<f32, String> {
    let bounds = arguments
        .iter()
        .map(|value| {
            value
                .as_number()
                .ok_or_else(|| format!("rand requires numbers, received {value}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (low, high) = match bounds.as_slice() {
        [high] => (0.0, *high),
        [low, high] => (*low, *high),
        _ => return Err("rand requires one or two bounds".to_owned()),
    };
    let low = low.ceil();
    let high = high.floor();
    if !low.is_finite() || !high.is_finite() || high < low {
        return Err(format!("invalid rand range {low} through {high}"));
    }
    Ok(low + (deterministic_unit(state) * (high - low + 1.0)).floor())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the unit sample is non-negative and strictly below one, yielding a valid list offset"
)]
fn pick_value(
    values: &[Value],
    weighted: &[bool],
    heap: &ValueHeap,
    state: &mut u64,
) -> Result<Value, String> {
    if weighted.len() == 1
        && !weighted[0]
        && let [Value::List(list)] = values
    {
        let list = heap.list(*list).map_err(|error| error.to_string())?;
        if list.is_empty() {
            return Ok(Value::Null);
        }
        let index = (deterministic_unit(state) * list.len() as f32).floor() as usize + 1;
        return list.get(index).cloned().map_err(|error| error.to_string());
    }
    let mut cursor = 0;
    let mut entries = Vec::with_capacity(weighted.len());
    let mut total = 0.0_f32;
    for is_weighted in weighted {
        let weight = if *is_weighted {
            let value = values
                .get(cursor)
                .ok_or_else(|| "invalid pick weights".to_owned())?;
            cursor += 1;
            value
                .as_number()
                .ok_or_else(|| format!("pick weight requires a number, received {value}"))?
                .max(0.0)
        } else {
            1.0
        };
        let candidate = values
            .get(cursor)
            .ok_or_else(|| "invalid pick candidates".to_owned())?
            .clone();
        cursor += 1;
        total += weight;
        entries.push((weight, candidate));
    }
    if total <= 0.0 {
        return Ok(Value::Null);
    }
    let mut point = deterministic_unit(state) * total;
    for (weight, candidate) in entries {
        if point < weight {
            return Ok(candidate);
        }
        point -= weight;
    }
    Ok(Value::Null)
}

/// Returns BYOND's legacy `length()` result for the runtime values accepted
/// by the headless VM. Text uses byte length because regular DM text indices
/// are byte indices; `_char` builtins are the explicit character-indexed API.
fn builtin_length(value: &Value, heap: &ValueHeap) -> Result<f32, String> {
    let length = match value {
        Value::Null => 0,
        Value::Text(text) => text.len(),
        Value::List(list) => heap.list(*list).map_err(|error| error.to_string())?.len(),
        _ => return Err(format!("length requires text or a list, received {value}")),
    };
    length
        .to_string()
        .parse::<f32>()
        .map_err(|error| format!("length cannot be represented as binary32: {error}"))
}

/// Produces the opaque reference text used by DM's `ref()` builtin.
///
/// BYOND reserves the `0xe...` range for list references; preserving that
/// convention matters to code which uses a list reference as an associative
/// key. Dream64's headless heap keeps identities live for the execution, so
/// its monotonic slot identity is sufficient for a stable reference.
fn ref_builtin(value: &Value) -> Value {
    let reference = match value {
        Value::Datum(datum) => format!("[0xd{:06x}]", datum.index() + 1),
        Value::List(list) => format!("[0xe{:06x}]", list.index() + 1),
        // `ref` identifies runtime heap objects; scalar values have no
        // object identity and therefore cannot yield a usable reference.
        Value::Null | Value::Number(_) | Value::Text(_) | Value::TypePath(_) => return Value::Null,
    };
    Value::text(reference)
}

/// Resolves BYOND's `get_step(atom_or_turf, direction)` against live turfs.
///
/// BYOND directions are bit flags: north/south affect Y and east/west affect
/// X, so diagonal values naturally combine their cardinal components.  A
/// source atom's materialized coordinates are sufficient even when its `loc`
/// has not been explicitly connected in the headless world model.
fn get_step_builtin(source: &Value, direction: &Value, heap: &ValueHeap) -> Result<Value, String> {
    let Value::Datum(source) = source else {
        return Ok(Value::Null);
    };
    let Some(direction) = direction.as_number() else {
        return Ok(Value::Null);
    };
    if !direction.is_finite() || direction.fract() != 0.0 {
        return Ok(Value::Null);
    }
    let Ok(direction) = direction.to_string().parse::<i16>() else {
        return Ok(Value::Null);
    };
    if !(0..=15).contains(&direction) {
        return Ok(Value::Null);
    }
    // Unknown bits do not name a world direction.  `get_step(source, 0)` is
    // useful as a normalized `get_turf` and returns the containing turf.
    let x = FieldName::parse("x").expect("built-in coordinate field is valid");
    let y = FieldName::parse("y").expect("built-in coordinate field is valid");
    let z = FieldName::parse("z").expect("built-in coordinate field is valid");
    let source = heap.datum(*source).map_err(|error| error.to_string())?;
    let coordinate = |field: &FieldName| -> Result<f32, String> {
        source
            .field(field)
            .map_err(|error| error.to_string())?
            .as_number()
            .ok_or_else(|| format!("get_step source coordinate {field} is not numeric"))
    };
    let source_x = coordinate(&x)?;
    let source_y = coordinate(&y)?;
    let source_z = coordinate(&z)?;
    let target_x = source_x + f32::from(u8::from(direction & 4 != 0))
        - f32::from(u8::from(direction & 8 != 0));
    let target_y = source_y + f32::from(u8::from(direction & 1 != 0))
        - f32::from(u8::from(direction & 2 != 0));
    for (datum, candidate) in heap.datums() {
        let path = candidate.type_path().as_str();
        if path != "/turf" && !path.starts_with("/turf/") {
            continue;
        }
        let matches = [(&x, target_x), (&y, target_y), (&z, source_z)]
            .into_iter()
            .all(|(field, expected)| {
                candidate
                    .field(field)
                    .is_ok_and(|value| value.as_number() == Some(expected))
            });
        if matches {
            return Ok(Value::Datum(datum));
        }
    }
    Ok(Value::Null)
}

/// Resolves BYOND's `block()` over materialized headless turfs.
fn block_builtin(arguments: &[Value], heap: &mut ValueHeap) -> Result<Value, String> {
    let list = heap.allocate_list();
    let x = FieldName::parse("x").expect("built-in coordinate field is valid");
    let y = FieldName::parse("y").expect("built-in coordinate field is valid");
    let z = FieldName::parse("z").expect("built-in coordinate field is valid");

    let datum_coordinates = |value: &Value, heap: &ValueHeap| -> Option<(f32, f32, f32)> {
        let Value::Datum(datum) = value else {
            return None;
        };
        let datum = heap.datum(*datum).ok()?;
        let path = datum.type_path().as_str();
        if path != "/turf" && !path.starts_with("/turf/") {
            return None;
        }
        Some((
            datum.field(&x).ok()?.as_number()?,
            datum.field(&y).ok()?.as_number()?,
            datum.field(&z).ok()?.as_number()?,
        ))
    };
    let numeric = |value: &Value| value.as_number().filter(|number| number.is_finite());

    let (start, end) = match arguments {
        [start, end] => {
            let Some(start) = datum_coordinates(start, heap) else {
                return Ok(Value::List(list));
            };
            let Some(end) = datum_coordinates(end, heap) else {
                return Ok(Value::List(list));
            };
            (start, end)
        }
        [start_x, start_y, start_z, rest @ ..] if rest.len() <= 3 => {
            let (Some(start_x), Some(start_y), Some(start_z)) =
                (numeric(start_x), numeric(start_y), numeric(start_z))
            else {
                return Ok(Value::List(list));
            };
            let end_x = rest
                .first()
                .filter(|value| !matches!(value, Value::Null))
                .and_then(numeric)
                .unwrap_or(start_x);
            let end_y = rest
                .get(1)
                .filter(|value| !matches!(value, Value::Null))
                .and_then(numeric)
                .unwrap_or(start_y);
            let end_z = rest
                .get(2)
                .filter(|value| !matches!(value, Value::Null))
                .and_then(numeric)
                .unwrap_or(start_z);
            ((start_x, start_y, start_z), (end_x, end_y, end_z))
        }
        _ => return Err("block requires two turfs or three through six coordinates".to_owned()),
    };

    // Accept either corner ordering while preserving the inclusive rectangular
    // volume described by the two endpoints. This is important to movement
    // code whose source/destination order naturally changes with direction.
    let low = (start.0.min(end.0), start.1.min(end.1), start.2.min(end.2));
    let high = (start.0.max(end.0), start.1.max(end.1), start.2.max(end.2));
    let matching = heap
        .datums()
        .filter_map(|(datum, candidate)| {
            let path = candidate.type_path().as_str();
            if path != "/turf" && !path.starts_with("/turf/") {
                return None;
            }
            let candidate_x = candidate.field(&x).ok()?.as_number()?;
            let candidate_y = candidate.field(&y).ok()?.as_number()?;
            let candidate_z = candidate.field(&z).ok()?.as_number()?;
            (candidate_x >= low.0
                && candidate_x <= high.0
                && candidate_y >= low.1
                && candidate_y <= high.1
                && candidate_z >= low.2
                && candidate_z <= high.2)
                .then_some(datum)
        })
        .collect::<Vec<_>>();
    let result = heap
        .list_mut(list)
        .expect("a newly allocated list handle must be live");
    for datum in matching {
        result.add(Value::Datum(datum));
    }
    Ok(Value::List(list))
}

/// Resolves BYOND's `range()` over the materialized headless world.
///
/// The regular `range(distance, center)` spelling and BYOND's accepted
/// reversed `range(center, distance)` spelling are both supported.  With one
/// argument, the current procedure's `src` is the center.  BYOND range is a
/// square tile radius (Chebyshev distance), not Euclidean distance; every
/// atom on matching same-z tiles is returned in deterministic heap allocation
/// order.  Areas are deliberately excluded because they describe tiles rather
/// than being located contents of one.
fn range_builtin(arguments: &[Value], src: &Value, heap: &mut ValueHeap) -> Result<Value, String> {
    let null_center = Value::Null;
    let (distance, center) = match arguments {
        [distance] => (distance.as_number(), src),
        [first, second] => match (first.as_number(), second.as_number()) {
            (Some(distance), None) => (Some(distance), second),
            (None, Some(distance)) => (Some(distance), first),
            // A number cannot be a materialized map location.  Keeping this
            // an empty result mirrors BYOND's non-loc center behavior while
            // avoiding a fabricated coordinate.
            _ => (None, &null_center),
        },
        _ => return Err("range accepts one or two arguments".to_owned()),
    };
    let list = heap.allocate_list();
    let Some(distance) = distance else {
        return Ok(Value::List(list));
    };
    if !distance.is_finite() || distance < 0.0 {
        return Ok(Value::List(list));
    }
    let Value::Datum(center) = center else {
        return Ok(Value::List(list));
    };
    let x = FieldName::parse("x").expect("built-in coordinate field is valid");
    let y = FieldName::parse("y").expect("built-in coordinate field is valid");
    let z = FieldName::parse("z").expect("built-in coordinate field is valid");
    let center = heap.datum(*center).map_err(|error| error.to_string())?;
    let coordinate = |field: &FieldName| center.field(field).ok().and_then(Value::as_number);
    let (Some(center_x), Some(center_y), Some(center_z)) =
        (coordinate(&x), coordinate(&y), coordinate(&z))
    else {
        return Ok(Value::List(list));
    };
    let distance = distance.floor();
    let matching = heap
        .datums()
        .filter_map(|(datum, candidate)| {
            let path = candidate.type_path().as_str();
            if path == "/area" || path.starts_with("/area/") {
                return None;
            }
            let candidate_x = candidate.field(&x).ok()?.as_number()?;
            let candidate_y = candidate.field(&y).ok()?.as_number()?;
            let candidate_z = candidate.field(&z).ok()?.as_number()?;
            (candidate_z.total_cmp(&center_z).is_eq()
                && (candidate_x - center_x).abs() <= distance
                && (candidate_y - center_y).abs() <= distance)
                .then_some(datum)
        })
        .collect::<Vec<_>>();
    let result = heap
        .list_mut(list)
        .expect("a newly allocated list handle must be live");
    for datum in matching {
        result.add(Value::Datum(datum));
    }
    Ok(Value::List(list))
}

/// Resolves BYOND's `typesof()` selector against the runtime's canonical type
/// catalog. The selected path itself is always present even for a deliberately
/// partial headless catalog, matching the inclusive nature of `typesof`.
fn typesof_builtin(
    value: &Value,
    heap: &ValueHeap,
    catalog: &std::collections::BTreeSet<TypePath>,
) -> Result<Vec<TypePath>, String> {
    let selector = match value {
        Value::TypePath(path) => path.clone(),
        Value::Datum(datum) => heap
            .datum(*datum)
            .map_err(|error| error.to_string())?
            .type_path()
            .clone(),
        value => {
            return Err(format!(
                "typesof requires a type path or datum, received {value}"
            ));
        }
    };
    let mut paths = catalog
        .iter()
        .filter(|path| {
            path.as_str() == selector.as_str()
                || path
                    .as_str()
                    .strip_prefix(selector.as_str())
                    .is_some_and(|suffix| suffix.starts_with('/'))
        })
        .cloned()
        .collect::<Vec<_>>();
    if !catalog.contains(&selector) {
        paths.insert(0, selector);
    }
    Ok(paths)
}

/// Evaluates the small family of BYOND value/type classifiers that need heap
/// access for datum runtime paths. Type paths deliberately participate in
/// `istype`: lifecycle helpers commonly validate deferred type values before
/// constructing them.  `isloc` is variadic, unlike the other simple
/// classifiers: all its supplied values must be atoms.
fn type_predicate_builtin(
    kind: TypePredicateKind,
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<bool, String> {
    let heap = &state.heap;
    let value = arguments
        .first()
        .ok_or_else(|| "type predicate requires a value".to_owned())?;
    match kind {
        TypePredicateKind::IsNull => Ok(matches!(value, Value::Null)),
        TypePredicateKind::IsNum => Ok(matches!(value, Value::Number(_))),
        TypePredicateKind::IsPath => {
            let Value::TypePath(candidate) = value else {
                return Ok(false);
            };
            let Some(target) = arguments.get(1) else {
                return Ok(true);
            };
            let Value::TypePath(target) = target else {
                return Ok(false);
            };
            Ok(is_subtype(state, candidate, target))
        }
        TypePredicateKind::IsList => Ok(matches!(value, Value::List(_))),
        TypePredicateKind::IsMovable => {
            let target = TypePath::parse("/atom/movable").expect("built-in movable path is valid");
            Ok(arguments.iter().all(|value| {
                let Value::Datum(datum) = value else {
                    return false;
                };
                heap.datum(*datum)
                    .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target))
            }))
        }
        TypePredicateKind::IsTurf => {
            let target = TypePath::parse("/turf").expect("built-in turf path is valid");
            Ok(arguments.iter().all(|value| {
                let Value::Datum(datum) = value else {
                    return false;
                };
                heap.datum(*datum)
                    .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target))
            }))
        }
        TypePredicateKind::IsIcon => match value {
            Value::Text(text) => Ok(text.to_ascii_lowercase().ends_with(".dmi")),
            Value::Datum(datum) => {
                let path = heap
                    .datum(*datum)
                    .map_err(|error| error.to_string())?
                    .type_path()
                    .as_str();
                Ok(path == "/icon" || path.starts_with("/icon/"))
            }
            _ => Ok(false),
        },
        TypePredicateKind::IsLoc => {
            let target = TypePath::parse("/atom").expect("built-in atom path is valid");
            Ok(arguments.iter().all(|value| {
                let Value::Datum(datum) = value else {
                    return false;
                };
                heap.datum(*datum)
                    .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target))
            }))
        }
        TypePredicateKind::IsType => {
            let Some(target) = arguments.get(1) else {
                return Ok(matches!(value, Value::Datum(_)));
            };
            let Value::TypePath(target) = target else {
                return Ok(false);
            };
            let candidate = match value {
                Value::TypePath(path) => path,
                Value::Datum(datum) => heap
                    .datum(*datum)
                    .map_err(|error| error.to_string())?
                    .type_path(),
                _ => return Ok(false),
            };
            Ok(is_subtype(state, candidate, target))
        }
    }
}

fn replacement_bounds(
    source: &str,
    arguments: &[Value],
    character_indices: bool,
) -> Result<(usize, usize), String> {
    let index_limit = if character_indices {
        source.chars().count()
    } else {
        source.len()
    };
    let start = optional_text_index(arguments.get(3), 1)?;
    let end = optional_text_index(arguments.get(4), 0)?;
    // BYOND text positions are 1-based and the end is exclusive; zero end
    // extends through the whole remaining text.
    let start = start.clamp(1, index_limit.saturating_add(1));
    let end = if end == 0 {
        index_limit.saturating_add(1)
    } else {
        end.clamp(start, index_limit.saturating_add(1))
    };
    if character_indices {
        Ok((
            character_offset(source, start.saturating_sub(1)),
            character_offset(source, end.saturating_sub(1)),
        ))
    } else {
        // Legacy DM indices count UTF-8 bytes. Clamp inward to valid Rust
        // boundaries instead of manufacturing invalid text slices.
        Ok((
            previous_char_boundary(source, start.saturating_sub(1)),
            previous_char_boundary(source, end.saturating_sub(1)),
        ))
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "DM text positions are non-negative integral binary32 values"
)]
fn optional_text_index(value: Option<&Value>, default: usize) -> Result<usize, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => Ok(number.to_f32().max(0.0) as usize),
        Some(value) => Err(format!(
            "replacetext bounds require a number, received {value}"
        )),
    }
}

fn character_offset(text: &str, character_index: usize) -> usize {
    text.char_indices()
        .nth(character_index)
        .map_or(text.len(), |(byte, _)| byte)
}

fn previous_char_boundary(text: &str, mut byte_index: usize) -> usize {
    byte_index = byte_index.min(text.len());
    while byte_index > 0 && !text.is_char_boundary(byte_index) {
        byte_index -= 1;
    }
    byte_index
}

fn replace_text_ascii_insensitive(target: &str, needle: &str, replacement: &str) -> String {
    if !needle.is_ascii() {
        // DM's Unicode case folding is more involved than Rust's simple
        // lowercase mapping. Preserve deterministic exact text for the rare
        // non-ASCII fallback rather than corrupting byte offsets.
        return target.replace(needle, replacement);
    }
    let needle_lower = needle.to_ascii_lowercase();
    let bytes = target.as_bytes();
    let mut output = String::with_capacity(target.len());
    let mut cursor = 0;
    while cursor < target.len() {
        let remaining = &target[cursor..];
        if remaining.len() >= needle.len()
            && remaining.as_bytes()[..needle.len()]
                .iter()
                .map(u8::to_ascii_lowercase)
                .eq(needle_lower.bytes())
        {
            output.push_str(replacement);
            cursor += needle.len();
        } else {
            let width = char::from(bytes[cursor]).len_utf8();
            output.push_str(&target[cursor..cursor + width]);
            cursor += width;
        }
    }
    output
}

fn runtime_truthy(heap: &ValueHeap, value: &Value) -> Result<bool, String> {
    heap.truthy(value).map_err(|error| error.to_string())
}

fn dynamic_call_target(
    module: &Module,
    state: &ExecutionState,
    receiver: &Value,
    selector: &Value,
    caller_context: &ExecutionContext,
) -> Result<(ProcedureId, ExecutionContext), String> {
    let selector = match selector {
        Value::Text(selector) => String::from(selector.as_ref()),
        Value::TypePath(selector) => selector.to_string(),
        _ => {
            return Err(format!(
                "call procedure selector must be text or a type path, received {selector}"
            ));
        }
    };

    let (base_path, context) = match receiver {
        Value::Null => ("/proc".to_owned(), caller_context.clone()),
        Value::Datum(datum) => (
            state
                .heap()
                .datum(*datum)
                .map_err(|error| error.to_string())?
                .type_path()
                .to_string(),
            ExecutionContext::new(receiver.clone(), caller_context.usr.clone()),
        ),
        Value::TypePath(path) => (path.to_string(), caller_context.clone()),
        _ => {
            return Err(format!(
                "call receiver must be a datum, type path, or null, received {receiver}"
            ));
        }
    };
    let selector_path = selector.trim_start_matches('/');
    let requested = if selector.starts_with('/') {
        selector.clone()
    } else {
        format!("{base_path}/proc/{selector_path}")
    };

    let mut candidate = requested.clone();
    loop {
        let prefix = format!("{candidate}@");
        if let Some((index, _)) = module
            .paths
            .iter()
            .enumerate()
            .rev()
            .find(|(_, path)| *path == &candidate || path.starts_with(&prefix))
        {
            let procedure = ProcedureId::from_index(index).map_err(|error| error.message)?;
            return Ok((procedure, context));
        }
        let Some(segment) = candidate.rfind("/proc/") else {
            break;
        };
        let owner = &candidate[..segment];
        if owner == "/proc" || owner.is_empty() {
            break;
        }
        let Some(parent_end) = owner.rfind('/') else {
            break;
        };
        candidate = format!("{}/proc/{selector_path}", &owner[..parent_end]);
    }
    Err(format!(
        "dynamic call could not resolve procedure {requested:?}"
    ))
}

fn read_list_value<'heap>(
    heap: &'heap ValueHeap,
    list: ListId,
    key: &Value,
) -> Result<&'heap Value, ValueError> {
    let values = heap.list(list)?;
    if matches!(key, Value::Number(_)) {
        let index = value_to_list_index(key).map_err(ValueError::InvalidListIndex)?;
        values.get(index)
    } else {
        values.get_key(key)
    }
}

fn write_list_value(
    heap: &mut ValueHeap,
    list: ListId,
    key: Value,
    value: Value,
) -> Result<(), ValueError> {
    let values = heap.list_mut(list)?;
    if matches!(key, Value::Number(_)) {
        let index = value_to_list_index(&key).map_err(ValueError::InvalidListIndex)?;
        values.set(index, value)?;
    } else {
        values.set_key(key, value);
    }
    Ok(())
}

fn value_to_list_index(value: &Value) -> Result<usize, String> {
    let Some(number) = value.as_number() else {
        return Err(format!("list index must be numeric, received {value}"));
    };
    if !number.is_finite() || number < 1.0 || number.fract() != 0.0 {
        return Err(format!(
            "list index must be a positive whole number, received {number}"
        ));
    }
    number
        .to_string()
        .parse()
        .map_err(|_| format!("list index {number} exceeds the host index range"))
}

fn pop(stack: &mut Vec<Value>) -> Result<Value, String> {
    stack
        .pop()
        .ok_or_else(|| "bytecode stack underflow".to_owned())
}

fn scalar_number_string(value: Value) -> Result<f32, String> {
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        value => Err(format!("numeric operation received {value}")),
    }
}

fn execute_scalar_add(left: Value, right: Value) -> Result<Value, String> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            Ok(Value::number(left.to_f32() + right.to_f32()))
        }
        (Value::Null, Value::Number(right)) => Ok(Value::number(right.to_f32())),
        (Value::Number(left), Value::Null) => Ok(Value::number(left.to_f32())),
        (Value::Null, Value::Null) => Ok(Value::number(0.0)),
        (Value::Text(left), Value::Text(right)) => Ok(Value::text(format!("{left}{right}"))),
        (left, right) => Err(format!(
            "addition requires compatible DM values, received {left} and {right}"
        )),
    }
}

fn execute_scalar_compound_assignment(
    operator: CompoundAssignmentOperator,
    left: Value,
    right: Value,
) -> Result<Value, String> {
    if matches!(operator, CompoundAssignmentOperator::Add)
        && matches!((&left, &right), (Value::Text(_), Value::Text(_)))
    {
        return execute_scalar_add(left, right);
    }
    let left = scalar_number_string(left)?;
    let right = scalar_number_string(right)?;
    let value = match operator {
        CompoundAssignmentOperator::Add => left + right,
        CompoundAssignmentOperator::Subtract => left - right,
        CompoundAssignmentOperator::Multiply => left * right,
        CompoundAssignmentOperator::Divide => left / right,
        CompoundAssignmentOperator::Remainder => left % right,
        CompoundAssignmentOperator::BitAnd => bitwise_binary(left, right, |a, b| a & b),
        CompoundAssignmentOperator::BitOr => bitwise_binary(left, right, |a, b| a | b),
        CompoundAssignmentOperator::BitXor => bitwise_binary(left, right, |a, b| a ^ b),
        CompoundAssignmentOperator::ShiftLeft => {
            bitwise_shift(left, right, |left, right| left << right)
        }
        CompoundAssignmentOperator::ShiftRight => {
            bitwise_shift(left, right, |left, right| left >> right)
        }
    };
    Ok(Value::number(value))
}

fn pop_number(stack: &mut Vec<Value>) -> Result<f32, String> {
    let value = pop(stack)?;
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        value => Err(format!("numeric operation received {value}")),
    }
}

/// Resolves a compact static call count or the count marker produced by an
/// immediately preceding `arglist()` expansion.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn runtime_argument_count(stack: &mut Vec<Value>, encoded: u16) -> Result<usize, String> {
    if encoded != EXPANDED_ARGUMENT_COUNT {
        return Ok(usize::from(encoded));
    }
    let value = stack
        .pop()
        .ok_or_else(|| "bytecode stack underflow".to_owned())?;
    let Value::Number(number) = value else {
        return Err("expanded call argument count is not numeric".to_owned());
    };
    let count = number.to_f32();
    if !count.is_finite() || count < 0.0 || count > f32::from(u16::MAX) || count.fract() != 0.0 {
        return Err("expanded call argument count is invalid".to_owned());
    }
    Ok(count as usize)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use dm_core::{DmNumberBits, SourceSpan};
    use dm_lexer::{SpannedToken, TokenKind, lex};
    use dm_syntax::parse;
    use dm_value::{FieldName, TypePath};

    use super::{
        CompoundListIndexOperator, ExecutionContext, ExecutionLimits, ExecutionState,
        InitializerBinding, Instruction, ProcedureSpec, Program, Value, compile_initializer,
        compile_module, compile_module_specs, compile_procedure,
        compile_procedure_with_resolver_and_fields, condition_tokens, execute, execute_in_context,
        execute_in_state, execute_module, execute_module_in_context, execute_module_in_state,
        execute_module_with_limits, execute_with_limits, execute_with_limits_in_state,
    };

    fn execute_source(source: &str, argument: f32) -> Value {
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        execute(&program, &[Value::number(argument)]).expect("procedure should execute")
    }

    #[test]
    fn condition_tokens_accepts_braced_macro_conditions_with_following_tokens() {
        let tokens = lex("if(!(flags_1 & INITIALIZED_1)) { var/previous = 1")
            .expect("condition source should lex");
        let condition = condition_tokens(&tokens[1..], "if").expect("condition should compile");
        assert!(matches!(condition[0].kind, TokenKind::Operator(ref op) if op == "!"));
    }

    #[test]
    fn absolute_type_path_expressions_lower_to_type_path_values() {
        let syntax =
            parse("/proc/type_path()\n\treturn /obj/item/tool\n").expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("type path expression should compile");

        assert_eq!(
            execute(&program, &[]),
            Ok(Value::TypePath(TypePath::parse("/obj/item/tool").unwrap()))
        );
    }

    #[test]
    fn resource_literal_expressions_lower_to_their_path_text() {
        let syntax = parse("/proc/resource_path()\n\treturn 'sound/effects/piano_hit.ogg'\n")
            .expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("resource literal should compile");

        assert_eq!(
            execute(&program, &[]),
            Ok(Value::text("sound/effects/piano_hit.ogg"))
        );
    }

    #[test]
    fn top_level_semicolons_split_macro_style_statements() {
        let syntax =
            parse("/proc/semicolon_statements()\n\tvar/value = 1; value += 2; return value\n")
                .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("semicolon-separated statements should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(3.0)));
    }

    #[test]
    fn compact_brace_macro_body_is_lowered_as_an_indented_block() {
        // This is the shape produced by Monkestation's lazy-list helpers at
        // the bottom of an already-indented `for`/`if` body.  The statement
        // following `}` must remain outside the inner conditional.
        let syntax = parse(
            "/proc/compact_macro(flag)\n\tvar/value = 0\n\tif(flag)\n\t\tif(!value) { value = 4; } value += 3;\n\treturn value\n",
        )
        .expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("compact macro body should compile");

        assert_eq!(
            execute(&program, &[Value::number(1.0)]),
            Ok(Value::number(7.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(0.0)]),
            Ok(Value::number(0.0))
        );
    }

    #[test]
    fn compact_do_while_macro_body_preserves_nested_block_indentation() {
        // Trait helper macros use a `do { ... } while (0)` wrapper to make a
        // multi-statement expansion behave as one source statement.  Its
        // nested brace blocks must remain children of the synthetic `do`
        // block, while the trailing `while` returns to the caller's level.
        let syntax = parse(
            "/proc/compact_do_macro(flag)\n\tvar/value = 0\n\tdo { var/local = 1; if(flag) { local += 2; } else { local += 4; } value = local; } while(0)\n\treturn value\n",
        )
        .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("compact do/while macro body should compile");

        assert_eq!(
            execute(&program, &[Value::number(1.0)]),
            Ok(Value::number(3.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(0.0)]),
            Ok(Value::number(5.0))
        );
    }

    #[test]
    fn conditional_accepts_a_preprocessor_retained_opening_brace() {
        let syntax = parse("/proc/brace_if(flag)\n\tif(flag) {\n\t\treturn 1\n\t}\n\treturn 0\n")
            .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("brace-terminated condition should compile");

        assert_eq!(
            execute(&program, &[Value::number(1.0)]),
            Ok(Value::number(1.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(0.0)]),
            Ok(Value::number(0.0))
        );
    }

    #[test]
    fn conditional_accepts_same_line_return_body() {
        let syntax = parse("/proc/inline_if(flag)\n\tif(flag) return 7\n\treturn 3\n")
            .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("same-line conditional body should compile");

        assert_eq!(
            execute(&program, &[Value::number(1.0)]),
            Ok(Value::number(7.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(0.0)]),
            Ok(Value::number(3.0))
        );
    }

    #[test]
    fn conditional_accepts_same_line_else_body() {
        let syntax = parse("/proc/inline_else(flag)\n\tif(flag) return 7\n\telse return 3\n")
            .expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("same-line else body should compile");

        assert_eq!(
            execute(&program, &[Value::number(1.0)]),
            Ok(Value::number(7.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(0.0)]),
            Ok(Value::number(3.0))
        );
    }

    #[test]
    fn conditional_preserves_else_if_condition_with_inline_body() {
        let syntax = parse(
            "/proc/inline_else_if(value)\n\tif(value == 1) return 1\n\telse if(value == 2) return 2\n\telse return 3\n",
        )
        .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("same-line else-if body should compile");

        assert_eq!(
            execute(&program, &[Value::number(1.0)]),
            Ok(Value::number(1.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(2.0)]),
            Ok(Value::number(2.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(3.0)]),
            Ok(Value::number(3.0))
        );
    }

    #[test]
    fn conditional_accepts_same_line_continue_body() {
        let syntax = parse(
            "/proc/inline_continue()\n\tvar/total = 0\n\tfor(var/i = 0; i < 4; i++)\n\t\tif(i == 2) continue\n\t\ttotal += i\n\treturn total\n",
        )
        .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("same-line continue body should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(4.0)));
    }

    #[test]
    fn new_type_path_allocates_a_datum_and_discards_constructor_arguments() {
        let syntax = parse(
            "/proc/build()\n\tvar/item = new /datum/example(41, \"ignored\")\n\treturn item\n",
        )
        .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("new should compile");
        assert!(matches!(
            program.instructions.as_slice(),
            [
                Instruction::MakeArgs,
                Instruction::StoreLocal(_),
                Instruction::PushTypePath(_),
                Instruction::PushNumber(_),
                Instruction::PushText(_),
                Instruction::AllocateDatum { argument_count: 2 },
                Instruction::StoreLocal(_),
                Instruction::LoadLocal(_),
                Instruction::Return,
            ]
        ));
        let mut state = ExecutionState::new();
        let value = execute_in_state(&program, &[], &mut state).expect("new should execute");
        let Value::Datum(datum) = value else {
            panic!("new should return a datum");
        };
        assert_eq!(
            state
                .heap()
                .datum(datum)
                .expect("datum must be live")
                .type_path(),
            &TypePath::parse("/datum/example").unwrap()
        );
    }

    #[test]
    fn runtime_new_type_and_proc_ref_macro_expansion_compile() {
        let syntax = parse(
            "/proc/build(starting_organ)\n\tvar/item = new starting_organ(src)\n\treturn list((nameof(.proc/on_entered)), item)\n",
        )
        .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("runtime new type and expanded PROC_REF should compile");
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::AllocateDatum { argument_count: 1 }
        )));
    }

    fn manual_program(instructions: Vec<Instruction>, parameter_count: usize) -> Program {
        let instruction_count = instructions.len();
        Program {
            parameter_count,
            local_count: parameter_count,
            instructions,
            source_spans: (0..instruction_count)
                .map(|index| SourceSpan::new(index * 10, index * 10 + 1))
                .collect(),
        }
    }

    fn field(name: &str) -> FieldName {
        FieldName::parse(name).unwrap()
    }

    fn expression_tokens(source: &str) -> Vec<SpannedToken> {
        lex(source)
            .expect("expression should lex")
            .into_iter()
            .filter(|token| {
                !matches!(
                    token.kind,
                    TokenKind::LineStart { .. } | TokenKind::Newline | TokenKind::LineContinuation
                )
            })
            .collect()
    }

    #[test]
    fn initializer_lowering_uses_explicit_bindings_and_source_spans() {
        let tokens = expression_tokens("base + src.increment + global.offset");
        let bindings =
            BTreeMap::from([("base".to_owned(), InitializerBinding::Global(field("base")))]);
        let initializer = compile_initializer(&tokens, &bindings, None)
            .expect("bound initializer should compile");
        let program = initializer
            .module()
            .procedure(initializer.entry())
            .expect("initializer entry should exist");

        assert_eq!(program.instructions.len(), program.source_spans.len());
        assert!(program.source_spans.iter().all(|span| !span.is_empty()));

        let mut state = ExecutionState::new();
        state.set_global(field("base"), Value::number(2.0));
        state.set_global(field("offset"), Value::number(3.0));
        let src = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());
        state
            .heap_mut()
            .set_datum_field(src, field("increment"), Value::number(4.0))
            .unwrap();
        let context = ExecutionContext::new(Value::Datum(src), Value::Null);

        assert_eq!(
            execute_module_in_context(
                initializer.module(),
                initializer.entry(),
                &[],
                &mut state,
                &context,
            ),
            Ok(Value::number(9.0))
        );
    }

    #[test]
    fn initializer_calls_only_real_linked_global_procedures() {
        let syntax =
            parse("/proc/double(value)\n\treturn value * 2\n").expect("procedure should parse");
        let procedures = compile_module(&syntax.definitions).expect("procedure should compile");
        let tokens = expression_tokens("double(4)");
        let initializer = compile_initializer(&tokens, &BTreeMap::new(), Some(&procedures))
            .expect("linked call should compile");

        assert_eq!(
            execute_module(initializer.module(), initializer.entry(), &[]),
            Ok(Value::number(8.0))
        );
        assert!(
            compile_initializer(
                &expression_tokens("invented_builtin()"),
                &BTreeMap::new(),
                None,
            )
            .is_err(),
            "unregistered names must not become fake built-ins"
        );
    }

    #[test]
    fn explicit_src_and_usr_fields_support_compound_assignment() {
        let source = "/proc/update()\n\tsrc.count += usr.increment\n\treturn src.count\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("fields should compile");
        let mut state = ExecutionState::new();
        let src = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/source").unwrap());
        let usr = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/mob/user").unwrap());
        state
            .heap_mut()
            .set_datum_field(src, field("count"), Value::number(3.0))
            .unwrap();
        state
            .heap_mut()
            .set_datum_field(usr, field("increment"), Value::number(2.0))
            .unwrap();
        let context = ExecutionContext::new(Value::Datum(src), Value::Datum(usr));

        assert_eq!(
            execute_in_context(&program, &[], &mut state, &context),
            Ok(Value::number(5.0))
        );
        assert!(
            state
                .heap()
                .datum_field(src, &field("count"))
                .unwrap()
                .semantic_eq(&Value::number(5.0))
        );
    }

    #[test]
    fn standalone_prefix_and_postfix_increments_are_valid_statements() {
        let source = "/proc/update()\n\tcount++\n\tvar/debt = 2\n\t--debt\n\treturn count - debt\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure_with_resolver_and_fields(
            &syntax.definitions[0],
            &HashMap::new(),
            &BTreeMap::from([("count".to_owned(), field("count"))]),
            &BTreeMap::new(),
        )
        .expect("increments should compile");
        let mut state = ExecutionState::new();
        let src = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/source").unwrap());
        state
            .heap_mut()
            .set_datum_field(src, field("count"), Value::number(3.0))
            .unwrap();
        assert_eq!(
            execute_in_context(
                &program,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(src), Value::Null),
            ),
            Ok(Value::number(3.0))
        );
    }

    #[test]
    fn src_and_usr_aliases_observe_the_same_datum_write() {
        let source = "/proc/alias()\n\tsrc.value = 7\n\treturn usr.value\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("fields should compile");
        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/shared").unwrap());
        let context = ExecutionContext::new(Value::Datum(datum), Value::Datum(datum));

        assert_eq!(
            execute_in_context(&program, &[], &mut state, &context),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn globals_persist_across_executions_and_compound_updates() {
        let set_source =
            parse("/proc/set_global()\n\tglobal.counter = 4\n\treturn global.counter\n").unwrap();
        let increment_source =
            parse("/proc/increment_global()\n\tglobal.counter += 1\n\treturn global.counter\n")
                .unwrap();
        let setter = compile_procedure(&set_source.definitions[0]).unwrap();
        let incrementer = compile_procedure(&increment_source.definitions[0]).unwrap();
        let mut state = ExecutionState::new();

        assert_eq!(
            execute_in_state(&setter, &[], &mut state),
            Ok(Value::number(4.0))
        );
        assert_eq!(
            execute_in_state(&incrementer, &[], &mut state),
            Ok(Value::number(5.0))
        );
        assert!(
            state
                .global(&field("counter"))
                .unwrap()
                .semantic_eq(&Value::number(5.0))
        );
        assert_eq!(state.globals().count(), 1);
    }

    #[test]
    fn glob_alias_lowers_and_executes_as_the_global_namespace() {
        let lowercase =
            parse("/proc/update()\n\tglobal.counter += 1\n\treturn global.counter\n").unwrap();
        let uppercase =
            parse("/proc/update()\n\tGLOB.counter += 1\n\treturn GLOB.counter\n").unwrap();
        let lowercase_program = compile_procedure(&lowercase.definitions[0]).unwrap();
        let uppercase_program = compile_procedure(&uppercase.definitions[0]).unwrap();

        assert_eq!(
            uppercase_program.instructions,
            lowercase_program.instructions
        );

        let mut state = ExecutionState::new();
        state.set_global(field("counter"), Value::number(4.0));
        assert_eq!(
            execute_in_state(&uppercase_program, &[], &mut state),
            Ok(Value::number(5.0))
        );
        assert!(
            state
                .global(&field("counter"))
                .unwrap()
                .semantic_eq(&Value::number(5.0))
        );
    }

    #[test]
    fn assignment_expressions_store_and_yield_the_assigned_value() {
        let source = parse(
            "/proc/locals_and_list(items)\n\tvar/local = 1\n\treturn (local = 5) + (items[1] = local)\n/proc/global_assignment()\n\treturn (GLOB.counter = 9)\n",
        )
        .unwrap();
        let module = compile_module(&source.definitions).unwrap();
        let local_entry = module.procedure_id("/proc/locals_and_list").unwrap();
        let global_entry = module.procedure_id("/proc/global_assignment").unwrap();
        let mut state = ExecutionState::new();
        let list = state.heap.allocate_list();
        state.heap.list_mut(list).unwrap().add(Value::number(0.0));

        assert_eq!(
            execute_module_in_state(&module, local_entry, &[Value::List(list)], &mut state),
            Ok(Value::number(10.0))
        );
        assert_eq!(
            state.heap.list(list).unwrap().positions().next(),
            Some((1, &Value::number(5.0)))
        );
        assert_eq!(
            execute_module_in_state(&module, global_entry, &[], &mut state),
            Ok(Value::number(9.0))
        );
        assert_eq!(state.global(&field("counter")), Some(&Value::number(9.0)));
    }

    #[test]
    fn nameof_procedure_reference_lowers_to_the_procedure_name() {
        let source = parse(
            "/proc/main()\n\treturn capture(nameof(.proc/on_signal))\n/proc/capture(value)\n\treturn value\n",
        )
        .unwrap();
        let module = compile_module(&source.definitions).unwrap();
        let entry = module.procedure_id("/proc/main").unwrap();

        assert_eq!(
            execute_module(&module, entry, &[]),
            Ok(Value::text("on_signal"))
        );
    }

    #[test]
    fn nameof_accepts_type_and_static_member_references() {
        let source = parse(
            "/proc/main()\n\treturn list(nameof(/datum/example.proc/run), nameof(type::field))\n",
        )
        .unwrap();
        let module = compile_module(&source.definitions).unwrap();
        let entry = module.procedure_id("/proc/main").unwrap();
        let mut state = ExecutionState::new();

        let result = execute_module_in_state(&module, entry, &[], &mut state).unwrap();
        let Value::List(list) = result else {
            panic!("expected list result");
        };
        assert_eq!(
            state
                .heap()
                .list(list)
                .unwrap()
                .positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>(),
            vec![Value::text("run"), Value::text("field")]
        );
    }

    #[test]
    fn named_and_parent_calls_preserve_object_context() {
        let source =
            parse("/proc/main()\n\treturn helper()\n/proc/helper()\n\treturn usr.value\n").unwrap();
        let module = compile_module(&source.definitions).unwrap();
        let entry = module.procedure_id("/proc/main").unwrap();
        let mut state = ExecutionState::new();
        let usr = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/mob/user").unwrap());
        state
            .heap_mut()
            .set_datum_field(usr, field("value"), Value::number(6.0))
            .unwrap();
        let context = ExecutionContext::new(Value::Null, Value::Datum(usr));
        assert_eq!(
            execute_module_in_context(&module, entry, &[], &mut state, &context),
            Ok(Value::number(6.0))
        );

        let parent_source =
            parse("/proc/base()\n\treturn src.value\n/proc/child()\n\treturn ..()\n").unwrap();
        let parent_module = compile_module_specs(&[
            ProcedureSpec {
                path: "/proc/base@0".to_owned(),
                definition: &parent_source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/proc/child@1".to_owned(),
                definition: &parent_source.definitions[1],
                parent: Some(0),
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ])
        .unwrap();
        let child = parent_module.procedure_id_at(1).unwrap();
        let src = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/source").unwrap());
        state
            .heap_mut()
            .set_datum_field(src, field("value"), Value::number(8.0))
            .unwrap();
        let context = ExecutionContext::new(Value::Datum(src), Value::Null);
        assert_eq!(
            execute_module_in_context(&parent_module, child, &[], &mut state, &context),
            Ok(Value::number(8.0))
        );

        let current_source = parse(
            "/proc/recurse(depth)\n\tif(depth <= 0)\n\t\treturn src.value\n\treturn .(depth - 1)\n",
        )
        .unwrap();
        let current_program = compile_procedure(&current_source.definitions[0]).unwrap();
        assert_eq!(
            execute_in_context(
                &current_program,
                &[Value::number(2.0)],
                &mut state,
                &context,
            ),
            Ok(Value::number(8.0))
        );
    }

    #[test]
    fn static_call_statement_executes_and_discards_its_result() {
        let source = parse(
            "/proc/entry()\n\thelper()\n\treturn global.calls\n/proc/helper()\n\tglobal.calls += 1\n\treturn 99\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry should exist");
        let mut state = ExecutionState::new();
        state.set_global(field("calls"), Value::number(0.0));

        assert_eq!(
            execute_module_in_context(
                &module,
                entry,
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn parent_call_statement_executes_and_discards_its_result() {
        let source = parse(
            "/proc/base()\n\tglobal.calls += 1\n\treturn 99\n/proc/child()\n\t..()\n\treturn global.calls\n",
        )
        .expect("source should parse");
        let module = compile_module_specs(&[
            ProcedureSpec {
                path: "/proc/base@0".to_owned(),
                definition: &source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/proc/child@1".to_owned(),
                definition: &source.definitions[1],
                parent: Some(0),
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ])
        .expect("resolved parent specs should compile");
        let child = module.procedure_id_at(1).expect("child should exist");
        let mut state = ExecutionState::new();
        state.set_global(field("calls"), Value::number(0.0));

        assert_eq!(
            execute_module_in_context(
                &module,
                child,
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(1.0))
        );
        let program = module.procedure(child).expect("child program should exist");
        assert!(program.instructions.windows(2).any(|instructions| matches!(
            instructions,
            [Instruction::CallParent { .. }, Instruction::Pop]
        )));
    }

    #[test]
    fn keyword_style_call_arguments_compile_in_source_order() {
        let source = parse(
            "/proc/entry()\n\treturn helper(first = 3, second = 4)\n/proc/helper(first, second)\n\treturn first * 10 + second\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions)
            .expect("keyword-style call arguments should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry procedure should resolve");

        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(34.0)));
    }

    #[test]
    fn keyword_style_arguments_in_discarded_calls_do_not_become_assignments() {
        let source = parse(
            "/proc/entry()\n\thelper(is_directional = TRUE, is_beam = TRUE)\n\treturn 7\n/proc/helper(first, second)\n\treturn first && second\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions)
            .expect("discarded keyword-style calls should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry procedure should resolve");

        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(7.0)));
    }

    #[test]
    fn keyword_style_arguments_in_datum_calls_do_not_become_assignments() {
        // Lifecycle code commonly invokes inherited datum procedures such as
        // `AddComponent(...)` rather than a global helper.  The argument
        // labels are still call syntax in that postfix form: they must not be
        // parsed as assignments to bare locals on the caller.
        let source = parse(
            "/proc/entry(receiver)\n\treceiver.AddComponent(/datum/component/overlay_lighting, is_directional = TRUE, is_beam = TRUE)\n\treturn 7\n",
        )
        .expect("source should parse");
        compile_procedure(&source.definitions[0])
            .expect("datum calls with keyword-style arguments should compile");
    }

    #[test]
    fn keyword_style_arguments_in_bare_datum_calls_do_not_become_assignments() {
        // An inherited datum call is commonly written without an explicit
        // receiver in an atom lifecycle hook.  This is the form used by
        // `/atom/movable/Initialize` in downstream SS13 codebases.
        let source = parse(
            "/atom/movable/proc/Initialize()\n\tAddComponent(/datum/component/overlay_lighting, is_directional = TRUE, is_beam = TRUE)\n/atom/proc/AddComponent(first, second, third)\n\treturn\n",
        )
        .expect("source should parse");
        compile_module_specs(&[
            ProcedureSpec {
                path: "/atom/movable/proc/Initialize@0".to_owned(),
                definition: &source.definitions[0],
                parent: None,
                static_calls: BTreeMap::from([("AddComponent".to_owned(), 1)]),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/atom/proc/AddComponent@0".to_owned(),
                definition: &source.definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ])
        .expect("bare datum calls with keyword-style arguments should compile");
    }

    #[test]
    fn macro_wrapped_named_arguments_become_textual_list_keys() {
        // Monkestation's `AddComponent` macro expands to this exact shape.
        // The labels survive only as associative list keys once expansion has
        // occurred, so they must not be lowered as caller locals.
        let source = parse(
            "/proc/entry()\n\t_AddComponent(list(/datum/component/overlay_lighting, is_directional = TRUE, is_beam = TRUE))\n\treturn 7\n/proc/_AddComponent(raw_args)\n\treturn 0\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions)
            .expect("macro-wrapped named arguments should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry procedure should resolve");

        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(7.0)));
    }

    #[test]
    fn weighted_pick_style_semicolons_select_a_builtin_candidate() {
        let source =
            parse("/proc/entry()\n\treturn pick(10; 3, 1; 4)\n").expect("source should parse");
        let module =
            compile_module(&source.definitions).expect("weighted pick syntax should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry should exist");

        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(3.0)));
        assert!(module.procedure(entry).expect("entry program should exist").instructions.iter().any(
            |instruction| matches!(instruction, Instruction::Pick { weighted } if weighted == &vec![true, true]),
        ));
    }

    #[test]
    fn repeated_nested_blocks_may_redeclare_macro_locals() {
        let source = parse(
            "/proc/repeated_scopes()\n\tvar/total = 0\n\tdo { var/_L = 1; total += _L; } while(0)\n\tdo { var/_L = 2; total += _L; } while(0)\n\treturn total\n",
        )
        .expect("repeated scoped locals should parse");
        let module = compile_module(&source.definitions)
            .expect("nested blocks should permit repeated local names");
        let entry = module.procedure_id("/proc/repeated_scopes").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(3.0)));
    }

    #[test]
    fn copytext_char_uses_character_positions_and_negative_offsets() {
        let source = parse(
            "/proc/middle()\n\treturn copytext_char(\"AéB\", 2, 3)\n/proc/tail()\n\treturn copytext_char(\"Hi there\", -5)\n",
        )
        .expect("copytext_char source should parse");
        let module = compile_module(&source.definitions).expect("copytext_char should compile");
        let middle = module.procedure_id("/proc/middle").expect("middle");
        let tail = module.procedure_id("/proc/tail").expect("tail");
        assert_eq!(execute_module(&module, middle, &[]), Ok(Value::text("é")));
        assert_eq!(execute_module(&module, tail, &[]), Ok(Value::text("there")));
    }

    #[test]
    fn block_enumerates_inclusive_turf_rectangles() {
        let source = parse("/proc/box(start, finish)\n\treturn block(start, finish)\n")
            .expect("block source should parse");
        let module = compile_module(&source.definitions).expect("block should compile");
        let entry = module.procedure_id("/proc/box").expect("box");
        let mut state = ExecutionState::new();
        let turf_path = TypePath::parse("/turf/test").expect("turf path");
        let mut turfs = Vec::new();
        for (x_value, y_value) in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)] {
            let turf = state.heap_mut().allocate_datum(turf_path.clone());
            state
                .heap_mut()
                .set_datum_field(turf, field("x"), Value::number(x_value))
                .unwrap();
            state
                .heap_mut()
                .set_datum_field(turf, field("y"), Value::number(y_value))
                .unwrap();
            state
                .heap_mut()
                .set_datum_field(turf, field("z"), Value::number(1.0))
                .unwrap();
            turfs.push(turf);
        }
        let result = execute_module_in_state(
            &module,
            entry,
            &[Value::Datum(turfs[3]), Value::Datum(turfs[0])],
            &mut state,
        )
        .expect("block should execute");
        let Value::List(list) = result else {
            panic!("block should return a list");
        };
        assert_eq!(state.heap().list(list).expect("block list").len(), 4);
    }

    #[test]
    fn random_builtins_are_deterministic_and_respect_their_bounds() {
        let source =
            parse("/proc/range()\n\treturn rand(4, 6)\n/proc/chance()\n\treturn prob(100)\n")
                .expect("source should parse");
        let module = compile_module(&source.definitions).expect("random builtins should compile");
        let range = module
            .procedure_id("/proc/range")
            .expect("range should exist");
        let chance = module
            .procedure_id("/proc/chance")
            .expect("chance should exist");
        let first = execute_module(&module, range, &[]).expect("rand should execute");
        let second =
            execute_module(&module, range, &[]).expect("fresh states should reproduce rand");
        assert_eq!(first, second);
        assert!(matches!(first.as_number(), Some(value) if (4.0..=6.0).contains(&value)));
        assert_eq!(execute_module(&module, chance, &[]), Ok(Value::number(1.0)));
    }

    #[test]
    fn round_builtin_preserves_byond_floor_and_nearest_multiple_forms() {
        let source = parse(
            "/proc/floor_form()\n\treturn round(-1.45)\n/proc/nearest()\n\treturn round(1.99, 1)\n/proc/step()\n\treturn round(1.45, 1.5)\n/proc/negative_tie()\n\treturn round(-1.5, 1)\n/proc/zero_multiple()\n\treturn round(-1.45, 0)\n/proc/negative_multiple()\n\treturn round(2.2, -0.5)\n",
        )
        .expect("round builtin source should parse");
        let module = compile_module(&source.definitions).expect("round builtin should compile");
        for (path, expected) in [
            ("/proc/floor_form", -2.0),
            ("/proc/nearest", 2.0),
            ("/proc/step", 1.5),
            ("/proc/negative_tie", -1.0),
            ("/proc/zero_multiple", -2.0),
            ("/proc/negative_multiple", 2.0),
        ] {
            let procedure = module
                .procedure_id(path)
                .expect("round procedure should exist");
            assert_eq!(
                execute_module(&module, procedure, &[]),
                Ok(Value::number(expected))
            );
        }
        assert!(module.procedures.iter().any(|program| {
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::Round { .. }))
        }));
    }

    #[test]
    fn calls_accept_a_trailing_argument_separator() {
        let source = parse(
            "/proc/entry()\n\treturn helper(3, 4,)\n/proc/helper(first, second)\n\treturn first * 10 + second\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions)
            .expect("a call with a trailing separator should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry procedure should resolve");

        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(34.0)));
    }

    #[test]
    fn dynamic_call_dispatches_receiver_methods_from_text_and_type_path_selectors() {
        let source = parse(
            "/datum/receiver/proc/entry(selector)\n\treturn call(src, selector)(4)\n/datum/receiver/proc/run(value)\n\treturn src.base + value\n",
        )
        .unwrap();
        let module = compile_module_specs(&[
            ProcedureSpec {
                path: "/datum/receiver/proc/entry@0".to_owned(),
                definition: &source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/receiver/proc/run@0".to_owned(),
                definition: &source.definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ])
        .unwrap();
        let entry = module.procedure_id_at(0).unwrap();
        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/receiver").unwrap());
        state
            .heap_mut()
            .set_datum_field(receiver, field("base"), Value::number(3.0))
            .unwrap();
        let context = ExecutionContext::new(Value::Datum(receiver), Value::Null);

        assert_eq!(
            execute_module_in_context(
                &module,
                entry,
                &[Value::Text("run".into())],
                &mut state,
                &context,
            ),
            Ok(Value::number(7.0))
        );
        assert_eq!(
            execute_module_in_context(
                &module,
                entry,
                &[Value::TypePath(
                    TypePath::parse("/datum/receiver/proc/run").unwrap(),
                )],
                &mut state,
                &context,
            ),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn null_conditional_field_index_and_call_short_circuit_without_rhs_evaluation() {
        let source = parse(
            "/datum/example/proc/read(value, list/values)\n\tvar/a = value?.field\n\tvar/b = values?[bump()]\n\tvar/c = value?:take(bump())\n\tvalue?.field = bump()\n\tvalues?[bump()] = bump()\n\treturn isnull(a) + isnull(b) + isnull(c) + GLOB.calls\n/datum/example/proc/take(value)\n\treturn value\n/proc/bump()\n\tGLOB.calls += 1\n\treturn 1\n",
        )
        .expect("null-conditional source should parse");
        let mut specs = Vec::new();
        specs.push(ProcedureSpec {
            path: "/datum/example/proc/read@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::from([("bump".to_owned(), 2)]),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::from([("calls".to_owned(), field("calls"))]),
        });
        specs.push(ProcedureSpec {
            path: "/datum/example/proc/take@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::from([("calls".to_owned(), field("calls"))]),
        });
        specs.push(ProcedureSpec {
            path: "/proc/bump@0".to_owned(),
            definition: &source.definitions[2],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::from([("calls".to_owned(), field("calls"))]),
        });
        let module = compile_module_specs(&specs).expect("null-conditional source should compile");
        let mut state = ExecutionState::new();
        state.set_global(field("calls"), Value::number(0.0));
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id_at(0).expect("read entry"),
                &[Value::Null, Value::Null],
                &mut state,
            ),
            Ok(Value::number(3.0))
        );
        assert_eq!(state.global(&field("calls")), Some(&Value::number(0.0)));
    }

    #[test]
    fn null_conditional_access_executes_normally_for_live_receivers() {
        let source = parse(
            "/datum/example/proc/read(list/values)\n\tvar/a = src?.field\n\tvar/b = values?[1]\n\treturn a + b\n",
        )
        .expect("live null-conditional source should parse");
        let module = compile_module_specs(&[ProcedureSpec {
            path: "/datum/example/proc/read@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::from([("field".to_owned(), field("field"))]),
            global_fields: BTreeMap::new(),
        }])
        .expect("live null-conditional source should compile");
        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());
        state
            .heap_mut()
            .set_datum_field(datum, field("field"), Value::number(4.0))
            .unwrap();
        let list = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .list_mut(list)
            .unwrap()
            .add(Value::number(5.0));
        assert_eq!(
            execute_module_in_context(
                &module,
                module.procedure_id_at(0).expect("read entry"),
                &[Value::List(list)],
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            ),
            Ok(Value::number(9.0))
        );
    }

    #[test]
    fn dotted_datum_calls_lower_to_dynamic_dispatch() {
        let source = parse(
            "/datum/receiver/proc/entry()\n\treturn src.run(4)\n/datum/receiver/proc/run(value)\n\treturn src.base + value\n",
        )
        .expect("source should parse");
        let module = compile_module_specs(&[
            ProcedureSpec {
                path: "/datum/receiver/proc/entry@0".to_owned(),
                definition: &source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/receiver/proc/run@0".to_owned(),
                definition: &source.definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ])
        .expect("dotted datum call should compile");
        let entry = module.procedure_id_at(0).expect("entry should exist");
        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/receiver").expect("type path"));
        state
            .heap_mut()
            .set_datum_field(receiver, field("base"), Value::number(3.0))
            .expect("datum should be live");

        assert_eq!(
            execute_module_in_context(
                &module,
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(receiver), Value::Null),
            ),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn bare_src_field_dotted_call_is_a_valid_side_effect_statement() {
        let source = parse(
            "/datum/item/proc/Initialize()\n\tatom_storage.set_holdable(4)\n\treturn 9\n/datum/storage/proc/set_holdable(value)\n\tsrc.value = value\n",
        )
        .expect("source should parse");
        let module = compile_module_specs(&[
            ProcedureSpec {
                path: "/datum/item/proc/Initialize@0".to_owned(),
                definition: &source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::from([("atom_storage".to_owned(), field("atom_storage"))]),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/storage/proc/set_holdable@0".to_owned(),
                definition: &source.definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ])
        .expect("bare field dotted call should compile");
        let mut state = ExecutionState::new();
        let item = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/item").expect("item type"));
        let storage = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/storage").expect("storage type"));
        state
            .heap_mut()
            .set_datum_field(item, field("atom_storage"), Value::Datum(storage))
            .expect("item should be live");

        assert_eq!(
            execute_module_in_context(
                &module,
                module.procedure_id_at(0).expect("Initialize should exist"),
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(item), Value::Null),
            ),
            Ok(Value::number(9.0))
        );
        assert!(
            state
                .heap()
                .datum_field(storage, &field("value"))
                .expect("storage should be live")
                .semantic_eq(&Value::number(4.0))
        );
    }

    #[test]
    fn field_errors_retain_source_mapping_for_null_missing_and_stale_receivers() {
        let syntax = parse("/proc/read()\n\treturn src.missing\n").unwrap();
        let span = syntax.definitions[0].body[0].span;
        let program = compile_procedure(&syntax.definitions[0]).unwrap();
        let mut state = ExecutionState::new();
        let null_error =
            execute_in_context(&program, &[], &mut state, &ExecutionContext::default())
                .unwrap_err();
        assert_eq!(null_error.message, "field read received null");
        assert_eq!(null_error.source_span, Some(span));

        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/source").unwrap());
        let context = ExecutionContext::new(Value::Datum(datum), Value::Null);
        let missing_error = execute_in_context(&program, &[], &mut state, &context).unwrap_err();
        assert_eq!(
            missing_error.message,
            "datum field FieldName(\"missing\") is absent"
        );
        assert_eq!(missing_error.source_span, Some(span));

        state.heap_mut().destroy_datum(datum).unwrap();
        let stale_error = execute_in_context(&program, &[], &mut state, &context).unwrap_err();
        assert_eq!(stale_error.message, format!("stale datum handle {datum:?}"));
        assert_eq!(stale_error.source_span, Some(span));
    }

    #[test]
    fn logical_assignment_short_circuits_locals_fields_and_list_entries() {
        let source = parse(
            "/datum/example/proc/run()\n\tvar/local\n\tlocal ||= 3\n\tvar/list/values = list()\n\tvalues[\"entry\"] ||= 4\n\tsrc.flag ||= 5\n\treturn local + values[\"entry\"] + src.flag\n",
        )
        .expect("logical assignment source should parse");
        let module = compile_module_specs(&[ProcedureSpec {
            path: "/datum/example/proc/run@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::from([("flag".to_owned(), field("flag"))]),
            global_fields: BTreeMap::new(),
        }])
        .expect("logical assignments should compile");
        let entry = module.procedure_id_at(0).expect("entry");
        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());
        state
            .heap_mut()
            .set_datum_field(datum, field("flag"), Value::Null)
            .unwrap();
        assert_eq!(
            execute_module_in_context(
                &module,
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            ),
            Ok(Value::number(12.0))
        );
    }

    #[test]
    fn plane_macro_nested_scope_keeps_cached_locals_visible() {
        let source = parse(
            "/proc/plane_macro(flag, other)\n\tvar/output = 0\n\tdo { if(flag) { var/_cached_plane = 7; var/_our_turf = other; if(_our_turf) { output = _cached_plane; } else if(other) { output = _cached_plane; } else { output = _cached_plane; } } else { output = 2; } } while(0)\n\treturn output\n",
        )
        .expect("plane macro source should parse");
        let module = compile_module(&source.definitions).expect("plane macro scope should compile");
        let entry = module.procedure_id("/proc/plane_macro").expect("entry");
        assert_eq!(
            execute_module(&module, entry, &[Value::number(1.0), Value::number(1.0)]),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn list_binary_operators_return_new_lists_without_mutating_the_left_operand() {
        let source = parse(
            "/proc/run()\n\tvar/list/a = list(1, 2, 2, 3)\n\tvar/list/b = list(2, 4)\n\tvar/list/added = a + b\n\tvar/list/subtracted = a - b\n\tvar/list/unioned = a | b\n\tvar/list/masked = a & b\n\tvar/list/xored = a ^ b\n\treturn a.len + added.len + subtracted.len + unioned.len + masked.len + xored.len + (a[2] == 2) + (unioned[4] == 4)\n",
        )
        .expect("list operator source should parse");
        let module = compile_module(&source.definitions).expect("list operators should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
            Ok(Value::number(24.0))
        );
    }

    #[test]
    fn compound_list_operators_mutate_shared_alias_identity() {
        let source = parse(
            "/proc/run()\n\tvar/list/a = list(1, 2)\n\tvar/list/alias = a\n\ta += list(2, 3)\n\tvar/after_add = alias.len\n\ta -= 2\n\tvar/after_remove = alias.len\n\ta |= list(3, 4)\n\tvar/after_union = alias.len\n\ta &= list(1, 4)\n\tvar/after_mask = alias.len\n\ta ^= list(4, 5)\n\treturn after_add + after_remove + after_union + after_mask + alias.len + (alias[1] == 1) + (alias[2] == 5)\n",
        )
        .expect("compound list operator source should parse");
        let module =
            compile_module(&source.definitions).expect("compound list operators should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
            Ok(Value::number(17.0))
        );
    }

    #[test]
    fn documented_list_methods_and_len_execute_natively() {
        let source = parse(
            "/proc/run()\n\tvar/list/values = list(\"a\", \"b\", \"c\")\n\tvalues.Add(list(\"d\", \"e\"))\n\tvar/list/copied = values.Copy(2, 5)\n\tvalues.Cut(2, 3)\n\tvar/found = values.Find(\"d\")\n\tvar/next_index = values.Insert(2, list(\"x\", \"y\"))\n\tvalues.Splice(-1, 0, \"z\")\n\tvalues.Swap(1, 6)\n\tvalues.len = 7\n\tvar/removed = values.Remove(\"d\")\n\tvar/removed_all = values.RemoveAll(\"x\")\n\treturn copied.len + (copied[1] == \"b\") + (copied[3] == \"d\") + found + next_index + removed + removed_all + values.len + (values[1] == \"z\") + (values[2] == \"y\")\n",
        )
        .expect("list method source should parse");
        let module = compile_module(&source.definitions).expect("list methods should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
            Ok(Value::number(21.0))
        );
    }

    #[test]
    fn list_copy_and_swap_keep_associative_values_attached_to_keys() {
        let source = parse(
            "/proc/run()\n\tvar/list/values = list(\"red\" = 1, \"blue\" = 2, \"green\" = 3)\n\tvar/list/copied = values.Copy()\n\tvalues.Swap(1, 3)\n\treturn (values[1] == \"green\") + (values[\"green\"] == 3) + (copied[1] == \"red\") + (copied[\"red\"] == 1)\n",
        )
        .expect("associative list method source should parse");
        let module =
            compile_module(&source.definitions).expect("associative list methods should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
            Ok(Value::number(4.0))
        );
    }

    #[test]
    fn documented_native_builtins_cover_text_math_and_type_helpers() {
        let source = parse(
            "/proc/native(kind)\n\tvar/path = text2path(\"/datum/child\")\n\tif(!path)\n\t\treturn 0\n\treturn (2 ** 3 ** 2) + floor(1.9) + abs(-2) + findlasttext(\"/datum/child\", \"/\") + initial(kind.flag)\n",
        )
        .expect("native builtin source should parse");
        let module = compile_module(&source.definitions).expect("native builtins should compile");
        let mut state = ExecutionState::new();
        let base = TypePath::parse("/datum/base").unwrap();
        let child = TypePath::parse("/datum/child").unwrap();
        state.set_type_paths([base.clone(), child.clone()]);
        state.set_type_parents(BTreeMap::from([
            (base.clone(), Some(TypePath::parse("/datum").unwrap())),
            (child.clone(), Some(base.clone())),
        ]));
        state.set_initial_values(BTreeMap::from([(
            child.clone(),
            BTreeMap::from([(field("flag"), Value::number(7.0))]),
        )]));
        let result = execute_module_in_state(
            &module,
            module.procedure_id("/proc/native").unwrap(),
            &[Value::TypePath(child)],
            &mut state,
        )
        .expect("native builtin procedure should execute");
        // 2 ** (3 ** 2) = 512; floor=1; abs=2; final slash is byte 7; initial=7.
        assert_eq!(result, Value::number(529.0));
    }

    #[test]
    fn type_predicates_follow_runtime_parent_catalog_not_path_spelling() {
        let source = parse(
            "/proc/check(value)\n\treturn istype(value, /atom/movable) && ismovable(value)\n",
        )
        .expect("predicate source should parse");
        let module = compile_module(&source.definitions).expect("predicate source should compile");
        let mut state = ExecutionState::new();
        let obj = TypePath::parse("/obj/item").unwrap();
        state.set_type_parents(BTreeMap::from([
            (obj.clone(), Some(TypePath::parse("/obj").unwrap())),
            (
                TypePath::parse("/obj").unwrap(),
                Some(TypePath::parse("/atom/movable").unwrap()),
            ),
            (
                TypePath::parse("/atom/movable").unwrap(),
                Some(TypePath::parse("/atom").unwrap()),
            ),
        ]));
        let datum = state.heap_mut().allocate_datum(obj);
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/check").unwrap(),
                &[Value::Datum(datum)],
                &mut state,
            ),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn direction_and_icon_builtins_cover_lifecycle_shapes() {
        let source = parse(
            "/proc/directions()\n\treturn NORTH + SOUTH + EAST + WEST + NORTHEAST + NORTHWEST + SOUTHEAST + SOUTHWEST\n/proc/icon_resource()\n\treturn isicon('icons/test.dmi')\n",
        )
        .expect("builtin source should parse");
        let module = compile_module(&source.definitions).expect("builtins should compile");
        assert_eq!(
            execute_module(
                &module,
                module.procedure_id("/proc/directions").unwrap(),
                &[]
            ),
            Ok(Value::number(45.0))
        );
        assert_eq!(
            execute_module(
                &module,
                module.procedure_id("/proc/icon_resource").unwrap(),
                &[]
            ),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn shared_value_migration_preserves_scalar_execution() {
        let program = manual_program(
            vec![
                Instruction::PushNumber(DmNumberBits::from_f32(2.0)),
                Instruction::PushNumber(DmNumberBits::from_f32(3.0)),
                Instruction::Add,
                Instruction::Return,
            ],
            0,
        );

        assert_eq!(execute(&program, &[]), Ok(Value::number(5.0)));
    }

    #[test]
    fn datum_type_field_reflects_the_heap_runtime_type() {
        let program = manual_program(
            vec![
                Instruction::LoadSrc,
                Instruction::LoadField(field("type")),
                Instruction::Return,
            ],
            0,
        );
        let mut state = ExecutionState::new();
        let path = TypePath::parse("/obj/machinery/example").unwrap();
        let datum = state.heap_mut().allocate_datum(path.clone());

        assert_eq!(
            execute_in_context(
                &program,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            ),
            Ok(Value::TypePath(path))
        );
    }

    #[test]
    fn list_construction_allocates_heap_storage_in_source_order() {
        let program = manual_program(
            vec![
                Instruction::PushNumber(DmNumberBits::from_f32(7.0)),
                Instruction::PushText("second".to_owned()),
                Instruction::MakeList(2),
                Instruction::Return,
            ],
            0,
        );
        let mut state = ExecutionState::new();
        let result = execute_in_state(&program, &[], &mut state).unwrap();
        let Value::List(list) = result else {
            panic!("MakeList must return a list handle");
        };

        let values = state.heap().list(list).unwrap();
        assert!(values.get(1).unwrap().semantic_eq(&Value::number(7.0)));
        assert!(values.get(2).unwrap().semantic_eq(&Value::text("second")));
    }

    #[test]
    fn list_aliases_observe_heap_mutation_across_executions() {
        let program = manual_program(
            vec![
                Instruction::LoadLocal(0),
                Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
                Instruction::IndexList,
                Instruction::Return,
            ],
            1,
        );
        let mut state = ExecutionState::new();
        let list = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .list_mut(list)
            .unwrap()
            .add(Value::number(4.0));
        let alias = Value::List(list);

        assert_eq!(
            execute_in_state(&program, std::slice::from_ref(&alias), &mut state),
            Ok(Value::number(4.0))
        );
        state
            .heap_mut()
            .list_mut(list)
            .unwrap()
            .set(1, Value::number(9.0))
            .unwrap();
        assert_eq!(
            execute_in_state(&program, &[alias], &mut state),
            Ok(Value::number(9.0))
        );
    }

    #[test]
    fn stale_list_indexing_maps_to_source_aware_runtime_error() {
        let program = manual_program(
            vec![
                Instruction::LoadLocal(0),
                Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
                Instruction::IndexList,
                Instruction::Return,
            ],
            1,
        );
        let mut state = ExecutionState::new();
        let stale_list = state.heap_mut().allocate_list();
        state.heap_mut().destroy_list(stale_list).unwrap();
        let error = execute_in_state(&program, &[Value::List(stale_list)], &mut state)
            .expect_err("a stale handle must never resolve through the VM");

        assert_eq!(error.message, format!("stale list handle {stale_list:?}"));
        assert_eq!(error.instruction, 2);
        assert_eq!(error.source_span, Some(SourceSpan::new(20, 21)));
        assert_eq!(error.call_stack.len(), 1);
    }

    #[test]
    fn list_instructions_consume_the_existing_shared_budget() {
        let program = manual_program(
            vec![
                Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
                Instruction::MakeList(1),
                Instruction::Return,
            ],
            0,
        );
        let mut state = ExecutionState::new();
        let error = execute_with_limits_in_state(
            &program,
            &[],
            ExecutionLimits {
                max_steps: 2,
                ..ExecutionLimits::default()
            },
            &mut state,
        )
        .expect_err("Return must require its own instruction-budget unit");

        assert_eq!(error.message, "instruction budget of 2 exhausted");
        assert_eq!(error.instruction, 2);
        assert_eq!(error.source_span, Some(SourceSpan::new(20, 21)));
    }

    #[test]
    fn compiles_locals_and_executes_binary32_arithmetic() {
        let source = "/proc/probe(input)\n\tvar/doubled = input * 2\n\treturn doubled + 3\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let result = execute(&program, &[Value::number(4.0)]).expect("procedure should execute");

        assert_eq!(result, Value::number(11.0));
        assert_eq!(program.instructions.len(), program.source_spans.len());
        // Every procedure now begins by materializing its implicit `args` list.
        assert_eq!(program.source_spans[0], syntax.definitions[0].span);
        assert_eq!(program.source_spans[2], syntax.definitions[0].body[0].span);
    }

    #[test]
    fn observes_operator_precedence_and_parentheses() {
        let result = execute_source("/proc/probe(input)\n\treturn (input + 3) * 2\n", 4.0);

        assert_eq!(result, Value::number(14.0));
    }

    #[test]
    fn parenthesized_expression_statements_discard_their_result() {
        let syntax =
            parse("/proc/probe(input)\n\tvar/value = 0\n\t((value = input + 1))\n\treturn value\n")
                .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("parenthesized expression statement should compile");

        assert_eq!(
            execute(&program, &[Value::number(41.0)]),
            Ok(Value::number(42.0))
        );
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::Pop))
        );
    }

    #[test]
    fn executes_bitwise_operators_with_dm_integer_coercion_and_precedence() {
        let source = "/proc/probe()\n\treturn (-1 & 6) + (7 ^ 3 | 8) + (9.9 & 3)\n";
        let syntax = parse(source).expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("bitwise expressions should compile");

        // -1 & 6 = 6, (7 ^ 3) | 8 = 12, and 9.9 truncates to 9 before
        // bitwise conjunction with 3, giving 1.
        assert_eq!(execute(&program, &[]), Ok(Value::number(19.0)));
        assert!(program.instructions.iter().any(|instruction| {
            matches!(
                instruction,
                Instruction::BitAnd | Instruction::BitOr | Instruction::BitXor
            )
        }));
    }

    #[test]
    fn executes_unary_bitwise_complement_with_dm_integer_coercion() {
        let source = "/proc/probe()\n\treturn ~9.9 + ~0\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("unary bitwise complement should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(-11.0)));
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::BitNot))
        );
    }

    #[test]
    fn bitwise_compound_assignments_update_locals_and_list_indices() {
        let source = "/proc/probe(items)\n\tvar/value = 14\n\tvalue &= 11\n\tvalue |= 16\n\titems[1] ^= value\n\treturn items[1]\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("bitwise compound assignments should compile");
        let mut state = ExecutionState::new();
        let list = state.heap.allocate_list();
        state.heap.list_mut(list).unwrap().add(Value::number(7.0));

        // ((14 & 11) | 16) = 26; 7 ^ 26 = 29.
        assert_eq!(
            execute_in_state(&program, &[Value::List(list)], &mut state),
            Ok(Value::number(29.0))
        );
    }

    #[test]
    fn shift_operators_and_compound_assignments_use_signed_32_bit_semantics() {
        let source = "/proc/probe(items)\n\tvar/value = 3 << 2\n\tvalue >>= 1\n\titems[1] <<= value\n\treturn (-8 >> 2) + items[1] + (1 << 33)\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("shift expressions and assignments should compile");
        let mut state = ExecutionState::new();
        let list = state.heap.allocate_list();
        state.heap.list_mut(list).unwrap().add(Value::number(1.0));

        // value is (3 << 2) >> 1 = 6; item becomes 1 << 6 = 64. Right
        // shifts preserve the sign bit, and a count of 33 masks to one.
        assert_eq!(
            execute_in_state(&program, &[Value::List(list)], &mut state),
            Ok(Value::number(64.0))
        );
        assert!(program.instructions.iter().any(|instruction| {
            matches!(
                instruction,
                Instruction::ShiftLeft | Instruction::ShiftRight
            )
        }));
    }

    #[test]
    fn conditional_expressions_associate_right() {
        let source = "/proc/probe(input)\n\treturn input == 1 ? 10 : input == 2 ? 20 : 30\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("conditional expressions should compile");

        assert_eq!(
            execute(&program, &[Value::number(1.0)]),
            Ok(Value::number(10.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(2.0)]),
            Ok(Value::number(20.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(3.0)]),
            Ok(Value::number(30.0))
        );

        let short_circuit = parse("/proc/short_circuit()\n\treturn TRUE ? 7 : 1 in 2\n")
            .expect("source should parse");
        let program = compile_procedure(&short_circuit.definitions[0])
            .expect("conditional expressions should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(7.0)));
    }

    #[test]
    fn compiles_in_as_a_relational_list_membership_operator() {
        let source = "/proc/probe(input)\n\treturn input + 1 in list(2, 4, \"key\" = 9)\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::Contains))
        );
        assert_eq!(
            execute(&program, &[Value::number(1.0)]),
            Ok(Value::number(1.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(3.0)]),
            Ok(Value::number(1.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(8.0)]),
            Ok(Value::number(0.0))
        );
    }

    #[test]
    fn in_checks_associative_keys_but_not_associative_values() {
        let source = "/proc/probe(input)\n\treturn input in list(\"key\" = 9)\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(
            execute(&program, &[Value::text("key")]),
            Ok(Value::number(1.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(9.0)]),
            Ok(Value::number(0.0))
        );
    }

    #[test]
    fn rejects_unknown_locals_during_compilation() {
        let syntax =
            parse("/proc/probe(input)\n\treturn missing + input\n").expect("source should parse");
        let error = compile_procedure(&syntax.definitions[0])
            .expect_err("unknown local should fail compilation");

        assert!(error.message.contains("unknown local"));
    }

    #[test]
    fn executes_assignment_and_nested_if_else_blocks() {
        let source = "/proc/clamp(input)\n\tvar/result = input\n\tif(result < 0)\n\t\tresult = 0\n\telse\n\t\tif(result > 10)\n\t\t\tresult = 10\n\treturn result\n";

        assert_eq!(execute_source(source, -2.0), Value::number(0.0));
        assert_eq!(execute_source(source, 7.0), Value::number(7.0));
        assert_eq!(execute_source(source, 18.0), Value::number(10.0));
    }

    #[test]
    fn recognizes_when_both_conditional_branches_return() {
        let source = "/proc/sign(input)\n\tif(input < 0)\n\t\treturn -1\n\telse\n\t\treturn 1\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(
            execute(&program, &[Value::number(-2.0)]),
            Ok(Value::number(-1.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(2.0)]),
            Ok(Value::number(1.0))
        );
        assert_eq!(program.instructions.len(), program.source_spans.len());
    }

    #[test]
    fn calls_forward_declared_procedures_with_positional_arguments() {
        let source = "/proc/main(input)\n\treturn add(input, 3)\n/proc/add(left, right)\n\treturn left + right\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/main")
            .expect("entry procedure should exist");

        assert_eq!(
            execute_module(&module, entry, &[Value::number(8.0)]),
            Ok(Value::number(11.0))
        );
    }

    #[test]
    fn arglist_expands_inside_static_and_dynamic_call_arguments() {
        let static_source = parse(
            "/proc/entry()\n\treturn combine(1, arglist(list(2, 3)), 4)\n/proc/combine(a, b, c, d)\n\treturn a + b + c + d\n",
        )
        .expect("static arglist source should parse");
        let module =
            compile_module(&static_source.definitions).expect("static arglist should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry should resolve");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(10.0)));

        let dynamic_source = parse(
            "/datum/receiver/proc/entry()\n\treturn call(src, \"combine\")(1, arglist(list(2, 3)), 4)\n/datum/receiver/proc/combine(a, b, c, d)\n\treturn a + b + c + d\n",
        )
        .expect("dynamic arglist source should parse");
        let module = compile_module_specs(&[
            ProcedureSpec {
                path: "/datum/receiver/proc/entry@0".to_owned(),
                definition: &dynamic_source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/receiver/proc/combine@0".to_owned(),
                definition: &dynamic_source.definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ])
        .expect("dynamic arglist should compile");
        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/receiver").expect("type path"));
        assert_eq!(
            execute_module_in_context(
                &module,
                module.procedure_id_at(0).expect("entry should resolve"),
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(receiver), Value::Null),
            ),
            Ok(Value::number(10.0))
        );
    }

    #[test]
    fn executes_recursive_calls_on_explicit_frames() {
        let source = "/proc/factorial(input)\n\tif(input <= 1)\n\t\treturn 1\n\treturn input * factorial(input - 1)\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/factorial")
            .expect("entry procedure should exist");

        assert_eq!(
            execute_module(&module, entry, &[Value::number(5.0)]),
            Ok(Value::number(120.0))
        );
    }

    #[test]
    fn binds_missing_arguments_to_null_and_retains_extra_arguments() {
        let source = "/proc/identity(input)\n\treturn input\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/identity")
            .expect("entry procedure should exist");

        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::Null));
        assert_eq!(
            execute_module(&module, entry, &[Value::number(7.0), Value::number(99.0)]),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn bounds_recursion_and_reports_the_source_mapped_call_stack() {
        let source = "/proc/recurse(input)\n\treturn recurse(input)\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/recurse")
            .expect("entry procedure should exist");
        let error = execute_module_with_limits(
            &module,
            entry,
            &[Value::number(1.0)],
            ExecutionLimits {
                max_call_depth: 3,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("unbounded recursion should reach the explicit limit");

        assert!(error.message.contains("maximum call depth 3"));
        assert_eq!(error.call_stack.len(), 3);
        assert!(error.source_span.is_some());
        assert!(
            error
                .call_stack
                .iter()
                .all(|trace| trace.procedure == "/proc/recurse" && trace.source_span.is_some())
        );
    }

    #[test]
    fn maps_callee_runtime_errors_and_preserves_caller_context() {
        let source = "/proc/main()\n\treturn broken()\n/proc/broken()\n\treturn \"text\" + 1\n";
        let syntax = parse(source).expect("source should parse");
        let expected_span = syntax.definitions[1].body[0].span;
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/main")
            .expect("entry procedure should exist");
        let error =
            execute_module(&module, entry, &[]).expect_err("numeric operation on text should fail");

        assert!(
            error
                .message
                .contains("addition requires compatible DM values")
        );
        assert_eq!(error.source_span, Some(expected_span));
        assert_eq!(error.call_stack.len(), 2);
        assert_eq!(error.call_stack[0].procedure, "/proc/main");
        assert_eq!(error.call_stack[1].procedure, "/proc/broken");
        assert_eq!(error.call_stack[1].source_span, Some(expected_span));
    }

    #[test]
    fn current_call_uses_explicit_positional_arguments() {
        let source =
            "/proc/countdown(value)\n\tif(value <= 0)\n\t\treturn 0\n\treturn 1 + .(value - 1)\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/countdown")
            .expect("entry procedure should exist");

        assert_eq!(
            execute_module(&module, entry, &[Value::number(4.0)]),
            Ok(Value::number(4.0))
        );
    }

    #[test]
    fn argumentless_current_call_reuses_original_frame_arguments() {
        let source = "/proc/recurse(value, stop)\n\tstop = 1\n\treturn .()\n";
        let syntax = parse(source).expect("source should parse");
        let call_span = syntax.definitions[0].body[1].span;
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/recurse")
            .expect("entry procedure should exist");
        let error = execute_module_with_limits(
            &module,
            entry,
            &[Value::number(7.0), Value::Null, Value::number(99.0)],
            ExecutionLimits {
                max_call_depth: 4,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("reused original arguments should keep recursing");

        assert!(error.message.contains("maximum call depth 4"));
        assert_eq!(error.source_span, Some(call_span));
        assert_eq!(error.call_stack.len(), 4);
        assert!(error.call_stack.iter().all(|trace| {
            trace.procedure == "/proc/recurse" && trace.source_span == Some(call_span)
        }));
    }

    #[test]
    fn unresolved_parent_call_reports_source_mapped_runtime_error() {
        let syntax = parse("/proc/child()\n\treturn ..()\n").expect("source should parse");
        let span = syntax.definitions[0].body[0].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let error = execute(&program, &[]).expect_err("unresolved parent should fail at runtime");

        assert_eq!(
            error.message,
            "parent procedure call has no resolved target"
        );
        assert_eq!(error.source_span, Some(span));
    }

    #[test]
    fn while_supports_zero_and_multiple_iterations() {
        let source = "/proc/count(limit)\n\tvar/result = 0\n\twhile(result < limit)\n\t\tresult = result + 1\n\treturn result\n";

        assert_eq!(execute_source(source, 0.0), Value::number(0.0));
        assert_eq!(execute_source(source, 5.0), Value::number(5.0));
    }

    #[test]
    fn do_while_executes_before_testing_and_routes_loop_control_to_condition() {
        let source = "/proc/count(limit)\n\tvar/result = 0\n\tdo\n\t\tresult = result + 1\n\t\tif(result == 2)\n\t\t\tcontinue\n\t\tif(result > limit)\n\t\t\tbreak\n\twhile(result <= limit)\n\treturn result\n";

        // A post-test loop enters even when the condition will be false.
        assert_eq!(execute_source(source, 0.0), Value::number(1.0));
        // `continue` tests the condition, and `break` exits without testing.
        assert_eq!(execute_source(source, 3.0), Value::number(4.0));
    }

    #[test]
    fn switch_matches_values_ranges_and_default_after_evaluating_selector_once() {
        let source = "/proc/classify(value)\n\tvar/calls = 0\n\tswitch(value + 0)\n\t\tif(1, 3)\n\t\t\treturn 10\n\t\tif(4 to 6)\n\t\t\treturn 20\n\t\telse\n\t\t\treturn 30\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("switch should compile");

        assert_eq!(
            execute(&program, &[Value::number(1.0)]),
            Ok(Value::number(10.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(3.0)]),
            Ok(Value::number(10.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(4.0)]),
            Ok(Value::number(20.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(6.0)]),
            Ok(Value::number(20.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(7.0)]),
            Ok(Value::number(30.0))
        );
    }

    #[test]
    fn switch_rejects_case_after_default() {
        let source = "/proc/invalid(value)\n\tswitch(value)\n\t\telse\n\t\t\treturn 1\n\t\tif(2)\n\t\t\treturn 2\n";
        let syntax = parse(source).expect("source should parse");
        let error = compile_procedure(&syntax.definitions[0])
            .expect_err("case after default must not compile");

        assert_eq!(error.message, "switch case cannot follow an else default");
    }

    #[test]
    fn do_requires_indented_body_and_trailing_while() {
        for (source, expected) in [
            (
                "/proc/invalid()\n\tdo\n",
                "do statement requires an indented body",
            ),
            (
                "/proc/invalid()\n\tdo\n\t\treturn 1\n",
                "do statement requires a trailing while condition",
            ),
        ] {
            let syntax = parse(source).expect("source should parse");
            let error = compile_procedure(&syntax.definitions[0])
                .expect_err("invalid do loop should not compile");

            assert_eq!(error.message, expected);
        }
    }

    #[test]
    fn break_and_continue_work_inside_nested_conditionals() {
        let source = "/proc/filter(limit)\n\tvar/index = 0\n\tvar/total = 0\n\twhile(index < limit)\n\t\tindex = index + 1\n\t\tif(index == 2)\n\t\t\tcontinue\n\t\tif(index > 4)\n\t\t\tbreak\n\t\ttotal = total + index\n\treturn total\n";
        let syntax = parse(source).expect("source should parse");
        let while_span = syntax.definitions[0].body[2].span;
        let continue_span = syntax.definitions[0].body[5].span;
        let break_span = syntax.definitions[0].body[7].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(
            execute(&program, &[Value::number(10.0)]),
            Ok(Value::number(8.0))
        );
        assert_eq!(program.instructions.len(), program.source_spans.len());
        assert!(program.instructions.iter().zip(&program.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::JumpIfFalse(_))
                && *span == while_span
        ));
        assert!(program.instructions.iter().zip(&program.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::Jump(_))
                && *span == continue_span
        ));
        assert!(program.instructions.iter().zip(&program.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::Jump(_))
                && *span == break_span
        ));
    }

    #[test]
    fn nested_loops_patch_break_and_continue_to_the_innermost_loop() {
        let source = "/proc/nested(limit)\n\tvar/outer = 0\n\tvar/total = 0\n\twhile(outer < limit)\n\t\touter = outer + 1\n\t\tvar/inner = 0\n\t\twhile(inner < 5)\n\t\t\tinner = inner + 1\n\t\t\tif(inner == 2)\n\t\t\t\tcontinue\n\t\t\tif(inner == 4)\n\t\t\t\tbreak\n\t\t\ttotal = total + 1\n\treturn total\n";

        assert_eq!(execute_source(source, 3.0), Value::number(6.0));
    }

    #[test]
    fn rejects_break_and_continue_outside_loops() {
        for (statement, expected) in [
            ("break", "break outside a loop"),
            ("continue", "continue outside a loop"),
        ] {
            let source = format!("/proc/invalid()\n\t{statement}\n");
            let syntax = parse(&source).expect("source should parse");
            let error = compile_procedure(&syntax.definitions[0])
                .expect_err("loop control outside a loop should fail");

            assert_eq!(error.message, expected);
        }
    }

    #[test]
    fn instruction_budget_terminates_an_infinite_while_with_source_context() {
        let source = "/proc/spin()\n\twhile(1)\n\t\tcontinue\n";
        let syntax = parse(source).expect("source should parse");
        let while_span = syntax.definitions[0].body[0].span;
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/spin")
            .expect("entry procedure should exist");
        let error = execute_module_with_limits(
            &module,
            entry,
            &[],
            ExecutionLimits {
                max_steps: 8,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("infinite loop should exhaust its instruction budget");

        assert_eq!(error.message, "instruction budget of 8 exhausted");
        assert_eq!(error.source_span, Some(while_span));
        assert_eq!(error.call_stack.len(), 1);
        assert_eq!(error.call_stack[0].procedure, "/proc/spin");
        assert_eq!(error.call_stack[0].source_span, Some(while_span));
    }

    #[test]
    fn exact_standalone_instruction_budget_completes_the_final_return() {
        let source = "/proc/increment(value)\n\treturn value + 1\n";
        let syntax = parse(source).expect("source should parse");
        let return_span = syntax.definitions[0].body[0].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let exact_steps = u64::try_from(program.instructions.len())
            .expect("test program instruction count should fit u64");

        assert_eq!(
            execute_with_limits(
                &program,
                &[Value::number(4.0)],
                ExecutionLimits {
                    max_steps: exact_steps,
                    ..ExecutionLimits::default()
                },
            ),
            Ok(Value::number(5.0))
        );
        let error = execute_with_limits(
            &program,
            &[Value::number(4.0)],
            ExecutionLimits {
                max_steps: exact_steps - 1,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("one fewer step should stop before Return");
        assert_eq!(error.source_span, Some(return_span));
        assert_eq!(error.call_stack[0].procedure, "<standalone>");
    }

    #[test]
    fn instruction_budget_is_shared_across_procedure_calls() {
        let source = "/proc/main()\n\treturn helper()\n/proc/helper()\n\treturn 7\n";
        let syntax = parse(source).expect("source should parse");
        let helper_span = syntax.definitions[1].body[0].span;
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/main")
            .expect("entry procedure should exist");

        assert_eq!(
            execute_module_with_limits(
                &module,
                entry,
                &[],
                ExecutionLimits {
                    max_steps: 8,
                    ..ExecutionLimits::default()
                },
            ),
            Ok(Value::number(7.0))
        );
        let error = execute_module_with_limits(
            &module,
            entry,
            &[],
            ExecutionLimits {
                max_steps: 5,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("caller and callee should consume one shared budget");

        assert_eq!(error.source_span, Some(helper_span));
        assert_eq!(error.call_stack.len(), 2);
        assert_eq!(error.call_stack[0].procedure, "/proc/main");
        assert_eq!(error.call_stack[1].procedure, "/proc/helper");
        assert_eq!(error.call_stack[1].source_span, Some(helper_span));
    }

    #[test]
    fn c_style_for_supports_scoped_initializer_and_postfix_increment() {
        let source = "/proc/sum(limit)\n\tvar/total = 0\n\tfor(var/i = 0; i < limit; i++)\n\t\ttotal = total + i\n\treturn total\n";

        assert_eq!(execute_source(source, 0.0), Value::number(0.0));
        assert_eq!(execute_source(source, 5.0), Value::number(10.0));

        let escaped =
            parse("/proc/invalid()\n\tfor(var/i = 0; i < 1; i++)\n\t\tcontinue\n\treturn i\n")
                .expect("source should parse");
        let error = compile_procedure(&escaped.definitions[0])
            .expect_err("for initializer should be scoped to its loop");
        assert_eq!(error.message, "unknown local \"i\"");
    }

    #[test]
    fn for_to_range_is_inclusive_and_continue_runs_its_increment() {
        let source = "/proc/sum(first, last)\n\tvar/total = 0\n\tfor(var/i in first to last)\n\t\tif(i == first)\n\t\t\tcontinue\n\t\ttotal += i\n\treturn total\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("range loop should compile");
        assert_eq!(
            execute(&program, &[Value::number(2.0), Value::number(5.0)]),
            Ok(Value::number(12.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(5.0), Value::number(2.0)]),
            Ok(Value::number(0.0))
        );
    }

    #[test]
    fn for_to_range_honors_explicit_positive_and_negative_steps() {
        let source = "/proc/ranges()\n\tvar/total = 0\n\tfor(var/i in 5 to 1 step -2)\n\t\ttotal += i\n\tfor(var/j in 1 to 5 step 2)\n\t\ttotal += j\n\treturn total\n";
        let syntax = parse(source).expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("stepped range loops should compile");

        // 5 + 3 + 1, then 1 + 3 + 5.
        assert_eq!(execute(&program, &[]), Ok(Value::number(18.0)));
    }

    #[test]
    fn for_to_range_evaluates_its_step_once() {
        let source = "/proc/step_once()\n\tvar/step = 2\n\tvar/total = 0\n\tfor(var/i in 1 to 5 step step)\n\t\ttotal += i\n\t\tstep = 1\n\treturn total\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("range step expression should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
    }

    #[test]
    fn c_style_for_supports_prefix_decrement_and_optional_clauses() {
        let decrement = "/proc/sum(limit)\n\tvar/total = 0\n\tfor(var/i = limit; i > 0; --i)\n\t\ttotal = total + i\n\treturn total\n";
        assert_eq!(execute_source(decrement, 3.0), Value::number(6.0));

        let optional = "/proc/once()\n\tfor(;;)\n\t\tbreak\n\treturn 9\n";
        let syntax = parse(optional).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
    }

    #[test]
    fn for_continue_runs_increment_and_break_exits_the_loop() {
        let source = "/proc/filter(limit)\n\tvar/total = 0\n\tfor(var/i = 0; i < limit; i++)\n\t\tif(i == 1)\n\t\t\tcontinue\n\t\tif(i == 4)\n\t\t\tbreak\n\t\ttotal = total + i\n\treturn total\n";
        let syntax = parse(source).expect("source should parse");
        let for_span = syntax.definitions[0].body[1].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(
            execute(&program, &[Value::number(10.0)]),
            Ok(Value::number(5.0))
        );
        assert!(program.instructions.iter().zip(&program.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::StoreLocal(_))
                && *span == for_span
        ));
    }

    #[test]
    fn nested_for_loops_patch_control_to_the_innermost_loop() {
        let source = "/proc/nested(limit)\n\tvar/total = 0\n\tfor(var/i = 0; i < limit; i++)\n\t\tfor(var/j = 0; j < 4; j++)\n\t\t\tif(j == 1)\n\t\t\t\tcontinue\n\t\t\tif(j == 3)\n\t\t\t\tbreak\n\t\t\ttotal = total + 1\n\treturn total\n";

        assert_eq!(execute_source(source, 3.0), Value::number(6.0));
    }

    #[test]
    fn infinite_for_obeys_step_budget_and_for_in_compiles() {
        let source = "/proc/spin()\n\tfor(;;)\n\t\tcontinue\n";
        let syntax = parse(source).expect("source should parse");
        let for_span = syntax.definitions[0].body[0].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let error = execute_with_limits(
            &program,
            &[],
            ExecutionLimits {
                max_steps: 7,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("infinite for should exhaust its step budget");
        assert_eq!(error.message, "instruction budget of 7 exhausted");
        assert_eq!(error.source_span, Some(for_span));

        let list_iteration =
            parse("/proc/list_loop(items)\n\tfor(var/item in items)\n\t\tcontinue\n")
                .expect("source should parse");
        let program = compile_procedure(&list_iteration.definitions[0])
            .expect("for-in list iteration should compile");
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::ListLength))
        );
    }

    #[test]
    fn typed_for_in_binding_ignores_as_qualifier() {
        let source = "/proc/typed_loop()\n\tfor(var/turf/area_turf as anything in list(1))\n\t\tarea_turf = null\n\treturn 7\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("typed for-in binding should use area_turf, not the as qualifier");

        assert_eq!(execute(&program, &[]), Ok(Value::number(7.0)));
    }

    #[test]
    fn list_literals_support_bracket_reads_and_writes() {
        let source =
            "/proc/list_access()\n\tvar/items = list(1, 2, 3)\n\titems[2] = 9\n\treturn items[2]\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("lists should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
    }

    #[test]
    fn list_assignment_preserves_alias_identity() {
        let source = "/proc/update(items)\n\titems[1] = 12\n\treturn items[1]\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("indexing should compile");
        let mut state = ExecutionState::new();
        let list = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .list_mut(list)
            .unwrap()
            .add(Value::number(1.0));

        assert_eq!(
            execute_in_state(&program, &[Value::List(list)], &mut state),
            Ok(Value::number(12.0))
        );
        assert!(
            state
                .heap()
                .list(list)
                .unwrap()
                .get(1)
                .unwrap()
                .semantic_eq(&Value::number(12.0))
        );
    }

    #[test]
    fn compound_list_index_assignment_updates_positional_and_associative_entries() {
        let source = "/proc/update(items)\n\titems[1] += 4\n\titems[\"score\"] *= 3\n\treturn items[1] + items[\"score\"]\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("compound list-index assignments should compile");
        let mut state = ExecutionState::new();
        let list = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .list_mut(list)
            .unwrap()
            .add(Value::number(2.0));
        state
            .heap_mut()
            .list_mut(list)
            .unwrap()
            .set_key(Value::text("score"), Value::number(5.0));

        assert!(program.instructions.iter().any(|instruction| {
            matches!(
                instruction,
                Instruction::CompoundListIndex(CompoundListIndexOperator::Add)
            )
        }));
        assert_eq!(
            execute_in_state(&program, &[Value::List(list)], &mut state),
            Ok(Value::number(21.0))
        );
        let values = state.heap().list(list).unwrap();
        assert!(values.get(1).unwrap().semantic_eq(&Value::number(6.0)));
        assert!(
            values
                .get_key(&Value::text("score"))
                .unwrap()
                .semantic_eq(&Value::number(15.0))
        );
    }

    #[test]
    fn associative_literals_lookup_update_and_iterate_in_source_order() {
        let lookup = "/proc/lookup()\n\tvar/items = list(1, \"first\" = 10, 2, \"second\" = 20)\n\titems[\"first\"] = 11\n\treturn items[\"first\"]\n";
        let syntax = parse(lookup).expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("associations should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(11.0)));

        let iteration = "/proc/order()\n\tvar/result = 0\n\tfor(var/item in list(1, \"key\" = 10, 2))\n\t\tif(item == \"key\")\n\t\t\tresult = result * 10 + 9\n\t\telse\n\t\t\tresult = result * 10 + item\n\treturn result\n";
        let syntax = parse(iteration).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("iteration should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(192.0)));
    }

    #[test]
    fn for_in_break_continue_and_nesting_target_the_innermost_loop() {
        let source = "/proc/nested_lists()\n\tvar/total = 0\n\tfor(var/outer in list(1, 2))\n\t\tfor(var/inner in list(1, 2, 3, 4))\n\t\t\tif(inner == 2)\n\t\t\t\tcontinue\n\t\t\tif(inner == 4)\n\t\t\t\tbreak\n\t\t\ttotal = total + outer * inner\n\treturn total\n";
        let syntax = parse(source).expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("nested lists should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(12.0)));
    }

    #[test]
    fn parameter_literal_default_applies_only_when_argument_is_omitted() {
        let source = "/proc/defaulted(value = 5)\n\treturn value\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(5.0)));
        assert_eq!(execute(&program, &[Value::Null]), Ok(Value::Null));
        assert_eq!(
            execute(&program, &[Value::number(9.0)]),
            Ok(Value::number(9.0))
        );

        let text = parse("/proc/text_default(value = \"fallback\")\n\treturn value\n")
            .expect("source should parse");
        let program = compile_procedure(&text.definitions[0]).expect("procedure should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::text("fallback")));
    }

    #[test]
    fn dm_boolean_constants_work_in_defaults_and_expressions() {
        let source = "/proc/booleans(enabled = TRUE, disabled = FALSE)\n\tif(disabled)\n\t\treturn 99\n\treturn enabled + TRUE\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("DM boolean constants should compile as numeric literals");

        assert_eq!(execute(&program, &[]), Ok(Value::number(2.0)));
        assert_eq!(
            execute(&program, &[Value::Null, Value::number(1.0)]),
            Ok(Value::number(99.0))
        );
    }

    #[test]
    fn dm_blend_constants_work_in_defaults_and_expressions() {
        let source = "/proc/blend(mode = BLEND_MULTIPLY)\n\treturn mode + BLEND_INSET_OVERLAY + BLEND_DEFAULT + BLEND_OVERLAY + BLEND_ADD + BLEND_SUBTRACT\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("BYOND blend constants should compile as numeric literals");

        assert_eq!(execute(&program, &[]), Ok(Value::number(15.0)));
        assert_eq!(
            execute(&program, &[Value::number(2.0)]),
            Ok(Value::number(13.0))
        );
    }

    #[test]
    fn dm_reset_appearance_constants_are_appearance_flag_bits() {
        let source =
            "/proc/appearance_flags()\n\treturn RESET_TRANSFORM | RESET_COLOR | RESET_ALPHA | 1\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("appearance constants should compile as BYOND numeric constants");

        assert_eq!(execute(&program, &[]), Ok(Value::number(57.0)));
    }

    #[test]
    fn dm_appearance_flags_are_the_documented_byond_bit_positions() {
        let source = "/proc/appearance_flags()\n\treturn KEEP_TOGETHER | KEEP_APART | LONG_GLIDE | RESET_TRANSFORM | RESET_COLOR | RESET_ALPHA | PIXEL_SCALE | TILE_BOUND | INHERIT_ID | NO_CLIENT_COLOR | RESET_CONTENTS | PLANE_MASTER | PASS_MOUSE | TILE_MOVER\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("appearance flags should compile as BYOND numeric constants");

        assert_eq!(execute(&program, &[]), Ok(Value::number(16_383.0)));
    }

    #[test]
    fn replacetext_builtin_family_replaces_text_with_byond_bounds() {
        let source = "/proc/rewrite()\n\tvar/exact = replacetextEx_char(\"Port Bow / port bow\", \"Port Bow\", \"Northwest\")\n\tvar/insensitive = replacetext_char(exact, \"port bow\", \"Southwest\")\n\treturn replacetextEx(insensitive, \"Northwest\", \"East\", 1, 10)\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("replacetext builtin family should compile");

        assert!(program.instructions.iter().any(|instruction| {
            matches!(
                instruction,
                Instruction::ReplaceText {
                    exact: true,
                    character_indices: true,
                    ..
                }
            )
        }));
        assert_eq!(execute(&program, &[]), Ok(Value::text("East / Southwest")));
    }

    #[test]
    fn typed_and_uninitialized_locals_start_as_null() {
        let source = "/proc/locals()\n\tvar/datum/example/typed\n\tvar/plain\n\tif(typed || plain)\n\t\treturn 0\n\treturn 7\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("typed locals without initializers should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(7.0)));
    }

    #[test]
    fn unnamed_varargs_parameter_reserves_its_argument_slot() {
        let source = "/proc/with_varargs(first, ...)\n\tvar/after = first\n\treturn after\n";
        let syntax = parse(source).expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("unnamed varargs should compile");

        assert_eq!(program.parameter_count, 2);
        // Two declared argument positions, one ordinary local, and implicit
        // per-call `args`.
        assert_eq!(program.local_count, 4);
        assert_eq!(
            execute(&program, &[Value::number(9.0)]),
            Ok(Value::number(9.0))
        );
    }

    #[test]
    fn implicit_args_is_a_per_call_list_of_all_supplied_values() {
        let source = "/proc/collect(first)\n\tif(length(args) != 3)\n\t\treturn 0\n\treturn args[3] + first\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("implicit args should compile as a local list");

        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::MakeArgs))
        );
        assert_eq!(
            execute(
                &program,
                &[Value::number(2.0), Value::number(5.0), Value::number(11.0)]
            ),
            Ok(Value::number(13.0))
        );
        assert_eq!(execute(&program, &[]), Ok(Value::number(0.0)));
    }

    #[test]
    fn multiple_parameter_defaults_evaluate_in_declaration_order() {
        let source = "/proc/combine(first = 1 + 1, second = 3, third = 4)\n\treturn first + second + third\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
        assert_eq!(
            execute(&program, &[Value::number(10.0)]),
            Ok(Value::number(17.0))
        );
        assert_eq!(
            execute(
                &program,
                &[Value::number(10.0), Value::Null, Value::number(1.0)],
            ),
            Ok(Value::number(11.0)),
            "explicit null suppresses the default and participates in arithmetic as numeric zero",
        );
    }

    #[test]
    fn defaults_interact_with_explicit_and_argument_reusing_current_calls() {
        let countdown = "/proc/countdown(value = 3)\n\tif(value <= 0)\n\t\treturn 0\n\treturn 1 + .(value - 1)\n";
        let syntax = parse(countdown).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(3.0)));

        let reapply = "/proc/reapply(value = 1)\n\tvalue = 0\n\treturn .()\n";
        let syntax = parse(reapply).expect("source should parse");
        let call_span = syntax.definitions[0].body[1].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let error = execute_with_limits(
            &program,
            &[],
            ExecutionLimits {
                max_call_depth: 3,
                ..ExecutionLimits::default()
            },
        )
        .expect_err(".() should reuse omission and reapply the default in each frame");
        assert!(error.message.contains("maximum call depth 3"));
        assert_eq!(error.source_span, Some(call_span));
        assert_eq!(error.call_stack.len(), 3);
    }

    #[test]
    fn parameter_defaults_are_general_runtime_expressions() {
        let source = parse(
            "/proc/add_one(value)\n\treturn value + 1\n\n/proc/defaulted(first = 2, second = add_one(first), third = second * 10)\n\treturn first + second + third\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions)
            .expect("defaults should support parameter references and procedure calls");

        // Defaults execute at invocation time, in parameter order.  Each
        // later default observes values supplied or defaulted for its
        // predecessors, while supplied arguments skip only their own default.
        assert_eq!(
            execute_module(&module, module.names["/proc/defaulted"], &[]),
            Ok(Value::number(35.0))
        );
        assert_eq!(
            execute_module(
                &module,
                module.names["/proc/defaulted"],
                &[Value::number(7.0)],
            ),
            Ok(Value::number(95.0))
        );
    }

    #[test]
    fn special_result_starts_null_and_is_returned_on_fallthrough() {
        let syntax = parse("/proc/empty()\n").expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::Null));
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::LoadResult))
        );
    }

    #[test]
    fn special_result_supports_reads_assignments_and_compound_assignments() {
        let source = "/proc/result()\n\t. = 2\n\t. += 3\n\t. *= 4\n\treturn .\n";
        let syntax = parse(source).expect("source should parse");
        let assignment_span = syntax.definitions[0].body[0].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(20.0)));
        assert!(program.instructions.iter().zip(&program.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::StoreResult)
                && *span == assignment_span
        ));
    }

    #[test]
    fn special_result_survives_branches_and_loops() {
        let source = "/proc/accumulate(input)\n\t. = 0\n\twhile(input > 0)\n\t\tif(input == 2)\n\t\t\t. += 10\n\t\telse\n\t\t\t. += input\n\t\tinput = input - 1\n";

        assert_eq!(execute_source(source, 3.0), Value::number(14.0));
    }

    #[test]
    fn explicit_return_takes_precedence_over_special_result() {
        let syntax = parse("/proc/result()\n\t. = 5\n\treturn 9\n").expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
    }

    #[test]
    fn type_predicate_builtins_classify_null_numbers_paths_and_subtypes() {
        for (source, expected) in [
            ("/proc/test()\n\treturn isnull(null)\n", 1.0),
            ("/proc/test()\n\treturn isnum(3)\n", 1.0),
            ("/proc/test()\n\treturn ispath(/datum/example)\n", 1.0),
            ("/proc/test()\n\treturn islist(list(1))\n", 1.0),
            ("/proc/test()\n\treturn islist(3)\n", 0.0),
            ("/proc/test()\n\treturn ismovable(new /atom/movable)\n", 1.0),
            (
                "/proc/test()\n\treturn ismovable(new /atom/movable/child)\n",
                1.0,
            ),
            ("/proc/test()\n\treturn ismovable(new /obj/item)\n", 1.0),
            ("/proc/test()\n\treturn ismovable(new /mob/living)\n", 1.0),
            ("/proc/test()\n\treturn ismovable(new /atom)\n", 0.0),
            ("/proc/test()\n\treturn ismovable(/obj/item)\n", 0.0),
            ("/proc/test()\n\treturn isturf(new /turf)\n", 1.0),
            ("/proc/test()\n\treturn isturf(new /turf/open/floor)\n", 1.0),
            ("/proc/test()\n\treturn isturf(new /obj/item)\n", 0.0),
            ("/proc/test()\n\treturn isturf(/turf/open/floor)\n", 0.0),
            ("/proc/test()\n\treturn isloc(new /area)\n", 1.0),
            ("/proc/test()\n\treturn isloc(new /turf/open/floor)\n", 1.0),
            ("/proc/test()\n\treturn isloc(new /obj/item)\n", 1.0),
            ("/proc/test()\n\treturn isloc(new /mob/living)\n", 1.0),
            (
                "/proc/test()\n\treturn isloc(new /turf, new /obj, new /mob)\n",
                1.0,
            ),
            ("/proc/test()\n\treturn isloc(new /turf, 3)\n", 0.0),
            ("/proc/test()\n\treturn isloc(/turf)\n", 0.0),
            (
                "/proc/test()\n\treturn istype(/datum/example, /datum)\n",
                1.0,
            ),
            (
                "/proc/test()\n\treturn istype(new /datum/example, /datum)\n",
                1.0,
            ),
            ("/proc/test()\n\treturn istype(3, /datum)\n", 0.0),
        ] {
            let syntax = parse(source).expect("source should parse");
            let program = compile_procedure(&syntax.definitions[0])
                .expect("type predicate builtin should compile");
            assert_eq!(execute(&program, &[]), Ok(Value::number(expected)));
            assert!(
                program
                    .instructions
                    .iter()
                    .any(|instruction| matches!(instruction, Instruction::TypePredicate { .. }))
            );
        }
    }

    #[test]
    fn waitfor_directives_are_headless_scheduling_metadata() {
        for value in ["FALSE", "TRUE", "0", "1"] {
            let syntax = parse(&format!(
                "/proc/scheduled()\n\tset waitfor = {value}\n\treturn 17\n"
            ))
            .expect("source should parse");
            let program = compile_procedure(&syntax.definitions[0])
                .expect("waitfor directive should compile");

            assert_eq!(execute(&program, &[]), Ok(Value::number(17.0)));
        }
    }

    #[test]
    fn unsupported_set_directives_remain_diagnostic() {
        let syntax = parse("/proc/invalid()\n\tset hidden = TRUE\n").expect("source should parse");
        let error = compile_procedure(&syntax.definitions[0])
            .expect_err("only the supported scheduling directive should compile");

        assert!(error.message.contains("hidden"));
    }

    #[test]
    fn crash_statement_compiles_and_a_false_guard_skips_it() {
        let syntax =
            parse("/proc/guarded()\n\tif(FALSE)\n\t\tCRASH(\"should not execute\")\n\treturn 17\n")
                .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("CRASH should compile even when its branch is not taken");

        assert_eq!(execute(&program, &[]), Ok(Value::number(17.0)));
    }

    #[test]
    fn crash_statement_returns_a_source_mapped_runtime_error() {
        let syntax = parse("/proc/fail()\n\tif(TRUE)\n\t\tCRASH(\"loading id is required\")\n")
            .expect("source should parse");
        let crash_span = syntax.definitions[0].body[1].span;
        let program =
            compile_procedure(&syntax.definitions[0]).expect("CRASH statement should compile");
        let error = execute(&program, &[]).expect_err("taken CRASH must stop execution");

        assert_eq!(error.message, "CRASH: \"loading id is required\"");
        assert_eq!(error.source_span, Some(crash_span));
        assert_eq!(error.call_stack.len(), 1);
        assert_eq!(error.call_stack[0].procedure, "<standalone>");
        assert_eq!(error.call_stack[0].source_span, Some(crash_span));
    }

    #[test]
    fn headless_locate_consumes_arguments_and_returns_null() {
        let syntax =
            parse("/proc/find()\n\treturn locate(1, 2, 3)\n").expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("headless locate should compile without a user procedure");

        assert!(
            program.instructions.iter().any(|instruction| matches!(
                instruction,
                Instruction::Locate { argument_count: 3 }
            ))
        );
        assert_eq!(execute(&program, &[]), Ok(Value::Null));
    }

    #[test]
    fn regex_builtin_constructs_a_regex_datum_with_pattern_and_flags() {
        let syntax = parse("/proc/build()\n\treturn regex(@\"[a-z]+\", \"ig\")\n")
            .expect("regex source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("the built-in regex constructor should compile");
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::MakeRegex { argument_count: 2 }
        )));

        let mut state = ExecutionState::new();
        let result =
            execute_in_state(&program, &[], &mut state).expect("regex constructor should execute");
        let Value::Datum(regex) = result else {
            panic!("regex constructor should return a datum");
        };
        let datum = state.heap().datum(regex).expect("regex datum should exist");
        assert_eq!(datum.type_path().to_string(), "/regex");
        assert_eq!(datum.field(&field("text")), Ok(&Value::text("[a-z]+")));
        assert_eq!(datum.field(&field("flags")), Ok(&Value::text("ig")));
    }

    #[test]
    fn mutable_appearance_builtin_constructs_its_builtin_datum() {
        let syntax =
            parse("/proc/build()\n\treturn mutable_appearance('icons/test.dmi', \"state\")\n")
                .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("the mutable_appearance constructor should compile");
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::MakeMutableAppearance { argument_count: 2 }
        )));

        let mut state = ExecutionState::new();
        let result = execute_in_state(&program, &[], &mut state)
            .expect("mutable_appearance constructor should execute");
        let Value::Datum(appearance) = result else {
            panic!("mutable_appearance constructor should return a datum");
        };
        assert_eq!(
            state
                .heap()
                .datum(appearance)
                .expect("mutable appearance datum should exist")
                .type_path()
                .to_string(),
            "/mutable_appearance"
        );
    }

    #[test]
    fn optional_field_and_index_operators_parse_as_access_expressions() {
        let fields = parse("/proc/probe(value)\n\treturn value?.name\n")
            .expect("optional field source should parse");
        compile_procedure(&fields.definitions[0]).expect("optional field access should compile");

        let index = parse("/proc/probe(value)\n\treturn value?[1]\n")
            .expect("optional index source should parse");
        compile_procedure(&index.definitions[0]).expect("optional index access should compile");
    }

    #[test]
    fn headless_locate_in_container_is_not_list_membership_and_supports_nesting() {
        let syntax =
            parse("/proc/find()\n\treturn locate(locate(1, 2, 3) in null, 4, 5) in null\n")
                .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("headless locate in a container should compile");

        assert_eq!(
            program
                .instructions
                .iter()
                .filter(|instruction| matches!(instruction, Instruction::LocateIn { .. }))
                .count(),
            2
        );
        assert_eq!(execute(&program, &[]), Ok(Value::Null));
    }

    #[test]
    fn length_builtin_counts_text_bytes_and_list_entries() {
        let source = "/proc/measure()\n\tvar/text_length = length(\"aé\")\n\tvar/list_length = length(list(10, 20, \"key\" = 30))\n\treturn text_length * 10 + list_length\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("length builtin should compile for text and lists");

        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::Length))
        );
        // DM's regular text operations use legacy byte positions, so `é`
        // contributes two UTF-8 bytes. Associative list entries contribute
        // one entry, like positional list values.
        assert_eq!(execute(&program, &[]), Ok(Value::number(33.0)));
    }

    #[test]
    fn ref_builtin_returns_stable_byond_style_heap_identity_text() {
        let syntax = parse(
            "/proc/list_ref()\n\tvar/item = list()\n\treturn ref(item)\n\n/proc/datum_ref()\n\treturn ref(new /datum/example)\n\n/proc/scalar_ref()\n\treturn ref(42)\n",
        )
        .expect("ref source should parse");
        let list_program =
            compile_procedure(&syntax.definitions[0]).expect("list ref builtin should compile");
        let datum_program =
            compile_procedure(&syntax.definitions[1]).expect("datum ref builtin should compile");
        let scalar_program =
            compile_procedure(&syntax.definitions[2]).expect("scalar ref builtin should compile");

        assert!(
            list_program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::Ref))
        );
        let mut state = ExecutionState::new();
        assert_eq!(
            execute_in_state(&list_program, &[], &mut state),
            // The procedure's implicit `args` list occupies the first list
            // slot, so the explicit list receives the next BYOND list ref.
            Ok(Value::text("[0xe000002]"))
        );
        assert_eq!(
            execute_in_state(&datum_program, &[], &mut state),
            Ok(Value::text("[0xd000001]"))
        );
        assert_eq!(
            execute_in_state(&scalar_program, &[], &mut state),
            Ok(Value::Null)
        );
    }

    #[test]
    fn get_step_finds_cardinal_diagonal_and_same_coordinate_turfs() {
        let syntax =
            parse("/proc/step_from(source, direction)\n\treturn get_step(source, direction)\n")
                .expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("get_step builtin should compile");
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::GetStep))
        );

        let mut state = ExecutionState::new();
        let origin = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/open/origin").expect("type path"));
        let north_east = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/open/north_east").expect("type path"));
        let west = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/open/west").expect("type path"));
        for (datum, x, y) in [
            (origin, 4.0, 9.0),
            (north_east, 5.0, 10.0),
            (west, 3.0, 9.0),
        ] {
            for (name, value) in [("x", x), ("y", y), ("z", 2.0)] {
                state
                    .heap_mut()
                    .set_datum_field(datum, field(name), Value::number(value))
                    .expect("coordinate should be set");
            }
        }

        assert_eq!(
            execute_in_state(
                &program,
                &[Value::Datum(origin), Value::number(5.0)],
                &mut state
            ),
            Ok(Value::Datum(north_east))
        );
        assert_eq!(
            execute_in_state(
                &program,
                &[Value::Datum(origin), Value::number(8.0)],
                &mut state
            ),
            Ok(Value::Datum(west))
        );
        assert_eq!(
            execute_in_state(
                &program,
                &[Value::Datum(origin), Value::number(0.0)],
                &mut state
            ),
            Ok(Value::Datum(origin))
        );
        assert_eq!(
            execute_in_state(
                &program,
                &[Value::Datum(origin), Value::number(1.0)],
                &mut state
            ),
            Ok(Value::Null)
        );
    }

    fn compile_range_programs() -> (Program, Program, Program) {
        let syntax = parse(
            "/proc/normal(distance, center)\n\treturn range(distance, center)\n/proc/reversed(center, distance)\n\treturn range(center, distance)\n/proc/implicit(distance)\n\treturn range(distance)\n",
        )
        .expect("range source should parse");
        let normal = compile_procedure(&syntax.definitions[0]).expect("range should compile");
        let reversed =
            compile_procedure(&syntax.definitions[1]).expect("reversed range should compile");
        let implicit =
            compile_procedure(&syntax.definitions[2]).expect("implicit range should compile");
        (normal, reversed, implicit)
    }

    #[test]
    fn range_returns_all_same_z_atoms_in_a_square_and_accepts_reversed_arguments() {
        let (normal, reversed, implicit) = compile_range_programs();
        assert!(
            normal
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::Range { argument_count: 2 }))
        );
        assert!(
            implicit
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::Range { argument_count: 1 }))
        );

        let mut state = ExecutionState::new();
        let center = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/open/center").expect("type path"));
        let adjacent = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/obj/item/adjacent").expect("type path"));
        let diagonal = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/mob/living/diagonal").expect("type path"));
        let far = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/open/far").expect("type path"));
        let other_z = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/obj/item/other_z").expect("type path"));
        let area = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/area/test").expect("type path"));
        for (datum, x, y, z) in [
            (center, 10.0, 10.0, 1.0),
            (adjacent, 11.0, 10.0, 1.0),
            (diagonal, 9.0, 9.0, 1.0),
            (far, 12.0, 10.0, 1.0),
            (other_z, 10.0, 10.0, 2.0),
            (area, 10.0, 10.0, 1.0),
        ] {
            for (name, value) in [("x", x), ("y", y), ("z", z)] {
                state
                    .heap_mut()
                    .set_datum_field(datum, field(name), Value::number(value))
                    .expect("coordinate should be set");
            }
        }
        let values = |value: Value, state: &ExecutionState| {
            let Value::List(list) = value else {
                panic!("range should return a list");
            };
            state
                .heap()
                .list(list)
                .expect("range list should be live")
                .positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>()
        };
        let normal_values = values(
            execute_in_state(
                &normal,
                &[Value::number(1.0), Value::Datum(center)],
                &mut state,
            )
            .expect("normal range should execute"),
            &state,
        );
        assert_eq!(
            normal_values,
            vec![
                Value::Datum(center),
                Value::Datum(adjacent),
                Value::Datum(diagonal)
            ]
        );
        let reversed_values = values(
            execute_in_state(
                &reversed,
                &[Value::Datum(center), Value::number(1.0)],
                &mut state,
            )
            .expect("reversed range should execute"),
            &state,
        );
        assert_eq!(reversed_values, normal_values);
        let context = ExecutionContext::new(Value::Datum(center), Value::Null);
        let implicit_values = values(
            execute_in_context(&implicit, &[Value::number(0.0)], &mut state, &context)
                .expect("implicit range should execute"),
            &state,
        );
        assert_eq!(implicit_values, vec![Value::Datum(center)]);
    }

    #[test]
    fn typesof_builtin_includes_the_selector_and_registered_descendants() {
        let syntax = parse("/proc/types()\n\treturn typesof(/datum)\n")
            .expect("typesof source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("typesof builtin should compile");
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::TypesOf))
        );

        let mut state = ExecutionState::new();
        state.set_type_paths([
            TypePath::parse("/obj").expect("type path"),
            TypePath::parse("/datum/child").expect("type path"),
            TypePath::parse("/datum").expect("type path"),
            TypePath::parse("/datum/child/grandchild").expect("type path"),
        ]);
        let result = execute_in_state(&program, &[], &mut state)
            .expect("typesof should execute against the catalog");
        let Value::List(list) = result else {
            panic!("typesof should return a list");
        };
        let values = state
            .heap()
            .list(list)
            .expect("typesof result list should be live")
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            vec![
                Value::TypePath(TypePath::parse("/datum").expect("type path")),
                Value::TypePath(TypePath::parse("/datum/child").expect("type path")),
                Value::TypePath(TypePath::parse("/datum/child/grandchild").expect("type path")),
            ]
        );
    }

    #[test]
    fn typesof_can_use_a_shared_immutable_catalog() {
        let catalog = Arc::new(
            [
                TypePath::parse("/datum").expect("type path"),
                TypePath::parse("/datum/child").expect("type path"),
            ]
            .into_iter()
            .collect(),
        );
        let mut state = ExecutionState::new();
        state.set_shared_type_paths(Arc::clone(&catalog));

        assert_eq!(Arc::strong_count(&catalog), 2);
        assert_eq!(
            state.type_paths().cloned().collect::<Vec<_>>(),
            vec![
                TypePath::parse("/datum").expect("type path"),
                TypePath::parse("/datum/child").expect("type path"),
            ]
        );
    }

    #[test]
    fn special_result_can_receive_resolved_parent_call() {
        let source = "/proc/base(value = 4)\n\t. = value\n/proc/child(value = 4)\n\t. = ..()\n";
        let syntax = parse(source).expect("source should parse");
        let parent_assignment_span = syntax.definitions[1].body[0].span;
        let module = compile_module_specs(&[
            ProcedureSpec {
                path: "/proc/base@0".to_owned(),
                definition: &syntax.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/proc/child@1".to_owned(),
                definition: &syntax.definitions[1],
                parent: Some(0),
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ])
        .expect("resolved parent specs should compile");
        let entry = module
            .procedure_id_at(1)
            .expect("child spec should have a VM identity");

        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(4.0)));
        let child = module.procedure(entry).expect("child program should exist");
        assert!(child.instructions.iter().zip(&child.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::StoreResult)
                && *span == parent_assignment_span
        ));
    }
}

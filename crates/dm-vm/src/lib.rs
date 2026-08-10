//! Portable stack bytecode and the deterministic reference interpreter.

#![cfg_attr(not(test), deny(missing_docs))]

mod builtins;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use builtins::{
    execute_external_call, execute_list_binary_operator, execute_list_compound_operator,
    execute_list_method, execute_output, execute_regex_method, execute_standard_builtin,
    is_regex_datum, is_subtype, standard_builtin_arity,
};

use dm_core::{DmNumberBits, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind, lex};
use dm_syntax::{Definition, DefinitionKind, SourceLine};
pub use dm_value::Value;
use dm_value::{DatumId, FieldName, ListId, ModifiedTypePath, TypePath, ValueError, ValueHeap};

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
    /// Pushes a local's pointer cell, creating one on first address-of use.
    AddressLocal(u16),
    /// Loads a local without transparently dereferencing a pointer cell.
    LoadLocalRaw(u16),
    /// Builds a first-class modified type path from evaluated override values.
    MakeModifiedTypePath {
        /// Override field names in source order.
        fields: Arc<[FieldName]>,
    },
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
    /// Constructs BYOND's built-in affine `/matrix` datum.
    MakeMatrix {
        /// Number of constructor arguments.
        argument_count: u8,
    },
    /// Constructs BYOND's built-in three-component `/vector` datum.
    MakeVector {
        /// Number of constructor arguments (zero through three).
        argument_count: u8,
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
    /// Calls an engine-native method on the current `src` datum.
    NativeSrcMethod {
        /// BYOND method selector.
        name: String,
        /// Positional argument count.
        argument_count: u16,
    },
    /// Sends a value to a file/log/output destination.
    Output,
    /// Reads the current value from a savefile or keyed savefile entry.
    Input,
    /// Invokes a function exported by an external BYOND-compatible library.
    /// Headless execution reports that no external-call host is installed.
    ExternalCall {
        /// Number of call arguments following the library and function selectors.
        argument_count: u16,
    },
    /// Applies one headless `animate()` step. Named appearance variables are
    /// retained because their names are part of the procedure's semantics.
    Animate {
        /// Name of each argument, or `None` for a positional argument.
        argument_names: Vec<Option<String>>,
    },
    /// Constructs a `/dm_filter` while retaining keyword property names.
    MakeFilter {
        /// Name of each argument, or `None` for a positional argument.
        argument_names: Vec<Option<String>>,
        /// Source argument positions supplied through `arglist()`.
        expanded_indices: Vec<u16>,
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
    /// Rolls one or more dice using BYOND's numeric or `NdS+offset` forms.
    Roll {
        /// Number of supplied arguments (one or two).
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
    /// Returns the adjacent turf in the direction of a target atom.
    GetStepTowards,
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
    /// Returns a snapshot list of live datums matching a type and descendants.
    TypeInstances(TypePath),
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
    /// Allocates a suffix-declared DM array from evaluated dimension sizes.
    MakeArray(u8),
    /// Allocates the current procedure's implicit `args` list.
    ///
    /// The list contains every value supplied to this call in positional
    /// order, including values beyond the declared parameter list.
    MakeArgs,
    /// Builds a list whose positional values and associative keys may intermix.
    MakeListEntries(Vec<ListEntryKind>),
    /// Builds a BYOND `/alist`, whose constructor entries are key/value pairs.
    MakeAssociativeListEntries(Vec<ListEntryKind>),
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
    /// Reuses a previously initialized procedure-static local and skips its
    /// declaration initializer, or falls through on first execution.
    LoadStaticLocalOrJump {
        /// Local slot receiving the persistent value.
        slot: u16,
        /// Instruction immediately after the declaration.
        target: usize,
    },
    /// Persists the just-evaluated initial value of a procedure-static local.
    InitializeStaticLocal(u16),
    /// Pushes the current frame's `src` value.
    LoadSrc,
    /// Pops and replaces the current frame's `src` value.
    StoreSrc,
    /// Pushes the current frame's `usr` value.
    LoadUsr,
    /// Pops and replaces the current frame's `usr` value.
    StoreUsr,
    /// Pushes a `/callee` snapshot for the calling frame, or null at the root.
    LoadCaller,
    /// Pops a datum receiver and pushes one named field.
    LoadField(FieldName),
    /// Pops a value and datum receiver, then writes one named field.
    StoreField(FieldName),
    /// Stores one datum field while preserving the assigned value on the stack.
    StoreFieldKeep(FieldName),
    /// Pushes one persistent runtime global.
    LoadGlobal(FieldName),
    /// Pushes BYOND's live associative `global.vars` namespace.
    LoadGlobalVars,
    /// Pushes the live associative `vars` reflection list for a datum.
    LoadDatumVars,
    /// Pushes the declaration-time value of a persistent global/static slot.
    LoadInitialGlobal(FieldName),
    /// Pops and stores one persistent runtime global.
    StoreGlobal(FieldName),
    /// Mutates a local by one and pushes either its old or new value.
    MutateLocal {
        /// Local slot to update.
        slot: u16,
        /// `1` for increment and `-1` for decrement.
        delta: i8,
        /// Whether the expression yields the updated value.
        prefix: bool,
    },
    /// Pops a receiver, mutates one field by one, and pushes the expression value.
    MutateField {
        /// Field to update.
        name: FieldName,
        /// `1` for increment and `-1` for decrement.
        delta: i8,
        /// Whether the expression yields the updated value.
        prefix: bool,
    },
    /// Mutates a persistent global by one and pushes the expression value.
    MutateGlobal {
        /// Global field to update.
        name: FieldName,
        /// `1` for increment and `-1` for decrement.
        delta: i8,
        /// Whether the expression yields the updated value.
        prefix: bool,
    },
    /// Pops a list key and list, mutates the entry, and pushes the expression value.
    MutateListIndex {
        /// `1` for increment and `-1` for decrement.
        delta: i8,
        /// Whether the expression yields the updated value.
        prefix: bool,
    },
    /// Mutates the current procedure's special result by one.
    MutateResult {
        /// `1` for increment and `-1` for decrement.
        delta: i8,
        /// Whether the expression yields the updated value.
        prefix: bool,
    },
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
    /// Installs an exception handler for the current frame.
    BeginTry {
        /// First instruction of the corresponding catch body.
        catch: usize,
        /// First instruction after the protected try body.
        end: usize,
        /// Optional local receiving the thrown DM value.
        local: Option<u16>,
    },
    /// Removes the innermost exception handler after normal completion.
    EndTry,
    /// Pops and throws an arbitrary DM value to the nearest active handler.
    Throw,
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
    /// Legacy integer remainder (`%`), after truncating both operands.
    Remainder,
    /// Fractional remainder (`%%`) without integer truncation.
    FractionalRemainder,
    /// 24-bit integer bitwise conjunction.
    BitAnd,
    /// 24-bit integer bitwise disjunction.
    BitOr,
    /// 24-bit integer bitwise exclusive disjunction.
    BitXor,
    /// 24-bit integer left shift.
    ShiftLeft,
    /// 24-bit logical right shift.
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
    /// Shallow BYOND equivalence comparison (`~=`).
    Equivalent,
    /// Negated shallow equivalence comparison (`~!`).
    NotEquivalent,
    /// Three-way comparison (`<=>`), yielding -1, 0, or 1.
    Compare,
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
        /// Whether a null receiver denotes the global `/proc` namespace.
        /// This is true only for the one-selector `call(proc)(...)` form;
        /// ordinary `datum.proc(...)` calls must diagnose a null datum.
        null_receiver_is_global: bool,
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
    /// Defers execution from an instruction in a cloned caller frame until a
    /// future scheduler tick. The delay value is consumed from the stack.
    Spawn {
        /// First instruction of the detached spawned body.
        entry: usize,
    },
    /// Suspends the complete current call chain for a scheduler delay.
    Sleep,
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
    /// Legacy integer remainder assignment (`%=`).
    Remainder,
    /// Fractional remainder assignment (`%%=`).
    FractionalRemainder,
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
    /// Legacy integer remainder assignment (`%=`).
    Remainder,
    /// Fractional remainder assignment (`%%=`).
    FractionalRemainder,
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
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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
    /// Whether callers wait when this procedure's call chain sleeps.
    /// `set waitfor = FALSE` clears this flag.
    pub wait_for: bool,
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
    deferred: Arc<HashMap<ProcedureId, DeferredProcedure>>,
    procedure_types: BTreeSet<TypePath>,
    initializer_call_names: Option<InitializerCallNameIndex>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InitializerCallNameIndex {
    names: HashMap<String, ProcedureId>,
    module_names_scanned: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DeferredProcedure {
    definition: Arc<Definition>,
    targets: Arc<HashMap<String, ProcedureId>>,
    src_fields: Arc<BTreeMap<String, FieldName>>,
    global_fields: Arc<BTreeMap<String, FieldName>>,
    global_types: Arc<BTreeMap<String, TypePath>>,
    preflight_error: Option<CompileError>,
    compiled: Arc<OnceLock<Result<Program, CompileError>>>,
}

/// An initializer expression linked as an entry point in a VM module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializerProgram {
    module: Module,
    entry: ProcedureId,
}

/// One per-instance initializer executed before `New()` on every allocation.
#[derive(Clone)]
pub struct InstanceInitializer {
    /// Destination datum field.
    pub field: FieldName,
    /// Entry in the shared linked initializer module.
    pub entry: ProcedureId,
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
    fn resolve_procedure(&self, procedure: ProcedureId) -> Result<&Program, String> {
        if let Some(deferred) = self.deferred.get(&procedure) {
            return deferred
                .compiled
                .get_or_init(|| {
                    if let Some(error) = &deferred.preflight_error {
                        return Err(error.clone());
                    }
                    compile_procedure_with_resolver_and_fields(
                        deferred.definition.as_ref(),
                        deferred.targets.as_ref(),
                        deferred.src_fields.as_ref(),
                        deferred.global_fields.as_ref(),
                        deferred.global_types.as_ref(),
                    )
                })
                .as_ref()
                .map_err(|error| error.message.clone());
        }
        self.procedures
            .get(procedure.index())
            .ok_or_else(|| format!("invalid procedure {}", procedure.index()))
    }

    /// Looks up a procedure by canonical path, such as `/proc/main`.
    #[must_use]
    pub fn procedure_id(&self, path: &str) -> Option<ProcedureId> {
        self.names.get(path).copied()
    }

    /// Returns a compiled procedure by module-local identity.
    #[must_use]
    pub fn procedure(&self, procedure: ProcedureId) -> Option<&Program> {
        self.resolve_procedure(procedure).ok()
    }

    /// Number of symbolically linked procedure bodies not compiled eagerly.
    #[must_use]
    pub fn deferred_procedure_count(&self) -> usize {
        self.deferred.len()
    }

    /// Number of deferred bodies materialized by execution or inspection.
    #[must_use]
    pub fn materialized_deferred_procedure_count(&self) -> usize {
        self.deferred
            .values()
            .filter(|procedure| procedure.compiled.get().is_some())
            .count()
    }

    /// Number of times the module name table was scanned to construct the
    /// reusable bare-call resolver used by appended initializer expressions.
    #[must_use]
    pub fn initializer_call_name_index_builds(&self) -> usize {
        usize::from(self.initializer_call_names.is_some())
    }

    /// Number of module-name entries inspected while constructing that index.
    #[must_use]
    pub fn initializer_call_name_symbols_scanned(&self) -> usize {
        self.initializer_call_names
            .as_ref()
            .map_or(0, |index| index.module_names_scanned)
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
        if !self.call_stack.is_empty() {
            formatter.write_str("\ncall stack:")?;
            for trace in self.call_stack.iter().rev() {
                write!(
                    formatter,
                    "\n  {} at instruction {}",
                    trace.procedure, trace.instruction
                )?;
                if let Some(span) = trace.source_span {
                    write!(formatter, " (source {}..{})", span.start, span.end)?;
                }
            }
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
    let mut module = procedures.cloned().unwrap_or_else(|| Module {
        procedures: Vec::new(),
        paths: Vec::new(),
        names: HashMap::new(),
        deferred: Arc::new(HashMap::new()),
        procedure_types: BTreeSet::new(),
        initializer_call_names: None,
    });
    let entry = compile_initializer_into_module(tokens, bindings, &mut module)?;
    Ok(InitializerProgram { module, entry })
}

/// Appends one initializer entry point to an existing linked module without
/// cloning or recompiling its project procedures.
pub fn compile_initializer_into_module(
    tokens: &[SpannedToken],
    bindings: &BTreeMap<String, InitializerBinding>,
    module: &mut Module,
) -> Result<ProcedureId, CompileError> {
    let mut expression = ExpressionParser::new(tokens).parse()?;
    bind_initializer_expression(&mut expression, bindings)?;
    let call_names = &module
        .initializer_call_names
        .get_or_insert_with(|| {
            let mut names = HashMap::new();
            for (path, procedure) in &module.names {
                if let Some(name) = path.strip_prefix("/proc/")
                    && !name.contains('/')
                {
                    names.insert(
                        name.split('@').next().unwrap_or(name).to_owned(),
                        *procedure,
                    );
                }
            }
            InitializerCallNameIndex {
                names,
                module_names_scanned: module.names.len(),
            }
        })
        .names;
    let mut instructions = Vec::new();
    emit_expression(
        &expression,
        &LocalTable::default(),
        &mut instructions,
        call_names,
    )?;
    instructions.push(Instruction::Return);
    let source_span = match (tokens.first(), tokens.last()) {
        (Some(first), Some(last)) => SourceSpan::new(first.span.start, last.span.end),
        _ => return Err(compile_error("expected an initializer expression")),
    };
    let program = Program {
        wait_for: true,
        parameter_count: 0,
        local_count: 0,
        source_spans: vec![source_span; instructions.len()],
        instructions,
    };
    let entry = ProcedureId::from_index(module.procedures.len())?;
    module.procedures.push(program);
    module.paths.push("<initializer>".to_owned());
    Ok(entry)
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
    compile_module_with_global_fields(definitions, &BTreeMap::new())
}

/// Compiles a module with an explicit registry of bare global variable names.
/// This is useful for syntax-only consumers that retain procedure definitions
/// separately from the declaration tree while preserving strict name checks.
pub fn compile_module_with_global_fields(
    definitions: &[Definition],
    global_fields: &BTreeMap<String, FieldName>,
) -> Result<Module, CompileError> {
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
        .map(|definition| {
            compile_procedure_with_resolver_and_fields(
                definition,
                &call_names,
                &BTreeMap::new(),
                global_fields,
                &BTreeMap::new(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Module {
        procedures,
        paths,
        names,
        deferred: Arc::new(HashMap::new()),
        procedure_types: definitions
            .iter()
            .filter(|definition| definition.kind == DefinitionKind::Procedure)
            .filter_map(|definition| TypePath::parse(&definition.path.to_string()).ok())
            .collect(),
        initializer_call_names: None,
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
    let global_types = vec![BTreeMap::new(); specs.len()];
    compile_module_specs_with_global_types(specs, &global_types)
}

/// Compiles procedure specs with declared global types used to infer bare
/// `new` expressions from their assignment destinations.
pub fn compile_module_specs_with_global_types(
    specs: &[ProcedureSpec<'_>],
    global_types: &[BTreeMap<String, TypePath>],
) -> Result<Module, CompileError> {
    if specs.len() != global_types.len() {
        return Err(compile_error(
            "procedure spec/global type table length mismatch",
        ));
    }
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
    let call_index = SpecCallIndex::build(&paths)?;

    let procedures = specs
        .iter()
        .zip(global_types)
        .map(|(spec, global_types)| {
            let mut targets = HashMap::new();
            if let Some(parent) = spec.parent {
                targets.insert("..".to_owned(), ProcedureId::from_index(parent)?);
            }
            targets.extend(static_call_targets(
                &spec.path,
                &call_index,
                referenced_static_call_names(spec.definition),
            ));
            for (selector, target) in &spec.static_calls {
                targets.insert(selector.clone(), ProcedureId::from_index(*target)?);
            }
            compile_procedure_with_resolver_and_fields(
                spec.definition,
                &targets,
                &spec.src_fields,
                &spec.global_fields,
                global_types,
            )
            .map_err(|error| compile_error(format!("{}: {}", spec.path, error.message)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Module {
        procedures,
        paths,
        names,
        deferred: Arc::new(HashMap::new()),
        procedure_types: specs
            .iter()
            .filter(|spec| spec.definition.kind == DefinitionKind::Procedure)
            .filter_map(|spec| {
                TypePath::parse(
                    spec.path
                        .split_once('@')
                        .map_or(&spec.path, |(base, _)| base),
                )
                .ok()
            })
            .collect(),
        initializer_call_names: None,
    })
}

/// Symbolically links every procedure spec while compiling only the requested
/// eager indices. Deferred bodies retain stable module-local identities and
/// are lowered exactly once when execution first dispatches to them.
///
/// This is intended for genuinely dynamic DM calls whose runtime receiver
/// cannot be proven statically. Linking all candidate symbols preserves
/// virtual dispatch without making cold boot compile every same-name body.
///
/// # Errors
///
/// Returns [`CompileError`] for an invalid spec table or an eager body that
/// cannot be lowered. A deferred-body lowering failure is reported when that
/// body is first selected.
pub fn compile_module_specs_selective(
    specs: &[ProcedureSpec<'_>],
    global_types: &[BTreeMap<String, TypePath>],
    eager_indices: &BTreeSet<usize>,
) -> Result<Module, CompileError> {
    compile_module_specs_selective_with_errors(specs, global_types, eager_indices, &BTreeMap::new())
}

/// Selective symbolic linking with source-aware semantic failures retained on
/// deferred symbols and raised only if runtime dispatch materializes them.
pub fn compile_module_specs_selective_with_errors(
    specs: &[ProcedureSpec<'_>],
    global_types: &[BTreeMap<String, TypePath>],
    eager_indices: &BTreeSet<usize>,
    deferred_errors: &BTreeMap<usize, CompileError>,
) -> Result<Module, CompileError> {
    if specs.len() != global_types.len() {
        return Err(compile_error(
            "procedure spec/global type table length mismatch",
        ));
    }
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
    let call_index = SpecCallIndex::build(&paths)?;
    let mut procedures = Vec::with_capacity(specs.len());
    let mut deferred = HashMap::new();
    for (index, (spec, global_types)) in specs.iter().zip(global_types).enumerate() {
        let targets = resolved_spec_targets(spec, &call_index)?;
        if eager_indices.contains(&index) {
            procedures.push(
                compile_procedure_with_resolver_and_fields(
                    spec.definition,
                    &targets,
                    &spec.src_fields,
                    &spec.global_fields,
                    global_types,
                )
                .map_err(|error| compile_error(format!("{}: {}", spec.path, error.message)))?,
            );
        } else {
            let procedure = ProcedureId::from_index(index)?;
            procedures.push(Program {
                wait_for: true,
                parameter_count: 0,
                local_count: 0,
                instructions: Vec::new(),
                source_spans: Vec::new(),
            });
            deferred.insert(
                procedure,
                DeferredProcedure {
                    definition: Arc::new(spec.definition.clone()),
                    targets: Arc::new(targets),
                    src_fields: Arc::new(spec.src_fields.clone()),
                    global_fields: Arc::new(spec.global_fields.clone()),
                    global_types: Arc::new(global_types.clone()),
                    preflight_error: deferred_errors.get(&index).cloned(),
                    compiled: Arc::new(OnceLock::new()),
                },
            );
        }
    }
    Ok(Module {
        procedures,
        paths,
        names,
        deferred: Arc::new(deferred),
        procedure_types: specs
            .iter()
            .filter(|spec| spec.definition.kind == DefinitionKind::Procedure)
            .filter_map(|spec| {
                TypePath::parse(
                    spec.path
                        .split_once('@')
                        .map_or(&spec.path, |(base, _)| base),
                )
                .ok()
            })
            .collect(),
        initializer_call_names: None,
    })
}

fn resolved_spec_targets(
    spec: &ProcedureSpec<'_>,
    call_index: &SpecCallIndex,
) -> Result<HashMap<String, ProcedureId>, CompileError> {
    let mut targets = HashMap::new();
    if let Some(parent) = spec.parent {
        targets.insert("..".to_owned(), ProcedureId::from_index(parent)?);
    }
    targets.extend(static_call_targets(
        &spec.path,
        call_index,
        referenced_static_call_names(spec.definition),
    ));
    for (selector, target) in &spec.static_calls {
        targets.insert(selector.clone(), ProcedureId::from_index(*target)?);
    }
    Ok(targets)
}

struct SpecCallIndex {
    latest_by_base_path: HashMap<String, ProcedureId>,
}

impl SpecCallIndex {
    fn build(paths: &[String]) -> Result<Self, CompileError> {
        let mut latest_by_base_path = HashMap::new();
        for (position, path) in paths.iter().enumerate() {
            let Some((_, _)) = path.rsplit_once("/proc/") else {
                continue;
            };
            let base_path = path.split_once('@').map_or(path.as_str(), |(base, _)| base);
            // Match the old reverse scan: the last spec for a base path wins.
            latest_by_base_path.insert(base_path.to_owned(), ProcedureId::from_index(position)?);
        }
        Ok(Self {
            latest_by_base_path,
        })
    }
}

fn static_call_targets(
    path: &str,
    index: &SpecCallIndex,
    selectors: impl IntoIterator<Item = String>,
) -> HashMap<String, ProcedureId> {
    let Some((owner, _)) = path.rsplit_once("/proc/") else {
        return HashMap::new();
    };
    let mut targets = HashMap::new();
    for name in selectors {
        let mut current_owner = owner;
        loop {
            let expected = if current_owner.is_empty() {
                format!("/proc/{name}")
            } else {
                format!("{current_owner}/proc/{name}")
            };
            if let Some(procedure) = index.latest_by_base_path.get(&expected) {
                targets.insert(name.clone(), *procedure);
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

fn referenced_static_call_names(definition: &Definition) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    let token_groups = std::iter::once(&definition.header)
        .chain(
            definition
                .parameters
                .iter()
                .map(|parameter| &parameter.tokens),
        )
        .chain(definition.body.iter().map(|line| &line.tokens));
    for tokens in token_groups {
        for pair in tokens.windows(2) {
            if let [
                SpannedToken {
                    kind: TokenKind::Identifier(name),
                    ..
                },
                SpannedToken {
                    kind: TokenKind::Punctuation('('),
                    ..
                },
            ] = pair
                && !matches!(
                    name.as_str(),
                    "if" | "for" | "while" | "switch" | "catch" | "spawn" | "new"
                )
            {
                names.insert(name.clone());
            }
        }
    }
    names
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
        &BTreeMap::new(),
    )
}

fn compile_procedure_with_resolver_and_fields(
    definition: &Definition,
    procedures: &HashMap<String, ProcedureId>,
    src_fields: &BTreeMap<String, FieldName>,
    global_fields: &BTreeMap<String, FieldName>,
    global_types: &BTreeMap<String, TypePath>,
) -> Result<Program, CompileError> {
    if !matches!(
        definition.kind,
        DefinitionKind::Procedure | DefinitionKind::ProcedureOverride | DefinitionKind::Verb
    ) {
        return Err(compile_error("definition is not executable"));
    }

    let mut locals = LocalTable::with_fields(src_fields, global_fields, global_types);
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
    let mut body = normalize_labeled_loops(split_top_level_semicolon_statements(&definition.body));
    if definition.kind == DefinitionKind::Verb {
        // `set hidden/category/name/...` lines on verbs are declaration
        // metadata, not executable assignments.
        body.retain(|line| {
            !matches!(line.tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(keyword)) if keyword == "set")
        });
    }
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
        wait_for: procedure_wait_for(definition),
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
    fn ends_with_type_path(tokens: &[SpannedToken]) -> bool {
        let mut index = tokens.len();
        let mut segments = 0usize;
        while index >= 2 {
            if !matches!(tokens[index - 1].kind, TokenKind::Identifier(_))
                || !matches!(&tokens[index - 2].kind, TokenKind::Operator(operator) if operator == "/")
            {
                break;
            }
            segments += 1;
            index -= 2;
        }
        segments > 0
    }

    let mut result = Vec::with_capacity(lines.len());
    // A preprocessor expansion may contain a backslash-continued compact
    // brace body. The syntax layer preserves those continuations as separate
    // physical SourceLines, so brace nesting must survive the line boundary.
    // Resetting it per line incorrectly made declarations from the opening
    // line unavailable to an `else` branch emitted on the following line.
    let mut brace_depth = 0usize;
    for line in lines {
        let mut statement = Vec::new();
        let mut grouping_depth = 0usize;
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
                TokenKind::Punctuation('{') if grouping_depth > 0 => {
                    grouping_depth += 1;
                    statement.push(token.clone());
                }
                TokenKind::Punctuation('{')
                    if grouping_depth == 0
                        && statement.iter().any(
                            |token| matches!(&token.kind, TokenKind::Identifier(name) if name == "new"),
                        )
                        || grouping_depth == 0 && ends_with_type_path(&statement) =>
                {
                    grouping_depth += 1;
                    statement.push(token.clone());
                }
                TokenKind::Punctuation('}') if grouping_depth > 0 => {
                    grouping_depth -= 1;
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

fn normalize_labeled_loops(mut lines: Vec<SourceLine>) -> Vec<SourceLine> {
    let mut index = 0;
    while index + 1 < lines.len() {
        let label = match lines[index].tokens.as_slice() {
            [
                SpannedToken {
                    kind: TokenKind::Identifier(label),
                    ..
                },
                SpannedToken {
                    kind: TokenKind::Operator(colon),
                    ..
                },
            ] if colon == ":" => label.clone(),
            _ => {
                index += 1;
                continue;
            }
        };
        let base = indentation(&lines[index]);
        let is_loop = matches!(
            lines[index + 1].tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if matches!(keyword.as_str(), "for" | "while" | "do")
        );
        if !is_loop || indentation(&lines[index + 1]) < base {
            index += 1;
            continue;
        }
        let indentation_delta = indentation(&lines[index + 1]) - base;
        lines.remove(index);
        lines[index].indentation.tabs = 0;
        lines[index].indentation.spaces = base;
        let mut active_loop_indents = vec![base];
        let mut cursor = index + 1;
        while cursor < lines.len() && indentation(&lines[cursor]) > base {
            lines[cursor].indentation.tabs = 0;
            lines[cursor].indentation.spaces =
                indentation(&lines[cursor]).saturating_sub(indentation_delta);
            let current_indent = indentation(&lines[cursor]);
            while active_loop_indents
                .last()
                .is_some_and(|indent| *indent >= current_indent)
            {
                active_loop_indents.pop();
            }
            if matches!(
                lines[cursor].tokens.first().map(|token| &token.kind),
                Some(TokenKind::Identifier(keyword)) if matches!(keyword.as_str(), "for" | "while" | "do")
            ) {
                active_loop_indents.push(current_indent);
            }
            if matches!(
                lines[cursor].tokens.as_slice(),
                [SpannedToken { kind: TokenKind::Identifier(keyword), .. }, SpannedToken { kind: TokenKind::Identifier(target), .. }]
                    if keyword == "break" && target == &label
            ) {
                lines[cursor].tokens[1].kind =
                    TokenKind::Number(active_loop_indents.len().to_string());
            }
            cursor += 1;
        }
        index = cursor;
    }
    lines
}

struct LocalTable<'fields> {
    names: HashMap<String, u16>,
    src_fields: &'fields BTreeMap<String, FieldName>,
    global_fields: &'fields BTreeMap<String, FieldName>,
    global_types: &'fields BTreeMap<String, TypePath>,
    slot_count: usize,
}

impl Default for LocalTable<'static> {
    fn default() -> Self {
        static EMPTY: std::sync::LazyLock<BTreeMap<String, FieldName>> =
            std::sync::LazyLock::new(BTreeMap::new);
        static EMPTY_TYPES: std::sync::LazyLock<BTreeMap<String, TypePath>> =
            std::sync::LazyLock::new(BTreeMap::new);
        Self::with_fields(&EMPTY, &EMPTY, &EMPTY_TYPES)
    }
}

impl<'fields> LocalTable<'fields> {
    fn with_fields(
        src_fields: &'fields BTreeMap<String, FieldName>,
        global_fields: &'fields BTreeMap<String, FieldName>,
        global_types: &'fields BTreeMap<String, TypePath>,
    ) -> Self {
        Self {
            names: HashMap::new(),
            src_fields,
            global_fields,
            global_types,
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

    fn global_type(&self, name: &str) -> Option<&TypePath> {
        self.global_types.get(name)
    }

    fn receiver_static(&self, receiver: &Expression, name: &FieldName) -> Option<&FieldName> {
        let receiver = match receiver {
            Expression::Src => "src",
            Expression::Local(receiver) => receiver.as_str(),
            Expression::GlobalField(receiver) => receiver.as_str(),
            _ => return None,
        };
        self.global_fields
            .get(&format!("{receiver}.{}", name.as_str()))
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
            TokenKind::Identifier(keyword) if keyword == "try" => {
                let (next_line, statement_falls_through) = compile_try(
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
            TokenKind::Identifier(keyword) if keyword == "catch" => {
                return Err(compile_error("catch without a matching try"));
            }
            TokenKind::Identifier(keyword) if keyword == "else" => {
                return Err(compile_error("else without a matching if"));
            }
            TokenKind::Identifier(keyword) if keyword == "break" => {
                let depth = match line.tokens.as_slice() {
                    [_] => 1,
                    [_, SpannedToken { kind: TokenKind::Number(depth), .. }] => depth
                        .parse::<usize>()
                        .map_err(|_| compile_error("invalid labeled break depth"))?,
                    _ => {
                        return Err(compile_error("break does not accept an expression"));
                    }
                };
                if loops.is_empty() {
                    return Err(compile_error("break outside a loop"));
                }
                if depth == 0 || depth > loops.len() {
                    return Err(compile_error("break does not accept an expression"));
                }
                let target_loop = loops.len() - depth;
                let Some(loop_context) = loops.get_mut(target_loop) else {
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
            TokenKind::Identifier(keyword) if keyword == "throw" => {
                if line.tokens.len() == 1 {
                    return Err(compile_error("throw requires an expression"));
                }
                let first_instruction = instructions.len();
                compile_expression(&line.tokens[1..], locals, instructions, procedures)?;
                instructions.push(Instruction::Throw);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
                falls_through = false;
            }
            // `waitfor` is procedure metadata captured on `Program`; it has no
            // executable assignment at the declaration site.
            TokenKind::Identifier(keyword)
                if keyword == "set" && is_waitfor_directive(&line.tokens) => {}
            TokenKind::Identifier(keyword)
                if keyword == "set"
                    && matches!(line.tokens.get(1).map(|token| &token.kind), Some(TokenKind::Identifier(_)))
                    && matches!(line.tokens.get(2).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "=") =>
            {
                // Verb/procedure `set` directives (`name`, `category`, `desc`,
                // `hidden`, and friends) are declaration metadata and do not
                // execute when the procedure is called.
            }
            TokenKind::Identifier(keyword) if keyword == "var" => {
                let first_instruction = instructions.len();
                compile_local_declarations(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(_) | TokenKind::Operator(_)
                if top_level_assignment(&line.tokens).is_some() =>
            {
                let first_instruction = instructions.len();
                compile_assignment_statement(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(_) if top_level_output(&line.tokens).is_some() => {
                let first_instruction = instructions.len();
                let output = top_level_output(&line.tokens).expect("output index was checked");
                compile_expression(&line.tokens[..output], locals, instructions, procedures)?;
                compile_expression(&line.tokens[output + 1..], locals, instructions, procedures)?;
                instructions.push(Instruction::Output);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(_) if top_level_input(&line.tokens).is_some() => {
                let first_instruction = instructions.len();
                let input = top_level_input(&line.tokens).expect("input index was checked");
                compile_expression(&line.tokens[..input], locals, instructions, procedures)?;
                instructions.push(Instruction::Input);
                let target = ExpressionParser::new(&line.tokens[input + 1..]).parse()?;
                match target {
                    Expression::Local(name) => {
                        if let Some(slot) = locals.get(&name) {
                            instructions.push(Instruction::StoreLocal(slot));
                        } else {
                            return Err(compile_error(format!(
                                "savefile input target {name:?} is not writable"
                            )));
                        }
                    }
                    _ => return Err(compile_error("savefile input target is not writable")),
                }
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
            TokenKind::Operator(operator) if matches!(operator.as_str(), "++" | "--") => {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Operator(operator) if operator == "." => {
                let first_instruction = instructions.len();
                if top_level_assignment(&line.tokens).is_some_and(|(index, _)| index == 1) {
                    compile_result_assignment(&line.tokens, locals, instructions, procedures)?;
                } else if top_level_assignment(&line.tokens).is_some() {
                    // The special result is also a regular expression value,
                    // so indexed writes such as `.[key] = value` use the same
                    // list-assignment lowering as any other expression.
                    compile_assignment_statement(
                        &line.tokens,
                        locals,
                        instructions,
                        procedures,
                    )?;
                } else {
                    compile_expression(&line.tokens, locals, instructions, procedures)?;
                    instructions.push(Instruction::Pop);
                }
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
            TokenKind::Identifier(keyword) if keyword == "spawn" => {
                let first_instruction = instructions.len();
                let after_keyword = &line.tokens[1..];
                let rest = if matches!(
                    after_keyword.first().map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    let mut spawn = ExpressionParser::new(after_keyword);
                    let arguments = spawn.parse_call_arguments()?;
                    if arguments.len() > 1 {
                        return Err(compile_error(
                            "spawn accepts at most one delay argument before the spawned expression",
                        ));
                    }
                    if let Some(delay) = arguments.first() {
                        emit_expression(delay, locals, instructions, procedures)?;
                    } else {
                        instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(0.0)));
                    }
                    &line.tokens[1 + spawn.index..]
                } else {
                    // BYOND's `spawn statement` and `spawn { ... }` forms are
                    // exactly `spawn(0)` with the parentheses omitted.
                    instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(0.0)));
                    after_keyword
                };
                let spawn_instruction = instructions.len();
                instructions.push(Instruction::Spawn { entry: usize::MAX });
                let skip_spawned_body = instructions.len();
                instructions.push(Instruction::Jump(usize::MAX));
                let spawned_entry = instructions.len();
                if rest.is_empty() {
                    source_spans.extend(std::iter::repeat_n(
                        line.span,
                        instructions.len() - first_instruction,
                    ));
                    let Some(first_body_line) = lines.get(line_index + 1) else {
                        return Err(compile_error("spawn requires a spawned statement"));
                    };
                    let body_indentation = indentation(first_body_line);
                    if body_indentation <= block_indentation {
                        return Err(compile_error(
                            "spawn requires an indented spawned statement",
                        ));
                    }
                    let (next_line, _) = compile_block(
                        lines,
                        line_index + 1,
                        body_indentation,
                        locals,
                        instructions,
                        source_spans,
                        procedures,
                        loops,
                    )?;
                    line_index = next_line;
                } else {
                    compile_expression(rest, locals, instructions, procedures)?;
                    instructions.push(Instruction::Pop);
                }
                instructions.push(Instruction::PushNull);
                instructions.push(Instruction::Return);
                let after_spawned_body = instructions.len();
                instructions[spawn_instruction] = Instruction::Spawn {
                    entry: spawned_entry,
                };
                instructions[skip_spawned_body] = Instruction::Jump(after_spawned_body);
                if rest.is_empty() {
                    source_spans.extend(std::iter::repeat_n(line.span, 2));
                } else {
                    source_spans.extend(std::iter::repeat_n(
                        line.span,
                        instructions.len() - first_instruction,
                    ));
                }
                if rest.is_empty() {
                    continue;
                }
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
            TokenKind::Identifier(_)
                if line.tokens.iter().any(|token| {
                    matches!(&token.kind, TokenKind::Operator(operator) if operator == "++" || operator == "--")
                }) =>
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

fn procedure_wait_for(definition: &Definition) -> bool {
    !definition.body.iter().any(|line| {
        matches!(
            line.tokens.as_slice(),
            [
                SpannedToken { kind: TokenKind::Identifier(set), .. },
                SpannedToken { kind: TokenKind::Identifier(name), .. },
                SpannedToken { kind: TokenKind::Operator(operator), .. },
                SpannedToken { kind: TokenKind::Identifier(value), .. }
            ] if set == "set" && name == "waitfor" && operator == "=" && value == "FALSE"
        ) || matches!(
            line.tokens.as_slice(),
            [
                SpannedToken { kind: TokenKind::Identifier(set), .. },
                SpannedToken { kind: TokenKind::Identifier(name), .. },
                SpannedToken { kind: TokenKind::Operator(operator), .. },
                SpannedToken { kind: TokenKind::Number(value), .. }
            ] if set == "set" && name == "waitfor" && operator == "=" && value == "0"
        )
    })
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
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator)
                if matches!(
                    operator.as_str(),
                    "=" | ":="
                        | "+="
                        | "-="
                        | "*="
                        | "/="
                        | "%="
                        | "%%="
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

fn top_level_output(tokens: &[SpannedToken]) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator) if operator == "<<" && depth == 0 => {
                return Some(index);
            }
            _ => {}
        }
    }
    None
}

fn top_level_input(tokens: &[SpannedToken]) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator) if operator == ">>" && depth == 0 => {
                return Some(index);
            }
            _ => {}
        }
    }
    None
}

#[allow(clippy::too_many_lines)]
fn compile_assignment_statement(
    tokens: &[SpannedToken],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let (assignment, operator) = top_level_assignment(tokens)
        .ok_or_else(|| compile_error("assignment statement requires '='"))?;
    let operator = if operator == ":=" { "=" } else { operator };
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
                    let mut value = ExpressionParser::new(&tokens[assignment + 1..]).parse()?;
                    if let Expression::New { type_path, .. } = &mut value
                        && type_path.is_none()
                        && let Some(inferred) = locals.global_type(&name)
                    {
                        *type_path = Some(Box::new(Expression::TypePath(inferred.clone())));
                    }
                    emit_expression(&value, locals, instructions, procedures)?;
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
            if let Some(storage) = locals
                .receiver_static(receiver.as_ref(), &name)
                .or_else(|| {
                    matches!(receiver.as_ref(), Expression::Src)
                        .then(|| locals.global_field(name.as_str()))
                        .flatten()
                })
            {
                if operator != "=" {
                    instructions.push(Instruction::LoadGlobal(storage.clone()));
                }
                compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::StoreGlobal(storage.clone()));
                return Ok(());
            }
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
        Expression::Src => {
            if operator != "=" {
                return Err(compile_error("src only supports direct assignment"));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            instructions.push(Instruction::StoreSrc);
        }
        Expression::Usr => {
            if operator != "=" {
                return Err(compile_error("usr only supports direct assignment"));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            instructions.push(Instruction::StoreUsr);
        }
        Expression::Result => {
            if operator != "=" {
                instructions.push(Instruction::LoadResult);
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreResult);
        }
        Expression::Unary {
            operator: unary_operator,
            operand,
        } if unary_operator == "*" => {
            if let Expression::Local(name) = operand.as_ref()
                && let Some(slot) = locals.get(name)
            {
                instructions.push(Instruction::LoadLocalRaw(slot));
            } else {
                emit_expression(&operand, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                return Err(compile_error(
                    "pointer dereference only supports direct assignment",
                ));
            }
            instructions.push(Instruction::SetListIndex);
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
        "%%=" => CompoundAssignmentOperator::FractionalRemainder,
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
        "%%=" => CompoundListIndexOperator::FractionalRemainder,
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

    loops.push(LoopContext {
        continue_target: Some(condition_target),
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = if let Some(body) = inline_conditional_body(&line.tokens) {
        let mut inline_line = line.clone();
        inline_line.tokens = body.to_vec();
        compile_block(
            std::slice::from_ref(&inline_line),
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
        .map(|(_, falls_through)| (line_index + 1, falls_through))
    } else {
        let child_index = line_index + 1;
        let Some(child) = lines.get(child_index) else {
            // BYOND permits an empty while whose condition performs all
            // useful work, including postfix/prefix mutation idioms.
            return finish_while_body(
                line_index + 1,
                condition_target,
                false_jump,
                line,
                loops,
                instructions,
                source_spans,
            );
        };
        let child_indentation = indentation(child);
        if child_indentation <= block_indentation {
            return finish_while_body(
                line_index + 1,
                condition_target,
                false_jump,
                line,
                loops,
                instructions,
                source_spans,
            );
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
        )
    };
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

#[allow(clippy::too_many_arguments)]
fn finish_while_body(
    after_body: usize,
    condition_target: usize,
    false_jump: usize,
    line: &SourceLine,
    loops: &mut Vec<LoopContext>,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
) -> Result<usize, CompileError> {
    let loop_context = loops.pop().expect("the active while context was pushed");
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, condition_target)?;
    }
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
    if let Some((local_name, type_path)) = for_type_parts(&line.tokens)? {
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
            true,
            &[],
            Some(&type_path),
        );
    }
    if let Some((first, second, iterable, declared)) = for_assoc_parts(&line.tokens)? {
        return compile_for_assoc(
            lines,
            line_index,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
            first,
            second,
            iterable,
            declared,
        );
    }
    if !for_header_uses_c_style(&line.tokens)
        && let Some((local_name, declared, start, end, step)) = for_to_parts(&line.tokens)?
    {
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
            declared,
            start,
            end,
            step,
        );
    }
    if let Some((local_name, declared, iterable)) = for_in_parts(&line.tokens)? {
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
            declared,
            iterable,
            None,
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
    let child_indentation = lines.get(child_index).map(indentation);
    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = if child_indentation.is_some_and(|indent| indent > block_indentation) {
        compile_block(
            lines,
            child_index,
            child_indentation.expect("checked"),
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
    } else {
        Ok((child_index, true))
    };
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

fn for_header_uses_c_style(tokens: &[SpannedToken]) -> bool {
    let mut depth = 0usize;
    let separators = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| match token.kind {
            TokenKind::Punctuation('(' | '[') => {
                depth += 1;
                None
            }
            TokenKind::Punctuation(')' | ']') => {
                depth = depth.saturating_sub(1);
                None
            }
            TokenKind::Punctuation(';' | ',') if depth == 1 => Some(index),
            _ => None,
        })
        .collect::<Vec<_>>();
    separators.len() >= 2
        || separators.first().is_some_and(|separator| {
            tokens[*separator + 1..tokens.len().saturating_sub(1)]
                .iter()
                .any(|_| true)
        })
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
    declared: bool,
    start: &[SpannedToken],
    end: &[SpannedToken],
    step: Option<&[SpannedToken]>,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let field_target = (!declared)
        .then(|| locals.src_field(local_name).cloned())
        .flatten();
    let item_slot = if declared {
        locals.declare(local_name.to_owned())?
    } else if let Some(slot) = locals.get(local_name) {
        slot
    } else if field_target.is_some() {
        locals.declare_hidden()?
    } else {
        return Err(compile_error(format!("unknown local {local_name:?}")));
    };
    let current_slot = locals.declare_hidden()?;
    let end_slot = locals.declare_hidden()?;
    let step_slot = locals.declare_hidden()?;

    let initialization_start = instructions.len();
    compile_expression(start, locals, instructions, procedures)?;
    instructions.push(Instruction::StoreLocal(current_slot));
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
        Instruction::LoadLocal(current_slot),
        Instruction::LoadLocal(end_slot),
        Instruction::LessEqual,
        Instruction::And,
        Instruction::LoadLocal(step_slot),
        Instruction::PushNumber(DmNumberBits::from_f32(0.0)),
        Instruction::Less,
        Instruction::LoadLocal(current_slot),
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
    // BYOND does not assign an existing iterator when the range is empty.
    // Keep the candidate in a hidden slot until the entry condition succeeds.
    push_instruction(
        instructions,
        source_spans,
        Instruction::LoadLocal(current_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::StoreLocal(item_slot),
        line.span,
    );
    if let Some(field) = &field_target {
        for instruction in [
            Instruction::LoadSrc,
            Instruction::LoadLocal(item_slot),
            Instruction::StoreField(field.clone()),
        ] {
            push_instruction(instructions, source_spans, instruction, line.span);
        }
    }
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
    if let Some(field) = &field_target {
        push_instruction(instructions, source_spans, Instruction::LoadSrc, line.span);
        push_instruction(
            instructions,
            source_spans,
            Instruction::LoadField(field.clone()),
            line.span,
        );
    } else {
        push_instruction(
            instructions,
            source_spans,
            Instruction::LoadLocal(item_slot),
            line.span,
        );
    }
    for instruction in [
        Instruction::LoadLocal(step_slot),
        Instruction::Add,
        Instruction::StoreLocal(current_slot),
        Instruction::Jump(condition_target),
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    if declared {
        locals.remove(local_name);
    }
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
    declared: bool,
    iterable: &[SpannedToken],
    type_instances: Option<&TypePath>,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let result_target = !declared && local_name == ".";
    let item_slot = if result_target {
        locals.declare_hidden()?
    } else if declared {
        locals.declare(local_name.to_owned())?
    } else {
        locals
            .get(local_name)
            .ok_or_else(|| compile_error(format!("unknown local {local_name:?}")))?
    };
    let list_slot = locals.declare_hidden()?;
    let index_slot = locals.declare_hidden()?;

    let initialization_start = instructions.len();
    if let Some(type_path) = type_instances {
        instructions.push(Instruction::TypeInstances(type_path.clone()));
    } else {
        compile_expression(iterable, locals, instructions, procedures)?;
    }
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
    if result_target {
        push_instruction(
            instructions,
            source_spans,
            Instruction::LoadLocal(item_slot),
            line.span,
        );
        push_instruction(
            instructions,
            source_spans,
            Instruction::StoreResult,
            line.span,
        );
    }

    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = if let Some(body) = inline_conditional_body(&line.tokens) {
        let mut inline_line = line.clone();
        inline_line.tokens = body.to_vec();
        compile_block(
            std::slice::from_ref(&inline_line),
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
        .map(|(_, falls_through)| (line_index + 1, falls_through))
    } else {
        let child_index = line_index + 1;
        let child = lines
            .get(child_index)
            .ok_or_else(|| compile_error("for-in statement requires an indented body"))?;
        let child_indentation = indentation(child);
        if child_indentation <= block_indentation {
            return Err(compile_error("for-in statement requires an indented body"));
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
        )
    };
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
    if declared {
        locals.remove(local_name);
    }
    Ok(after_body)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_for_assoc(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
    first: &[SpannedToken],
    second: &[SpannedToken],
    iterable: &[SpannedToken],
    declared: bool,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let (first_target, first_name) = parse_for_target(first, declared, locals)?;
    let (second_target, second_name) = parse_for_target(second, declared, locals)?;
    let list_slot = locals.declare_hidden()?;
    let index_slot = locals.declare_hidden()?;
    let key_slot = locals.declare_hidden()?;
    let value_slot = locals.declare_hidden()?;
    let start = instructions.len();
    compile_expression(iterable, locals, instructions, procedures)?;
    instructions.push(Instruction::StoreLocal(list_slot));
    instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
    instructions.push(Instruction::StoreLocal(index_slot));
    source_spans.extend(std::iter::repeat_n(line.span, instructions.len() - start));
    let condition = instructions.len();
    for instruction in [
        Instruction::LoadLocal(index_slot),
        Instruction::LoadLocal(list_slot),
        Instruction::ListLength,
        Instruction::LessEqual,
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
    for instruction in [
        Instruction::LoadLocal(list_slot),
        Instruction::LoadLocal(index_slot),
        Instruction::IndexList,
        Instruction::StoreLocal(key_slot),
        Instruction::LoadLocal(list_slot),
        Instruction::LoadLocal(key_slot),
        Instruction::IndexList,
        Instruction::StoreLocal(value_slot),
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    emit_for_target_store(&first_target, key_slot, locals, instructions, procedures)?;
    emit_for_target_store(&second_target, value_slot, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - source_spans.len(),
    ));
    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("for-in statement requires an indented body"))?;
    let child_indent = indentation(child);
    if child_indent <= block_indentation {
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
        child_indent,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    let context = loops.pop().expect("assoc loop context pushed");
    let (after_body, _) = body?;
    let increment = instructions.len();
    for jump in context.continue_jumps {
        patch_jump(instructions, jump, increment)?;
    }
    for instruction in [
        Instruction::LoadLocal(index_slot),
        Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
        Instruction::Add,
        Instruction::StoreLocal(index_slot),
        Instruction::Jump(condition),
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    let end = instructions.len();
    patch_jump(instructions, false_jump, end)?;
    for jump in context.break_jumps {
        patch_jump(instructions, jump, end)?;
    }
    if let Some(name) = first_name {
        locals.remove(&name);
    }
    if let Some(name) = second_name {
        locals.remove(&name);
    }
    Ok(after_body)
}

fn parse_for_target(
    tokens: &[SpannedToken],
    declared: bool,
    locals: &mut LocalTable,
) -> Result<(Expression, Option<String>), CompileError> {
    if declared {
        let name = tokens
            .iter()
            .rev()
            .find_map(|token| match &token.kind {
                TokenKind::Identifier(name) if name != "var" => Some(name.clone()),
                _ => None,
            })
            .ok_or_else(|| compile_error("associative loop declaration has no name"))?;
        locals.declare(name.clone())?;
        return Ok((Expression::Local(name.clone()), Some(name)));
    }
    Ok((ExpressionParser::new(tokens).parse()?, None))
}

fn emit_for_target_store(
    target: &Expression,
    slot: u16,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match target {
        Expression::Local(name) => {
            let target = locals
                .get(name)
                .ok_or_else(|| compile_error(format!("unknown local {name:?}")))?;
            instructions.push(Instruction::LoadLocal(slot));
            instructions.push(Instruction::StoreLocal(target));
        }
        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::LoadLocal(slot));
            instructions.push(Instruction::SetListIndex);
        }
        Expression::SafeIndex { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::LoadLocal(slot));
            instructions.push(Instruction::SetListIndex);
            let end_jump = instructions.len();
            instructions.push(Instruction::Jump(usize::MAX));
            let null_target = instructions.len();
            instructions.push(Instruction::Pop);
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(null_target);
            instructions[end_jump] = Instruction::Jump(end);
        }
        _ => return Err(compile_error("associative loop target is not writable")),
    }
    Ok(())
}

fn for_type_parts(tokens: &[SpannedToken]) -> Result<Option<(String, TypePath)>, CompileError> {
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
    let inner = &header[1..header.len() - 1];
    if !matches!(inner.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "var")
        || inner.iter().any(|token| {
            matches!(&token.kind,
            TokenKind::Identifier(name) if matches!(name.as_str(), "in" | "to"))
                || matches!(token.kind, TokenKind::Punctuation(',' | ';'))
        })
    {
        return Ok(None);
    }
    let names = inner
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(name) if name != "var" => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if names.len() < 2 {
        return Ok(None);
    }
    let local = names.last().expect("length checked").clone();
    let path = format!("/{}", names[..names.len() - 1].join("/"));
    let path = TypePath::parse(&path).map_err(|error| compile_error(error.to_string()))?;
    Ok(Some((local, path)))
}

#[allow(clippy::type_complexity)]
fn for_assoc_parts(
    tokens: &[SpannedToken],
) -> Result<Option<(&[SpannedToken], &[SpannedToken], &[SpannedToken], bool)>, CompileError> {
    let header = &tokens[1..];
    if !matches!(
        header.first().map(|t| &t.kind),
        Some(TokenKind::Punctuation('('))
    ) || !matches!(
        header.last().map(|t| &t.kind),
        Some(TokenKind::Punctuation(')'))
    ) {
        return Ok(None);
    }
    let inner = &header[1..header.len() - 1];
    let Some(in_pos) = inner
        .iter()
        .position(|t| matches!(&t.kind, TokenKind::Identifier(n) if n == "in"))
    else {
        return Ok(None);
    };
    let targets = &inner[..in_pos];
    let iterable = &inner[in_pos + 1..];
    let mut depth = 0usize;
    let mut comma = None;
    for (index, token) in targets.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[') => depth += 1,
            TokenKind::Punctuation(')' | ']') => depth = depth.saturating_sub(1),
            TokenKind::Punctuation(',') if depth == 0 => comma = Some(index),
            _ => {}
        }
    }
    let Some(comma) = comma else {
        return Ok(None);
    };
    if iterable.is_empty() || targets[..comma].is_empty() || targets[comma + 1..].is_empty() {
        return Err(compile_error(
            "associative for-in requires two targets and an iterable",
        ));
    }
    let declared =
        matches!(targets.first().map(|t| &t.kind), Some(TokenKind::Identifier(n)) if n == "var");
    Ok(Some((
        &targets[..comma],
        &targets[comma + 1..],
        iterable,
        declared,
    )))
}

fn for_in_parts(
    tokens: &[SpannedToken],
) -> Result<Option<(String, bool, &[SpannedToken])>, CompileError> {
    let header = &tokens[1..];
    if !matches!(
        header.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) {
        return Ok(None);
    }
    let mut depth = 0usize;
    let mut closing = None;
    for (index, token) in header.iter().enumerate() {
        match token.kind {
            TokenKind::Punctuation('(') => depth += 1,
            TokenKind::Punctuation(')') => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    closing = Some(index);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(closing) = closing else {
        return Ok(None);
    };
    let clauses = &header[1..closing];
    let clauses = if matches!(
        clauses.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(';'))
    ) && clauses[..clauses.len().saturating_sub(1)]
        .iter()
        .all(|token| token.kind != TokenKind::Punctuation(';'))
    {
        &clauses[..clauses.len() - 1]
    } else if clauses
        .iter()
        .any(|token| token.kind == TokenKind::Punctuation(';'))
    {
        return Ok(None);
    } else {
        clauses
    };
    let separators = top_level_keyword_positions(clauses, "in");
    if separators.len() > 1 {
        return Err(compile_error(
            "for-in header contains multiple 'in' keywords",
        ));
    }
    let Some(separator) = separators.first().copied() else {
        return Ok(None);
    };
    let declaration = &clauses[..separator];
    let iterable = &clauses[separator + 1..];
    if iterable.is_empty() {
        return Err(compile_error("for-in requires an iterable expression"));
    }
    let declared = matches!(
        declaration.first().map(|token| &token.kind),
        Some(TokenKind::Identifier(identifier)) if identifier == "var"
    );
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
    let local_name = if matches!(declaration, [SpannedToken { kind: TokenKind::Operator(operator), .. }] if operator == ".") {
        Some(".".to_owned())
    } else { declaration[..declaration_end]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) if identifier != "var" => Some(identifier.clone()),
            _ => None,
        }) } .ok_or_else(|| compile_error("for-in variable declaration has no name"))?;
    Ok(Some((local_name, declared, iterable)))
}

/// Recognizes `for(var/name in first to last [step increment])`, rather than treating the
/// range's `to` keyword as the beginning of a normal iterable expression.
#[allow(clippy::type_complexity)]
fn for_to_parts(
    tokens: &[SpannedToken],
) -> Result<
    Option<(
        String,
        bool,
        &[SpannedToken],
        &[SpannedToken],
        Option<&[SpannedToken]>,
    )>,
    CompileError,
> {
    let (local_name, declared, iterable) = if let Some(parts) = for_in_parts(tokens)? {
        parts
    } else {
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
        let separators = top_level_keyword_positions(clauses, "to");
        let [to_separator] = separators.as_slice() else {
            return Ok(None);
        };
        let before_to = &clauses[..*to_separator];
        let Some(assignment) = before_to.iter().rposition(
            |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
        ) else {
            return Ok(None);
        };
        let declaration = &before_to[..assignment];
        let declared = matches!(
            declaration.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(identifier)) if identifier == "var"
        );
        let local_name = declaration
            .iter()
            .rev()
            .find_map(|token| match &token.kind {
                TokenKind::Identifier(identifier) if identifier != "var" => {
                    Some(identifier.clone())
                }
                _ => None,
            })
            .ok_or_else(|| compile_error("for-to variable declaration has no name"))?;
        let start = &before_to[assignment + 1..];
        let iterable = &clauses[assignment + 1..];
        debug_assert!(iterable.starts_with(start));
        (local_name, declared, iterable)
    };
    let separators = top_level_keyword_positions(iterable, "to");
    let [separator] = separators.as_slice() else {
        return Ok(None);
    };
    let start = &iterable[..*separator];
    let after_to = &iterable[*separator + 1..];
    let after_to = after_to
        .iter()
        .position(|token| token.kind == TokenKind::Punctuation(';'))
        .map_or(after_to, |end| &after_to[..end]);
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
    Ok(Some((local_name, declared, start, end, step)))
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
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
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
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Punctuation(';' | ',') if depth == 0 => separators.push(index),
            _ => {}
        }
    }
    if separators.is_empty() && clauses.is_empty() {
        return Ok([clauses, clauses, clauses]);
    }
    if separators.len() == 1 {
        let separator = separators[0];
        return Ok([
            &clauses[..separator],
            &clauses[separator + 1..],
            &clauses[0..0],
        ]);
    }
    if separators.len() != 2 {
        if clauses.iter().any(
            |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "in"),
        ) {
            return Err(compile_error("for-in list iteration is not implemented"));
        }
        return Err(compile_error(
            "C-style for requires initializer, condition, and increment clauses separated by ';' or ','",
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
        // In C-style headers BYOND accepts a declaration followed by an
        // `in range` type-filter-looking suffix. It does not iterate that
        // range; the suffix qualifies the initializer and the declared value
        // remains the ordinary left-hand initializer.
        let tokens = top_level_keyword_positions(tokens, "in")
            .first()
            .map_or(tokens, |separator| &tokens[..*separator]);
        let separators = tokens
            .iter()
            .enumerate()
            .filter_map(|(index, token)| {
                matches!(&token.kind, TokenKind::Operator(operator) if operator == "&&")
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        if !separators.is_empty() {
            let mut start = 0usize;
            let mut last = None;
            for end in separators.into_iter().chain(std::iter::once(tokens.len())) {
                let declaration = &tokens[start..end];
                if !matches!(declaration.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "var")
                {
                    return Err(compile_error(
                        "combined for initializer must contain variable declarations",
                    ));
                }
                last = Some(compile_local(
                    declaration,
                    locals,
                    instructions,
                    procedures,
                )?);
                start = end + 1;
            }
            return Ok(last);
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
        let global = locals.global_field(name).cloned();
        if let Some(slot) = local {
            instructions.push(Instruction::LoadLocal(slot));
        } else if let Some(field) = &field {
            instructions.push(Instruction::LoadSrc);
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::LoadField(field.clone()));
        } else if let Some(global) = &global {
            instructions.push(Instruction::LoadGlobal(global.clone()));
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
        } else if field.is_some() {
            instructions.push(Instruction::StoreField(field.expect("field was checked")));
        } else {
            instructions.push(Instruction::StoreGlobal(
                global.expect("global was checked"),
            ));
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

#[allow(clippy::too_many_arguments)]
fn compile_try(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    let try_line = &lines[line_index];
    if try_line.tokens.len() != 1 {
        return Err(compile_error("try does not accept an expression"));
    }
    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("try statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("try statement requires an indented body"));
    }

    let handler_instruction = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::BeginTry {
            catch: usize::MAX,
            end: usize::MAX,
            local: None,
        },
        try_line.span,
    );
    let (catch_index, try_falls_through) = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    )?;
    let catch_line = lines
        .get(catch_index)
        .filter(|line| {
            indentation(line) == block_indentation
                && matches!(line.tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(keyword)) if keyword == "catch")
        })
        .ok_or_else(|| compile_error("try requires a matching catch"))?;
    let catch_local_name = parse_catch_local(&catch_line.tokens)?;
    let catch_local = catch_local_name
        .as_ref()
        .map(|_| locals.declare_hidden())
        .transpose()?;

    let protected_end = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::EndTry,
        catch_line.span,
    );
    let end_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::Jump(usize::MAX),
        catch_line.span,
    );
    let catch_target = instructions.len();
    instructions[handler_instruction] = Instruction::BeginTry {
        catch: catch_target,
        end: protected_end,
        local: catch_local,
    };

    let catch_child_index = catch_index + 1;
    let catch_indentation = lines.get(catch_child_index).map(indentation);
    // An empty catch is legal (`catch` followed by the next sibling
    // statement) and simply consumes the thrown value. A try itself may not
    // be empty, which also preserves BYOND's OD0015 diagnostic for an empty
    // try/catch pair.
    if catch_indentation.is_none_or(|indentation| indentation <= block_indentation) {
        let end_target = instructions.len();
        patch_jump(instructions, end_jump, end_target)?;
        return Ok((catch_child_index, true));
    }
    let catch_indentation = catch_indentation.expect("indentation was checked");
    let saved_names = locals.names.clone();
    if let (Some(name), Some(slot)) = (catch_local_name, catch_local) {
        locals.names.insert(name, slot);
    }
    let (next_line, catch_falls_through) = compile_block(
        lines,
        catch_child_index,
        catch_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    )?;
    locals.names = saved_names;
    let end_target = instructions.len();
    patch_jump(instructions, end_jump, end_target)?;
    Ok((next_line, try_falls_through || catch_falls_through))
}

fn parse_catch_local(tokens: &[SpannedToken]) -> Result<Option<String>, CompileError> {
    if tokens.len() == 1 {
        return Ok(None);
    }
    if !matches!(
        tokens.get(1).map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) || !matches!(
        tokens.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(')'))
    ) {
        return Err(compile_error("catch variable requires parentheses"));
    }
    let inner = &tokens[2..tokens.len() - 1];
    if !matches!(inner.first().map(|token| &token.kind), Some(TokenKind::Identifier(keyword)) if keyword == "var")
    {
        return Err(compile_error(
            "catch binding must be a variable declaration",
        ));
    }
    let name = inner.iter().rev().find_map(|token| match &token.kind {
        TokenKind::Identifier(name) if name != "var" => Some(name.clone()),
        _ => None,
    });
    name.map(Some)
        .ok_or_else(|| compile_error("catch variable declaration requires a name"))
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
        && matches!(body.first().map(|token| &token.kind), Some(TokenKind::Identifier(keyword)) if keyword == "do")
    {
        // Macro expansions frequently produce `if(condition) do { ... }
        // while(0)`. The brace normalizer has already placed the compact do
        // body on subsequent logical lines, so retain that tail while
        // replacing only the leading conditional with its inline statement.
        let mut inline_lines = lines[line_index..].to_vec();
        inline_lines[0].tokens = body.to_vec();
        let consumed = compile_do_while(
            &inline_lines,
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        (line_index + consumed, true)
    } else if let Some(body) = inline_conditional_body(&line.tokens) {
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
        // `else for(...)` and `else while(...)` keep their controlled body on
        // the following indented lines. Preserve the remaining source rather
        // than compiling only a synthetic header line.
        let mut inline_lines = lines[after_then..].to_vec();
        inline_lines[0].tokens = body.to_vec();
        let (consumed, falls_through) = compile_block(
            &inline_lines,
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        (after_then + consumed, falls_through)
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
        let inline_case_body = if is_default && case_line.tokens.len() > 1 {
            Some(&case_line.tokens[1..])
        } else {
            inline_conditional_body(&case_line.tokens)
        };
        let after_body = if let Some(body) = inline_case_body {
            let mut inline_line = case_line.clone();
            inline_line.tokens = body.to_vec();
            compile_block(
                std::slice::from_ref(&inline_line),
                0,
                case_indentation,
                locals,
                instructions,
                source_spans,
                procedures,
                loops,
            )?;
            next_case_index + 1
        } else {
            let body_index = next_case_index + 1;
            let body_indentation = lines.get(body_index).map(indentation);
            if body_indentation.is_some_and(|indent| indent > case_indentation) {
                compile_block(
                    lines,
                    body_index,
                    body_indentation.expect("checked"),
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?
                .0
            } else {
                // A macro may deliberately expand a case body to a lone
                // semicolon (`EMPTY_BLOCK_GUARD`). The syntax normalizer
                // removes that empty statement; the case remains a valid
                // no-op and falls through to the end of the switch.
                body_index
            }
        };
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
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    compile_error("switch case contains unmatched closing punctuation")
                })?;
            }
            TokenKind::Punctuation(punctuation) if *punctuation == separator && depth == 0 => {
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
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
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
        match &token.kind {
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
    let suffix = tokens[1..declaration_end]
        .iter()
        .position(|token| matches!(token.kind, TokenKind::Punctuation('[')))
        .map_or(declaration_end, |offset| 1 + offset);
    let name = tokens[1..suffix]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.clone()),
            _ => None,
        })
        .ok_or_else(|| compile_error("local declaration has no name"))?;
    let slot = locals.declare(name.clone())?;
    let is_static = tokens[1..declaration_end].iter().any(
        |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "static"),
    );
    let static_jump = is_static.then(|| {
        let index = instructions.len();
        instructions.push(Instruction::LoadStaticLocalOrJump {
            slot,
            target: usize::MAX,
        });
        index
    });
    if let Some(assignment) = assignment {
        let mut value = ExpressionParser::new(&tokens[assignment + 1..]).parse()?;
        if let Expression::New { type_path, .. } = &mut value
            && type_path.is_none()
            && let Some(inferred) = declared_local_type(tokens, &name)
        {
            *type_path = Some(Box::new(Expression::TypePath(inferred)));
        }
        emit_expression(&value, locals, instructions, procedures)?;
    } else if suffix < declaration_end {
        let mut dimensions = 0u8;
        let mut cursor = suffix;
        while cursor < declaration_end {
            if !matches!(tokens[cursor].kind, TokenKind::Punctuation('[')) {
                cursor += 1;
                continue;
            }
            let mut bracket_depth = 1usize;
            let close = (cursor + 1..declaration_end)
                .find(|&index| {
                    match tokens[index].kind {
                        TokenKind::Punctuation('[') => bracket_depth += 1,
                        TokenKind::Punctuation(']') => {
                            bracket_depth = bracket_depth.saturating_sub(1)
                        }
                        _ => {}
                    }
                    bracket_depth == 0
                })
                .ok_or_else(|| compile_error("array declaration has an unclosed dimension"))?;
            compile_expression(&tokens[cursor + 1..close], locals, instructions, procedures)?;
            dimensions = dimensions
                .checked_add(1)
                .ok_or_else(|| compile_error("too many array dimensions"))?;
            cursor = close + 1;
        }
        instructions.push(Instruction::MakeArray(dimensions));
    } else {
        // Typed and untyped local declarations without an initializer begin
        // as null in DM.
        instructions.push(Instruction::PushNull);
    }
    if is_static {
        instructions.push(Instruction::InitializeStaticLocal(slot));
    }
    instructions.push(Instruction::StoreLocal(slot));
    if let Some(jump) = static_jump {
        let target = instructions.len();
        instructions[jump] = Instruction::LoadStaticLocalOrJump { slot, target };
    }
    Ok(name)
}

fn declared_local_type(tokens: &[SpannedToken], name: &str) -> Option<TypePath> {
    let name_index = tokens.iter().rposition(
        |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == name),
    )?;
    let var_index = tokens[..name_index].iter().position(
        |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "var"),
    )?;
    let segments = tokens[var_index + 1..name_index]
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(identifier)
                if !matches!(identifier.as_str(), "static" | "global" | "tmp" | "final") =>
            {
                Some(identifier.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    (!segments.is_empty())
        .then(|| TypePath::parse(&format!("/{}", segments.join("/"))).ok())
        .flatten()
}

fn compile_local_declarations(
    tokens: &[SpannedToken],
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let mut depth = 0_usize;
    let mut start = 1_usize;
    let mut parts = Vec::new();
    for (index, token) in tokens.iter().enumerate().skip(1) {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Punctuation(',') if depth == 0 => {
                parts.push(&tokens[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(&tokens[start..]);
    for part in parts {
        if part.is_empty() {
            return Err(compile_error("local declaration after ',' is empty"));
        }
        let mut declaration = Vec::with_capacity(part.len() + 1);
        declaration.push(tokens[0].clone());
        declaration.extend_from_slice(part);
        compile_local(&declaration, locals, instructions, procedures)?;
    }
    Ok(())
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
    ModifiedTypePath {
        base: TypePath,
        overrides: Vec<(FieldName, Self)>,
    },
    New {
        type_path: Option<Box<Self>>,
        arguments: Vec<Self>,
        overrides: Vec<(FieldName, Self)>,
    },
    Regex {
        arguments: Vec<Self>,
    },
    MutableAppearance {
        arguments: Vec<Self>,
    },
    Matrix {
        arguments: Vec<Self>,
    },
    Vector {
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
    NativeSrcMethod {
        name: String,
        arguments: Vec<Self>,
    },
    ExternalCall {
        library: Box<Self>,
        function: Box<Self>,
        arguments: Vec<Self>,
    },
    Animate {
        arguments: Vec<(Option<String>, Self)>,
    },
    Filter {
        arguments: Vec<(Option<String>, Self)>,
    },
    Crash(Box<Self>),
    Sleep(Box<Self>),
    Initial(Box<Self>),
    Block {
        arguments: Vec<Self>,
    },
    Rand {
        arguments: Vec<Self>,
    },
    Roll {
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
    GetStepTowards {
        source: Box<Self>,
        target: Box<Self>,
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
    Caller,
    World,
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
        null_receiver_is_global: bool,
    },
    SafeDynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
    },
    List(Vec<ListExpressionEntry>),
    AssociativeList(Vec<ListExpressionEntry>),
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
    Mutation {
        target: Box<Self>,
        delta: i8,
        prefix: bool,
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
        "FALSE"
        | "BLEND_DEFAULT"
        | "MATRIX_COPY"
        | "MOB_PERSPECTIVE"
        | "TOPDOWN_MAP"
        | "LINEAR_EASING"
        | "COLORSPACE_RGB"
        | "MOUSE_INACTIVE_POINTER"
        | "NO_STEPS"
        | "PROFILE_START"
        | "PROFILE_REFRESH"
        | "FILTER_OVERLAY"
        | "ICON_ADD" => Some(0.0),
        "FLOAT_LAYER" => Some(-1.0),
        "TRUE"
        | "MASK_INVERSE"
        | "ICON_SUBTRACT"
        | "BLEND_OVERLAY"
        | "KEEP_TOGETHER"
        | "NORTH"
        | "EYE_PERSPECTIVE"
        | "AREA_LAYER"
        | "SINE_EASING"
        | "ANIMATION_END_NOW"
        | "COLORSPACE_HSV"
        | "VIS_INHERIT_ICON"
        | "MOUSE_ACTIVE_POINTER"
        | "FORWARD_STEPS"
        | "BLIND"
        | "PROFILE_STOP" => Some(1.0),
        "CONTROL_FREAK_SKIN" => Some(1.0),
        "CONTROL_FREAK_MACROS" => Some(2.0),
        "JSON_PRETTY_PRINT" => Some(1.0),
        "BLEND_ADD"
        | "KEEP_APART"
        | "SOUTH"
        | "EDGE_PERSPECTIVE"
        | "TURF_LAYER"
        | "CIRCULAR_EASING"
        | "ANIMATION_LINEAR_TRANSFORM"
        | "COLORSPACE_HSL"
        | "VIS_INHERIT_ICON_STATE"
        | "SLIDE_STEPS"
        | "PROFILE_CLEAR"
        | "PROFILE_RESTART"
        | "ICON_MULTIPLY" => Some(2.0),
        "BLEND_SUBTRACT" | "OBJ_LAYER" | "CUBIC_EASING" | "COLORSPACE_HCY"
        | "MOUSE_DRAG_POINTER" | "SYNC_STEPS" | "ICON_OVERLAY" => Some(3.0),
        "BLEND_MULTIPLY" | "LONG_GLIDE" | "EAST" | "MATRIX_INVERT" | "MOB_LAYER"
        | "BOUNCE_EASING" | "ANIMATION_PARALLEL" | "VIS_INHERIT_DIR" | "MOUSE_DROP_POINTER"
        | "SEE_MOBS" | "SEEMOBS" | "PROFILE_AVERAGE" => Some(4.0),
        "BLEND_INSET_OVERLAY"
        | "NORTHEAST"
        | "MATRIX_ROTATE"
        | "FLY_LAYER"
        | "ELASTIC_EASING"
        | "MOUSE_ARROW_POINTER"
        | "ICON_OR" => Some(5.0),
        "SOUTHEAST"
        | "MATRIX_SCALE"
        | "BACK_EASING"
        | "MOUSE_CROSSHAIRS_POINTER"
        | "ICON_UNDERLAY" => Some(6.0),
        "MATRIX_TRANSLATE" | "QUAD_EASING" | "MOUSE_HAND_POINTER" => Some(7.0),
        "WEST" | "RESET_TRANSFORM" | "JUMP_EASING" | "ANIMATION_SLICE" | "VIS_INHERIT_LAYER"
        | "SEE_OBJS" | "SEEOBJS" => Some(8.0),
        "NORTHWEST" => Some(9.0),
        "SOUTHWEST" => Some(10.0),
        "UP" | "RESET_COLOR" | "ANIMATION_END_LOOP" | "VIS_INHERIT_PLANE" | "SEE_TURFS"
        | "SEETURFS" => Some(16.0),
        "DOWN" | "RESET_ALPHA" | "VIS_INHERIT_ID" | "SEE_SELF" => Some(32.0),
        // Appearance flags are BYOND bitflags. Keep the complete contiguous
        // built-in flag family here rather than teaching project code about
        // individual flags as each one is encountered.
        // These make an overlay/image ignore the corresponding value
        // inherited from its parent.
        "PIXEL_SCALE" | "EASE_IN" | "VIS_UNDERLAY" | "SEE_INFRA" => Some(64.0),
        "TILE_BOUND" | "MATRIX_MODIFY" | "EASE_OUT" | "VIS_HIDE" => Some(128.0),
        "INHERIT_ID" | "ANIMATION_RELATIVE" | "SEE_PIXELS" => Some(256.0),
        "NO_CLIENT_COLOR" | "ANIMATION_CONTINUE" | "SEE_THRU" => Some(512.0),
        "RESET_CONTENTS" | "SEE_BLACKNESS" => Some(1024.0),
        "PLANE_MASTER" => Some(2048.0),
        "PASS_MOUSE" => Some(4096.0),
        "TILE_MOVER" => Some(8192.0),
        "EFFECTS_LAYER" => Some(5000.0),
        "TOPDOWN_LAYER" => Some(10000.0),
        "BACKGROUND_LAYER" => Some(20000.0),
        "FLOAT_PLANE" => Some(-32767.0),
        "TILED_ICON_MAP" => Some(32768.0),
        _ => None,
    }
}

struct ExpressionParser<'a> {
    tokens: &'a [SpannedToken],
    index: usize,
    /// While parsing the true arm of `?:`, a bare colon terminates that arm
    /// instead of selecting a dynamic field.  Outside that one context DM's
    /// `datum:field` syntax remains a normal postfix operation, including in
    /// the false arm (`condition ? datum : datum:type`).
    conditional_true_arm: bool,
}

impl<'a> ExpressionParser<'a> {
    const fn new(tokens: &'a [SpannedToken]) -> Self {
        Self {
            tokens,
            index: 0,
            conditional_true_arm: false,
        }
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
            "=" | ":="
                | "+="
                | "-="
                | "*="
                | "/="
                | "%="
                | "%%="
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
        let operator = if operator == ":=" {
            "=".to_owned()
        } else {
            operator.clone()
        };
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
        let enclosing_true_arm = self.conditional_true_arm;
        self.conditional_true_arm = true;
        let when_true = self.parse_assignment()?;
        match self.tokens.get(self.index).map(|token| &token.kind) {
            Some(TokenKind::Operator(operator)) if operator == ":" => self.index += 1,
            _ => return Err(compile_error("expected ':' in conditional expression")),
        }
        // The false arm is still inside an enclosing true arm, if there is
        // one.  In `a ? b ? c : d : e`, that outer colon must terminate the
        // nested expression rather than becoming dynamic access `d:e`.
        self.conditional_true_arm = enclosing_true_arm;
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
                // `value in lower to upper` is BYOND's inclusive range
                // predicate. `to` is a keyword delimiter rather than a
                // general arithmetic operator, so lower it directly to the
                // two comparisons while the left operand is still available.
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Identifier(keyword)) if keyword == "to"
                ) {
                    self.index += 1;
                    let upper = self.parse_binary(right_precedence)?;
                    Expression::Binary {
                        operator: "&&".to_owned(),
                        left: Box::new(Expression::Binary {
                            operator: ">=".to_owned(),
                            left: Box::new(left.clone()),
                            right: Box::new(right),
                        }),
                        right: Box::new(Expression::Binary {
                            operator: "<=".to_owned(),
                            left: Box::new(left),
                            right: Box::new(upper),
                        }),
                    }
                } else {
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
        // Prefix mutation is an expression in DM, not merely a statement:
        // `values[++i]` first updates i, then uses the new value. Reuse the
        // assignment lowering so every assignable target retains its normal
        // single-evaluation behavior.
        if let Some(operator @ ("++" | "--")) = self.current_operator() {
            let operator = operator.to_owned();
            self.index += 1;
            let target = self.parse_unary()?;
            return Ok(Expression::Mutation {
                target: Box::new(target),
                delta: if operator == "++" { 1 } else { -1 },
                prefix: true,
            });
        }
        if let Some(operator @ ("!" | "+" | "-" | "~" | "&" | "*")) = self.current_operator() {
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
                // An index is a full DM expression. In particular, ternaries
                // and assignments are legal here (`values[flag ? a : b]`).
                // Parsing only the binary-precedence layer left the `?` in
                // front of the closing bracket and produced a misleading
                // "expected ']'" diagnostic.
                let index = self.parse_assignment()?;
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
            if matches!(self.current_operator(), Some("::")) {
                self.index += 1;
                let Some(TokenKind::Identifier(qualified)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected identifier after '::'"));
                };
                let qualified = qualified.clone();
                self.index += 1;
                if qualified == "name"
                    && let Expression::TypePath(path) = &expression
                    && let Some((_, procedure_name)) = path.as_str().rsplit_once("/proc/")
                {
                    expression = Expression::Text(procedure_name.to_owned());
                } else if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    expression = Expression::Call {
                        procedure: qualified,
                        arguments: self.parse_call_arguments()?,
                    };
                } else {
                    let name = FieldName::parse(&qualified)
                        .map_err(|error| compile_error(error.to_string()))?;
                    expression = Expression::Initial(Box::new(Expression::Field {
                        receiver: Box::new(expression),
                        name,
                    }));
                }
                continue;
            }
            if matches!(self.current_operator(), Some("." | "?." | "?:"))
                || (matches!(self.current_operator(), Some(":"))
                    && (!self.conditional_true_arm || self.colon_member_is_lexically_attached())
                    && matches!(
                        self.tokens.get(self.index + 1).map(|token| &token.kind),
                        Some(TokenKind::Identifier(_))
                    ))
            {
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
            // `input(...) as null|anything in choices` is an input-picker
            // suffix, not a cast of the call expression. The type union is UI
            // metadata and `in choices` supplies the displayed candidates;
            // the native input call already represents the selected value in
            // headless execution. Parse and consume both parts here.
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Identifier(keyword)) if keyword == "as"
            ) {
                self.index += 1;
                while !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Identifier(keyword)) if keyword == "in"
                ) {
                    if self.index >= self.tokens.len() {
                        return Ok(expression);
                    }
                    self.index += 1;
                }
                self.index += 1;
                let _choices = self.parse_assignment()?;
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
                            null_receiver_is_global: false,
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
                    // A second argument list invokes the procedure selector
                    // produced by the preceding expression.  DreamMaker uses
                    // this for `call_ext(library, function)(arguments)` as
                    // well as ordinary `call(...)(...)` selectors.
                    other => Expression::DynamicCall {
                        target: Box::new(Expression::Null),
                        procedure: Box::new(other),
                        arguments: self.parse_call_arguments()?,
                        null_receiver_is_global: true,
                    },
                };
                continue;
            }
            if let Some(operator @ ("++" | "--")) = self.current_operator() {
                let delta = if operator == "++" { 1 } else { -1 };
                self.index += 1;
                expression = Expression::Mutation {
                    target: Box::new(expression),
                    delta,
                    prefix: false,
                };
                continue;
            }
            break;
        }
        Ok(expression)
    }

    fn colon_member_is_lexically_attached(&self) -> bool {
        let Some(colon) = self.tokens.get(self.index) else {
            return false;
        };
        let Some(name) = self.tokens.get(self.index + 1) else {
            return false;
        };
        colon.span.end == name.span.start
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
                        // BYOND accepts a canonical type path with a trailing
                        // slash (commonly used as an associative-list key).
                        // The slash has already been consumed; canonicalize it
                        // away once at least one real segment was collected.
                        if !path.is_empty() {
                            break;
                        }
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
                let base =
                    TypePath::parse(&path).map_err(|error| compile_error(error.to_string()))?;
                let overrides = self.parse_modified_type_overrides()?;
                if overrides.is_empty() {
                    Ok(Expression::TypePath(base))
                } else {
                    Ok(Expression::ModifiedTypePath { base, overrides })
                }
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
            TokenKind::String(text) => parse_interpolated_string(text),
            TokenKind::RawString(text) | TokenKind::TextBlock(text) => {
                Ok(Expression::Text(text.clone()))
            }
            TokenKind::Resource(text) => {
                let normalized = text.replace('\\', "/");
                Ok(Expression::Text(
                    normalized
                        .strip_prefix("./")
                        .unwrap_or(&normalized)
                        .to_owned(),
                ))
            }
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
            TokenKind::Operator(operator) if operator == "::" => {
                let Some(TokenKind::Identifier(name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected global identifier after '::'"));
                };
                let name = name.clone();
                self.index += 1;
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    Ok(Expression::Call {
                        procedure: name,
                        arguments: self.parse_call_arguments()?,
                    })
                } else {
                    FieldName::parse(&name)
                        .map(Expression::GlobalField)
                        .map_err(|error| compile_error(error.to_string()))
                }
            }
            TokenKind::Identifier(identifier) if identifier == "src" => Ok(Expression::Src),
            TokenKind::Identifier(identifier) if identifier == "usr" => Ok(Expression::Usr),
            TokenKind::Identifier(identifier) if identifier == "caller" => Ok(Expression::Caller),
            TokenKind::Identifier(identifier) if identifier == "world" => Ok(Expression::World),
            TokenKind::Identifier(identifier) if identifier == "locs" => Ok(Expression::Field {
                receiver: Box::new(Expression::Src),
                name: FieldName::parse("locs").expect("built-in locs field name is valid"),
            }),
            TokenKind::Identifier(identifier) if identifier == "vars" => Ok(Expression::Field {
                receiver: Box::new(Expression::Src),
                name: FieldName::parse("vars").expect("built-in vars field name is valid"),
            }),
            // Only lowercase `global` is BYOND's built-in namespace. `GLOB`
            // in SS13 codebases is an ordinary declared global datum.
            TokenKind::Identifier(identifier) if identifier == "global" => {
                Ok(Expression::GlobalNamespace)
            }
            TokenKind::Identifier(identifier) if matches!(self.tokens.get(self.index).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "::") =>
            {
                let mut qualifiers = Vec::new();
                let mut next_token = self.tokens.get(self.index).map(|token| &token.kind);
                while let Some(TokenKind::Operator(operator)) = next_token {
                    if operator != "::" {
                        break;
                    }
                    self.index += 1;
                    let token = self
                        .tokens
                        .get(self.index)
                        .ok_or_else(|| compile_error("expected namespace qualifier after '::'"))?;
                    let TokenKind::Identifier(qualified) = &token.kind else {
                        return Err(compile_error("expected identifier after '::'"));
                    };
                    qualifiers.push(qualified.clone());
                    self.index += 1;
                    next_token = self.tokens.get(self.index).map(|token| &token.kind);
                }

                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    let arguments = self.parse_call_arguments()?;
                    Ok(Expression::Call {
                        procedure: qualifiers
                            .last()
                            .expect("namespace chain has a qualifier")
                            .clone(),
                        arguments,
                    })
                } else {
                    let mut receiver = Expression::Local(identifier.clone());
                    for qualifier in qualifiers {
                        let name = FieldName::parse(&qualifier)
                            .map_err(|error| compile_error(error.to_string()))?;
                        receiver = Expression::Initial(Box::new(Expression::Field {
                            receiver: Box::new(receiver),
                            name,
                        }));
                    }
                    Ok(receiver)
                }
            }
            TokenKind::Identifier(identifier) if identifier == "new" => {
                // `new /path(args)` is the common explicit form.  An
                // unqualified `new(args)` constructs the current datum type.
                // Keep the constructor arguments in the AST even though the
                // headless VM currently only establishes object identity.
                if matches!(self.current_operator(), Some("/")) {
                    let type_path = self.parse_primary()?;
                    let overrides = self.parse_modified_type_overrides()?;
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
                        overrides,
                    })
                } else if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    Ok(Expression::New {
                        type_path: None,
                        arguments: self.parse_call_arguments()?,
                        overrides: Vec::new(),
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
                        overrides: Vec::new(),
                    })
                } else {
                    Ok(Expression::New {
                        type_path: None,
                        arguments: Vec::new(),
                        overrides: Vec::new(),
                    })
                }
            }
            TokenKind::Identifier(identifier) if identifier == "call_ext" => {
                let selectors = self.parse_call_arguments()?;
                let [library, function] = selectors.as_slice() else {
                    return Err(compile_error(
                        "call_ext requires a library and exported function selector",
                    ));
                };
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    return Err(compile_error("call_ext selector requires an argument list"));
                }
                Ok(Expression::ExternalCall {
                    library: Box::new(library.clone()),
                    function: Box::new(function.clone()),
                    arguments: self.parse_call_arguments()?,
                })
            }
            TokenKind::Identifier(identifier) if identifier == "call" => {
                let selectors = self.parse_call_arguments()?;
                let (target, procedure, null_receiver_is_global) = match selectors.as_slice() {
                    [procedure] => (Expression::Null, procedure.clone(), true),
                    [target, procedure] => (target.clone(), procedure.clone(), false),
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
                    null_receiver_is_global,
                })
            }
            TokenKind::Identifier(identifier)
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) =>
            {
                if identifier == "CRASH" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "CRASH requires exactly one argument, received {}",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Crash(Box::new(
                        arguments.pop().expect("CRASH argument count was validated"),
                    )))
                } else if identifier == "list" {
                    Ok(Expression::List(self.parse_list_arguments()?))
                } else if identifier == "alist" {
                    Ok(Expression::AssociativeList(self.parse_list_arguments()?))
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
                } else if identifier == "matrix" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() > 6 {
                        return Err(compile_error("matrix accepts at most six arguments"));
                    }
                    Ok(Expression::Matrix { arguments })
                } else if identifier == "vector" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() > 3 {
                        return Err(compile_error("vector accepts at most three arguments"));
                    }
                    Ok(Expression::Vector { arguments })
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
                } else if identifier == "get_step_towards" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 2 {
                        return Err(compile_error(format!(
                            "get_step_towards requires exactly a source and target, received {} arguments",
                            arguments.len()
                        )));
                    }
                    let target = arguments.pop().expect("argument count validated");
                    let source = arguments.pop().expect("argument count validated");
                    Ok(Expression::GetStepTowards {
                        source: Box::new(source),
                        target: Box::new(target),
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
                    if arguments.len() > 2 {
                        return Err(compile_error(format!(
                            "rand accepts zero, one, or two numeric bounds, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Rand { arguments })
                } else if identifier == "roll" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "roll requires dice or a dice count and side count, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Roll { arguments })
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
                } else if identifier == "sleep" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "sleep requires exactly one delay, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Sleep(Box::new(
                        arguments.pop().expect("sleep argument count was validated"),
                    )))
                } else if identifier == "locate" {
                    Ok(Expression::Locate {
                        arguments: self.parse_call_arguments()?,
                    })
                } else if identifier == "animate" {
                    Ok(Expression::Animate {
                        arguments: self.parse_named_call_arguments()?,
                    })
                } else if identifier == "filter" {
                    Ok(Expression::Filter {
                        arguments: self.parse_named_call_arguments()?,
                    })
                } else if identifier == "nameof" {
                    self.parse_nameof_expression()
                } else if matches!(
                    identifier.as_str(),
                    "MapColors"
                        | "Blend"
                        | "SetIntensity"
                        | "Scale"
                        | "Crop"
                        | "Shift"
                        | "Width"
                        | "Height"
                        | "DrawBox"
                        | "Insert"
                        | "GetPixel"
                        | "Turn"
                ) {
                    Ok(Expression::NativeSrcMethod {
                        name: identifier.clone(),
                        arguments: self.parse_call_arguments()?,
                    })
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
        Ok(self
            .parse_named_call_arguments()?
            .into_iter()
            .map(|(_, expression)| expression)
            .collect())
    }

    fn parse_named_call_arguments(
        &mut self,
    ) -> Result<Vec<(Option<String>, Expression)>, CompileError> {
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ) {
            return Err(compile_error("expected '(' before call arguments"));
        }
        self.index += 1;
        let mut arguments = Vec::new();
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            loop {
                // Empty positional slots are legal DM and mean that the
                // callee's default value should be used.  The VM represents
                // an omitted interior slot as null until its call ABI carries
                // a distinct omission marker.
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation(','))
                ) {
                    arguments.push((None, Expression::Null));
                    self.index += 1;
                    if matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation(')'))
                    ) {
                        break;
                    }
                    continue;
                }
                // BYOND permits keyword-style call arguments, e.g.
                // `do_after(user, 4 SECONDS, target = src)`.  The current
                // execution ABI is positional, but retaining the source
                // order here is still the correct lowering for its existing
                // subset and, importantly, lets the compiler continue on to
                // report the next unsupported construct instead of rejecting
                // the call syntax itself.
                let name = match (
                    self.tokens.get(self.index).map(|token| &token.kind),
                    self.tokens.get(self.index + 1).map(|token| &token.kind),
                ) {
                    (Some(TokenKind::Identifier(name)), Some(TokenKind::Operator(operator)))
                        if operator == "=" =>
                    {
                        Some(name.clone())
                    }
                    (Some(TokenKind::String(name)), Some(TokenKind::Operator(operator)))
                        if operator == "=" =>
                    {
                        Some(name.clone())
                    }
                    _ => None,
                };
                if name.is_some() {
                    self.index += 2;
                }
                arguments.push((name, self.parse_assignment()?));
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
                        return Err(compile_error(format!(
                            "expected ',' or ')' after procedure argument, received {:?}",
                            self.tokens.get(self.index).map(|token| &token.kind)
                        )));
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

    fn parse_modified_type_overrides(
        &mut self,
    ) -> Result<Vec<(FieldName, Expression)>, CompileError> {
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('{'))
        ) {
            return Ok(Vec::new());
        }
        self.index += 1;
        let mut overrides = Vec::new();
        loop {
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('}'))
            ) {
                self.index += 1;
                return Ok(overrides);
            }
            let Some(TokenKind::Identifier(name)) =
                self.tokens.get(self.index).map(|token| &token.kind)
            else {
                return Err(compile_error("modified type requires a field name"));
            };
            let name = FieldName::parse(name).map_err(|error| compile_error(error.to_string()))?;
            self.index += 1;
            if !matches!(self.current_operator(), Some("=")) {
                return Err(compile_error("modified type field requires '='"));
            }
            self.index += 1;
            let start = self.index;
            let mut depth = 0_usize;
            while let Some(token) = self.tokens.get(self.index) {
                match token.kind {
                    TokenKind::Punctuation('(' | '[') => depth += 1,
                    TokenKind::Punctuation(')' | ']') => depth = depth.saturating_sub(1),
                    TokenKind::Punctuation('}' | ';') if depth == 0 => break,
                    _ => {}
                }
                self.index += 1;
            }
            if start == self.index {
                return Err(compile_error("modified type field value is empty"));
            }
            let value = ExpressionParser::new(&self.tokens[start..self.index]).parse()?;
            overrides.push((name, value));
            match self.tokens.get(self.index).map(|token| &token.kind) {
                Some(TokenKind::Punctuation(';')) => self.index += 1,
                Some(TokenKind::Punctuation('}')) => {}
                _ => return Err(compile_error("modified type requires ';' or '}'")),
            }
        }
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
    let value = if matches!(normalized.as_str(), "1#INF" | "1.#INF") {
        f32::INFINITY
    } else if matches!(normalized.as_str(), "1#IND" | "1.#IND") {
        f32::NAN
    } else if let Some(hexadecimal) = normalized
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

fn parse_interpolated_string(text: &str) -> Result<Expression, CompileError> {
    let mut arguments = Vec::new();
    let mut cursor = 0_usize;
    while let Some(relative_open) = text[cursor..].find('[') {
        let open = cursor + relative_open;
        if open > 0 && text.as_bytes()[open - 1] == b'\\' {
            cursor = open + 1;
            continue;
        }
        let Some(close) = interpolated_expression_close(text, open + 1) else {
            break;
        };
        if text[open + 1..close].trim().is_empty() {
            cursor = close + 1;
            continue;
        }
        if open > cursor {
            arguments.push(Expression::Text(text[cursor..open].to_owned()));
        }
        let tokens = lex(&text[open + 1..close])
            .map_err(|error| {
                compile_error(format!("invalid embedded expression: {}", error.message))
            })?
            .into_iter()
            .filter(|token| !matches!(token.kind, TokenKind::LineStart { .. } | TokenKind::Newline))
            .collect::<Vec<_>>();
        arguments.push(Expression::StandardBuiltin {
            name: "text".to_owned(),
            arguments: vec![
                Expression::Text("[]".to_owned()),
                ExpressionParser::new(&tokens).parse()?,
            ],
        });
        cursor = close + 1;
    }
    if arguments.is_empty() {
        return Ok(Expression::Text(text.to_owned()));
    }
    if cursor < text.len() {
        arguments.push(Expression::Text(text[cursor..].to_owned()));
    }
    Ok(Expression::StandardBuiltin {
        name: "addtext".to_owned(),
        arguments,
    })
}

fn interpolated_expression_close(text: &str, start: usize) -> Option<usize> {
    let mut depth = 1usize;
    let mut cursor = start;
    let mut quote = None;
    let mut escaped = false;
    while cursor < text.len() {
        let character = text[cursor..].chars().next()?;
        if let Some(delimiter) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == delimiter {
                quote = None;
            }
        } else {
            match character {
                '"' | '\'' => quote = Some(character),
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(cursor);
                    }
                }
                _ => {}
            }
        }
        cursor += character.len_utf8();
    }
    None
}

const fn binary_precedence(operator: &str) -> Option<u8> {
    match operator.as_bytes() {
        b"||" => Some(1),
        b"&&" => Some(2),
        b"|" => Some(3),
        b"^" => Some(4),
        b"&" => Some(5),
        b"==" | b"!=" | b"<>" | b"~=" | b"~!" => Some(6),
        b"<<" | b">>" | b"<" | b"<=" | b">" | b">=" | b"<=>" | b"in" => Some(7),
        b"+" | b"-" => Some(8),
        b"*" | b"/" | b"%" | b"%%" => Some(9),
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
    // In DM's list-constructor grammar, a bare identifier to the left of `=`
    // is a named/text key even when a local, field, or global with the same
    // spelling exists. Dynamic keys use an explicit expression instead.
    if let Expression::Local(name) = key {
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
        Expression::ModifiedTypePath { base, overrides } => {
            instructions.push(Instruction::PushTypePath(base.clone()));
            for (_, value) in overrides {
                emit_expression(value, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeModifiedTypePath {
                fields: overrides
                    .iter()
                    .map(|(field, _)| field.clone())
                    .collect::<Vec<_>>()
                    .into(),
            });
        }
        Expression::New {
            type_path,
            arguments,
            overrides,
        } => {
            let Some(type_path) = type_path else {
                return Err(compile_error(
                    "inferred new has no statically resolved destination type",
                ));
            };
            emit_expression(type_path, locals, instructions, procedures)?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::AllocateDatum { argument_count });
            for (name, value) in overrides {
                instructions.push(Instruction::Duplicate);
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::StoreField(name.clone()));
            }
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
        Expression::Matrix { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeMatrix {
                argument_count: u8::try_from(arguments.len())
                    .expect("matrix argument count was validated"),
            });
        }
        Expression::Vector { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeVector {
                argument_count: u8::try_from(arguments.len())
                    .expect("vector argument count was validated"),
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
        Expression::GetStepTowards { source, target } => {
            emit_expression(source, locals, instructions, procedures)?;
            emit_expression(target, locals, instructions, procedures)?;
            instructions.push(Instruction::GetStepTowards);
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
        Expression::Roll { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Roll {
                argument_count: u8::try_from(arguments.len())
                    .expect("roll argument count was validated by the parser"),
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
        Expression::Caller => instructions.push(Instruction::LoadCaller),
        Expression::World => instructions.push(Instruction::LoadGlobal(
            FieldName::parse("world").expect("built-in world global name is valid"),
        )),
        Expression::GlobalNamespace => {
            return Err(compile_error("global namespace requires a field name"));
        }
        Expression::Field { receiver, name } => {
            if name.as_str() == "vars" {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::LoadDatumVars);
            } else if let Some(storage) =
                locals.receiver_static(receiver.as_ref(), name).or_else(|| {
                    matches!(receiver.as_ref(), Expression::Src)
                        .then(|| locals.global_field(name.as_str()))
                        .flatten()
                })
            {
                instructions.push(Instruction::LoadGlobal(storage.clone()));
            } else {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::LoadField(name.clone()));
            }
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
            if name.as_str() == "vars" {
                instructions.push(Instruction::LoadGlobalVars);
            } else {
                instructions.push(Instruction::LoadGlobal(name.clone()));
            }
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
                if let Expression::ArgList(value) = argument {
                    // A single expanded list is already the native ABI used
                    // by list-aware builtins such as min/max.
                    emit_expression(value, locals, instructions, procedures)?;
                } else {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
            }
            // DM source may deliberately replace a global procedure whose name
            // also has an engine fallback (tgstation's /proc/qdel is the
            // important case). A real project procedure wins over the native
            // fallback exactly like any other global proc declaration.
            if let Some(procedure) = procedures.get(name).copied() {
                instructions.push(Instruction::Call {
                    procedure,
                    argument_count,
                });
            } else {
                instructions.push(Instruction::StandardBuiltin {
                    name: name.clone(),
                    argument_count,
                });
            }
        }
        Expression::NativeSrcMethod { name, arguments } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("native method has more than 65535 arguments"))?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::NativeSrcMethod {
                name: name.clone(),
                argument_count,
            });
        }
        Expression::ExternalCall {
            library,
            function,
            arguments,
        } => {
            emit_expression(library, locals, instructions, procedures)?;
            emit_expression(function, locals, instructions, procedures)?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::ExternalCall {
                argument_count: u16::try_from(arguments.len())
                    .map_err(|_| compile_error("external call has more than 65535 arguments"))?,
            });
        }
        Expression::Animate { arguments } => {
            for (_, argument) in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Animate {
                argument_names: arguments.iter().map(|(name, _)| name.clone()).collect(),
            });
        }
        Expression::Filter { arguments } => {
            let mut expanded_indices = Vec::new();
            for (index, (_, argument)) in arguments.iter().enumerate() {
                if let Expression::ArgList(value) = argument {
                    expanded_indices.push(to_local_index(index)?);
                    emit_expression(value, locals, instructions, procedures)?;
                } else {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
            }
            instructions.push(Instruction::MakeFilter {
                argument_names: arguments.iter().map(|(name, _)| name.clone()).collect(),
                expanded_indices,
            });
        }
        Expression::Crash(message) => {
            emit_expression(message, locals, instructions, procedures)?;
            instructions.push(Instruction::Crash);
            // Keep expression stack shape valid for unreachable continuation.
            instructions.push(Instruction::PushNull);
        }
        Expression::Sleep(delay) => {
            emit_expression(delay, locals, instructions, procedures)?;
            instructions.push(Instruction::Sleep);
        }
        Expression::Initial(reference) => match reference.as_ref() {
            Expression::Field { receiver, name } => {
                if let Some(storage) = locals.receiver_static(receiver, name) {
                    // Static initialization is materialized before procedures
                    // run and occupies its qualified persistent slot.
                    instructions.push(Instruction::LoadInitialGlobal(storage.clone()));
                } else {
                    emit_expression(receiver, locals, instructions, procedures)?;
                    instructions.push(Instruction::InitialField(name.clone()));
                }
            }
            Expression::Local(name) => {
                let field = locals.src_field(name).ok_or_else(|| {
                    compile_error(format!("initial target {name:?} is not an instance field"))
                })?;
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::InitialField(field.clone()));
            }
            Expression::SafeField { receiver, name } => {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let null_jump = instructions.len();
                instructions.push(Instruction::JumpIfNull(usize::MAX));
                instructions.push(Instruction::InitialField(name.clone()));
                let end = instructions.len();
                instructions[null_jump] = Instruction::JumpIfNull(end);
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
            null_receiver_is_global,
        } => {
            emit_expression(target, locals, instructions, procedures)?;
            emit_expression(procedure, locals, instructions, procedures)?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic {
                argument_count,
                null_receiver_is_global: *null_receiver_is_global,
            });
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
            instructions.push(Instruction::CallDynamic {
                argument_count,
                null_receiver_is_global: false,
            });
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
        Expression::AssociativeList(entries) => {
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
            instructions.push(Instruction::MakeAssociativeListEntries(kinds));
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
            if operator == "&"
                && let Expression::Local(name) = operand.as_ref()
                && let Some(slot) = locals.get(name)
            {
                instructions.push(Instruction::AddressLocal(slot));
                return Ok(());
            }
            if operator == "*"
                && let Expression::Local(name) = operand.as_ref()
                && let Some(slot) = locals.get(name)
            {
                instructions.push(Instruction::LoadLocalRaw(slot));
                instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
                instructions.push(Instruction::IndexList);
                return Ok(());
            }
            emit_expression(operand, locals, instructions, procedures)?;
            match operator.as_str() {
                "+" => {}
                "-" => instructions.push(Instruction::Negate),
                "!" => instructions.push(Instruction::Not),
                "~" => instructions.push(Instruction::BitNot),
                "&" => instructions.push(Instruction::MakeList(1)),
                "*" => {
                    instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
                    instructions.push(Instruction::IndexList);
                }
                _ => {
                    return Err(compile_error(format!(
                        "unsupported unary operator {operator}"
                    )));
                }
            }
        }
        Expression::Mutation {
            target,
            delta,
            prefix,
        } => emit_mutation_expression(target, *delta, *prefix, locals, instructions, procedures)?,
        Expression::Binary {
            operator,
            left,
            right,
        } => {
            if operator == "&&" {
                emit_expression(left, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let false_jump = instructions.len();
                instructions.push(Instruction::JumpIfFalse(usize::MAX));
                instructions.push(Instruction::Pop);
                emit_expression(right, locals, instructions, procedures)?;
                let end = instructions.len();
                patch_jump(instructions, false_jump, end)?;
            } else if operator == "||" {
                emit_expression(left, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let false_jump = instructions.len();
                instructions.push(Instruction::JumpIfFalse(usize::MAX));
                let end_jump = instructions.len();
                instructions.push(Instruction::Jump(usize::MAX));
                let false_target = instructions.len();
                patch_jump(instructions, false_jump, false_target)?;
                instructions.push(Instruction::Pop);
                emit_expression(right, locals, instructions, procedures)?;
                let end = instructions.len();
                patch_jump(instructions, end_jump, end)?;
            } else {
                emit_expression(left, locals, instructions, procedures)?;
                emit_expression(right, locals, instructions, procedures)?;
                instructions.push(match operator.as_str() {
                    "+" => Instruction::Add,
                    "-" => Instruction::Subtract,
                    "*" => Instruction::Multiply,
                    "**" => Instruction::Power,
                    "/" => Instruction::Divide,
                    "%" => Instruction::Remainder,
                    "%%" => Instruction::FractionalRemainder,
                    "&" => Instruction::BitAnd,
                    "|" => Instruction::BitOr,
                    "^" => Instruction::BitXor,
                    "<<" => Instruction::ShiftLeft,
                    ">>" => Instruction::ShiftRight,
                    "==" => Instruction::Equal,
                    "!=" | "<>" => Instruction::NotEqual,
                    "~=" => Instruction::Equivalent,
                    "~!" => Instruction::NotEquivalent,
                    "<=>" => Instruction::Compare,
                    "in" => Instruction::Contains,
                    "<" => Instruction::Less,
                    "<=" => Instruction::LessEqual,
                    ">" => Instruction::Greater,
                    ">=" => Instruction::GreaterEqual,
                    _ => {
                        return Err(compile_error(format!(
                            "unsupported binary operator {operator}"
                        )));
                    }
                });
            }
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

fn emit_mutation_expression(
    target: &Expression,
    delta: i8,
    prefix: bool,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match target {
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                instructions.push(Instruction::MutateLocal {
                    slot,
                    delta,
                    prefix,
                });
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::MutateField {
                    name: field.clone(),
                    delta,
                    prefix,
                });
            } else if let Some(global) = locals.global_field(name) {
                instructions.push(Instruction::MutateGlobal {
                    name: global.clone(),
                    delta,
                    prefix,
                });
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::GlobalField(name) => instructions.push(Instruction::MutateGlobal {
            name: name.clone(),
            delta,
            prefix,
        }),
        Expression::Field { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::MutateField {
                name: name.clone(),
                delta,
                prefix,
            });
        }
        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::MutateListIndex { delta, prefix });
        }
        Expression::Result => instructions.push(Instruction::MutateResult { delta, prefix }),
        _ => return Err(compile_error("increment/decrement target is not writable")),
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn emit_assignment_expression(
    target: &Expression,
    operator: &str,
    value: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match target {
        Expression::Result => {
            if operator != "=" {
                instructions.push(Instruction::LoadResult);
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreResult);
        }
        Expression::Usr => {
            if operator != "=" {
                return Err(compile_error("usr only supports direct assignment"));
            }
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreUsr);
        }
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
        Expression::Src => {
            if operator != "=" {
                return Err(compile_error("src only supports direct assignment"));
            }
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreSrc);
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
        Expression::World => {}
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
        Expression::Field { receiver, name } | Expression::SafeField { receiver, name } => {
            if let Expression::Local(receiver_name) = receiver.as_ref()
                && let Some(InitializerBinding::Global(storage)) =
                    bindings.get(&format!("{receiver_name}.{}", name.as_str()))
            {
                *expression = Expression::GlobalField(storage.clone());
            } else {
                bind_initializer_expression(receiver, bindings)?;
            }
        }
        Expression::Call { arguments, .. }
        | Expression::StandardBuiltin { arguments, .. }
        | Expression::NativeSrcMethod { arguments, .. }
        | Expression::Regex { arguments }
        | Expression::MutableAppearance { arguments }
        | Expression::Matrix { arguments }
        | Expression::Vector { arguments }
        | Expression::ReplaceText { arguments, .. }
        | Expression::CopyText { arguments, .. }
        | Expression::Block { arguments }
        | Expression::Rand { arguments }
        | Expression::Roll { arguments }
        | Expression::Round { arguments }
        | Expression::Range { arguments }
        | Expression::TypePredicate { arguments, .. }
        | Expression::Locate { arguments } => {
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::ExternalCall {
            library,
            function,
            arguments,
        } => {
            bind_initializer_expression(library, bindings)?;
            bind_initializer_expression(function, bindings)?;
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::Animate { arguments } => {
            for (_, argument) in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::Filter { arguments } => {
            for (_, argument) in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::Length { value }
        | Expression::Ref { value }
        | Expression::TypesOf { value }
        | Expression::Initial(value)
        | Expression::Sleep(value)
        | Expression::Crash(value) => {
            bind_initializer_expression(value, bindings)?;
        }
        Expression::ArgList(value) => bind_initializer_expression(value, bindings)?,
        Expression::GetStep { source, direction } => {
            bind_initializer_expression(source, bindings)?;
            bind_initializer_expression(direction, bindings)?;
        }
        Expression::GetStepTowards { source, target } => {
            bind_initializer_expression(source, bindings)?;
            bind_initializer_expression(target, bindings)?;
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
            overrides,
        } => {
            if let Some(type_path) = type_path {
                bind_initializer_expression(type_path, bindings)?;
            }
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
            for (_, value) in overrides {
                bind_initializer_expression(value, bindings)?;
            }
        }
        Expression::ModifiedTypePath { overrides, .. } => {
            for (_, value) in overrides {
                bind_initializer_expression(value, bindings)?;
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
            ..
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
        Expression::List(entries) | Expression::AssociativeList(entries) => {
            for entry in entries {
                match entry {
                    ListExpressionEntry::Positional(value) => {
                        bind_initializer_expression(value, bindings)?;
                    }
                    ListExpressionEntry::Associative { key, value } => {
                        // A bare key in `list(name = value)` is named-argument
                        // syntax and therefore the text "name", even if an
                        // initializer binding with that spelling exists.
                        let bare_text_key = matches!(key, Expression::Local(_));
                        if !bare_text_key {
                            bind_initializer_expression(key, bindings)?;
                        }
                        bind_initializer_expression(value, bindings)?;
                    }
                }
            }
        }
        Expression::Index { list, index } | Expression::SafeIndex { list, index } => {
            bind_initializer_expression(list, bindings)?;
            bind_initializer_expression(index, bindings)?;
        }
        Expression::Unary { operand, .. }
        | Expression::Mutation {
            target: operand, ..
        } => {
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
        Expression::CurrentCall { .. }
        | Expression::ParentCall { .. }
        | Expression::Result
        | Expression::Caller => {
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
    associative_lists: HashSet<ListId>,
    reference_lists: HashSet<ListId>,
    savefiles: HashMap<DatumId, SavefileState>,
    savefile_entries: HashMap<DatumId, (DatumId, String)>,
    global_vars_proxy: Option<ListId>,
    datum_vars_proxies: HashMap<ListId, DatumId>,
    datum_vars_by_datum: HashMap<DatumId, ListId>,
    contents_owners: HashMap<ListId, DatumId>,
    shared_fields: Arc<BTreeMap<TypePath, BTreeMap<FieldName, FieldName>>>,
    instance_initializers: Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
    instance_initializer_module: Option<Arc<Module>>,
    globals: BTreeMap<FieldName, Value>,
    initial_globals: BTreeMap<FieldName, Value>,
    type_paths: Arc<std::collections::BTreeSet<TypePath>>,
    type_parents: Arc<BTreeMap<TypePath, Option<TypePath>>>,
    initial_values: Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>>,
    project_root: Option<Arc<PathBuf>>,
    random_state: u64,
    scheduler_tick: u64,
    scheduler_sequence: u64,
    scheduled_spawns: Vec<ScheduledSpawn>,
    // Datums whose BYOND `Del()` hook is currently executing. This prevents a
    // reentrant `del(src)` from dispatching the same hook indefinitely.
    deleting_datums: HashSet<DatumId>,
    last_animation_target: Option<Value>,
    environment_overrides: BTreeMap<String, Option<Value>>,
    external_timers: BTreeMap<String, Instant>,
    procedure_static_locals: BTreeMap<(String, u16), Value>,
    /// Authoritative coordinate-to-cell identities for the mutable headless map.
    world_turfs: BTreeMap<(i32, i32, i32), DatumId>,
    world_areas: BTreeMap<(i32, i32, i32), DatumId>,
    default_world_area: Option<DatumId>,
}

#[derive(Clone, Debug, Default)]
struct SavefileState {
    entries: HashMap<String, Value>,
    cd: String,
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
        let mut state = Self {
            heap,
            associative_lists: HashSet::new(),
            reference_lists: HashSet::new(),
            savefiles: HashMap::new(),
            savefile_entries: HashMap::new(),
            global_vars_proxy: None,
            datum_vars_proxies: HashMap::new(),
            datum_vars_by_datum: HashMap::new(),
            contents_owners: HashMap::new(),
            shared_fields: Arc::new(BTreeMap::new()),
            instance_initializers: Arc::new(BTreeMap::new()),
            instance_initializer_module: None,
            globals: BTreeMap::new(),
            initial_globals: BTreeMap::new(),
            type_paths: Arc::new(std::collections::BTreeSet::new()),
            type_parents: Arc::new(BTreeMap::new()),
            initial_values: Arc::new(BTreeMap::new()),
            project_root: None,
            random_state: 0,
            scheduler_tick: 0,
            scheduler_sequence: 0,
            scheduled_spawns: Vec::new(),
            deleting_datums: HashSet::new(),
            last_animation_target: None,
            environment_overrides: BTreeMap::new(),
            external_timers: BTreeMap::new(),
            procedure_static_locals: BTreeMap::new(),
            world_turfs: BTreeMap::new(),
            world_areas: BTreeMap::new(),
            default_world_area: None,
        };
        state.rebuild_world_geometry();
        state
    }

    fn rebuild_world_geometry(&mut self) {
        self.world_turfs.clear();
        self.world_areas.clear();
        self.contents_owners.clear();
        let contents = FieldName::parse("contents").expect("built-in contents field");
        for (id, datum) in self.heap.datums() {
            if let Ok(Value::List(list)) = datum.field(&contents) {
                self.contents_owners.insert(*list, id);
            }
        }
        let x = FieldName::parse("x").expect("built-in coordinate field");
        let y = FieldName::parse("y").expect("built-in coordinate field");
        let z = FieldName::parse("z").expect("built-in coordinate field");
        let loc = FieldName::parse("loc").expect("built-in loc field");
        for (id, datum) in self.heap.datums() {
            let path = datum.type_path().as_str();
            if path != "/turf" && !path.starts_with("/turf/") {
                continue;
            }
            let coordinate = [datum.field(&x), datum.field(&y), datum.field(&z)]
                .map(|value| value.ok().and_then(Value::as_number))
                .map(|value| value.filter(|value| value.is_finite() && value.fract() == 0.0));
            let [Some(x), Some(y), Some(z)] = coordinate else {
                continue;
            };
            #[allow(clippy::cast_possible_truncation)]
            let coordinate = (x as i32, y as i32, z as i32);
            self.world_turfs.insert(coordinate, id);
            if let Ok(Value::Datum(area)) = datum.field(&loc) {
                self.world_areas.insert(coordinate, *area);
            }
        }
    }

    fn world_dimension(&self, world: DatumId, name: &str) -> Result<i32, String> {
        let value = self
            .heap
            .datum_field(
                world,
                &FieldName::parse(name).expect("built-in world dimension field"),
            )
            .ok()
            .and_then(Value::as_number)
            .unwrap_or(1.0);
        if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > i32::MAX as f32 {
            return Err(format!(
                "world.{name} must be a positive integer, received {value}"
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        Ok(value as i32)
    }

    fn world_type_field(
        &self,
        world: DatumId,
        name: &str,
        fallback: &str,
    ) -> Result<TypePath, String> {
        match self.heap.datum_field(
            world,
            &FieldName::parse(name).expect("built-in world type field"),
        ) {
            Ok(Value::TypePath(path)) => Ok(path.clone()),
            Ok(Value::ModifiedTypePath(path)) => Ok(path.base().clone()),
            Ok(Value::Null) | Err(_) => {
                TypePath::parse(fallback).map_err(|error| error.to_string())
            }
            Ok(value) => Err(format!(
                "world.{name} must be a type path, received {value}"
            )),
        }
    }

    fn ensure_contents(&mut self, datum: DatumId) -> Result<ListId, String> {
        let contents = FieldName::parse("contents").expect("built-in contents field");
        if let Ok(Value::List(list)) = self.heap.datum_field(datum, &contents) {
            self.contents_owners.insert(*list, datum);
            return Ok(*list);
        }
        let list = self.heap.allocate_list();
        self.heap
            .set_datum_field(datum, contents, Value::List(list))
            .map_err(|error| error.to_string())?;
        self.contents_owners.insert(list, datum);
        Ok(list)
    }

    pub(crate) fn contents_owner(&self, list: ListId) -> Option<DatumId> {
        self.contents_owners.get(&list).copied()
    }

    fn default_area_for_world(&mut self, world: DatumId) -> Result<DatumId, String> {
        if let Some(area) = self.default_world_area
            && self.heap.datum(area).is_ok()
        {
            return Ok(area);
        }
        let path = self.world_type_field(world, "area", "/area")?;
        let existing = self
            .heap
            .datums()
            .find_map(|(id, datum)| (datum.type_path() == &path).then_some(id));
        let area = match existing {
            Some(area) => area,
            None => allocate_initialized_datum(self, path)?,
        };
        self.ensure_contents(area)?;
        let world_contents = self.ensure_contents(world)?;
        let contents = self
            .heap
            .list_mut(world_contents)
            .map_err(|error| error.to_string())?;
        if !contents.contains(&Value::Datum(area)) {
            contents.add(Value::Datum(area));
        }
        self.default_world_area = Some(area);
        Ok(area)
    }

    fn remove_world_cell(
        &mut self,
        world: DatumId,
        coordinate: (i32, i32, i32),
    ) -> Result<(), String> {
        let Some(turf) = self.world_turfs.remove(&coordinate) else {
            self.world_areas.remove(&coordinate);
            return Ok(());
        };
        if let Some(area) = self.world_areas.remove(&coordinate) {
            let contents = self.ensure_contents(area)?;
            self.heap
                .list_mut(contents)
                .map_err(|error| error.to_string())?
                .remove_first(&Value::Datum(turf));
        }
        let contents = self.ensure_contents(world)?;
        self.heap
            .list_mut(contents)
            .map_err(|error| error.to_string())?
            .remove_first(&Value::Datum(turf));
        self.heap
            .destroy_datum(turf)
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub(crate) fn resize_world_geometry(
        &mut self,
        world: DatumId,
        dimensions: (i32, i32, i32),
    ) -> Result<(), String> {
        let (maxx, maxy, maxz) = dimensions;
        if maxx < 1 || maxy < 1 || maxz < 1 {
            return Err("world dimensions must be positive integers".to_owned());
        }
        let removed = self
            .world_turfs
            .keys()
            .copied()
            .filter(|(x, y, z)| *x > maxx || *y > maxy || *z > maxz)
            .collect::<Vec<_>>();
        for coordinate in removed {
            self.remove_world_cell(world, coordinate)?;
        }

        let area = self.default_area_for_world(world)?;
        let turf_path = self.world_type_field(world, "turf", "/turf")?;
        let area_contents = self.ensure_contents(area)?;
        let world_contents = self.ensure_contents(world)?;
        let coordinate_fields = ["x", "y", "z"]
            .map(|name| FieldName::parse(name).expect("built-in coordinate field is valid"));
        let loc = FieldName::parse("loc").expect("built-in loc field is valid");
        for z in 1..=maxz {
            for y in 1..=maxy {
                for x in 1..=maxx {
                    let coordinate = (x, y, z);
                    if self.world_turfs.contains_key(&coordinate) {
                        continue;
                    }
                    let turf = allocate_initialized_datum(self, turf_path.clone())?;
                    for (field, value) in coordinate_fields.iter().zip([x, y, z]) {
                        self.heap
                            .set_datum_field(turf, field.clone(), Value::number(value as f32))
                            .map_err(|error| error.to_string())?;
                    }
                    self.heap
                        .set_datum_field(turf, loc.clone(), Value::Datum(area))
                        .map_err(|error| error.to_string())?;
                    self.ensure_contents(turf)?;
                    self.heap
                        .list_mut(area_contents)
                        .map_err(|error| error.to_string())?
                        .add(Value::Datum(turf));
                    self.heap
                        .list_mut(world_contents)
                        .map_err(|error| error.to_string())?
                        .add(Value::Datum(turf));
                    self.world_turfs.insert(coordinate, turf);
                    self.world_areas.insert(coordinate, area);
                }
            }
        }
        for (name, value) in [("maxx", maxx), ("maxy", maxy), ("maxz", maxz)] {
            self.heap
                .set_datum_field(
                    world,
                    FieldName::parse(name).expect("built-in world dimension field"),
                    Value::number(value as f32),
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn turf_at(&self, x: i32, y: i32, z: i32) -> Option<DatumId> {
        self.world_turfs.get(&(x, y, z)).copied()
    }

    fn note_turf_area(&mut self, turf: DatumId, area: DatumId) {
        let coordinate = ["x", "y", "z"]
            .map(|name| FieldName::parse(name).expect("built-in coordinate field"))
            .map(|field| {
                self.heap
                    .datum_field(turf, &field)
                    .ok()
                    .and_then(Value::as_number)
            });
        let [Some(x), Some(y), Some(z)] = coordinate else {
            return;
        };
        if [x, y, z]
            .into_iter()
            .any(|value| !value.is_finite() || value.fract() != 0.0)
        {
            return;
        }
        #[allow(clippy::cast_possible_truncation)]
        self.world_areas
            .insert((x as i32, y as i32, z as i32), area);
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

    pub(crate) fn environment_override(&self, name: &str) -> Option<&Option<Value>> {
        self.environment_overrides.get(name)
    }

    pub(crate) fn set_environment_override(&mut self, name: String, value: Option<Value>) {
        self.environment_overrides.insert(name, value);
    }

    pub(crate) fn reset_external_timer(&mut self, name: String) {
        self.external_timers.insert(name, Instant::now());
    }

    pub(crate) fn external_timer_milliseconds(&self, name: &str) -> f64 {
        self.external_timers
            .get(name)
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
    }

    pub(crate) fn is_associative_list(&self, list: ListId) -> bool {
        self.associative_lists.contains(&list)
    }

    pub(crate) fn mark_associative_list(&mut self, list: ListId) {
        self.associative_lists.insert(list);
    }

    pub(crate) fn refresh_vars_proxy(&mut self, list: ListId) -> Result<(), String> {
        let Some(datum) = self.datum_vars_proxies.get(&list).copied() else {
            return Ok(());
        };
        let keys = self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, key)| key.clone())
            .collect::<Vec<_>>();
        for key in keys {
            let Value::Text(name) = &key else {
                continue;
            };
            let field = FieldName::parse(name).map_err(|error| error.to_string())?;
            let value = datum_shared_storage(self, datum, &field)
                .and_then(|storage| self.global(&storage).cloned())
                .or_else(|| self.heap.datum_field(datum, &field).ok().cloned())
                .unwrap_or(Value::Null);
            self.heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .set_key(key, value);
        }
        Ok(())
    }

    /// Reads a persistent runtime global.
    #[must_use]
    pub fn global(&self, name: &FieldName) -> Option<&Value> {
        self.globals.get(name)
    }

    /// Inserts or replaces a persistent runtime global.
    pub fn set_global(&mut self, name: FieldName, value: Value) -> Option<Value> {
        let is_new = !self.globals.contains_key(&name);
        let previous = self.globals.insert(name.clone(), value);
        if is_new
            && let Some(list) = self.global_vars_proxy
            && let Ok(values) = self.heap.list_mut(list)
        {
            values.add(Value::text(name.as_str()));
        }
        previous
    }

    /// Records a declaration-time global/static value used by `initial()`.
    pub fn set_initial_global(&mut self, name: FieldName, value: Value) -> Option<Value> {
        self.initial_globals.insert(name, value)
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

    /// Replaces the runtime type-parent catalog with shared immutable metadata.
    pub fn set_shared_type_parents(&mut self, parents: Arc<BTreeMap<TypePath, Option<TypePath>>>) {
        self.type_parents = parents;
    }

    /// Replaces effective compile-time initial field values for every runtime type.
    pub fn set_initial_values(&mut self, values: BTreeMap<TypePath, BTreeMap<FieldName, Value>>) {
        self.initial_values = Arc::new(values);
    }

    /// Replaces effective initial values with shared immutable metadata.
    pub fn set_shared_initial_values(
        &mut self,
        values: Arc<BTreeMap<TypePath, BTreeMap<FieldName, Value>>>,
    ) {
        self.initial_values = values;
    }

    /// Installs inherited reflection names for owner-qualified shared fields.
    pub fn set_shared_fields(
        &mut self,
        fields: Arc<BTreeMap<TypePath, BTreeMap<FieldName, FieldName>>>,
    ) {
        self.shared_fields = fields;
    }

    /// Installs direct per-type initializer programs used by runtime `new`.
    pub fn set_instance_initializers(
        &mut self,
        initializers: Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
        module: Option<Arc<Module>>,
    ) {
        self.instance_initializers = initializers;
        self.instance_initializer_module = module;
    }

    /// Replaces runtime-new initializer metadata and returns the previous catalog.
    pub fn replace_instance_initializers(
        &mut self,
        initializers: Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>>,
    ) -> Arc<BTreeMap<TypePath, Vec<InstanceInitializer>>> {
        std::mem::replace(&mut self.instance_initializers, initializers)
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

    /// Returns the current deterministic scheduler tick.
    #[must_use]
    pub const fn scheduler_tick(&self) -> u64 {
        self.scheduler_tick
    }

    /// Returns the number of suspended or spawned tasks awaiting dispatch.
    #[must_use]
    pub const fn scheduled_task_count(&self) -> usize {
        self.scheduled_spawns.len()
    }

    /// Returns the earliest tick at which pending scheduler work is due.
    #[must_use]
    pub fn next_scheduled_tick(&self) -> Option<u64> {
        self.scheduled_spawns.iter().map(|task| task.due_tick).min()
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

#[derive(Clone, Debug)]
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
    exception_handlers: Vec<ExceptionHandler>,
    // A waitfor=FALSE boundary detaches from its caller only once. Later
    // sleeps in the already-detached continuation yield normally.
    detached_waitfor: bool,
    static_locals: HashSet<u16>,
}

#[derive(Clone, Debug)]
struct ExceptionHandler {
    start: usize,
    end: usize,
    catch: usize,
    local: Option<u16>,
    stack_depth: usize,
}

#[derive(Clone, Debug)]
struct ScheduledSpawn {
    due_tick: u64,
    sequence: u64,
    frames: Vec<CallFrame>,
}

enum FrameRunOutcome {
    Complete(Value),
    Yielded { frames: Vec<CallFrame>, delay: u64 },
}

fn materialize_callee_chain(
    module: &Module,
    state: &mut ExecutionState,
    callers: &[CallFrame],
) -> Result<Value, String> {
    let callee_path = TypePath::parse("/callee").expect("built-in /callee path");
    let mut previous = Value::Null;
    for frame in callers {
        let args = state.heap.allocate_list();
        for argument in &frame.arguments {
            state
                .heap
                .list_mut(args)
                .map_err(|error| error.to_string())?
                .add(argument.clone());
        }
        let datum = state.heap.allocate_datum(callee_path.clone());
        let procedure = module
            .procedure_path(frame.procedure)
            .unwrap_or("/proc")
            .split('@')
            .next()
            .unwrap_or("/proc");
        let procedure_value = TypePath::parse(procedure)
            .map(Value::TypePath)
            .unwrap_or_else(|_| Value::text(procedure));
        for (name, value) in [
            ("caller", previous.clone()),
            ("src", frame.src.clone()),
            ("usr", frame.usr.clone()),
            ("args", Value::List(args)),
            ("type", procedure_value),
            ("file", Value::Null),
            ("line", Value::number(0.0)),
        ] {
            state
                .heap
                .set_datum_field(
                    datum,
                    FieldName::parse(name).expect("built-in callee field"),
                    value,
                )
                .map_err(|error| error.to_string())?;
        }
        previous = Value::Datum(datum);
    }
    Ok(previous)
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
        deferred: Arc::new(HashMap::new()),
        procedure_types: BTreeSet::new(),
        initializer_call_names: None,
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
        deferred: Arc::new(HashMap::new()),
        procedure_types: BTreeSet::new(),
        initializer_call_names: None,
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
    let program = module
        .resolve_procedure(entry)
        .map_err(|message| RuntimeError {
            message,
            instruction: 0,
            source_span: None,
            call_stack: Vec::new(),
        })?;
    if limits.max_call_depth == 0 {
        return Err(RuntimeError {
            message: "maximum call depth must be at least one".to_owned(),
            instruction: 0,
            source_span: program.source_spans.first().copied(),
            call_stack: vec![trace(module, entry, 0)],
        });
    }

    let frames = vec![make_frame(entry, program, arguments, context)];
    finish_frame_run(module, run_frames(module, frames, limits, state)?, state)
}

fn finish_frame_run(
    module: &Module,
    outcome: FrameRunOutcome,
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    match outcome {
        FrameRunOutcome::Complete(value) => Ok(value),
        FrameRunOutcome::Yielded { frames, delay } => {
            schedule_frames(state, frames, delay);
            let _ = module;
            Ok(Value::Null)
        }
    }
}

/// Advances the deterministic scheduler and runs every spawned body whose
/// delay has elapsed. Tasks with the same deadline retain source order.
///
/// # Errors
///
/// Returns a runtime error when a due spawned body fails.
pub fn advance_scheduler(
    module: &Module,
    ticks: u64,
    limits: ExecutionLimits,
    state: &mut ExecutionState,
) -> Result<Vec<Value>, RuntimeError> {
    state.scheduler_tick = state.scheduler_tick.saturating_add(ticks);
    advance_headless_world_clock(state, ticks);
    state
        .scheduled_spawns
        .sort_by_key(|spawn| (spawn.due_tick, spawn.sequence));
    let due_count = state
        .scheduled_spawns
        .partition_point(|spawn| spawn.due_tick <= state.scheduler_tick);
    let due = state
        .scheduled_spawns
        .drain(..due_count)
        .collect::<Vec<_>>();
    // BYOND exposes elapsed host-tick percentage. The deterministic headless
    // VM has no wall-clock budget, so due work observes one fully active tick
    // and the quiescent boundary observes zero.
    set_world_numeric_field(
        state,
        "tick_usage",
        if due.is_empty() { 0.0 } else { 100.0 },
    );
    let mut completed = Vec::new();
    for spawn in due {
        let outcome = match run_frames(module, spawn.frames, limits, state) {
            Ok(outcome) => outcome,
            Err(error) => {
                set_world_numeric_field(state, "tick_usage", 0.0);
                return Err(error);
            }
        };
        match outcome {
            FrameRunOutcome::Complete(value) => completed.push(value),
            FrameRunOutcome::Yielded { frames, delay } => schedule_frames(state, frames, delay),
        }
    }
    set_world_numeric_field(state, "tick_usage", 0.0);
    Ok(completed)
}

fn schedule_frames(state: &mut ExecutionState, frames: Vec<CallFrame>, delay: u64) {
    let sequence = state.scheduler_sequence;
    state.scheduler_sequence = state.scheduler_sequence.saturating_add(1);
    let tick_lag = world_numeric_field(state, "tick_lag")
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    let delay_ticks = if delay == 0 {
        0
    } else {
        ((delay as f64) / f64::from(tick_lag)).ceil() as u64
    };
    state.scheduled_spawns.push(ScheduledSpawn {
        due_tick: state.scheduler_tick.saturating_add(delay_ticks),
        sequence,
        frames,
    });
}

fn world_datum(state: &ExecutionState) -> Option<DatumId> {
    state
        .global(&FieldName::parse("world").expect("built-in world global"))
        .and_then(|value| match value {
            Value::Datum(world) => Some(*world),
            _ => None,
        })
}

fn world_numeric_field(state: &ExecutionState, name: &str) -> Option<f32> {
    state
        .heap
        .datum_field(world_datum(state)?, &FieldName::parse(name).ok()?)
        .ok()?
        .as_number()
}

fn set_world_numeric_field(state: &mut ExecutionState, name: &str, value: f32) {
    let Some(world) = world_datum(state) else {
        return;
    };
    let _ = state.heap.set_datum_field(
        world,
        FieldName::parse(name).expect("built-in world numeric field"),
        Value::number(value),
    );
}

fn advance_headless_world_clock(state: &mut ExecutionState, ticks: u64) {
    if ticks == 0 {
        return;
    }
    let tick_lag = world_numeric_field(state, "tick_lag")
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    let elapsed = (ticks as f64 * f64::from(tick_lag)) as f32;
    let time = world_numeric_field(state, "time").unwrap_or(0.0) + elapsed;
    let timeofday =
        (world_numeric_field(state, "timeofday").unwrap_or(0.0) + elapsed).rem_euclid(864_000.0);
    set_world_numeric_field(state, "time", time);
    set_world_numeric_field(state, "timeofday", timeofday);
}

#[allow(clippy::too_many_lines)]
fn run_frames(
    module: &Module,
    mut frames: Vec<CallFrame>,
    limits: ExecutionLimits,
    state: &mut ExecutionState,
) -> Result<FrameRunOutcome, RuntimeError> {
    let mut remaining_steps = limits.max_steps;
    loop {
        let frame_index = frames.len() - 1;
        let procedure = frames[frame_index].procedure;
        let instruction_index = frames[frame_index].instruction;
        let program = module
            .resolve_procedure(procedure)
            .map_err(|message| execution_error(module, &frames, message))?;
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
            Instruction::MakeModifiedTypePath { fields } => {
                let stack = &mut frames[frame_index].stack;
                if stack.len() < fields.len() + 1 {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let values_start = stack.len() - fields.len();
                let base_index = values_start - 1;
                let Value::TypePath(base) = stack[base_index].clone() else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "modified type requires a base type path",
                    ));
                };
                let overrides = fields
                    .iter()
                    .cloned()
                    .zip(stack[values_start..].iter().cloned())
                    .collect();
                stack.truncate(base_index);
                stack.push(Value::ModifiedTypePath(Arc::new(ModifiedTypePath::new(
                    base, overrides,
                ))));
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
                        // BYOND treats `arglist(null)` as an empty argument
                        // vector. Callback.Invoke relies on this when neither
                        // its constructor nor invocation supplied arguments.
                        if matches!(value, Value::Null) {
                            continue;
                        }
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
                let (type_path, overrides) = match stack[type_path_index].clone() {
                    Value::TypePath(path) => (path, None),
                    Value::ModifiedTypePath(modified) => (modified.base().clone(), Some(modified)),
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("new requires a type path, received {value}"),
                        ));
                    }
                };
                let arguments = stack[arguments_start..].to_vec();
                stack.truncate(type_path_index);
                let is_movable = builtins::is_movable_path(type_path.as_str());
                let allocated = if type_path.as_str() == "/list" {
                    Value::List(
                        construct_sized_list(&arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?,
                    )
                } else {
                    let datum = if type_path.as_str() == "/matrix" {
                        construct_matrix(&arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    } else if type_path.as_str() == "/vector" {
                        construct_vector(&arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    } else if type_path.as_str() == "/regex" {
                        let datum = allocate_initialized_datum(state, type_path.clone())
                            .map_err(|message| execution_error(module, &frames, message))?;
                        for (name, value) in [
                            ("text", arguments.first().cloned().unwrap_or(Value::Null)),
                            ("flags", arguments.get(1).cloned().unwrap_or(Value::Null)),
                            ("match", Value::Null),
                            ("index", Value::number(0.0)),
                            ("group", Value::Null),
                        ] {
                            state
                                .heap_mut()
                                .set_datum_field(
                                    datum,
                                    FieldName::parse(name).expect("regex field is valid"),
                                    value,
                                )
                                .map_err(|error| {
                                    execution_error(module, &frames, error.to_string())
                                })?;
                        }
                        datum
                    } else {
                        let datum =
                            allocate_or_replace_engine_datum(state, type_path.clone(), &arguments)
                                .map_err(|message| execution_error(module, &frames, message))?;
                        datum
                    };
                    Value::Datum(datum)
                };
                if let (Value::Datum(datum), Some(modified)) = (&allocated, overrides) {
                    for (field, value) in modified.overrides() {
                        state
                            .heap_mut()
                            .set_datum_field(*datum, field.clone(), value.clone())
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    }
                }
                if let Value::Datum(datum) = &allocated {
                    if is_movable
                        && let Some(Value::Datum(location)) = arguments.first()
                        && state.heap.datum(*location).is_ok_and(|datum| {
                            let path = datum.type_path().as_str();
                            path == "/turf" || path.starts_with("/turf/")
                        })
                    {
                        builtins::move_movable_to_turf(state, *datum, *location)
                            .map_err(|message| execution_error(module, &frames, message))?;
                    }
                    invoke_constructor_if_present(
                        module,
                        state,
                        *datum,
                        &arguments,
                        &frame_context(&frames[frame_index]),
                    )
                    .map_err(|mut error| {
                        error.call_stack.insert(
                            0,
                            trace(
                                module,
                                frames[frame_index].procedure,
                                frames[frame_index].instruction.saturating_sub(1),
                            ),
                        );
                        error
                    })?;
                }
                frames[frame_index].stack.push(allocated);
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
                let arguments_start = frames[frame_index].stack.len() - count;
                let arguments = frames[frame_index].stack[arguments_start..].to_vec();
                frames[frame_index].stack.truncate(arguments_start);
                let datum = allocate_or_replace_engine_datum(state, type_path.clone(), &arguments)
                    .map_err(|message| execution_error(module, &frames, message))?;
                if builtins::is_movable_path(type_path.as_str())
                    && let Some(Value::Datum(location)) = arguments.first()
                    && state.heap.datum(*location).is_ok_and(|datum| {
                        let path = datum.type_path().as_str();
                        path == "/turf" || path.starts_with("/turf/")
                    })
                {
                    builtins::move_movable_to_turf(state, datum, *location)
                        .map_err(|message| execution_error(module, &frames, message))?;
                }
                invoke_constructor_if_present(
                    module,
                    state,
                    datum,
                    &arguments,
                    &frame_context(&frames[frame_index]),
                )
                .map_err(|mut error| {
                    error.call_stack.insert(
                        0,
                        trace(
                            module,
                            frames[frame_index].procedure,
                            frames[frame_index].instruction.saturating_sub(1),
                        ),
                    );
                    error
                })?;
                frames[frame_index].stack.push(Value::Datum(datum));
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
            Instruction::MakeMatrix { argument_count } => {
                let count = usize::from(argument_count);
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid matrix constructor stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let datum = construct_matrix(&arguments, &mut state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::Datum(datum));
            }
            Instruction::MakeVector { argument_count } => {
                let count = usize::from(argument_count);
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid vector constructor stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let datum = construct_vector(&arguments, &mut state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::Datum(datum));
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
                let value = if name == "del" {
                    execute_del(
                        module,
                        &arguments,
                        state,
                        &frame_context(&frames[frame_index]),
                    )?
                } else {
                    execute_standard_builtin(&name, &arguments, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::NativeSrcMethod {
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
                let Value::Datum(src) = frames[frame_index].src else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("native method {name} requires a datum src"),
                    ));
                };
                let value = match name.as_str() {
                    "MapColors" if is_icon_datum(src, &state.heap) => {
                        apply_icon_map_colors(src, &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?;
                        Value::Null
                    }
                    "Blend" if is_icon_datum(src, &state.heap) => {
                        apply_icon_blend(src, &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?;
                        Value::Null
                    }
                    "SetIntensity" if is_icon_datum(src, &state.heap) => {
                        apply_icon_set_intensity(src, &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?;
                        Value::Null
                    }
                    method if is_icon_datum(src, &state.heap) => {
                        execute_icon_method(src, method, &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    }
                    "Turn" if is_matrix_datum(src, &state.heap) => {
                        execute_matrix_method(src, "Turn", &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    }
                    _ => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("unknown native method {name} for src"),
                        ));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Output => {
                let value = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let target = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                if let Value::Datum(entry) = target
                    && let Some((savefile, key)) = state.savefile_entries.get(&entry).cloned()
                {
                    state
                        .savefiles
                        .entry(savefile)
                        .or_default()
                        .entries
                        .insert(key, value);
                } else {
                    execute_output(&target, &value, state)
                        .map_err(|message| execution_error(module, &frames, message))?;
                }
            }
            Instruction::Input => {
                let target = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = match target {
                    Value::Datum(entry) if state.savefile_entries.contains_key(&entry) => {
                        let (savefile, key) = state.savefile_entries[&entry].clone();
                        state
                            .savefiles
                            .get(&savefile)
                            .and_then(|savefile| savefile.entries.get(&key))
                            .cloned()
                            .unwrap_or(Value::Null)
                    }
                    Value::Datum(savefile)
                        if state.heap.datum(savefile).is_ok_and(|datum| {
                            let path = datum.type_path().as_str();
                            path == "/savefile" || path.starts_with("/savefile/")
                        }) =>
                    {
                        let savefile = state.savefiles.entry(savefile).or_default();
                        let key = if savefile.cd.is_empty() {
                            "/"
                        } else {
                            &savefile.cd
                        };
                        savefile.entries.get(key).cloned().unwrap_or(Value::Null)
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("savefile input received {value}"),
                        ));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::ExternalCall { argument_count } => {
                let count = usize::from(argument_count) + 2;
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(
                        module,
                        &frames,
                        "external call stack underflow",
                    ));
                }
                let values = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = execute_external_call(&values[0], &values[1], &values[2..], state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::Animate { argument_names } => {
                let count = argument_names.len();
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(module, &frames, "animate stack underflow"));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = execute_animate(&argument_names, &arguments, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::MakeFilter {
                argument_names,
                expanded_indices,
            } => {
                let count = argument_names.len();
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(module, &frames, "filter stack underflow"));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let filter = state
                    .heap_mut()
                    .allocate_datum(TypePath::parse("/dm_filter").expect("canonical filter path"));
                let mut fields = Vec::new();
                for (index, (name, value)) in argument_names.iter().zip(arguments).enumerate() {
                    if expanded_indices
                        .binary_search(
                            &to_local_index(index).expect("filter argument count is u16"),
                        )
                        .is_ok()
                    {
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
                        fields.extend(list.associations().filter_map(|(key, value)| match key {
                            Value::Text(key) => Some((key.to_string(), value.clone())),
                            _ => None,
                        }));
                        continue;
                    }
                    let field = name.clone().unwrap_or_else(|| {
                        if index == 0 {
                            "type".to_owned()
                        } else {
                            format!("arg{}", index + 1)
                        }
                    });
                    fields.push((field, value));
                }
                for (field, value) in fields {
                    state
                        .heap_mut()
                        .set_datum_field(
                            filter,
                            FieldName::parse(&field).map_err(|error| {
                                execution_error(module, &frames, error.to_string())
                            })?,
                            value,
                        )
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                }
                frames[frame_index].stack.push(Value::Datum(filter));
            }
            Instruction::Sleep => {
                let delay = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let delay = delay.as_number().ok_or_else(|| {
                    execution_error(
                        module,
                        &frames,
                        format!("sleep delay must be numeric, received {delay}"),
                    )
                })?;
                let delay = if delay.is_finite() && delay > 0.0 {
                    (delay.floor() as u64).max(1)
                } else {
                    0
                };
                frames[frame_index].stack.push(Value::Null);
                frames[frame_index].instruction += 1;
                if let Some(detach_at) = frames.iter().rposition(|frame| {
                    !frame.detached_waitfor
                        && module
                            .procedure(frame.procedure)
                            .is_some_and(|program| !program.wait_for)
                }) {
                    let detached_result = frames[detach_at].result.clone();
                    let mut detached = frames.split_off(detach_at);
                    detached[0].detached_waitfor = true;
                    schedule_frames(state, detached, delay);
                    if let Some(caller) = frames.last_mut() {
                        // The caller continues exactly as if the waitfor=0
                        // procedure returned its current `.` value. The
                        // detached continuation's eventual return is ignored.
                        caller.stack.push(detached_result);
                        caller.instruction += 1;
                        continue;
                    }
                    return Ok(FrameRunOutcome::Complete(detached_result));
                }
                return Ok(FrameRunOutcome::Yielded { frames, delay });
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
            Instruction::GetStepTowards => {
                let target = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let source = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let direction = direction_towards_builtin(&source, &target, &state.heap)
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
                let paths = if let Value::TypePath(root) = &selector
                    && (root.as_str() == "/proc" || root.as_str().ends_with("/proc"))
                {
                    let prefix = format!("{}/", root.as_str());
                    module
                        .procedure_types
                        .iter()
                        .filter(|path| {
                            path.as_str() == root.as_str() || path.as_str().starts_with(&prefix)
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                } else {
                    typesof_builtin(&selector, &state.heap, &state.type_paths)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
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
            Instruction::TypeInstances(target) => {
                let matches = state
                    .heap
                    .datums()
                    .filter(|(_, datum)| is_subtype(state, datum.type_path(), &target))
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>();
                let list = state.heap.allocate_list();
                for datum in matches {
                    state
                        .heap
                        .list_mut(list)
                        .expect("new type-instance list is live")
                        .add(Value::Datum(datum));
                }
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::Rand { argument_count } => {
                let count = usize::from(argument_count);
                if count > 2 || frames[frame_index].stack.len() < count {
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
            Instruction::Roll { argument_count } => {
                let count = usize::from(argument_count);
                if !(1..=2).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid roll builtin stack",
                    ));
                }
                let stack_length = frames[frame_index].stack.len();
                let arguments = frames[frame_index].stack.split_off(stack_length - count);
                let value = roll_dice(&arguments, &mut state.random_state)
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
            Instruction::MakeArray(dimension_count) => {
                let count = usize::from(dimension_count);
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let stack_len = frames[frame_index].stack.len();
                let values = frames[frame_index].stack.split_off(stack_len - count);
                let mut sizes = Vec::with_capacity(count);
                for value in values {
                    let Some(size) = value.as_number() else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "array dimension must be numeric",
                        ));
                    };
                    sizes.push(size.max(0.0).floor() as usize);
                }
                let array = allocate_dm_array(&mut state.heap, &sizes, 0);
                frames[frame_index].stack.push(Value::List(array));
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
            Instruction::MakeAssociativeListEntries(kinds) => {
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
                        "alist literal is too large",
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
                let entries = state.heap.list_mut(list).expect("new alist is live");
                let mut values = values.into_iter();
                for kind in kinds {
                    match kind {
                        ListEntryKind::Positional => {
                            let key = values.next().expect("alist entry count was validated");
                            entries.set_key(key, Value::Null);
                        }
                        ListEntryKind::Associative => {
                            let key = values.next().expect("alist key count was validated");
                            let value = values.next().expect("alist value count was validated");
                            entries.set_key(key, value);
                        }
                    }
                }
                state.mark_associative_list(list);
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::IndexList => {
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if let Value::Datum(savefile) = receiver
                    && state.heap.datum(savefile).is_ok_and(|datum| {
                        let path = datum.type_path().as_str();
                        path == "/savefile" || path.starts_with("/savefile/")
                    })
                {
                    let key = match key {
                        Value::Text(key) => key.to_string(),
                        value => value.to_string(),
                    };
                    let key = savefile_resolve_path(
                        &state.savefiles.entry(savefile).or_default().cd,
                        &key,
                    );
                    let entry = state
                        .heap
                        .allocate_datum(TypePath::parse("/savefile/entry").unwrap());
                    state.savefile_entries.insert(entry, (savefile, key));
                    frames[frame_index].stack.push(Value::Datum(entry));
                    frames[frame_index].instruction += 1;
                    continue;
                }
                let list = match receiver {
                    Value::List(list) => list,
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list index operation received {value}"),
                        ));
                    }
                };
                let value = if state.global_vars_proxy == Some(list) {
                    match &key {
                        Value::Text(name) => FieldName::parse(name)
                            .ok()
                            .and_then(|name| state.global(&name).cloned())
                            .unwrap_or(Value::Null),
                        _ => read_list_value(&state.heap, list, &key)
                            .cloned()
                            .unwrap_or(Value::Null),
                    }
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    match &key {
                        Value::Text(name) => {
                            let field = FieldName::parse(name).ok();
                            let shared = field
                                .as_ref()
                                .and_then(|field| datum_shared_storage(state, datum, field));
                            shared
                                .and_then(|storage| state.global(&storage).cloned())
                                .or_else(|| {
                                    field.and_then(|field| {
                                        state.heap.datum_field(datum, &field).ok().cloned()
                                    })
                                })
                                .unwrap_or(Value::Null)
                        }
                        _ => read_list_value(&state.heap, list, &key)
                            .cloned()
                            .unwrap_or(Value::Null),
                    }
                } else {
                    match read_list_value(&state.heap, list, &key) {
                        Ok(value) => value.clone(),
                        // BYOND associative lookup returns null for an absent key.
                        // Lazy-list idioms such as `lists[target] ||= list()` rely
                        // on this before inserting the new association.
                        Err(ValueError::MissingKey) => Value::Null,
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
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
                if state.global_vars_proxy == Some(list) {
                    let Value::Text(name) = key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "global.vars writes require a text key",
                        ));
                    };
                    let name = FieldName::parse(&name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    state.set_global(name, value);
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    write_datum_vars(state, datum, list, key, value)
                        .map_err(|message| execution_error(module, &frames, message))?;
                } else if let Err(error) = write_list_value(&mut state.heap, list, key, value) {
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
                if state.global_vars_proxy == Some(list) {
                    let Value::Text(name) = key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "global.vars writes require a text key",
                        ));
                    };
                    let name = FieldName::parse(&name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    state.set_global(name, value.clone());
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    write_datum_vars(state, datum, list, key, value.clone())
                        .map_err(|message| execution_error(module, &frames, message))?;
                } else if let Err(error) =
                    write_list_value(&mut state.heap, list, key, value.clone())
                {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
                frames[frame_index].stack.push(value);
            }
            Instruction::CompoundListIndex(operator) => {
                let right = match pop(&mut frames[frame_index].stack) {
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
                let current = if state.global_vars_proxy == Some(list) {
                    let Value::Text(name) = &key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "global.vars writes require a text key",
                        ));
                    };
                    FieldName::parse(name)
                        .ok()
                        .and_then(|name| state.global(&name).cloned())
                        .unwrap_or(Value::Null)
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    let Value::Text(name) = &key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "datum.vars writes require a text key",
                        ));
                    };
                    let field = FieldName::parse(name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    datum_shared_storage(state, datum, &field)
                        .and_then(|storage| state.global(&storage).cloned())
                        .or_else(|| state.heap.datum_field(datum, &field).ok().cloned())
                        .unwrap_or(Value::Null)
                } else {
                    match read_list_value(&state.heap, list, &key) {
                        Ok(value) => value.clone(),
                        Err(ValueError::MissingKey) => Value::Null,
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    }
                };
                let value = match (&current, &right, operator) {
                    (Value::Null, Value::List(_), CompoundListIndexOperator::Add) => right,
                    (Value::List(current), _, CompoundListIndexOperator::Add) => {
                        execute_list_compound_operator(
                            CompoundAssignmentOperator::Add,
                            *current,
                            &right,
                            state,
                        )
                        .map_err(|message| execution_error(module, &frames, message))?
                    }
                    _ => {
                        let left = current.as_number().ok_or_else(|| {
                            execution_error(
                                module,
                                &frames,
                                format!("numeric operation received {current}"),
                            )
                        })?;
                        let right = right.as_number().ok_or_else(|| {
                            execution_error(
                                module,
                                &frames,
                                format!("numeric operation received {right}"),
                            )
                        })?;
                        Value::number(execute_compound_list_index_operation(operator, left, right))
                    }
                };
                if state.global_vars_proxy == Some(list) {
                    let Value::Text(name) = key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "global.vars writes require a text key",
                        ));
                    };
                    let name = FieldName::parse(&name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    state.set_global(name, value);
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    write_datum_vars(state, datum, list, key, value)
                        .map_err(|message| execution_error(module, &frames, message))?;
                } else if let Err(error) = write_list_value(&mut state.heap, list, key, value) {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
            }
            Instruction::ListLength => {
                let length = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::Null) => 0,
                    Ok(Value::List(list)) => match state.heap.list(list) {
                        Ok(values) => values.len(),
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    },
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list length operation received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
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
            Instruction::StoreSrc => {
                let src = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].src = src;
            }
            Instruction::LoadUsr => {
                let usr = frames[frame_index].usr.clone();
                frames[frame_index].stack.push(usr);
            }
            Instruction::LoadCaller => {
                let caller = if frame_index == 0 {
                    Value::Null
                } else {
                    materialize_callee_chain(module, state, &frames[..frame_index])
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(caller);
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
                        } else if runtime_type.as_str() == "/savefile"
                            || runtime_type.as_str().starts_with("/savefile/")
                        {
                            match name.as_str() {
                                "cd" => Value::text(
                                    savefile_current_directory(
                                        &state.savefiles.entry(datum).or_default().cd,
                                    )
                                    .to_owned(),
                                ),
                                "eof" => {
                                    let savefile = state.savefiles.entry(datum).or_default();
                                    let path = savefile_current_directory(&savefile.cd);
                                    Value::number(if savefile.entries.contains_key(path) {
                                        0.0
                                    } else {
                                        1.0
                                    })
                                }
                                "dir" => {
                                    let children = savefile_directory_entries(
                                        state.savefiles.entry(datum).or_default(),
                                    );
                                    let list = state.heap.allocate_list();
                                    let values = state.heap.list_mut(list).map_err(|error| {
                                        execution_error(module, &frames, error.to_string())
                                    })?;
                                    for child in children {
                                        values.add(Value::text(child));
                                    }
                                    Value::List(list)
                                }
                                _ => match state.heap.datum_field(datum, &name) {
                                    Ok(value) => value.clone(),
                                    Err(error) => {
                                        return Err(execution_error(
                                            module,
                                            &frames,
                                            error.to_string(),
                                        ));
                                    }
                                },
                            }
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
                        assign_datum_field(state, datum, name.clone(), value.clone())
                            .map_err(|message| execution_error(module, &frames, message))?;
                    }
                    Value::List(list) if name.as_str() == "len" => {
                        let new_len = match &value {
                            Value::Number(number) if number.to_f32().is_finite() => {
                                let length = number.to_f32().trunc();
                                if length < 0.0 {
                                    return Err(execution_error(
                                        module,
                                        &frames,
                                        "list length cannot be negative",
                                    ));
                                }
                                length.to_string().parse::<usize>().unwrap_or(usize::MAX)
                            }
                            _ => 0,
                        };
                        if state.is_associative_list(list) && new_len != 0 {
                            return Err(execution_error(
                                module,
                                &frames,
                                "alist length can only be assigned zero",
                            ));
                        }
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
            Instruction::LoadGlobalVars => {
                let list = if let Some(list) = state.global_vars_proxy {
                    list
                } else {
                    let list = state.heap.allocate_list();
                    for name in state.globals.keys() {
                        state
                            .heap
                            .list_mut(list)
                            .expect("new global.vars proxy is live")
                            .add(Value::text(name.as_str()));
                    }
                    state.mark_associative_list(list);
                    state.global_vars_proxy = Some(list);
                    list
                };
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::LoadDatumVars => {
                let datum = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::Datum(datum)) => datum,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("vars requires a datum, received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let list = if let Some(list) = state.datum_vars_by_datum.get(&datum).copied() {
                    list
                } else {
                    let runtime_type = state
                        .heap
                        .datum(datum)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .type_path()
                        .clone();
                    let instance = state
                        .heap
                        .datum_fields(datum)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .map(|(field, value)| (field.clone(), value.clone()))
                        .collect::<Vec<_>>();
                    let shared = state
                        .shared_fields
                        .get(&runtime_type)
                        .cloned()
                        .unwrap_or_default();
                    let list = state.heap.allocate_list();
                    for (field, value) in instance {
                        state
                            .heap
                            .list_mut(list)
                            .expect("new datum.vars proxy is live")
                            .set_key(Value::text(field.as_str()), value);
                    }
                    for (name, storage) in shared {
                        let value = state.global(&storage).cloned().unwrap_or(Value::Null);
                        state
                            .heap
                            .list_mut(list)
                            .expect("new datum.vars proxy is live")
                            .set_key(Value::text(name.as_str()), value);
                    }
                    state.mark_associative_list(list);
                    state.datum_vars_proxies.insert(list, datum);
                    state.datum_vars_by_datum.insert(datum, list);
                    list
                };
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::LoadInitialGlobal(name) => {
                let value = state
                    .initial_globals
                    .get(&name)
                    .cloned()
                    .unwrap_or(Value::Null);
                frames[frame_index].stack.push(value);
            }
            Instruction::StoreGlobal(name) => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                state.set_global(name, value);
            }
            Instruction::MutateLocal {
                slot,
                delta,
                prefix,
            } => {
                let Some(current) = frames[frame_index].locals.get(usize::from(slot)).cloned()
                else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].locals[usize::from(slot)] = updated;
                frames[frame_index].stack.push(result);
            }
            Instruction::MutateGlobal {
                name,
                delta,
                prefix,
            } => {
                let current = state.global(&name).cloned().unwrap_or(Value::Null);
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                state.set_global(name, updated);
                frames[frame_index].stack.push(result);
            }
            Instruction::MutateResult { delta, prefix } => {
                let current = frames[frame_index].result.clone();
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].result = updated;
                frames[frame_index].stack.push(result);
            }
            Instruction::MutateField {
                name,
                delta,
                prefix,
            } => {
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let current = match &receiver {
                    Value::Datum(datum) => state
                        .heap
                        .datum_field(*datum, &name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .clone(),
                    Value::List(list) if name.as_str() == "len" => {
                        let len = state
                            .heap
                            .list(*list)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?
                            .len();
                        Value::number(len as f32)
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!(
                                "increment/decrement field requires a datum or list.len, received {value}"
                            ),
                        ));
                    }
                };
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                match receiver {
                    Value::Datum(datum) => {
                        state
                            .heap
                            .set_datum_field(datum, name, updated)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    }
                    Value::List(list) => {
                        let length = updated.as_number().unwrap_or(0.0).trunc();
                        if length < 0.0 {
                            return Err(execution_error(
                                module,
                                &frames,
                                "list length cannot be negative",
                            ));
                        }
                        if state.is_associative_list(list) && length != 0.0 {
                            return Err(execution_error(
                                module,
                                &frames,
                                "alist length can only be assigned zero",
                            ));
                        }
                        let new_len = length as usize;
                        state
                            .heap
                            .list_mut(list)
                            .and_then(|values| values.resize(new_len))
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    }
                    _ => unreachable!("receiver was validated above"),
                }
                frames[frame_index].stack.push(result);
            }
            Instruction::MutateListIndex { delta, prefix } => {
                let key = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let list = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::List(list)) => list,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list mutation requires a list, received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let current = read_list_value(&state.heap, list, &key)
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?
                    .clone();
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                write_list_value(&mut state.heap, list, key, updated)
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                frames[frame_index].stack.push(result);
            }
            Instruction::Duplicate => {
                let Some(value) = frames[frame_index].stack.last().cloned() else {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::AddressLocal(slot) => {
                let Some(local) = frames[frame_index].locals.get_mut(usize::from(slot)) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                let reference = match local {
                    Value::List(list) if state.reference_lists.contains(list) => *list,
                    value => {
                        let list = state.heap.allocate_list();
                        state
                            .heap
                            .list_mut(list)
                            .expect("new pointer cell is live")
                            .add(value.clone());
                        state.reference_lists.insert(list);
                        *value = Value::List(list);
                        list
                    }
                };
                frames[frame_index].stack.push(Value::List(reference));
            }
            Instruction::LoadLocalRaw(slot) => {
                let Some(value) = frames[frame_index].locals.get(usize::from(slot)).cloned() else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::LoadLocal(slot) => {
                let Some(mut value) = frames[frame_index].locals.get(usize::from(slot)).cloned()
                else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                if let Value::List(list) = value
                    && state.reference_lists.contains(&list)
                {
                    value = state
                        .heap
                        .list(list)
                        .and_then(|values| values.get(1))
                        .cloned()
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                }
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
                if let Value::List(list) = local
                    && state.reference_lists.contains(list)
                {
                    state
                        .heap
                        .list_mut(*list)
                        .and_then(|values| values.set(1, value.clone()))
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                } else {
                    *local = value.clone();
                }
                if frames[frame_index].static_locals.contains(&slot) {
                    let path = module
                        .procedure_path(frames[frame_index].procedure)
                        .unwrap_or("<unknown procedure>")
                        .to_owned();
                    state.procedure_static_locals.insert((path, slot), value);
                }
            }
            Instruction::LoadStaticLocalOrJump { slot, target } => {
                let path = module
                    .procedure_path(frames[frame_index].procedure)
                    .unwrap_or("<unknown procedure>")
                    .to_owned();
                if let Some(value) = state.procedure_static_locals.get(&(path, slot)).cloned() {
                    let Some(local) = frames[frame_index].locals.get_mut(usize::from(slot)) else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("invalid static local slot {slot}"),
                        ));
                    };
                    *local = value;
                    frames[frame_index].static_locals.insert(slot);
                    frames[frame_index].instruction = target.saturating_sub(1);
                }
            }
            Instruction::InitializeStaticLocal(slot) => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let path = module
                    .procedure_path(frames[frame_index].procedure)
                    .unwrap_or("<unknown procedure>")
                    .to_owned();
                state
                    .procedure_static_locals
                    .insert((path, slot), value.clone());
                frames[frame_index].static_locals.insert(slot);
                frames[frame_index].stack.push(value);
            }
            Instruction::LoadResult => {
                let result = frames[frame_index].result.clone();
                frames[frame_index].stack.push(result);
            }
            Instruction::StoreUsr => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index].usr = value;
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
            Instruction::BeginTry { catch, end, local } => {
                if catch >= program.instructions.len() || end >= program.instructions.len() {
                    return Err(execution_error(
                        module,
                        &frames,
                        "exception handler target is outside the procedure",
                    ));
                }
                let stack_depth = frames[frame_index].stack.len();
                frames[frame_index]
                    .exception_handlers
                    .push(ExceptionHandler {
                        start: instruction_index + 1,
                        end,
                        catch,
                        local,
                        stack_depth,
                    });
            }
            Instruction::EndTry => {
                if frames[frame_index].exception_handlers.pop().is_none() {
                    return Err(execution_error(
                        module,
                        &frames,
                        "EndTry without an active exception handler",
                    ));
                }
            }
            Instruction::Throw => {
                let thrown = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let mut handler = None;
                for candidate_frame in (0..frames.len()).rev() {
                    let current = frames[candidate_frame].instruction;
                    if let Some(position) = frames[candidate_frame]
                        .exception_handlers
                        .iter()
                        .rposition(|handler| handler.start <= current && current <= handler.end)
                    {
                        handler = Some((candidate_frame, position));
                        break;
                    }
                }
                let Some((handler_frame, handler_position)) = handler else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("uncaught exception: {thrown}"),
                    ));
                };
                frames.truncate(handler_frame + 1);
                let handler = frames[handler_frame]
                    .exception_handlers
                    .remove(handler_position);
                frames[handler_frame]
                    .exception_handlers
                    .truncate(handler_position);
                frames[handler_frame].stack.truncate(handler.stack_depth);
                if let Some(slot) = handler.local {
                    let Some(local) = frames[handler_frame].locals.get_mut(usize::from(slot))
                    else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("invalid catch local {slot}"),
                        ));
                    };
                    *local = thrown;
                }
                frames[handler_frame].instruction = handler.catch;
                continue;
            }
            Instruction::Locate { argument_count } => {
                let count = usize::from(argument_count);
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments = frames[frame_index].stack.split_off(stack_length - count);
                let located = if let [x, y, z] = arguments.as_slice() {
                    let integer = |value: &Value| {
                        value.as_number().and_then(|value| {
                            (value.is_finite()
                                && value.fract() == 0.0
                                && value >= i32::MIN as f32
                                && value <= i32::MAX as f32)
                                .then(|| {
                                    #[allow(clippy::cast_possible_truncation)]
                                    {
                                        value as i32
                                    }
                                })
                        })
                    };
                    match (integer(x), integer(y), integer(z)) {
                        (Some(x), Some(y), Some(z)) => {
                            state.turf_at(x, y, z).map_or(Value::Null, Value::Datum)
                        }
                        _ => Value::Null,
                    }
                } else {
                    Value::Null
                };
                frames[frame_index].stack.push(located);
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
                let value = if let Value::Datum(datum) = left {
                    if is_matrix_datum(datum, &state.heap) {
                        execute_matrix_compound(operator, datum, &right, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    } else if is_vector_datum(datum, &state.heap) {
                        execute_vector_compound(operator, datum, &right, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    } else {
                        execute_scalar_compound_assignment(operator, Value::Datum(datum), right)
                            .map_err(|message| execution_error(module, &frames, message))?
                    }
                } else if let Value::List(list) = left {
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
                let value = if let (Value::Datum(left), Value::Datum(right)) = (&left, &right)
                    && is_matrix_datum(*left, &state.heap)
                    && is_matrix_datum(*right, &state.heap)
                {
                    let left_values = matrix_components(*left, &state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let right_values = matrix_components(*right, &state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let datum = allocate_matrix(
                        std::array::from_fn(|index| left_values[index] + right_values[index]),
                        &mut state.heap,
                    )
                    .map_err(|message| execution_error(module, &frames, message))?;
                    Value::Datum(datum)
                } else if let (Value::Datum(left), Value::Datum(right)) = (&left, &right)
                    && is_vector_datum(*left, &state.heap)
                    && is_vector_datum(*right, &state.heap)
                {
                    Value::Datum(
                        allocate_vector(
                            vector_zip(*left, *right, &state.heap, |a, b| a + b)
                                .map_err(|message| execution_error(module, &frames, message))?,
                            &mut state.heap,
                        )
                        .map_err(|message| execution_error(module, &frames, message))?,
                    )
                } else if let Value::List(list) = left {
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
                let value = if matches!(instruction, Instruction::Subtract)
                    && let (Value::Datum(left), Value::Datum(right)) = (&left, &right)
                    && is_matrix_datum(*left, &state.heap)
                    && is_matrix_datum(*right, &state.heap)
                {
                    let left_values = matrix_components(*left, &state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let right_values = matrix_components(*right, &state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let datum = allocate_matrix(
                        std::array::from_fn(|index| left_values[index] - right_values[index]),
                        &mut state.heap,
                    )
                    .map_err(|message| execution_error(module, &frames, message))?;
                    Value::Datum(datum)
                } else if matches!(instruction, Instruction::Subtract)
                    && let (Value::Datum(left), Value::Datum(right)) = (&left, &right)
                    && is_vector_datum(*left, &state.heap)
                    && is_vector_datum(*right, &state.heap)
                {
                    Value::Datum(
                        allocate_vector(
                            vector_zip(*left, *right, &state.heap, |a, b| a - b)
                                .map_err(|message| execution_error(module, &frames, message))?,
                            &mut state.heap,
                        )
                        .map_err(|message| execution_error(module, &frames, message))?,
                    )
                } else if let Value::List(list) = left {
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
            | Instruction::FractionalRemainder
            | Instruction::ShiftLeft
            | Instruction::ShiftRight => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let vector_operator = match instruction {
                    Instruction::Multiply => Some("*"),
                    Instruction::Divide => Some("/"),
                    _ => None,
                };
                let value = if let Value::Datum(datum) = left
                    && is_vector_datum(datum, &state.heap)
                    && let Some(operator) = vector_operator
                {
                    execute_vector_binary(operator, datum, &right, &mut state.heap)
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
            Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let comparison = compare_values(&left, &right)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let result = match instruction {
                    Instruction::Less => comparison.is_some_and(std::cmp::Ordering::is_lt),
                    Instruction::LessEqual => comparison.is_some_and(std::cmp::Ordering::is_le),
                    Instruction::Greater => comparison.is_some_and(std::cmp::Ordering::is_gt),
                    Instruction::GreaterEqual => comparison.is_some_and(std::cmp::Ordering::is_ge),
                    _ => unreachable!(),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::Compare => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let comparison = compare_values(&left, &right)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = comparison.map_or(0.0, |value| match value {
                    std::cmp::Ordering::Less => -1.0,
                    std::cmp::Ordering::Equal => 0.0,
                    std::cmp::Ordering::Greater => 1.0,
                });
                frames[frame_index].stack.push(Value::number(value));
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
            Instruction::Equivalent | Instruction::NotEquivalent => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let equivalent = values_equivalent(&left, &right, &state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let result = if matches!(instruction, Instruction::NotEquivalent) {
                    !equivalent
                } else {
                    equivalent
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::Contains => {
                let container = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let needle = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let contains = if let Value::List(list) = container {
                    state
                        .heap
                        .list(list)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .positions()
                        .any(|(_, value)| values_equal(&needle, value))
                } else {
                    false
                };
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
                procedure: mut target,
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
                let mut context = frame_context(&frames[frame_index]);
                if let Some(path) = module.procedure_path(target)
                    && let Some((_, selector)) = path.rsplit_once("/proc/")
                    && !path.starts_with("/proc/")
                    && matches!(frames[frame_index].src, Value::Datum(_))
                {
                    let selector = selector.split('@').next().unwrap_or(selector);
                    let (dynamic_target, dynamic_context) = dynamic_call_target(
                        module,
                        state,
                        &frames[frame_index].src,
                        &Value::text(selector),
                        &context,
                        false,
                    )
                    .map_err(|message| execution_error(module, &frames, message))?;
                    target = dynamic_target;
                    context = dynamic_context;
                }
                let target_program = module
                    .resolve_procedure(target)
                    .map_err(|message| execution_error(module, &frames, message))?;
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
                let target_program = module
                    .resolve_procedure(target)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let context = frame_context(&frames[frame_index]);
                frames.push(make_frame(target, target_program, &arguments, &context));
                continue;
            }
            Instruction::CallDynamic {
                argument_count,
                null_receiver_is_global,
            } => {
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
                } else if let (Value::Datum(savefile), Value::Text(method)) = (&receiver, &selector)
                    && state.heap.datum(*savefile).is_ok_and(|datum| {
                        let path = datum.type_path().as_str();
                        path == "/savefile" || path.starts_with("/savefile/")
                    })
                    && method.as_ref() == "ExportText"
                {
                    let key = arguments
                        .first()
                        .and_then(|value| match value {
                            Value::Text(value) => Some(value.as_ref()),
                            _ => None,
                        })
                        .unwrap_or("");
                    let encoded = state
                        .savefiles
                        .get(savefile)
                        .and_then(|savefile| {
                            let path = savefile_resolve_path(&savefile.cd, key);
                            savefile.entries.get(&path)
                        })
                        .map_or_else(String::new, savefile_export_value);
                    frames[frame_index]
                        .stack
                        .push(Value::text(format!("{key} = {{\"\n{encoded}\n\"}}\n\n")));
                } else if let (Value::Datum(datum), Value::Text(method)) = (&receiver, &selector)
                    && is_matrix_datum(*datum, &state.heap)
                {
                    let result = execute_matrix_method(*datum, method, &arguments, &mut state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    frames[frame_index].stack.push(result);
                } else if let (Value::Datum(datum), Value::Text(method)) = (&receiver, &selector)
                    && is_vector_datum(*datum, &state.heap)
                {
                    let result = execute_vector_method(*datum, method, &arguments, &mut state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    frames[frame_index].stack.push(result);
                } else if let (Value::Datum(datum), Value::Text(method)) = (&receiver, &selector)
                    && is_regex_datum(*datum, state)
                {
                    let result = execute_regex_method(*datum, method, &arguments, state)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    frames[frame_index].stack.push(result);
                } else if let (Value::Datum(datum), Value::Text(method)) = (&receiver, &selector)
                    && is_icon_datum(*datum, &state.heap)
                    && matches!(
                        method.as_ref(),
                        "MapColors"
                            | "Blend"
                            | "SetIntensity"
                            | "Scale"
                            | "Crop"
                            | "Shift"
                            | "Width"
                            | "Height"
                            | "DrawBox"
                            | "Insert"
                            | "GetPixel"
                    )
                {
                    let result = match method.as_ref() {
                        "MapColors" => apply_icon_map_colors(*datum, &arguments, &mut state.heap)
                            .map(|()| Value::Null),
                        "Blend" => apply_icon_blend(*datum, &arguments, &mut state.heap)
                            .map(|()| Value::Null),
                        "SetIntensity" => {
                            apply_icon_set_intensity(*datum, &arguments, &mut state.heap)
                                .map(|()| Value::Null)
                        }
                        method => execute_icon_method(*datum, method, &arguments, &mut state.heap),
                    }
                    .map_err(|message| execution_error(module, &frames, message))?;
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
                    let (target, context) = dynamic_call_target(
                        module,
                        state,
                        &receiver,
                        &selector,
                        &caller_context,
                        null_receiver_is_global,
                    )
                    .map_err(|message| execution_error(module, &frames, message))?;
                    let target_program = module
                        .resolve_procedure(target)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    frames.push(make_frame(target, target_program, &arguments, &context));
                    continue;
                }
            }
            Instruction::Spawn { entry } => {
                let delay = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let delay = delay.as_number().ok_or_else(|| {
                    execution_error(
                        module,
                        &frames,
                        format!("spawn delay must be numeric, received {delay}"),
                    )
                })?;
                let mut spawned = frames[frame_index].clone();
                spawned.instruction = entry;
                spawned.stack.clear();
                if delay.is_sign_negative() {
                    match run_frames(module, vec![spawned], limits, state)? {
                        FrameRunOutcome::Complete(_) => {}
                        FrameRunOutcome::Yielded { frames, delay } => {
                            schedule_frames(state, frames, delay);
                        }
                    }
                    continue;
                }
                let delay = if delay.is_finite() && delay > 0.0 {
                    (delay.floor() as u64).max(1)
                } else {
                    0
                };
                schedule_frames(state, vec![spawned], delay);
            }
            Instruction::Return => {
                let result = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames.pop();
                let Some(caller) = frames.last_mut() else {
                    return Ok(FrameRunOutcome::Complete(result));
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
        exception_handlers: Vec::new(),
        detached_waitfor: false,
        static_locals: HashSet::new(),
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
        Instruction::Remainder => integer_remainder(left, right),
        Instruction::FractionalRemainder => fractional_remainder(left, right),
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
        CompoundListIndexOperator::Remainder => integer_remainder(left, right),
        CompoundListIndexOperator::FractionalRemainder => fractional_remainder(left, right),
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

const DM_BIT_MASK: u32 = (1 << 24) - 1;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn dm_u24(value: f32) -> u32 {
    (value.trunc() as i64 as u32) & DM_BIT_MASK
}

#[allow(clippy::cast_precision_loss)]
fn bitwise_binary(left: f32, right: f32, operation: impl FnOnce(u32, u32) -> u32) -> f32 {
    (operation(dm_u24(left), dm_u24(right)) & DM_BIT_MASK) as f32
}

#[allow(clippy::cast_precision_loss)]
fn bitwise_not(value: f32) -> f32 {
    ((!dm_u24(value)) & DM_BIT_MASK) as f32
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn bitwise_shift(left: f32, right: f32, operation: impl FnOnce(u32, u32) -> u32) -> f32 {
    let count = right.trunc().max(0.0) as u32;
    if count >= 24 {
        return 0.0;
    }
    (operation(dm_u24(left), count) & DM_BIT_MASK) as f32
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn integer_remainder(left: f32, right: f32) -> f32 {
    let left = left.trunc() as i32;
    let right = right.trunc() as i32;
    if right == 0 {
        f32::NAN
    } else {
        (left % right) as f32
    }
}

fn fractional_remainder(left: f32, right: f32) -> f32 {
    if right == 0.0 {
        f32::NAN
    } else {
        right * (left / right).fract()
    }
}

fn compare_values(left: &Value, right: &Value) -> Result<Option<std::cmp::Ordering>, String> {
    match (left, right) {
        (Value::Text(left), Value::Text(right)) => Ok(Some(left.as_ref().cmp(right.as_ref()))),
        (Value::Null | Value::Number(_), Value::Null | Value::Number(_)) => {
            let left = match left {
                Value::Null => 0.0,
                Value::Number(number) => number.to_f32(),
                _ => unreachable!(),
            };
            let right = match right {
                Value::Null => 0.0,
                Value::Number(number) => number.to_f32(),
                _ => unreachable!(),
            };
            Ok(left.partial_cmp(&right))
        }
        _ => Err(format!(
            "comparison requires two numbers or two text values, received {left} and {right}"
        )),
    }
}

fn values_equivalent(left: &Value, right: &Value, heap: &ValueHeap) -> Result<bool, String> {
    if let (Value::Datum(left), Value::Datum(right)) = (left, right)
        && is_matrix_datum(*left, heap)
        && is_matrix_datum(*right, heap)
    {
        return Ok(matrix_components(*left, heap)? == matrix_components(*right, heap)?);
    }
    let (Value::List(left_id), Value::List(right_id)) = (left, right) else {
        return Ok(left.semantic_eq(right));
    };
    let left = heap.list(*left_id).map_err(|error| error.to_string())?;
    let right = heap.list(*right_id).map_err(|error| error.to_string())?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for index in 1..=left.len() {
        let left_key = left.get(index).map_err(|error| error.to_string())?;
        let right_key = right.get(index).map_err(|error| error.to_string())?;
        if !left_key.semantic_eq(right_key) {
            return Ok(false);
        }
        let left_assoc = left.get_key(left_key).cloned().unwrap_or(Value::Null);
        let right_assoc = right.get_key(right_key).cloned().unwrap_or(Value::Null);
        if !left_assoc.semantic_eq(&right_assoc) {
            return Ok(false);
        }
    }
    Ok(true)
}

const VECTOR_FIELDS: [&str; 3] = ["x", "y", "z"];

fn is_vector_datum(datum: DatumId, heap: &ValueHeap) -> bool {
    heap.datum(datum)
        .is_ok_and(|datum| datum.type_path().as_str() == "/vector")
}

fn is_icon_datum(datum: DatumId, heap: &ValueHeap) -> bool {
    heap.datum(datum)
        .is_ok_and(|datum| datum.type_path().as_str() == "/icon")
}

fn icon_dimension(icon: DatumId, name: &str, heap: &ValueHeap) -> f32 {
    heap.datum_field(
        icon,
        &FieldName::parse(name).expect("internal icon dimension field is valid"),
    )
    .ok()
    .and_then(Value::as_number)
    .unwrap_or(32.0)
}

fn icon_number(argument: Option<&Value>, method: &str, name: &str) -> Result<f32, String> {
    argument
        .and_then(Value::as_number)
        .ok_or_else(|| format!("icon.{method} requires numeric {name}"))
}

fn record_icon_operation(
    icon: DatumId,
    method: &str,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    let operation = heap.allocate_list();
    {
        let values = heap
            .list_mut(operation)
            .map_err(|error| error.to_string())?;
        values.add(Value::text(method));
        for argument in arguments {
            values.add(argument.clone());
        }
    }
    let field = FieldName::parse("_dream64_icon_operations")
        .expect("internal icon operation field is valid");
    let operations = match heap.datum_field(icon, &field) {
        Ok(Value::List(operations)) => *operations,
        _ => {
            let operations = heap.allocate_list();
            heap.set_datum_field(icon, field, Value::List(operations))
                .map_err(|error| error.to_string())?;
            operations
        }
    };
    heap.list_mut(operations)
        .map_err(|error| error.to_string())?
        .add(Value::List(operation));
    Ok(())
}

fn execute_icon_method(
    icon: DatumId,
    method: &str,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let width_field = FieldName::parse("_dream64_width").expect("internal icon width is valid");
    let height_field = FieldName::parse("_dream64_height").expect("internal icon height is valid");
    match method {
        "Width" if arguments.is_empty() => {
            Ok(Value::number(icon_dimension(icon, "_dream64_width", heap)))
        }
        "Height" if arguments.is_empty() => {
            Ok(Value::number(icon_dimension(icon, "_dream64_height", heap)))
        }
        "Scale" if (1..=2).contains(&arguments.len()) => {
            let width = icon_number(arguments.first(), method, "width")?;
            let height = icon_number(arguments.get(1), method, "height").unwrap_or(width);
            heap.set_datum_field(icon, width_field, Value::number(width))
                .map_err(|error| error.to_string())?;
            heap.set_datum_field(icon, height_field, Value::number(height))
                .map_err(|error| error.to_string())?;
            record_icon_operation(icon, method, arguments, heap)?;
            Ok(Value::Null)
        }
        "Crop" if arguments.len() == 4 => {
            let x1 = icon_number(arguments.first(), method, "x1")?;
            let y1 = icon_number(arguments.get(1), method, "y1")?;
            let x2 = icon_number(arguments.get(2), method, "x2")?;
            let y2 = icon_number(arguments.get(3), method, "y2")?;
            heap.set_datum_field(icon, width_field, Value::number((x2 - x1).abs() + 1.0))
                .map_err(|error| error.to_string())?;
            heap.set_datum_field(icon, height_field, Value::number((y2 - y1).abs() + 1.0))
                .map_err(|error| error.to_string())?;
            record_icon_operation(icon, method, arguments, heap)?;
            Ok(Value::Null)
        }
        "Shift" | "DrawBox" | "Insert"
            if (method == "Shift" && (2..=3).contains(&arguments.len()))
                || (method == "DrawBox" && (1..=5).contains(&arguments.len()))
                || (method == "Insert" && (1..=6).contains(&arguments.len())) =>
        {
            record_icon_operation(icon, method, arguments, heap)?;
            Ok(Value::Null)
        }
        // Pixel decoding is renderer/resource-provider work. BYOND yields null
        // for a transparent or unavailable pixel, which is the truthful
        // headless result while retaining every mutating operation above.
        "GetPixel" if (2..=5).contains(&arguments.len()) => Ok(Value::Null),
        _ => Err(format!(
            "icon.{method} received unsupported arguments ({})",
            arguments.len()
        )),
    }
}

fn apply_icon_map_colors(
    icon: DatumId,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    if !matches!(arguments.len(), 4 | 5 | 12 | 20) {
        return Err(format!(
            "icon.MapColors requires 4, 5, 12, or 20 arguments, received {}",
            arguments.len()
        ));
    }
    let matrix = heap.allocate_list();
    for value in arguments {
        heap.list_mut(matrix)
            .map_err(|error| error.to_string())?
            .add(value.clone());
    }
    heap.set_datum_field(
        icon,
        FieldName::parse("_dream64_color_matrix").expect("headless icon field is valid"),
        Value::List(matrix),
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

fn apply_icon_blend(
    icon: DatumId,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    if !(1..=4).contains(&arguments.len()) {
        return Err(format!(
            "icon.Blend requires an icon/color and up to mode, x, y; received {} arguments",
            arguments.len()
        ));
    }
    let history_field =
        FieldName::parse("_dream64_blends").expect("headless icon blend field is valid");
    let history = match heap.datum_field(icon, &history_field) {
        Ok(Value::List(history)) => *history,
        _ => {
            let history = heap.allocate_list();
            heap.set_datum_field(icon, history_field, Value::List(history))
                .map_err(|error| error.to_string())?;
            history
        }
    };
    let operation = heap.allocate_list();
    for value in [
        arguments[0].clone(),
        arguments.get(1).cloned().unwrap_or(Value::number(0.0)),
        arguments.get(2).cloned().unwrap_or(Value::number(1.0)),
        arguments.get(3).cloned().unwrap_or(Value::number(1.0)),
    ] {
        heap.list_mut(operation)
            .map_err(|error| error.to_string())?
            .add(value);
    }
    heap.list_mut(history)
        .map_err(|error| error.to_string())?
        .add(Value::List(operation));
    Ok(())
}

fn apply_icon_set_intensity(
    icon: DatumId,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    if !(1..=3).contains(&arguments.len()) {
        return Err(format!(
            "icon.SetIntensity requires r and optional g and b, received {} arguments",
            arguments.len()
        ));
    }
    let red = arguments[0]
        .as_number()
        .ok_or_else(|| "icon.SetIntensity red component must be numeric".to_owned())?;
    let green = arguments
        .get(1)
        .unwrap_or(&arguments[0])
        .as_number()
        .ok_or_else(|| "icon.SetIntensity green component must be numeric".to_owned())?;
    let blue = arguments
        .get(2)
        .unwrap_or(&arguments[0])
        .as_number()
        .ok_or_else(|| "icon.SetIntensity blue component must be numeric".to_owned())?;
    apply_icon_map_colors(
        icon,
        &[
            Value::number(red),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(green),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(blue),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(0.0),
        ],
        heap,
    )
}

fn vector_components(datum: DatumId, heap: &ValueHeap) -> Result<[f32; 3], String> {
    if !is_vector_datum(datum, heap) {
        return Err("vector operation requires a /vector datum".to_owned());
    }
    let mut values = [0.0; 3];
    for (index, name) in VECTOR_FIELDS.iter().enumerate() {
        let field = FieldName::parse(name).expect("vector field is valid");
        values[index] = heap
            .datum_field(datum, &field)
            .map_err(|error| error.to_string())?
            .as_number()
            .unwrap_or(0.0);
    }
    Ok(values)
}

fn write_vector(datum: DatumId, values: [f32; 3], heap: &mut ValueHeap) -> Result<(), String> {
    for (name, value) in VECTOR_FIELDS.into_iter().zip(values) {
        heap.set_datum_field(
            datum,
            FieldName::parse(name).expect("vector field is valid"),
            Value::number(value),
        )
        .map_err(|error| error.to_string())?;
    }
    let magnitude = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    for (name, value) in [("len", 3.0), ("size", magnitude)] {
        heap.set_datum_field(
            datum,
            FieldName::parse(name).expect("vector metadata field is valid"),
            Value::number(value),
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn allocate_vector(values: [f32; 3], heap: &mut ValueHeap) -> Result<DatumId, String> {
    let datum = heap.allocate_datum(TypePath::parse("/vector").expect("vector path is valid"));
    write_vector(datum, values, heap)?;
    Ok(datum)
}

fn construct_vector(arguments: &[Value], heap: &mut ValueHeap) -> Result<DatumId, String> {
    if arguments.len() > 3 {
        return Err("vector accepts at most three arguments".to_owned());
    }
    let mut values = [0.0; 3];
    for (index, value) in arguments.iter().enumerate() {
        values[index] = value.as_number().unwrap_or(0.0);
    }
    allocate_vector(values, heap)
}

fn vector_zip(
    left: DatumId,
    right: DatumId,
    heap: &ValueHeap,
    operation: impl Fn(f32, f32) -> f32,
) -> Result<[f32; 3], String> {
    let left = vector_components(left, heap)?;
    let right = vector_components(right, heap)?;
    Ok(std::array::from_fn(|index| {
        operation(left[index], right[index])
    }))
}

fn execute_vector_binary(
    operator: &str,
    datum: DatumId,
    right: &Value,
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let left = vector_components(datum, heap)?;
    let right_values = match right {
        Value::Datum(other) if is_vector_datum(*other, heap) => vector_components(*other, heap)?,
        value => [value.as_number().unwrap_or(0.0); 3],
    };
    let values = match operator {
        "*" => std::array::from_fn(|index| left[index] * right_values[index]),
        "/" => std::array::from_fn(|index| left[index] / right_values[index]),
        _ => return Err(format!("unsupported vector operator {operator}")),
    };
    Ok(Value::Datum(allocate_vector(values, heap)?))
}

fn execute_vector_compound(
    operator: CompoundAssignmentOperator,
    datum: DatumId,
    right: &Value,
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let left = vector_components(datum, heap)?;
    let right_values = match right {
        Value::Datum(other) if is_vector_datum(*other, heap) => vector_components(*other, heap)?,
        value => [value.as_number().unwrap_or(0.0); 3],
    };
    let values = match operator {
        CompoundAssignmentOperator::Add => {
            std::array::from_fn(|index| left[index] + right_values[index])
        }
        CompoundAssignmentOperator::Subtract => {
            std::array::from_fn(|index| left[index] - right_values[index])
        }
        CompoundAssignmentOperator::Multiply => {
            std::array::from_fn(|index| left[index] * right_values[index])
        }
        CompoundAssignmentOperator::Divide => {
            std::array::from_fn(|index| left[index] / right_values[index])
        }
        _ => return Err("unsupported vector compound operator".to_owned()),
    };
    write_vector(datum, values, heap)?;
    Ok(Value::Datum(datum))
}

fn execute_vector_method(
    datum: DatumId,
    method: &str,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let current = vector_components(datum, heap)?;
    match method.to_ascii_lowercase().as_str() {
        "dot" => {
            let Some(Value::Datum(other)) = arguments.first() else {
                return Err("vector.Dot requires a vector".to_owned());
            };
            let other = vector_components(*other, heap)?;
            Ok(Value::number(
                current.iter().zip(other).map(|(a, b)| a * b).sum::<f32>(),
            ))
        }
        "interpolate" => {
            let Some(Value::Datum(other)) = arguments.first() else {
                return Err("vector.Interpolate requires a vector and factor".to_owned());
            };
            let other = vector_components(*other, heap)?;
            let factor = arguments.get(1).and_then(Value::as_number).unwrap_or(0.0);
            let values = std::array::from_fn(|index| {
                current[index] + (other[index] - current[index]) * factor
            });
            Ok(Value::Datum(allocate_vector(values, heap)?))
        }
        "normalize" => {
            let magnitude = current
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let values = if magnitude == 0.0 {
                current
            } else {
                current.map(|value| value / magnitude)
            };
            write_vector(datum, values, heap)?;
            Ok(Value::Datum(datum))
        }
        _ => Err(format!("unknown /vector procedure {method:?}")),
    }
}

const MATRIX_FIELDS: [&str; 6] = ["a", "b", "c", "d", "e", "f"];

fn matrix_numeric(value: &Value) -> f32 {
    value.as_number().unwrap_or(0.0)
}

fn is_matrix_datum(datum: DatumId, heap: &ValueHeap) -> bool {
    heap.datum(datum)
        .is_ok_and(|datum| datum.type_path().as_str() == "/matrix")
}

fn matrix_components(datum: DatumId, heap: &ValueHeap) -> Result<[f32; 6], String> {
    if !is_matrix_datum(datum, heap) {
        return Err("matrix operation requires a /matrix datum".to_owned());
    }
    let mut values = [0.0; 6];
    for (index, name) in MATRIX_FIELDS.iter().enumerate() {
        let field = FieldName::parse(name).expect("matrix field is valid");
        values[index] = matrix_numeric(heap.datum_field(datum, &field).map_err(|e| e.to_string())?);
    }
    Ok(values)
}

fn write_matrix(datum: DatumId, values: [f32; 6], heap: &mut ValueHeap) -> Result<(), String> {
    for (name, value) in MATRIX_FIELDS.into_iter().zip(values) {
        heap.set_datum_field(
            datum,
            FieldName::parse(name).expect("matrix field is valid"),
            Value::number(value),
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn allocate_matrix(values: [f32; 6], heap: &mut ValueHeap) -> Result<DatumId, String> {
    let datum = heap.allocate_datum(TypePath::parse("/matrix").expect("matrix path is valid"));
    write_matrix(datum, values, heap)?;
    Ok(datum)
}

/// Applies constructor state owned by BYOND's engine types. These arguments
/// are not a project-defined `New()` call: contextual `var/icon/I = new(...)`
/// must retain the same resource fields as the `icon(...)` builtin.
fn initialize_engine_resource(
    state: &mut ExecutionState,
    datum: DatumId,
    type_path: &TypePath,
    arguments: &[Value],
) -> Result<(), String> {
    let fields: &[&str] = match type_path.as_str() {
        "/icon" => &["icon", "icon_state", "dir", "frame", "moving"],
        _ => return Ok(()),
    };
    for (name, value) in fields.iter().zip(arguments) {
        state
            .heap
            .set_datum_field(
                datum,
                FieldName::parse(name).expect("engine resource field is valid"),
                value.clone(),
            )
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn allocate_or_replace_engine_datum(
    state: &mut ExecutionState,
    type_path: TypePath,
    arguments: &[Value],
) -> Result<DatumId, String> {
    let path = type_path.as_str();
    let is_turf = path == "/turf" || path.starts_with("/turf/");
    if is_turf
        && let Some(Value::Datum(existing)) = arguments.first()
        && state.heap.datum(*existing).is_ok_and(|datum| {
            let path = datum.type_path().as_str();
            path == "/turf" || path.starts_with("/turf/")
        })
    {
        initialize_existing_datum(state, *existing, type_path.clone(), true)?;
        initialize_engine_resource(state, *existing, &type_path, arguments)?;
        return Ok(*existing);
    }
    let datum = allocate_initialized_datum(state, type_path.clone())?;
    initialize_engine_resource(state, datum, &type_path, arguments)?;
    Ok(datum)
}

fn allocate_initialized_datum(
    state: &mut ExecutionState,
    type_path: TypePath,
) -> Result<DatumId, String> {
    let datum = state.heap.allocate_datum(type_path.clone());
    initialize_existing_datum(state, datum, type_path, false)?;
    Ok(datum)
}

fn initialize_existing_datum(
    state: &mut ExecutionState,
    datum: DatumId,
    type_path: TypePath,
    preserve_cell: bool,
) -> Result<(), String> {
    let initial_values = state
        .initial_values
        .get(&type_path)
        .cloned()
        .unwrap_or_default();
    let is_atom = is_atom_type_path(&type_path);
    if preserve_cell {
        let preserved = ["x", "y", "z", "loc", "contents"];
        let fields = state
            .heap
            .datum_fields(datum)
            .map_err(|error| error.to_string())?
            .map(|(name, _)| name.clone())
            .filter(|name| !preserved.contains(&name.as_str()))
            .collect::<Vec<_>>();
        for field in fields {
            state
                .heap
                .delete_datum_field(datum, &field)
                .map_err(|error| error.to_string())?;
        }
        state
            .heap
            .set_datum_type_path(datum, type_path.clone())
            .map_err(|error| error.to_string())?;
    }
    for (name, value) in initial_values {
        if preserve_cell && ["x", "y", "z", "loc", "contents"].contains(&name.as_str()) {
            continue;
        }
        state
            .heap
            .set_datum_field(datum, name, value)
            .map_err(|error| error.to_string())?;
    }
    let mut hierarchy = Vec::new();
    let mut current = Some(type_path.clone());
    while let Some(path) = current {
        hierarchy.push(path.clone());
        current = state.type_parent(&path).cloned();
    }
    hierarchy.reverse();
    let plans = hierarchy
        .into_iter()
        .flat_map(|path| {
            state
                .instance_initializers
                .get(&path)
                .cloned()
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let initializer_module = state.instance_initializer_module.clone();
    for initializer in plans {
        let module = initializer_module
            .as_ref()
            .ok_or_else(|| "runtime instance initializer module is absent".to_owned())?;
        let value = execute_module_in_context(
            module,
            initializer.entry,
            &[],
            state,
            &ExecutionContext::new(Value::Datum(datum), Value::Null),
        )
        .map_err(|error| error.to_string())?;
        state
            .heap
            .set_datum_field(datum, initializer.field, value)
            .map_err(|error| error.to_string())?;
    }
    if is_atom {
        let contents = FieldName::parse("contents").expect("built-in contents field");
        state.ensure_contents(datum)?;
        if preserve_cell {
            return Ok(());
        }
        let world = FieldName::parse("world").expect("built-in world global");
        let world_contents = state
            .global(&world)
            .and_then(|value| match value {
                Value::Datum(world) => Some(*world),
                _ => None,
            })
            .and_then(|world| state.heap.datum_field(world, &contents).ok())
            .and_then(|value| match value {
                Value::List(list) => Some(*list),
                _ => None,
            });
        if let Some(list) = world_contents {
            state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .add(Value::Datum(datum));
        }
    }
    Ok(())
}

fn is_atom_type_path(path: &TypePath) -> bool {
    let path = path.as_str();
    ["/atom", "/area", "/turf", "/obj", "/mob"]
        .into_iter()
        .any(|root| {
            path == root
                || path
                    .strip_prefix(root)
                    .is_some_and(|rest| rest.starts_with('/'))
        })
}

fn construct_matrix(arguments: &[Value], heap: &mut ValueHeap) -> Result<DatumId, String> {
    match arguments {
        [] => return allocate_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], heap),
        [Value::Datum(source)] if is_matrix_datum(*source, heap) => {
            return allocate_matrix(matrix_components(*source, heap)?, heap);
        }
        [a, b, c, d, e, f] => {
            return allocate_matrix([a, b, c, d, e, f].map(matrix_numeric), heap);
        }
        _ => {}
    }
    let mode_value = arguments
        .last()
        .and_then(Value::as_number)
        .ok_or_else(|| "matrix operation mode must be numeric".to_owned())?
        as i32;
    let mode = mode_value & 127;
    let modify = mode_value & 128 != 0;
    let source = arguments.first().and_then(|value| match value {
        Value::Datum(datum) if is_matrix_datum(*datum, heap) => Some(*datum),
        _ => None,
    });
    let mut values = source.map_or(Ok([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), |datum| {
        matrix_components(datum, heap)
    })?;
    match mode {
        0 => {
            let source = source.ok_or_else(|| "MATRIX_COPY requires a matrix".to_owned())?;
            values = matrix_components(source, heap)?;
        }
        4 => {
            let determinant = values[0] * values[4] - values[1] * values[3];
            if determinant == 0.0 {
                return Err("cannot invert a singular matrix".to_owned());
            }
            values = [
                values[4] / determinant,
                -values[1] / determinant,
                (values[1] * values[5] - values[4] * values[2]) / determinant,
                -values[3] / determinant,
                values[0] / determinant,
                (values[3] * values[2] - values[0] * values[5]) / determinant,
            ];
        }
        5 => {
            let offset = usize::from(source.is_some());
            let radians = matrix_numeric(&arguments[offset]).to_radians();
            let mut cosine = radians.cos();
            let mut sine = radians.sin();
            if cosine.abs() < 1.0e-6 {
                cosine = 0.0;
            }
            if sine.abs() < 1.0e-6 {
                sine = 0.0;
            }
            values = matrix_product(values, [cosine, sine, 0.0, -sine, cosine, 0.0]);
        }
        6 => {
            let offset = usize::from(source.is_some());
            let x = matrix_numeric(&arguments[offset]);
            let y = if arguments.len() - offset >= 3 {
                matrix_numeric(&arguments[offset + 1])
            } else {
                x
            };
            values = matrix_product(values, [x, 0.0, 0.0, 0.0, y, 0.0]);
        }
        7 => {
            let offset = usize::from(source.is_some());
            let x = matrix_numeric(&arguments[offset]);
            let y = if arguments.len() - offset >= 3 {
                matrix_numeric(&arguments[offset + 1])
            } else {
                x
            };
            values[2] += x;
            values[5] += y;
        }
        _ => return Err(format!("unknown matrix operation mode {mode}")),
    }
    if modify {
        let datum = source.ok_or_else(|| "MATRIX_MODIFY requires a matrix".to_owned())?;
        write_matrix(datum, values, heap)?;
        Ok(datum)
    } else {
        allocate_matrix(values, heap)
    }
}

fn execute_matrix_method(
    datum: DatumId,
    method: &str,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let current = matrix_components(datum, heap)?;
    let updated = match method.to_ascii_lowercase().as_str() {
        "add" | "subtract" => {
            let Value::Datum(other) = arguments.first().unwrap_or(&Value::Null) else {
                return Err(format!("matrix.{method} requires a matrix"));
            };
            let other = matrix_components(*other, heap)?;
            let sign = if method.eq_ignore_ascii_case("add") {
                1.0
            } else {
                -1.0
            };
            std::array::from_fn(|index| current[index] + sign * other[index])
        }
        "multiply" => match arguments.first().unwrap_or(&Value::Null) {
            Value::Null => current,
            Value::Datum(other) if is_matrix_datum(*other, heap) => {
                matrix_product(current, matrix_components(*other, heap)?)
            }
            value => current.map(|component| component * matrix_numeric(value)),
        },
        "scale" => {
            let factor = arguments.first().map_or(0.0, matrix_numeric);
            current.map(|component| component * factor)
        }
        "translate" => {
            let Some(x) = arguments.first().and_then(Value::as_number) else {
                return Ok(Value::Datum(datum));
            };
            let y = arguments.get(1).and_then(Value::as_number).unwrap_or(x);
            [
                current[0],
                current[1],
                current[2] + x,
                current[3],
                current[4],
                current[5] + y,
            ]
        }
        "turn" => {
            let degrees = arguments.first().map_or(0.0, matrix_numeric).to_radians();
            let mut cosine = degrees.cos();
            let mut sine = degrees.sin();
            if cosine.abs() < 1.0e-6 {
                cosine = 0.0;
            }
            if sine.abs() < 1.0e-6 {
                sine = 0.0;
            }
            let rotation = [cosine, sine, 0.0, -sine, cosine, 0.0];
            matrix_product(current, rotation)
        }
        "invert" => {
            let determinant = current[0] * current[4] - current[1] * current[3];
            if determinant == 0.0 {
                return Err("cannot invert a singular matrix".to_owned());
            }
            [
                current[4] / determinant,
                -current[1] / determinant,
                (current[1] * current[5] - current[4] * current[2]) / determinant,
                -current[3] / determinant,
                current[0] / determinant,
                (current[3] * current[2] - current[0] * current[5]) / determinant,
            ]
        }
        _ => return Err(format!("unknown /matrix procedure {method:?}")),
    };
    write_matrix(datum, updated, heap)?;
    Ok(Value::Datum(datum))
}

fn matrix_product(left: [f32; 6], right: [f32; 6]) -> [f32; 6] {
    [
        left[0] * right[0] + left[3] * right[1],
        left[1] * right[0] + left[4] * right[1],
        left[2] * right[0] + left[5] * right[1] + right[2],
        left[0] * right[3] + left[3] * right[4],
        left[1] * right[3] + left[4] * right[4],
        left[2] * right[3] + left[5] * right[4] + right[5],
    ]
}

fn execute_matrix_compound(
    operator: CompoundAssignmentOperator,
    datum: DatumId,
    right: &Value,
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let left = matrix_components(datum, heap)?;
    let updated = match operator {
        CompoundAssignmentOperator::Add | CompoundAssignmentOperator::Subtract => {
            let Value::Datum(other) = right else {
                return Err("matrix addition/subtraction requires another matrix".to_owned());
            };
            let other = matrix_components(*other, heap)?;
            let sign = if matches!(operator, CompoundAssignmentOperator::Add) {
                1.0
            } else {
                -1.0
            };
            std::array::from_fn(|index| left[index] + sign * other[index])
        }
        CompoundAssignmentOperator::Multiply => match right {
            Value::Datum(other) if is_matrix_datum(*other, heap) => {
                matrix_product(left, matrix_components(*other, heap)?)
            }
            value => left.map(|component| component * matrix_numeric(value)),
        },
        CompoundAssignmentOperator::Divide => {
            let divisor = matrix_numeric(right);
            left.map(|component| component / divisor)
        }
        _ => return Err("unsupported compound matrix operator".to_owned()),
    };
    write_matrix(datum, updated, heap)?;
    Ok(Value::Datum(datum))
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
        [] => return Ok(deterministic_unit(state)),
        [high] => (0.0, *high),
        [low, high] => (*low, *high),
        _ => return Err("rand accepts zero, one, or two bounds".to_owned()),
    };
    let low = low.ceil();
    let high = high.floor();
    if !low.is_finite() || !high.is_finite() || high < low {
        return Err(format!("invalid rand range {low} through {high}"));
    }
    Ok(low + (deterministic_unit(state) * (high - low + 1.0)).floor())
}

fn roll_dice(arguments: &[Value], state: &mut u64) -> Result<f32, String> {
    let (count, sides, offset) = match arguments {
        [Value::Text(dice)] => {
            let dice = dice.trim();
            let (count, remainder) = dice
                .split_once(['d', 'D'])
                .ok_or_else(|| format!("invalid dice expression {dice:?}"))?;
            let sign = remainder
                .char_indices()
                .skip(1)
                .find(|(_, character)| matches!(character, '+' | '-'))
                .map(|(index, _)| index);
            let (sides, offset) = sign.map_or((remainder, "0"), |index| remainder.split_at(index));
            (
                count
                    .parse::<i32>()
                    .map_err(|_| format!("invalid dice count {count:?}"))?,
                sides
                    .parse::<i32>()
                    .map_err(|_| format!("invalid dice sides {sides:?}"))?,
                offset
                    .parse::<i32>()
                    .map_err(|_| format!("invalid dice offset {offset:?}"))?,
            )
        }
        [sides] => (
            1,
            sides
                .as_number()
                .ok_or_else(|| format!("roll requires a number or dice text, received {sides}"))?
                .trunc() as i32,
            0,
        ),
        [count, sides] => (
            count
                .as_number()
                .ok_or_else(|| format!("roll count requires a number, received {count}"))?
                .trunc() as i32,
            sides
                .as_number()
                .ok_or_else(|| format!("roll sides requires a number, received {sides}"))?
                .trunc() as i32,
            0,
        ),
        _ => return Err("roll requires one or two arguments".to_owned()),
    };
    if count < 0 || sides < 1 {
        return Err(format!("invalid dice dimensions {count}d{sides}"));
    }
    let mut total = offset as f32;
    for _ in 0..count {
        total += random_integer(&[Value::number(1.0), Value::number(sides as f32)], state)?;
    }
    Ok(total)
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
        Value::Null
        | Value::Number(_)
        | Value::Text(_)
        | Value::TypePath(_)
        | Value::ModifiedTypePath(_) => return Value::Null,
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

fn direction_towards_builtin(
    source: &Value,
    target: &Value,
    heap: &ValueHeap,
) -> Result<Value, String> {
    let (Value::Datum(source), Value::Datum(target)) = (source, target) else {
        return Ok(Value::number(0.0));
    };
    let coordinate = |datum: DatumId, name: &str| -> Result<f32, String> {
        heap.datum_field(
            datum,
            &FieldName::parse(name).expect("built-in coordinate field is valid"),
        )
        .map_err(|error| error.to_string())?
        .as_number()
        .ok_or_else(|| format!("get_step_towards coordinate {name} is not numeric"))
    };
    if coordinate(*source, "z")? != coordinate(*target, "z")? {
        return Ok(Value::number(0.0));
    }
    let dx = coordinate(*target, "x")? - coordinate(*source, "x")?;
    let dy = coordinate(*target, "y")? - coordinate(*source, "y")?;
    let mut direction = 0_u8;
    if dy > 0.0 {
        direction |= 1;
    } else if dy < 0.0 {
        direction |= 2;
    }
    if dx > 0.0 {
        direction |= 4;
    } else if dx < 0.0 {
        direction |= 8;
    }
    Ok(Value::number(f32::from(direction)))
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
        // BYOND 516 filters null selectors out. This matters for helper
        // routines that expand a caller-provided list of roots one at a time.
        Value::Null => return Ok(Vec::new()),
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
            if let Value::List(list) = value {
                return Ok(target.as_str() == "/list"
                    || (target.as_str() == "/alist" && state.is_associative_list(*list)));
            }
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
    null_receiver_is_global: bool,
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
        Value::Null if null_receiver_is_global => ("/proc".to_owned(), caller_context.clone()),
        Value::Null => {
            return Err("cannot call a procedure on null".to_owned());
        }
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
    let selector_path = selector
        .trim_start_matches('/')
        .strip_prefix("proc/")
        .unwrap_or_else(|| selector.trim_start_matches('/'));
    let requested = if selector.starts_with('/') {
        selector.clone()
    } else if base_path == "/proc" {
        format!("/proc/{selector_path}")
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

fn execute_del(
    module: &Module,
    arguments: &[Value],
    state: &mut ExecutionState,
    caller_context: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    let Some(value) = arguments.first() else {
        return Ok(Value::Null);
    };
    let Value::Datum(datum) = value else {
        return execute_standard_builtin("del", arguments, state).map_err(|message| RuntimeError {
            message,
            instruction: 0,
            source_span: None,
            call_stack: Vec::new(),
        });
    };

    // A Del() body is allowed to delete its own src. In that case invalidate
    // the handle immediately, while the outer deletion remains responsible for
    // treating its eventual stale finalization as success.
    if !state.deleting_datums.insert(*datum) {
        let _ = state.heap_mut().destroy_datum(*datum);
        return Ok(Value::Null);
    }

    let receiver = Value::Datum(*datum);
    let hook = dynamic_call_target(
        module,
        state,
        &receiver,
        &Value::text("Del"),
        caller_context,
        false,
    );
    let hook_result = match hook {
        Ok((procedure, context)) => {
            execute_module_in_context(module, procedure, &[], state, &context).map(|_| ())
        }
        Err(_) => Ok(()),
    };

    // BYOND invalidates the object after Del() regardless of its return value.
    // Runtime failure likewise must not resurrect a half-cleaned-up datum.
    let _ = state.heap_mut().destroy_datum(*datum);
    state.deleting_datums.remove(datum);
    hook_result.map(|()| Value::Null)
}

fn invoke_constructor_if_present(
    module: &Module,
    state: &mut ExecutionState,
    datum: DatumId,
    arguments: &[Value],
    caller_context: &ExecutionContext,
) -> Result<(), RuntimeError> {
    let receiver = Value::Datum(datum);
    let selector = Value::text("New");
    let Ok((constructor, context)) =
        dynamic_call_target(module, state, &receiver, &selector, caller_context, false)
    else {
        return Ok(());
    };
    execute_module_in_context(module, constructor, arguments, state, &context).map(|_| ())
}

fn construct_sized_list(arguments: &[Value], heap: &mut ValueHeap) -> Result<ListId, String> {
    fn dimension(value: &Value) -> Result<usize, String> {
        let number = value
            .as_number()
            .ok_or_else(|| format!("list dimension must be numeric, received {value}"))?;
        if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
            return Err(format!(
                "list dimension must be a non-negative integer, received {number}"
            ));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let size = number as usize;
        if size as f32 != number {
            return Err(format!("list dimension is too large: {number}"));
        }
        Ok(size)
    }

    let list = heap.allocate_list();
    let Some((first, remaining)) = arguments.split_first() else {
        return Ok(list);
    };
    let size = dimension(first)?;
    if remaining.is_empty() {
        heap.list_mut(list)
            .map_err(|error| error.to_string())?
            .resize(size)
            .map_err(|error| error.to_string())?;
    } else {
        for _ in 0..size {
            let child = construct_sized_list(remaining, heap)?;
            heap.list_mut(list)
                .map_err(|error| error.to_string())?
                .add(Value::List(child));
        }
    }
    Ok(list)
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

fn savefile_export_value(value: &Value) -> String {
    match value {
        // Headless rendering does not own PNG pixels. Preserve BYOND's
        // base64-shaped savefile contract with a deterministic payload so
        // callers can cache and transport the result.
        Value::Datum(_) => "ZHJlYW02NA==".to_owned(),
        Value::Text(value) => value.to_string(),
        value => value.to_string(),
    }
}

fn savefile_current_directory(cd: &str) -> &str {
    if cd.is_empty() { "/" } else { cd }
}

fn savefile_resolve_path(cd: &str, path: &str) -> String {
    let joined = if path.starts_with('/') {
        path.to_owned()
    } else {
        format!(
            "{}/{}",
            savefile_current_directory(cd).trim_end_matches('/'),
            path
        )
    };
    let parts = joined
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .fold(Vec::new(), |mut parts, part| {
            if part == ".." {
                parts.pop();
            } else {
                parts.push(part);
            }
            parts
        });
    if parts.is_empty() {
        "/".to_owned()
    } else {
        format!("/{}", parts.join("/"))
    }
}

fn savefile_directory_entries(savefile: &SavefileState) -> Vec<String> {
    let directory = savefile_current_directory(&savefile.cd);
    let prefix = if directory == "/" {
        "/".to_owned()
    } else {
        format!("{}/", directory.trim_end_matches('/'))
    };
    let mut children = savefile
        .entries
        .keys()
        .filter_map(|path| path.strip_prefix(&prefix))
        .filter_map(|remainder| remainder.split('/').next())
        .filter(|child| !child.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    children.sort();
    children.dedup();
    children
}

fn datum_shared_storage(
    state: &ExecutionState,
    datum: DatumId,
    field: &FieldName,
) -> Option<FieldName> {
    let path = state.heap.datum(datum).ok()?.type_path();
    state.shared_fields.get(path)?.get(field).cloned()
}

fn assign_datum_field(
    state: &mut ExecutionState,
    datum: DatumId,
    field: FieldName,
    value: Value,
) -> Result<(), String> {
    let is_savefile = state.heap.datum(datum).is_ok_and(|datum| {
        let path = datum.type_path().as_str();
        path == "/savefile" || path.starts_with("/savefile/")
    });
    if is_savefile && field.as_str() == "cd" {
        let requested = match &value {
            Value::Text(value) => value.as_ref(),
            value => return Err(format!("savefile.cd requires text, received {value}")),
        };
        let current = state.savefiles.entry(datum).or_default().cd.clone();
        state.savefiles.entry(datum).or_default().cd = savefile_resolve_path(&current, requested);
    }
    let is_world = state
        .heap
        .datum(datum)
        .is_ok_and(|datum| datum.type_path().as_str() == "/world");
    if field.as_str() == "loc" {
        let old_loc = state
            .heap
            .datum_field(datum, &field)
            .ok()
            .and_then(|value| match value {
                Value::Datum(loc) => Some(*loc),
                _ => None,
            });
        let new_loc = match &value {
            Value::Datum(loc) => Some(*loc),
            Value::Null => None,
            value => {
                return Err(format!(
                    "loc assignment requires a datum or null, received {value}"
                ));
            }
        };
        let is_movable = state
            .heap
            .datum(datum)
            .is_ok_and(|datum| builtins::is_movable_path(datum.type_path().as_str()));
        let new_loc_is_turf = new_loc.is_some_and(|loc| {
            state.heap.datum(loc).is_ok_and(|datum| {
                let path = datum.type_path().as_str();
                path == "/turf" || path.starts_with("/turf/")
            })
        });
        if is_movable && new_loc_is_turf {
            builtins::move_movable_to_turf(state, datum, new_loc.expect("turf loc exists"))?;
            return Ok(());
        }
        if old_loc != new_loc {
            builtins::synchronize_moved_atom_contents(state, datum, old_loc, new_loc)?;
        }
    }
    if is_world && matches!(field.as_str(), "maxx" | "maxy" | "maxz") {
        let requested = value
            .as_number()
            .ok_or_else(|| format!("world.{} requires a numeric value", field.as_str()))?;
        if !requested.is_finite()
            || requested.fract() != 0.0
            || requested < 1.0
            || requested > i32::MAX as f32
        {
            return Err(format!(
                "world.{} must be a positive integer, received {requested}",
                field.as_str()
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        let requested = requested as i32;
        let mut dimensions = (
            state.world_dimension(datum, "maxx")?,
            state.world_dimension(datum, "maxy")?,
            state.world_dimension(datum, "maxz")?,
        );
        match field.as_str() {
            "maxx" => dimensions.0 = requested,
            "maxy" => dimensions.1 = requested,
            "maxz" => dimensions.2 = requested,
            _ => unreachable!(),
        }
        state.resize_world_geometry(datum, dimensions)?;
    }
    state
        .heap
        .set_datum_field(datum, field.clone(), value.clone())
        .map_err(|error| error.to_string())?;
    if is_world {
        let reciprocal = match (field.as_str(), value.as_number()) {
            ("tick_lag", Some(value)) if value.is_finite() && value > 0.0 => {
                Some(("fps", 10.0 / value))
            }
            ("fps", Some(value)) if value.is_finite() && value > 0.0 => {
                Some(("tick_lag", 10.0 / value))
            }
            _ => None,
        };
        if let Some((field, reciprocal)) = reciprocal {
            let _ = state.heap.set_datum_field(
                datum,
                FieldName::parse(field).expect("built-in world timing field"),
                Value::number(reciprocal),
            );
        }
    }
    Ok(())
}

fn write_datum_vars(
    state: &mut ExecutionState,
    datum: DatumId,
    list: ListId,
    key: Value,
    value: Value,
) -> Result<(), String> {
    let Value::Text(name) = &key else {
        return Err("datum.vars writes require a text key".to_owned());
    };
    let field = FieldName::parse(name).map_err(|error| error.to_string())?;
    if let Some(storage) = datum_shared_storage(state, datum, &field) {
        state.set_global(storage, value.clone());
    } else {
        assign_datum_field(state, datum, field, value.clone())?;
    }
    write_list_value(&mut state.heap, list, key, value).map_err(|error| error.to_string())
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

fn allocate_dm_array(heap: &mut ValueHeap, sizes: &[usize], depth: usize) -> ListId {
    let list = heap.allocate_list();
    for _ in 0..sizes.get(depth).copied().unwrap_or(0) {
        let value = if depth + 1 < sizes.len() {
            Value::List(allocate_dm_array(heap, sizes, depth + 1))
        } else {
            Value::Null
        };
        heap.list_mut(list)
            .expect("new array list is live")
            .add(value);
    }
    list
}

fn execute_animate(
    names: &[Option<String>],
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let explicit_target = names
        .iter()
        .zip(arguments)
        .find(|(name, _)| name.is_none())
        .map(|(_, value)| value.clone());
    let target = explicit_target
        .clone()
        .or_else(|| state.last_animation_target.clone());
    if let Some(target) = explicit_target {
        state.last_animation_target = Some(target);
    }
    let Some(Value::Datum(target)) = target else {
        // Rendering-only calls against null or unsupported client-side values
        // have no persistent effect in a headless world.
        return Ok(Value::Null);
    };

    const CONTROL_ARGUMENTS: &[&str] = &[
        "time",
        "loop",
        "easing",
        "flags",
        "delay",
        "tag",
        "command",
        "appearance",
        "var_list",
        "object",
    ];
    for (name, value) in names.iter().zip(arguments) {
        let Some(name) = name else { continue };
        if CONTROL_ARGUMENTS.contains(&name.to_ascii_lowercase().as_str()) {
            if name.eq_ignore_ascii_case("var_list") {
                let Value::List(list) = value else { continue };
                let fields = state
                    .heap
                    .list(*list)
                    .map_err(|error| error.to_string())?
                    .associations()
                    .filter_map(|(key, value)| match key {
                        Value::Text(key) => {
                            FieldName::parse(key).ok().map(|key| (key, value.clone()))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let datum = state
                    .heap
                    .datum_mut(target)
                    .map_err(|error| error.to_string())?;
                for (field, value) in fields {
                    datum.set_field(field, value);
                }
            }
            continue;
        }
        let field = FieldName::parse(name).map_err(|error| error.to_string())?;
        state
            .heap
            .datum_mut(target)
            .map_err(|error| error.to_string())?
            .set_field(field, value.clone());
    }
    Ok(Value::Null)
}

fn scalar_number_string(value: Value) -> Result<f32, String> {
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        value => Err(format!("numeric operation received {value}")),
    }
}

fn mutate_scalar_value(value: Value, delta: i8, prefix: bool) -> Result<(Value, Value), String> {
    let old_result = value.clone();
    let old_number = match value {
        Value::Null | Value::Text(_) => 0.0,
        Value::Number(number) => number.to_f32(),
        value => {
            return Err(format!(
                "increment/decrement requires a scalar value, received {value}"
            ));
        }
    };
    let updated = Value::number(old_number + f32::from(delta));
    let result = if prefix { updated.clone() } else { old_result };
    Ok((result, updated))
}

fn execute_scalar_add(left: Value, right: Value) -> Result<Value, String> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            Ok(Value::number(left.to_f32() + right.to_f32()))
        }
        (Value::Null, Value::Number(right)) => Ok(Value::number(right.to_f32())),
        (Value::Number(left), Value::Null) => Ok(Value::number(left.to_f32())),
        (Value::Null, Value::Null) => Ok(Value::number(0.0)),
        // A declaration-only `/list` variable begins as null, and BYOND's
        // `field += list(value)` idiom initializes it to that list. Logging
        // queues and many SS13 lazy collections depend on this coercion.
        (Value::Null, right @ Value::List(_)) => Ok(right),
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
        && matches!((&left, &right), (Value::Null, Value::List(_)))
    {
        return Ok(right);
    }
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
        CompoundAssignmentOperator::Remainder => integer_remainder(left, right),
        CompoundAssignmentOperator::FractionalRemainder => fractional_remainder(left, right),
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
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::sync::Arc;

    use dm_core::{DmNumberBits, SourceSpan};
    use dm_lexer::{SpannedToken, TokenKind, lex};
    use dm_syntax::parse;
    use dm_value::{FieldName, TypePath};

    use super::{
        CompoundListIndexOperator, ExecutionContext, ExecutionLimits, ExecutionState,
        InitializerBinding, Instruction, ProcedureSpec, Program, Value, advance_scheduler,
        allocate_initialized_datum, allocate_matrix, compile_initializer,
        compile_initializer_into_module, compile_module, compile_module_specs,
        compile_module_specs_selective, compile_module_specs_selective_with_errors,
        compile_module_with_global_fields, compile_procedure,
        compile_procedure_with_resolver_and_fields, condition_tokens, dm_builtin_numeric_constant,
        execute, execute_in_context, execute_in_state, execute_module, execute_module_in_context,
        execute_module_in_state, execute_module_with_limits, execute_with_limits,
        execute_with_limits_in_state, interpolated_expression_close, matrix_components,
    };

    #[test]
    fn builtin_mob_sight_flag_family_has_byond_bit_values() {
        for (name, expected) in [
            ("BLIND", 1.0),
            ("SEE_MOBS", 4.0),
            ("SEEMOBS", 4.0),
            ("SEE_OBJS", 8.0),
            ("SEEOBJS", 8.0),
            ("SEE_TURFS", 16.0),
            ("SEETURFS", 16.0),
            ("SEE_SELF", 32.0),
            ("SEE_INFRA", 64.0),
            ("SEE_PIXELS", 256.0),
            ("SEE_THRU", 512.0),
            ("SEE_BLACKNESS", 1024.0),
        ] {
            assert_eq!(dm_builtin_numeric_constant(name), Some(expected), "{name}");
        }
    }

    #[test]
    fn interpolation_close_skips_brackets_inside_nested_quotes() {
        let expression = r#"src ? "nested[value]" : fallback]tail"#;
        let close = interpolated_expression_close(expression, 0).expect("outer close should exist");
        assert_eq!(
            &expression[..=close],
            r#"src ? "nested[value]" : fallback]"#
        );
    }

    fn execute_source(source: &str, argument: f32) -> Value {
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        execute(&program, &[Value::number(argument)]).expect("procedure should execute")
    }

    #[test]
    fn procedure_specs_resolve_implicit_owner_calls_through_the_path_index() {
        let source = parse(
            "/datum/example/proc/value()\n\treturn 17\n/datum/example/proc/read()\n\treturn value()\n",
        )
        .expect("source should parse");
        let specs = [
            ProcedureSpec {
                path: "/datum/example/proc/value@0".to_owned(),
                definition: &source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/example/proc/read@0".to_owned(),
                definition: &source.definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ];
        let module = compile_module_specs(&specs).expect("implicit owner call should resolve");
        let entry = module
            .procedure_id("/datum/example/proc/read@0")
            .expect("read entry should exist");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(17.0)));
    }

    #[test]
    fn text_template_fills_empty_and_whitespace_holes_and_honors_escaped_brackets() {
        let syntax = parse(
            "/proc/run()\n\treturn text(\"before [] [ ] \\[literal\\] after\", \"one\", 2)\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("text() should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(
            execute_module(&module, entry, &[]),
            Ok(Value::text("before one 2 [literal] after"))
        );
    }

    #[test]
    fn crash_expression_is_lazy_behind_null_conditional_access() {
        let syntax = parse(
            "/proc/run()\n\tvar/value = null\n\tvalue?.field = CRASH(\"skipped rhs\")\n\tvar/result = value?.method(CRASH(\"skipped argument\"))\n\treturn isnull(result)\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("CRASH expression should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
    }

    #[test]
    fn newlist_constructs_one_fresh_datum_for_each_type_path() {
        let syntax = parse(
            "/proc/run()\n\tvar/list/items = newlist(/datum/one, /datum/two)\n\treturn length(items) + istype(items[1], /datum/one) + istype(items[2], /datum/two)\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("newlist should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(4.0)));
    }

    #[test]
    fn src_assignment_rebinds_subsequent_bare_method_dispatch() {
        let syntax = parse(
            "/datum/A/proc/who()\n\treturn 1\n/datum/B/proc/who()\n\treturn 2\n/datum/A/proc/test()\n\tsrc = new /datum/B\n\treturn who()\n/proc/run()\n\tvar/datum/A/item = new /datum/A\n\treturn item.test()\n",
        )
        .expect("source should parse");
        let definitions = &syntax.definitions;
        let specs = [
            ProcedureSpec {
                path: "/datum/A/proc/who".to_owned(),
                definition: &definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/B/proc/who".to_owned(),
                definition: &definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/A/proc/test".to_owned(),
                definition: &definitions[2],
                parent: None,
                static_calls: BTreeMap::from([("who".to_owned(), 0)]),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/proc/run".to_owned(),
                definition: &definitions[3],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ];
        let module = compile_module_specs(&specs).expect("src rebinding family compiles");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(2.0)));
    }

    #[test]
    fn exact_list_allocation_constructs_heap_list_identity() {
        let syntax = parse("/proc/run()\n\tvar/list/items = new /list\n\treturn islist(items)\n")
            .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("new /list compiles");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
    }

    #[test]
    fn modified_type_construction_applies_overrides_after_declared_initial_values() {
        let syntax = parse(
            "/proc/run()\n\tvar/datum/plain = new /datum/example\n\tvar/datum/changed = new /datum/example {a=6;b=8}\n\treturn plain.a + plain.b + changed.a + changed.b\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("modified type should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        let path = TypePath::parse("/datum/example").expect("type path");
        let mut state = ExecutionState::new();
        state.set_initial_values(BTreeMap::from([(
            path,
            BTreeMap::from([
                (field("a"), Value::number(5.0)),
                (field("b"), Value::number(7.0)),
            ]),
        )]));
        assert_eq!(
            execute_module_in_state(&module, entry, &[], &mut state),
            Ok(Value::number(26.0))
        );
    }

    #[test]
    fn modified_type_paths_are_list_keys_and_dynamic_new_operands() {
        let syntax = parse(
            "/proc/run()\n\tvar/amount = 15\n\tvar/list/cache = list(/datum/example{a = amount} = 4)\n\tvar/kind = /datum/example{a = amount}\n\tvar/datum/created = new kind\n\treturn cache[kind] + created.a + created.b\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("modified path values compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        let path = TypePath::parse("/datum/example").expect("type path");
        let mut state = ExecutionState::new();
        state.set_initial_values(BTreeMap::from([(
            path,
            BTreeMap::from([
                (field("a"), Value::number(1.0)),
                (field("b"), Value::number(2.0)),
            ]),
        )]));

        assert_eq!(
            execute_module_in_state(&module, entry, &[], &mut state),
            Ok(Value::number(21.0)),
            "the modified path must retain its evaluated key identity and override defaults after allocation"
        );
    }

    #[test]
    fn infinity_constants_interpolate_and_complex_raw_strings_use_custom_delimiters() {
        let syntax = parse(
            "/proc/run()\n\tvar/a = 1#INF\n\tvar/b = -1#INF\n\tvar/c = -1#IND\n\tvar/raw = @(END)\nhello worldEND\n\treturn (\"[a]\" == \"inf\") + (\"[b]\" == \"-inf\") + (\"[c]\" == \"nan\") + (raw == \"hello world\")\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("constant expressions compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(4.0)));
    }

    #[test]
    fn assign_into_is_direct_assignment_and_output_statement_does_not_shift_receiver() {
        let syntax = parse(
            "/proc/run()\n\tvar/value = 5\n\tvalue := 10\n\tvalue << 1\n\tvar/other = 3\n\tother := null\n\treturn value + isnull(other)\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("operator statements compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(11.0)));
    }

    #[test]
    fn comma_locals_logical_assignment_and_procedure_scope_name_follow_dm_rules() {
        let syntax = parse(
            "/datum/proc/foo()\n\tset name = \"display\"\n\treturn\n/proc/run()\n\tvar/v1,v2\n\tv1 = 0\n\tv2 = 1\n\tv1 ||= 5\n\tv2 &&= 7\n\treturn v1 + v2 + (/datum/proc/foo::name == \"foo\")\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("operator parser family compiles");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(13.0)));
    }

    #[test]
    fn alist_constructs_ordered_key_value_storage_and_preserves_its_runtime_type() {
        let syntax = parse(
            "/proc/run()\n\tvar/alist/inner = alist(\"one\" = 1, \"two\" = 2)\n\tvar/alist/items = alist(\"left\" = inner, \"right\" = 3)\n\titems += alist(\"right\" = 9, \"extra\" = 4)\n\tvar/alist/copy = items.Copy()\n\tif(!istype(items[\"left\"], /alist)) return 0\n\tif(items[\"right\"] != 3 || items[\"extra\"] != 4) return 0\n\tif(length(items) != 3 || !istype(copy, /alist)) return 0\n\treturn copy[\"left\"][\"two\"]\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("alist family should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(2.0)));
    }

    #[test]
    fn list_length_is_writable_and_values_cut_filters_associations_by_numeric_value() {
        let syntax = parse(
            "/proc/run()\n\tvar/list/items = list(\"a\" = 1, \"b\" = 2, \"c\" = 0)\n\tvar/removed = values_cut_over(items, 1, TRUE)\n\tvar/list/plain = list(1, 2, 3, 4)\n\tplain.len--\n\tplain.len -= 1\n\tplain.len = 1\n\treturn removed * 10 + length(items) + length(plain)\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("list mutation family compiles");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(22.0)));

        let negative = parse("/proc/run()\n\tvar/list/items = list()\n\titems.len--\n")
            .expect("negative source parses");
        let negative = compile_module(&negative.definitions).expect("negative source compiles");
        let entry = negative.procedure_id("/proc/run").expect("entry");
        assert!(
            execute_module(&negative, entry, &[])
                .expect_err("negative length must fail")
                .message
                .contains("cannot be negative")
        );
    }

    #[test]
    fn condition_tokens_accepts_braced_macro_conditions_with_following_tokens() {
        let tokens = lex("if(!(flags_1 & INITIALIZED_1)) { var/previous = 1")
            .expect("condition source should lex");
        let condition = condition_tokens(&tokens[1..], "if").expect("condition should compile");
        assert!(matches!(condition[0].kind, TokenKind::Operator(ref op) if op == "!"));
    }

    #[test]
    fn try_catch_binds_arbitrary_thrown_values_and_skips_catch_normally() {
        let syntax = parse(
            "/proc/run(should_throw)\n\tvar/result = 1\n\ttry\n\t\tif (should_throw)\n\t\t\tthrow 5\n\t\tresult = 2\n\tcatch(var/error)\n\t\tresult = error + 10\n\treturn result\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("try/catch should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(
            execute_module(&module, entry, &[Value::number(0.0)]),
            Ok(Value::number(2.0))
        );
        assert_eq!(
            execute_module(&module, entry, &[Value::number(1.0)]),
            Ok(Value::number(15.0))
        );
    }

    #[test]
    fn thrown_values_unwind_calls_and_nested_handlers_choose_the_nearest_catch() {
        let syntax = parse(
            "/proc/run()\n\tvar/result\n\ttry\n\t\ttry\n\t\t\thelper()\n\t\tcatch(var/inner)\n\t\t\tresult = inner + 1\n\t\t\tthrow 10\n\tcatch(var/outer)\n\t\tresult += outer\n\treturn result\n/proc/helper()\n\tthrow 5\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("nested try/catch should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(16.0)));
    }

    #[test]
    fn catch_without_binding_consumes_the_exception_and_uncaught_throw_errors() {
        let caught = parse("/proc/run()\n\ttry\n\t\tthrow \"test\"\n\tcatch\n\t\treturn 7\n")
            .expect("source should parse");
        let caught = compile_module(&caught.definitions).expect("catch should compile");
        let entry = caught.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&caught, entry, &[]), Ok(Value::number(7.0)));

        let uncaught = parse("/proc/run()\n\tthrow \"test\"\n").expect("source should parse");
        let uncaught = compile_module(&uncaught.definitions).expect("throw should compile");
        let entry = uncaught.procedure_id("/proc/run").expect("entry");
        let error = execute_module(&uncaught, entry, &[]).expect_err("throw should escape");
        assert!(error.message.contains("uncaught exception:"));
        assert!(error.message.contains("test"));
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
    fn sized_list_construction_supports_gc_and_master_initialization_shapes() {
        let syntax = parse(
            "/proc/build_gc_queues(count)\n\tvar/list/queues = new /list(count)\n\tfor(var/i in 1 to count)\n\t\tqueues[i] = list()\n\treturn queues\n/proc/build_stages(count)\n\tvar/list/stages = new(count)\n\tfor(var/i in 1 to count)\n\t\tstages[i] = list(i)\n\treturn stages\n",
        )
        .expect("sized list construction should parse");
        let module = compile_module(&syntax.definitions).expect("sized lists should compile");

        let mut state = ExecutionState::new();
        for (path, count) in [("/proc/build_gc_queues", 5.0), ("/proc/build_stages", 2.0)] {
            let Value::List(list) = execute_module_in_state(
                &module,
                module.procedure_id(path).unwrap(),
                &[Value::number(count)],
                &mut state,
            )
            .expect("sized list writes should stay in bounds") else {
                panic!("sized construction should return a list");
            };
            assert_eq!(state.heap().list(list).unwrap().len(), count as usize);
            assert!(
                state
                    .heap()
                    .list(list)
                    .unwrap()
                    .positions()
                    .all(|(_, value)| matches!(value, Value::List(_)))
            );
        }
    }

    #[test]
    fn multidimensional_new_list_builds_independent_null_filled_rows() {
        let syntax = parse("/proc/build()\n\treturn new /list(2, 3)\n")
            .expect("multidimensional list should parse");
        let program = compile_procedure(&syntax.definitions[0]).unwrap();
        let mut state = ExecutionState::new();
        let Value::List(outer) = execute_in_state(&program, &[], &mut state).unwrap() else {
            panic!("multidimensional construction should return a list");
        };
        let rows = state
            .heap()
            .list(outer)
            .unwrap()
            .positions()
            .map(|(_, value)| match value {
                Value::List(row) => *row,
                _ => panic!("outer positions should be row lists"),
            })
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_ne!(rows[0], rows[1]);
        for row in rows {
            let row = state.heap().list(row).unwrap();
            assert_eq!(row.len(), 3);
            assert!(
                row.positions()
                    .all(|(_, value)| matches!(value, Value::Null))
            );
        }
    }

    #[test]
    fn sized_list_rejects_fractional_negative_and_text_dimensions() {
        let syntax = parse("/proc/build(size)\n\treturn new /list(size)\n").unwrap();
        let program = compile_procedure(&syntax.definitions[0]).unwrap();
        for invalid in [Value::number(-1.0), Value::number(1.5), Value::text("3")] {
            let error = execute(&program, &[invalid]).expect_err("invalid dimension must fail");
            assert!(error.message.contains("list dimension"));
        }
    }

    #[test]
    fn trailing_slash_type_path_is_canonicalized_in_list_keys() {
        let syntax = parse("/proc/build()\n\treturn list(/datum/example/ = 7)\n")
            .expect("trailing-slash type path should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("type-path list should compile");
        let mut state = ExecutionState::new();
        let Value::List(list) =
            execute_in_state(&program, &[], &mut state).expect("initializer should execute")
        else {
            panic!("initializer should return a list");
        };
        let key = Value::TypePath(TypePath::parse("/datum/example").unwrap());
        assert_eq!(
            state.heap().list(list).unwrap().get_key(&key),
            Ok(&Value::number(7.0))
        );
    }

    #[test]
    fn runtime_created_atoms_register_with_world_and_receive_contents() {
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        let world_contents = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .set_datum_field(world, field("contents"), Value::List(world_contents))
            .unwrap();
        state.set_global(field("world"), Value::Datum(world));

        let atom =
            allocate_initialized_datum(&mut state, TypePath::parse("/obj/item/runtime").unwrap())
                .expect("runtime atom should allocate");
        let datum_contents = state.heap().datum_field(atom, &field("contents")).unwrap();

        assert!(matches!(datum_contents, Value::List(_)));
        assert!(
            state
                .heap()
                .list(world_contents)
                .unwrap()
                .contains(&Value::Datum(atom))
        );
    }

    #[test]
    fn direct_loc_assignment_synchronizes_container_contents() {
        let syntax =
            parse("/proc/move(atom, target)\n\tatom.loc = target\n\treturn atom.loc\n").unwrap();
        let program = compile_procedure(&syntax.definitions[0]).unwrap();
        let mut state = ExecutionState::new();
        let old = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/old").unwrap());
        let new = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/new").unwrap());
        let atom = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/obj/item").unwrap());
        let old_contents = state.heap_mut().allocate_list();
        let new_contents = state.heap_mut().allocate_list();
        for (container, list) in [(old, old_contents), (new, new_contents)] {
            state
                .heap_mut()
                .set_datum_field(container, field("contents"), Value::List(list))
                .unwrap();
        }
        state
            .heap_mut()
            .list_mut(old_contents)
            .unwrap()
            .add(Value::Datum(atom));
        state
            .heap_mut()
            .set_datum_field(atom, field("loc"), Value::Datum(old))
            .unwrap();

        assert_eq!(
            execute_in_state(
                &program,
                &[Value::Datum(atom), Value::Datum(new)],
                &mut state
            ),
            Ok(Value::Datum(new))
        );
        assert!(
            !state
                .heap()
                .list(old_contents)
                .unwrap()
                .contains(&Value::Datum(atom))
        );
        assert!(
            state
                .heap()
                .list(new_contents)
                .unwrap()
                .contains(&Value::Datum(atom))
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
            wait_for: true,
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
    fn appended_initializers_scan_module_call_names_once() {
        let source = (0..64)
            .map(|index| format!("/proc/p{index}()\n\treturn {index}\n"))
            .collect::<String>();
        let syntax = parse(&source).expect("procedures should parse");
        let mut module = compile_module(&syntax.definitions).expect("procedures should compile");

        for _ in 0..32 {
            compile_initializer_into_module(
                &expression_tokens("p0()"),
                &BTreeMap::new(),
                &mut module,
            )
            .expect("initializer should append");
        }

        assert_eq!(module.initializer_call_name_index_builds(), 1);
        assert_eq!(module.initializer_call_name_symbols_scanned(), 64);
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
    fn standalone_increment_updates_a_bound_global() {
        let syntax = parse("/proc/update()\n\tuid++\n\treturn uid\n").expect("source should parse");
        let program = compile_procedure_with_resolver_and_fields(
            &syntax.definitions[0],
            &HashMap::new(),
            &BTreeMap::new(),
            &BTreeMap::from([("uid".to_owned(), field("qualified_uid"))]),
            &BTreeMap::new(),
        )
        .expect("a bound global increment should compile");
        let mut state = ExecutionState::new();
        state.set_global(field("qualified_uid"), Value::number(4.0));

        assert_eq!(
            execute_in_state(&program, &[], &mut state),
            Ok(Value::number(5.0))
        );
        assert_eq!(
            state.global(&field("qualified_uid")),
            Some(&Value::number(5.0))
        );
    }

    #[test]
    fn link_builtin_preserves_headless_redirect_payload() {
        let syntax =
            parse("/proc/run()\n\treturn link(\"byond://server\")\n").expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("link should compile as a builtin");
        let entry = module.procedure_id("/proc/run").expect("run entry");
        assert_eq!(
            execute_module(&module, entry, &[]),
            Ok(Value::text("byond://server"))
        );
    }

    #[test]
    fn clamp_accepts_reversed_numeric_bounds() {
        let source = "/proc/test()\n\treturn clamp(15, 10, 0)\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("clamp should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(10.0)));
    }

    #[test]
    fn clamp_list_returns_new_clamped_numeric_values() {
        let source = "/proc/test()\n\tvar/list/input = list(-10, \"skip\", 5, 40)\n\tvar/list/output = clamp(input, 1, 10)\n\treturn output[1] * 100 + output[2] * 10 + output[3]\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("list clamp should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(160.0)));
    }

    #[test]
    fn inverse_trig_builtins_use_dm_degrees_and_fallbacks() {
        let source = "/proc/test()\n\treturn round(arctan(3, 4)) + round(arctan(-1, 1)) + round(arcsin(1)) + round(arccos(0)) + arcsin(2)\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("inverse trig builtins should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(368.0)));
    }

    #[test]
    fn prefix_increment_is_an_expression_for_list_indexing() {
        let source =
            "/proc/test()\n\tvar/list/values = list(10, 20)\n\tvar/i = 0\n\treturn values[++i]\n";
        let syntax = parse(source).expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("prefix increment should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(10.0)));
    }

    #[test]
    fn increment_expressions_follow_byond_coercion_and_return_rules() {
        let source = "/proc/test()\n\tvar/a = 1\n\tvar/old = a++\n\tvar/new_value = ++a\n\tvar/text_value = \"bad\"\n\tvar/text_new = ++text_value\n\tvar/null_value = null\n\tvar/null_new = ++null_value\n\tvar/list/values = list(1)\n\tvar/list_old = values[1]++\n\tvar/list_new = values[1]\n\treturn old * 10000 + new_value * 1000 + text_new * 100 + null_new * 10 + list_old + list_new\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("increment expressions should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(13_113.0)));
    }

    #[test]
    fn decrement_expressions_preserve_postfix_old_value() {
        let source = "/proc/test()\n\tvar/value = 3\n\tvar/old = value--\n\tvar/new_value = --value\n\treturn old * 10 + new_value\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("decrement expressions should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(31.0)));
    }

    #[test]
    fn field_increment_expressions_mutate_once_and_return_correct_value() {
        let source = "/proc/test()\n\tvar/old = count++\n\tvar/current = ++count\n\treturn old * 10 + current\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure_with_resolver_and_fields(
            &syntax.definitions[0],
            &HashMap::new(),
            &BTreeMap::from([("count".to_owned(), field("count"))]),
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .expect("field mutation should compile");
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
            Ok(Value::number(35.0))
        );
        assert_eq!(
            state.heap().datum_field(src, &field("count")),
            Ok(&Value::number(5.0))
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
    fn lowercase_global_namespace_remains_distinct_from_declared_glob_datum() {
        let syntax = parse(
            "/datum/revision/proc/load()\n\treturn 7\n/datum/controller/global_vars/proc/InitGlobalrevdata()\n\tsrc.revdata = new /datum/revision\n/datum/controller/global_vars/Initialize()\n\tfor(var/glob_proc in typesof(/datum/controller/global_vars/proc))\n\t\tcall(src, glob_proc)()\n/proc/early_log()\n\tGLOB.config_error_log = \"early.log\"\n\treturn GLOB.config_error_log\n/proc/run()\n\tGLOB.Initialize()\n\tglobal.counter += 1\n\treturn GLOB.revdata.load() + global.counter\n",
        )
        .unwrap();
        let module = compile_module_with_global_fields(
            &syntax.definitions,
            &BTreeMap::from([
                ("GLOB".to_owned(), field("GLOB")),
                ("counter".to_owned(), field("counter")),
                (
                    "GLOB.config_error_log".to_owned(),
                    FieldName::static_storage("/datum/controller/global_vars/var/config_error_log"),
                ),
            ]),
        )
        .unwrap();
        let mut state = ExecutionState::new();
        state.set_global(field("GLOB"), Value::Null);
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/early_log").unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::text("early.log"))
        );
        let glob = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/controller/global_vars").unwrap());
        state
            .heap_mut()
            .set_datum_field(glob, field("revdata"), Value::Null)
            .unwrap();
        state.set_global(field("GLOB"), Value::Datum(glob));
        state.set_global(field("counter"), Value::number(4.0));
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/run").unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::number(12.0))
        );
        assert!(matches!(
            state.heap().datum_field(glob, &field("revdata")),
            Ok(Value::Datum(_))
        ));
        assert_eq!(state.global(&field("counter")), Some(&Value::number(5.0)));
    }

    #[test]
    fn assignment_expressions_store_and_yield_the_assigned_value() {
        let source = parse(
            "/proc/locals_and_list(items)\n\tvar/local = 1\n\treturn (local = 5) + (items[1] = local)\n/proc/global_assignment()\n\treturn (global.counter = 9)\n",
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
    fn symbolic_dynamic_target_compiles_once_and_survives_scheduler_yield() {
        let source = parse(
            "/proc/entry(receiver)\n\treturn receiver.run()\n/datum/child/proc/run()\n\tsleep(1)\n\treturn 9\n",
        )
        .unwrap();
        let specs = [
            ProcedureSpec {
                path: "/proc/entry@0".to_owned(),
                definition: &source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/child/proc/run@0".to_owned(),
                definition: &source.definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ];
        let module = compile_module_specs_selective(
            &specs,
            &[BTreeMap::new(), BTreeMap::new()],
            &BTreeSet::from([0]),
        )
        .unwrap();
        assert_eq!(module.deferred_procedure_count(), 1);
        assert_eq!(module.materialized_deferred_procedure_count(), 0);
        let deferred_id = module.procedure_id_at(1).unwrap();
        let cloned_module = module.clone();
        assert!(Arc::ptr_eq(&module.deferred, &cloned_module.deferred));
        let original_deferred = module.deferred.get(&deferred_id).unwrap();
        let cloned_deferred = cloned_module.deferred.get(&deferred_id).unwrap();
        assert!(Arc::ptr_eq(
            &original_deferred.definition,
            &cloned_deferred.definition
        ));
        assert!(Arc::ptr_eq(
            &original_deferred.targets,
            &cloned_deferred.targets
        ));
        assert!(Arc::ptr_eq(
            &original_deferred.compiled,
            &cloned_deferred.compiled
        ));

        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/child").unwrap());
        let entry = module.procedure_id_at(0).unwrap();
        assert_eq!(
            execute_module_in_context(
                &module,
                entry,
                &[Value::Datum(receiver)],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::Null)
        );
        assert_eq!(module.materialized_deferred_procedure_count(), 1);
        assert_eq!(state.scheduled_task_count(), 1);
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::number(9.0)])
        );
        assert_eq!(module.materialized_deferred_procedure_count(), 1);
    }

    #[test]
    fn deferred_semantic_error_blocks_only_when_runtime_selects_symbol() {
        let source = parse(
            "/proc/entry(receiver)\n\treturn receiver.run()\n/datum/child/proc/run()\n\treturn 9\n",
        )
        .unwrap();
        let specs = [
            ProcedureSpec {
                path: "/proc/entry@0".to_owned(),
                definition: &source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/child/proc/run@0".to_owned(),
                definition: &source.definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ];
        let module = compile_module_specs_selective_with_errors(
            &specs,
            &[BTreeMap::new(), BTreeMap::new()],
            &BTreeSet::from([0]),
            &BTreeMap::from([(
                1,
                super::CompileError {
                    message: "deferred source semantic failure".to_owned(),
                },
            )]),
        )
        .expect("unselected deferred semantic error must not block module linking");
        assert_eq!(module.materialized_deferred_procedure_count(), 0);

        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/child").unwrap());
        let error = execute_module_in_context(
            &module,
            module.procedure_id_at(0).unwrap(),
            &[Value::Datum(receiver)],
            &mut state,
            &ExecutionContext::default(),
        )
        .expect_err("selecting the invalid deferred symbol must fail");
        assert_eq!(error.message, "deferred source semantic failure");
        assert_eq!(module.materialized_deferred_procedure_count(), 1);
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
    fn namespace_qualified_call_is_parsed_as_static_call() {
        let source =
            parse("/proc/entry()\n\tTypeA::helper()\n\treturn 11\n/proc/helper()\n\treturn 11\n")
                .expect("source should parse");
        let module = compile_module(&source.definitions)
            .expect("namespace-qualified static calls should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry should resolve");
        let helper = module
            .procedure_id("/proc/helper")
            .expect("helper should resolve");

        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(11.0)));
        let entry_program = module.procedure(entry).expect("entry program should exist");
        assert!(
            entry_program.instructions.iter().any(|instruction| {
                matches!(instruction, Instruction::Call { procedure, .. } if *procedure == helper)
            }),
            "namespace-qualified call should resolve to a real static call",
        );
    }

    #[test]
    fn spawn_statement_runs_only_when_its_scheduler_delay_elapses() {
        let source = parse(
            "/proc/entry()\n\tspawn(1)\n\t\thelper()\n\treturn 11\n/proc/helper()\n\treturn 22\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions).expect("spawn statement should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry should resolve");

        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_state(&module, entry, &[], &mut state),
            Ok(Value::number(11.0))
        );
        assert_eq!(
            advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
            Ok(Vec::new())
        );
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::Null])
        );
    }

    #[test]
    fn scheduler_advances_world_clock_with_tick_lag_and_resets_tick_usage() {
        let source = parse(
            "/proc/entry()\n\tspawn(3)\n\t\treturn_usage()\n/proc/return_usage()\n\tworld.observed = world.tick_usage\n",
        )
        .unwrap();
        let module = compile_module(&source.definitions).unwrap();
        let entry = module.procedure_id("/proc/entry").unwrap();
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        for (name, value) in [
            ("tick_lag", 2.0),
            ("fps", 5.0),
            ("time", 0.0),
            ("timeofday", 863_999.0),
            ("tick_usage", 0.0),
        ] {
            state
                .heap_mut()
                .set_datum_field(world, field(name), Value::number(value))
                .unwrap();
        }
        state.set_global(field("world"), Value::Datum(world));

        assert_eq!(
            execute_module_in_state(&module, entry, &[], &mut state),
            Ok(Value::Null)
        );
        assert_eq!(state.next_scheduled_tick(), Some(2));
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![])
        );
        assert_eq!(crate::world_numeric_field(&state, "time"), Some(2.0));
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::Null])
        );
        assert_eq!(
            state.heap().datum_field(world, &field("observed")),
            Ok(&Value::number(100.0))
        );
        assert_eq!(crate::world_numeric_field(&state, "time"), Some(4.0));
        assert_eq!(crate::world_numeric_field(&state, "timeofday"), Some(3.0));
        assert_eq!(crate::world_numeric_field(&state, "tick_usage"), Some(0.0));
    }

    #[test]
    fn world_fps_and_tick_lag_assignments_remain_reciprocal() {
        let source = parse(
            "/proc/set_fps()\n\tworld.fps = 20\n\treturn world.tick_lag\n/proc/set_lag()\n\tworld.tick_lag = 2\n\treturn world.fps\n",
        )
        .unwrap();
        let module = compile_module(&source.definitions).unwrap();
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        state.set_global(field("world"), Value::Datum(world));

        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/set_fps").unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::number(0.5))
        );
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/set_lag").unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::number(5.0))
        );
    }

    #[test]
    fn spawn_without_parentheses_defaults_to_zero_delay_for_inline_and_block_bodies() {
        for source in [
            "/proc/entry()\n\tspawn helper()\n\treturn 1\n/proc/helper()\n\treturn 2\n",
            "/proc/entry()\n\tspawn {\n\t\thelper()\n\t}\n\treturn 1\n/proc/helper()\n\treturn 2\n",
        ] {
            let syntax = parse(source).expect("source should parse");
            let module =
                compile_module(&syntax.definitions).expect("parenthesis-free spawn should compile");
            let entry = module.procedure_id("/proc/entry").expect("entry");
            let mut state = ExecutionState::new();
            assert_eq!(
                execute_module_in_state(&module, entry, &[], &mut state),
                Ok(Value::number(1.0))
            );
            assert_eq!(
                advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
                Ok(vec![Value::Null])
            );
        }
    }

    #[test]
    fn sleep_yields_and_resumes_the_full_procedure_frame() {
        let source = parse("/proc/entry()\n\tvar/value = sleep(1)\n\treturn value + 11\n")
            .expect("source should parse");
        let module = compile_module(&source.definitions).expect("sleep should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry should resolve");
        let mut state = ExecutionState::new();

        assert_eq!(
            execute_module_in_state(&module, entry, &[], &mut state),
            Ok(Value::Null),
            "a yielding entry returns control to the scheduler"
        );
        assert_eq!(
            advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
            Ok(Vec::new())
        );
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::number(11.0)])
        );
    }

    #[test]
    fn sleep_preserves_callers_waiting_on_a_nested_call() {
        let source =
            parse("/proc/entry()\n\treturn helper() + 1\n/proc/helper()\n\tsleep(1)\n\treturn 2\n")
                .expect("source should parse");
        let module = compile_module(&source.definitions).expect("sleep should compile");
        let entry = module
            .procedure_id("/proc/entry")
            .expect("entry should resolve");
        let mut state = ExecutionState::new();

        assert_eq!(
            execute_module_in_state(&module, entry, &[], &mut state),
            Ok(Value::Null)
        );
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::number(3.0)])
        );
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
        let source = parse(
            "/proc/unit()\n\treturn rand()\n/proc/range()\n\treturn rand(4, 6)\n/proc/chance()\n\treturn prob(100)\n",
        )
                .expect("source should parse");
        let module = compile_module(&source.definitions).expect("random builtins should compile");
        let range = module
            .procedure_id("/proc/range")
            .expect("range should exist");
        let unit = module
            .procedure_id("/proc/unit")
            .expect("unit should exist");
        let chance = module
            .procedure_id("/proc/chance")
            .expect("chance should exist");
        let first = execute_module(&module, range, &[]).expect("rand should execute");
        let second =
            execute_module(&module, range, &[]).expect("fresh states should reproduce rand");
        assert_eq!(first, second);
        assert!(matches!(first.as_number(), Some(value) if (4.0..=6.0).contains(&value)));
        let unit_value = execute_module(&module, unit, &[]).expect("rand() should execute");
        assert!(
            matches!(unit_value.as_number(), Some(value) if (0.0..1.0).contains(&value)),
            "rand() returned {unit_value}"
        );
        assert_eq!(execute_module(&module, chance, &[]), Ok(Value::number(1.0)));
    }

    #[test]
    fn rand_seed_resets_the_stream_consumed_by_random_builtins() {
        let source = parse(
            "/proc/seeded(seed)\n\trand_seed(seed)\n\treturn rand(1, 1000000) * 100 + pick(10, 20, 30) + prob(50)\n",
        )
        .expect("rand_seed source should parse");
        let module = compile_module(&source.definitions).expect("rand_seed should compile");
        let entry = module.procedure_id("/proc/seeded").expect("seeded proc");
        let mut state = ExecutionState::new();
        let first =
            execute_module_in_state(&module, entry, &[Value::number(29051994.0)], &mut state)
                .expect("first seeded sequence");
        let repeated =
            execute_module_in_state(&module, entry, &[Value::number(29051994.0)], &mut state)
                .expect("repeated seeded sequence");
        assert_eq!(first, repeated, "reseeding must reproduce the whole stream");
    }

    #[test]
    fn roll_supports_numeric_and_encoded_dice_forms() {
        let source = dm_syntax::parse(
            "/proc/numeric()\n\treturn roll(3, 6)\n/proc/encoded()\n\treturn roll(\"2d4+5\")\n",
        )
        .expect("dice source should parse");
        let module = compile_module(&source.definitions).expect("roll should compile");
        let numeric = execute_module(&module, module.procedure_id("/proc/numeric").unwrap(), &[])
            .expect("numeric dice should execute");
        let encoded = execute_module(&module, module.procedure_id("/proc/encoded").unwrap(), &[])
            .expect("encoded dice should execute");
        assert!(
            numeric
                .as_number()
                .is_some_and(|value| (3.0..=18.0).contains(&value))
        );
        assert!(
            encoded
                .as_number()
                .is_some_and(|value| (7.0..=13.0).contains(&value))
        );
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
    fn calls_accept_omitted_positional_arguments() {
        let source = parse(
            "/proc/entry()\n\treturn helper(3,, 4)\n/proc/helper(first, omitted, third)\n\treturn first * 10 + third + isnull(omitted)\n",
        )
        .expect("source should parse");
        let module =
            compile_module(&source.definitions).expect("interior omitted arguments should compile");
        let entry = module.procedure_id("/proc/entry").unwrap();
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(35.0)));
    }

    #[test]
    fn expression_produced_procedure_selectors_are_invocable() {
        let source = parse(
            "/proc/entry()\n\treturn selector()(4)\n/proc/selector()\n\treturn \"/proc/helper\"\n/proc/helper(value)\n\treturn value + 3\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions)
            .expect("a procedure selector returned by an expression should compile");
        let entry = module.procedure_id("/proc/entry").unwrap();
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(7.0)));
    }

    #[test]
    fn call_ext_retains_both_selectors_and_call_arguments() {
        let source =
            parse("/proc/entry()\n\treturn call_ext(\"bridge.dll\", \"run\")(1, \"two\")\n")
                .expect("call_ext source should parse");
        let program = compile_procedure(&source.definitions[0])
            .expect("call_ext selector and invocation should compile");
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::ExternalCall { argument_count: 2 }
        )));
        let error = execute(&program, &[]).expect_err("headless execution has no host bridge");
        assert!(error.message.contains("bridge.dll"));
        assert!(error.message.contains("run"));
        assert!(error.message.contains("installed host bridge"));
    }

    #[test]
    fn special_result_supports_indexed_assignment() {
        let source =
            parse("/proc/entry()\n\t. = list()\n\t.[\"answer\"] = 42\n\treturn .[\"answer\"]\n")
                .expect("source should parse");
        let program = compile_procedure(&source.definitions[0])
            .expect("indexed special-result assignment should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(42.0)));
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
    fn dynamic_call_canonicalizes_global_proc_selectors_without_double_proc_segment() {
        let source = parse(
            "/proc/entry(selector)\n\treturn call(selector)(4)\n/proc/Log(value)\n\treturn value + 3\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions).expect("module should compile");
        let entry = module.procedure_id("/proc/entry").unwrap();

        for selector in [
            Value::text("Log"),
            Value::text("proc/Log"),
            Value::TypePath(TypePath::parse("/proc/Log").unwrap()),
        ] {
            assert_eq!(
                execute_module(&module, entry, &[selector]),
                Ok(Value::number(7.0))
            );
        }
    }

    #[test]
    fn dynamic_member_call_on_null_is_not_reinterpreted_as_a_global_proc() {
        let source = parse(
            "/proc/entry()\n\tvar/datum/logger = null\n\treturn logger.Log(4)\n/proc/Log(value)\n\treturn value + 3\n",
        )
        .expect("source should parse");
        let module = compile_module(&source.definitions).expect("module should compile");
        let error = execute_module(&module, module.procedure_id("/proc/entry").unwrap(), &[])
            .expect_err("null member calls must remain datum calls");
        assert!(error.message.contains("procedure on null"));
    }

    #[test]
    fn bare_owner_proc_with_arglist_binds_current_src_not_global_namespace() {
        let source = parse(
            "/datum/log_holder/proc/init_logging()\n\tvar/list/arg_list = list(4)\n\treturn Log(arglist(arg_list))\n/datum/log_holder/proc/Log(value)\n\treturn src.base + value\n",
        )
        .expect("Monk-shaped logger source should parse");
        let module = compile_module_specs(&[
            ProcedureSpec {
                path: "/datum/log_holder/proc/init_logging@0".to_owned(),
                definition: &source.definitions[0],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
            ProcedureSpec {
                path: "/datum/log_holder/proc/Log@0".to_owned(),
                definition: &source.definitions[1],
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            },
        ])
        .expect("bare Log call should owner-resolve");
        let mut state = ExecutionState::new();
        let logger = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/log_holder").unwrap());
        state
            .heap_mut()
            .set_datum_field(logger, field("base"), Value::number(3.0))
            .unwrap();

        assert_eq!(
            execute_module_in_context(
                &module,
                module.procedure_id_at(0).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(logger), Value::Null),
            ),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    #[allow(clippy::vec_init_then_push)]
    fn null_conditional_field_index_and_call_short_circuit_without_rhs_evaluation() {
        let source = parse(
            "/datum/example/proc/read(value, list/values)\n\tvar/a = value?.field\n\tvar/b = values?[bump()]\n\tvar/c = value?:take(bump())\n\tvalue?.field = bump()\n\tvalues?[bump()] = bump()\n\treturn isnull(a) + isnull(b) + isnull(c) + global.calls\n/datum/example/proc/take(value)\n\treturn value\n/proc/bump()\n\tglobal.calls += 1\n\treturn 1\n",
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
            "/proc/plane_macro(flag, other)\n\tvar/output = 0\n\tdo { if(flag) { var/_cached_plane = 7; var/_our_turf = other; if(_our_turf) { var/key = \"[_cached_plane]\"; output = _cached_plane; } else if(other) { output = _cached_plane; } else { output = _cached_plane; } } else { output = 2; } } while(0)\n\treturn output\n",
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
    fn compact_brace_scope_survives_physical_macro_lines() {
        let source = parse(
            "/proc/plane_macro(flag, inner)\n\tvar/output = 0\n\tif(flag) { var/_cached_plane = 7; if(inner) { output = 1; } else if(flag) { output = 2;\n\t} else { output = _cached_plane; } } else { output = 3; }\n\treturn output\n",
        )
        .expect("continued compact macro body should parse");
        let module = compile_module(&source.definitions)
            .expect("continued compact macro scope should compile");
        let entry = module.procedure_id("/proc/plane_macro").expect("entry");
        assert_eq!(
            execute_module(&module, entry, &[Value::number(1.0), Value::Null]),
            Ok(Value::number(2.0))
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
    fn null_plus_equals_list_initializes_a_lazy_list() {
        let source = parse("/proc/queue(value)\n\tvar/list/waiting\n\twaiting += list(value)\n\treturn waiting[1]\n")
        .expect("lazy list source should parse");
        let module = compile_module(&source.definitions).expect("lazy list should compile");
        let entry = module.procedure_id("/proc/queue").expect("queue proc");
        assert_eq!(
            execute_module(&module, entry, &[Value::number(7.0)]),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn missing_association_plus_equals_list_initializes_nested_collection() {
        let source = parse(
            "/proc/queue()\n\tvar/list/groups = list()\n\tgroups[\"master\"] += list(/datum/one)\n\tgroups[\"master\"] += list(/datum/two)\n\treturn groups[\"master\"].len\n",
        )
        .expect("nested lazy list source should parse");
        let module = compile_module(&source.definitions).expect("nested lazy list should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/queue").unwrap(), &[]),
            Ok(Value::number(2.0))
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
    fn namespaced_runtime_type_value_reads_its_initial_field() {
        let source = parse("/proc/read_mode(component_type)\n\treturn component_type::dupe_mode\n")
            .expect("namespaced value source should parse");
        let module = compile_module(&source.definitions)
            .expect("namespaced runtime type value should compile");
        let component = TypePath::parse("/datum/component/example").unwrap();
        let mut state = ExecutionState::new();
        state.set_initial_values(BTreeMap::from([(
            component.clone(),
            BTreeMap::from([(field("dupe_mode"), Value::number(3.0))]),
        )]));
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/read_mode").unwrap(),
                &[Value::TypePath(component)],
                &mut state,
            ),
            Ok(Value::number(3.0))
        );
    }

    #[test]
    fn scope_operator_supports_type_src_and_global_values() {
        let type_source = parse("/proc/read()\n\treturn /datum/example::flag\n")
            .expect("type scope source should parse");
        let type_program =
            compile_procedure(&type_source.definitions[0]).expect("type scope should compile");
        let path = TypePath::parse("/datum/example").unwrap();
        let mut state = ExecutionState::new();
        state.set_initial_values(BTreeMap::from([(
            path.clone(),
            BTreeMap::from([(field("flag"), Value::number(7.0))]),
        )]));
        assert_eq!(
            execute_in_context(&type_program, &[], &mut state, &ExecutionContext::default(),),
            Ok(Value::number(7.0))
        );

        let global_source =
            parse("/proc/read()\n\treturn ::answer\n").expect("global scope source should parse");
        let global_program =
            compile_procedure(&global_source.definitions[0]).expect("global scope should compile");
        state.set_global(field("answer"), Value::number(42.0));
        assert_eq!(
            execute_in_context(
                &global_program,
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(42.0))
        );

        let src_source =
            parse("/proc/read()\n\treturn src::flag\n").expect("src scope source should parse");
        let src_program =
            compile_procedure(&src_source.definitions[0]).expect("src scope should compile");
        let src = state.heap_mut().allocate_datum(path);
        assert_eq!(
            execute_in_context(
                &src_program,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(src), Value::Null),
            ),
            Ok(Value::number(7.0))
        );
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
        let source = "/proc/probe()\n\treturn (0xFFFFFF & 6) + (7 ^ 3 | 8) + (9.9 & 3)\n";
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

        // ~9 and ~0 are 24-bit complements. Their binary32 sum rounds to
        // the nearest representable value at this magnitude.
        assert_eq!(execute(&program, &[]), Ok(Value::number(33_554_420.0)));
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
    fn shift_operators_and_compound_assignments_use_byond_24_bit_semantics() {
        let source = "/proc/probe(items)\n\tvar/value = 3 << 2\n\tvalue >>= 1\n\titems[1] <<= value\n\treturn (8 >> 2) + items[1] + (1 << 33)\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("shift expressions and assignments should compile");
        let mut state = ExecutionState::new();
        let list = state.heap.allocate_list();
        state.heap.list_mut(list).unwrap().add(Value::number(1.0));

        // value is (3 << 2) >> 1 = 6; item becomes 1 << 6 = 64.
        // 8 >> 2 is 2, and counts >=24 shift every effective bit away.
        assert_eq!(
            execute_in_state(&program, &[Value::List(list)], &mut state),
            Ok(Value::number(66.0))
        );
        assert!(program.instructions.iter().any(|instruction| {
            matches!(
                instruction,
                Instruction::ShiftLeft | Instruction::ShiftRight
            )
        }));
    }

    #[test]
    fn documented_pure_standard_procs_cover_sort_params_and_number_text() {
        let source = parse(
            "/proc/probe()\n\tvar/list/p = params2list(\"a=one+two&b=%26\")\n\tif(p[\"a\"] != \"one two\" || p[\"b\"] != \"&\")\n\t\treturn 0\n\tif(list2params(p) != \"a=one+two&b=%26\")\n\t\treturn 0\n\tif(lentext(\"abc\") != 3)\n\t\treturn 0\n\tif(sorttext(\"A\", \"b\") != 1 || sorttextEx(\"a\", \"B\") != -1)\n\t\treturn 0\n\tif(num2text(11, 2, 16) != \"0b\")\n\t\treturn 0\n\treturn 1\n",
        )
        .expect("pure standard-proc source should parse");
        let module =
            compile_module(&source.definitions).expect("pure standard procs should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn documented_operator_semantics_cover_short_circuit_modulo_compare_and_equivalence() {
        let source = parse(
            "/proc/probe()\n\tvar/list/a = list(\"key\" = 7, 2)\n\tvar/list/b = list(\"key\" = 7, 2)\n\tvar/list/c = list(\"key\" = 8, 2)\n\tvar/legacy = 5.9 % 2.1\n\tvar/fractional = 5.5 %% 2\n\tlegacy %= 2\n\tfractional %%= 1.25\n\tif((a ~= b) != 1 || (a ~! c) != 1)\n\t\treturn -100\n\tif((3 <=> 4) != -1 || (\"b\" <=> \"a\") != 1 || (1 <> 2) != 1)\n\t\treturn -101\n\tif((99 in null) != 0)\n\t\treturn -102\n\tvar/or_value = \"\" || \"fallback\"\n\tvar/and_value = \"left\" && \"right\"\n\tvar/skip_or = 1 || list()[99]\n\tvar/skip_and = 0 && list()[99]\n\tif(or_value != \"fallback\" || and_value != \"right\" || skip_or != 1 || skip_and != 0)\n\t\treturn -103\n\treturn legacy + fractional\n",
        )
        .expect("documented operator source should parse");
        let module =
            compile_module(&source.definitions).expect("documented operators should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
            Ok(Value::number(1.25))
        );
    }

    #[test]
    fn bitwise_operators_use_byonds_24_effective_bits() {
        let source = parse(
            "/proc/probe()\n\tvar/a = ~0\n\tvar/b = 1 << 24\n\tvar/c = 0xFFFFFF >> 23\n\treturn a + b + c\n",
        )
        .expect("bitwise source should parse");
        let module = compile_module(&source.definitions).expect("bitwise source should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
            Ok(Value::number(16_777_216.0))
        );
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

        // `:` is both the conditional delimiter and DreamMaker's dynamic
        // member operator.  A member access in the false arm must not be
        // mistaken for a second conditional delimiter.
        let dynamic_false_arm =
            parse("/proc/dynamic_false_arm(input)\n\treturn input ? 7 : input:type\n")
                .expect("dynamic-member conditional source should parse");
        let program = compile_procedure(&dynamic_false_arm.definitions[0])
            .expect("dynamic member in the false arm should compile");
        assert!(
            execute(&program, &[Value::Null])
                .expect_err("reading a dynamic field from null should fail at runtime")
                .message
                .contains("field read received null")
        );

        let nested_false_arm = parse("/proc/nested(a, b)\n\treturn a ? (b ? 10 : 20) : 30\n")
            .expect("nested conditional source should parse");
        let program = compile_procedure(&nested_false_arm.definitions[0])
            .expect("an outer delimiter after a nested false arm should compile");
        assert_eq!(
            execute(&program, &[Value::number(1.0), Value::number(0.0)]),
            Ok(Value::number(20.0))
        );

        let macro_nested = parse(
            "/proc/macro_nested(a, b, c, d, e, f, g)\n\treturn ((a) ? (b?[\"x\"] ? -9 : (-9) - (((c) ? (d ? e[f] : g) : 0) + 1)) : (-9))\n",
        )
        .expect("macro-expanded nested conditional source should parse");
        compile_procedure(&macro_nested.definitions[0])
            .expect("nested conditional delimiters should remain distinct from dynamic access");
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
    fn arglist_null_expands_to_zero_callback_arguments() {
        let source = parse(
            "/proc/invoke(arguments)\n\treturn call(/proc/target)(arglist(arguments))\n/proc/target(value = 9)\n\treturn value\n",
        )
        .expect("callback-shaped arglist source should parse");
        let module = compile_module(&source.definitions).expect("arglist(null) should compile");
        let entry = module.procedure_id("/proc/invoke").expect("invoke proc");
        assert_eq!(
            execute_module(&module, entry, &[Value::Null]),
            Ok(Value::number(9.0))
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
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("call stack:\n  /proc/broken at instruction"));
        assert!(diagnostic.contains("\n  /proc/main at instruction"));
        assert!(diagnostic.contains(&format!(
            "(source {}..{})",
            expected_span.start, expected_span.end
        )));
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
    fn do_while_accepts_byond_single_statement_body() {
        // The DM reference defines the body as a Statement, which may be a
        // block or one statement. One level of indentation is sufficient; it
        // does not require a nested multi-line block.
        let source = "/proc/count(limit)\n\tvar/result = 0\n\tdo\n\t\tresult += 1\n\twhile(result < limit)\n\treturn result\n";

        assert_eq!(execute_source(source, 0.0), Value::number(1.0));
        assert_eq!(execute_source(source, 4.0), Value::number(4.0));
    }

    #[test]
    fn do_while_accepts_multiline_braced_macro_body() {
        // Continued macros and generated DM commonly spell statement blocks
        // with braces. The lexer retains the whole delimited region as one
        // logical line, then compact-statement normalization must recover the
        // same structure as an indented DM block.
        let source = "/proc/count(limit)\n\tvar/result = 0\n\tdo {\n\t\tresult += 1;\n\t\tif(result == 2) {\n\t\t\tcontinue;\n\t\t}\n\t\tif(result > limit) {\n\t\t\tbreak;\n\t\t}\n\t} while(result <= limit)\n\treturn result\n";

        assert_eq!(execute_source(source, 0.0), Value::number(1.0));
        assert_eq!(execute_source(source, 3.0), Value::number(4.0));
    }

    #[test]
    fn conditional_accepts_inline_braced_do_while_macro_statement() {
        let source = "/proc/run(enabled)\n\tvar/result = 0\n\tif(enabled) do { result += 2; } while(0); result += 1\n\treturn result\n";

        assert_eq!(execute_source(source, 0.0), Value::number(1.0));
        assert_eq!(execute_source(source, 1.0), Value::number(3.0));
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
    fn labeled_loop_break_exits_the_selected_loop() {
        let source = "/proc/run()\n\tvar/result = 0\n\touter:\n\t\tfor(var/x in 1 to 3)\n\t\t\tfor(var/y in 1 to 3)\n\t\t\t\tresult += 1\n\t\t\t\tbreak outer\n\treturn result\n";

        assert_eq!(execute_source(source, 0.0), Value::number(1.0));
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

        let comma_source = "/proc/sum(limit)\n\tvar/total = 0\n\tfor(var/i = 0, i < limit, i++)\n\t\ttotal += i\n\treturn total\n";
        assert_eq!(execute_source(comma_source, 5.0), Value::number(10.0));
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

        let empty = "/proc/count()\n\tvar/i = 0\n\tfor()\n\t\tif(i > 3)\n\t\t\tbreak\n\t\ti++\n\treturn i\n";
        let syntax = parse(empty).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("empty for should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(4.0)));

        let one_separator = "/proc/count()\n\tvar/i = 1\n\tvar/count = 0\n\tfor(, i++ <= 3)\n\t\tcount++\n\treturn count\n";
        let syntax = parse(one_separator).expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("short comma for should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(3.0)));
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
        assert_eq!(
            execute(&program, &[Value::Null]),
            Ok(Value::Null),
            "BYOND treats for-in over null as an empty iteration"
        );
    }

    #[test]
    fn for_in_and_for_to_accept_existing_iterator_locals() {
        let list_source = "/proc/sum()\n\tvar/item\n\tvar/total = 0\n\tfor(item in list(1, 2, 3))\n\t\ttotal += item\n\treturn total\n";
        let syntax = parse(list_source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("existing for-in iterator should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(6.0)));

        let range_source = "/proc/sum()\n\tvar/item\n\tvar/total = 0\n\tfor(item in 1 to 3)\n\t\ttotal += item\n\treturn total\n";
        let syntax = parse(range_source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("existing for-to iterator should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(6.0)));

        let assignment_range = "/proc/sum()\n\tvar/total = 0\n\tfor(var/item = 1 to 8 step 3)\n\t\ttotal += item\n\treturn total\n";
        let syntax = parse(assignment_range).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("assignment-style for-to should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(12.0)));

        let empty_range =
            "/proc/check()\n\tvar/item = -1\n\tfor(item = 1 to 0)\n\t\tcontinue\n\treturn item\n";
        let syntax = parse(empty_range).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("empty existing-variable range should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(-1.0)));
    }

    #[test]
    fn type_for_loop_enumerates_only_live_matching_datums() {
        let syntax = parse(
            "/proc/count()\n\tvar/total = 0\n\tfor(var/datum/a/item)\n\t\ttotal++\n\treturn total\n",
        )
        .expect("type loop should parse");
        let module = compile_module(&syntax.definitions).expect("type loop should compile");
        let mut state = ExecutionState::new();
        let datum = TypePath::parse("/datum").unwrap();
        let a = TypePath::parse("/datum/a").unwrap();
        let child = TypePath::parse("/datum/a/child").unwrap();
        let b = TypePath::parse("/datum/b").unwrap();
        state.set_type_parents(BTreeMap::from([
            (datum.clone(), None),
            (a.clone(), Some(datum.clone())),
            (child.clone(), Some(a.clone())),
            (b.clone(), Some(datum)),
        ]));
        state.heap_mut().allocate_datum(a);
        state.heap_mut().allocate_datum(child);
        state.heap_mut().allocate_datum(b);
        let entry = module.procedure_id("/proc/count").expect("entry");
        assert_eq!(
            execute_module_in_state(&module, entry, &[], &mut state),
            Ok(Value::number(2.0))
        );
    }

    #[test]
    fn associative_for_loop_binds_keys_values_and_writable_targets() {
        let syntax = parse(
            "/proc/run()\n\tvar/list/items = list(\"a\", \"b\" = 5)\n\tvar/total = 0\n\tfor(var/key, value in items)\n\t\ttotal += (key == \"a\") + value\n\tvar/existing_key\n\tvar/existing_value\n\tfor(existing_key, existing_value in items)\n\t\ttotal += 0\n\tvar/list/out = list(null, null)\n\tfor(out[1], out[2] in items)\n\t\ttotal += 0\n\treturn total + (existing_key == \"b\") + (existing_value == 5) + (out[1] == \"b\") + (out[2] == 5)\n",
        )
        .expect("associative loop should parse");
        let module = compile_module(&syntax.definitions).expect("associative loop should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(10.0)));
    }

    #[test]
    fn exotic_c_style_for_headers_follow_byond_declaration_and_range_fakeouts() {
        let syntax = parse(
            "/proc/run()\n\tvar/out1 = 0\n\tfor(var/x = 2 in 1 to 20; x < 6; x++)\n\t\tout1 += x\n\tvar/out2 = 0\n\tfor(var/y in 1 to 5;)\n\t\tout2 += y\n\tvar/out3 = 0\n\tfor(var/z = 5 in 1 to 20; z < 10)\n\t\tout3 += z\n\t\tout3++\n\t\tif(out3 > 10)\n\t\t\tbreak\n\tvar/out4 = 0\n\tfor(var/a && var/b, a < b + 4, a += 2)\n\t\tout4++\n\treturn out1 * 1000 + out2 * 100 + out3 * 10 + out4\n",
        )
        .expect("exotic loops should parse");
        let module = compile_module(&syntax.definitions).expect("exotic loops should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(
            execute_module(&module, entry, &[]),
            Ok(Value::number(15_622.0))
        );
    }

    #[test]
    fn range_for_can_reuse_bare_and_explicit_src_field_iterators() {
        for iterator in ["idx", "src.idx"] {
            let source = format!(
                "/datum/example/proc/run()\n\tfor({iterator} in 1 to 5)\n\t\tc += idx\n\treturn c\n"
            );
            let syntax = parse(&source).expect("field range loop should parse");
            let fields = BTreeMap::from([
                ("idx".to_owned(), field("idx")),
                ("c".to_owned(), field("c")),
            ]);
            let program = compile_procedure_with_resolver_and_fields(
                &syntax.definitions[0],
                &HashMap::new(),
                &fields,
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .expect("field range loop should compile");
            let mut state = ExecutionState::new();
            let src = state
                .heap_mut()
                .allocate_datum(TypePath::parse("/datum/example").unwrap());
            state
                .heap_mut()
                .set_datum_field(src, field("idx"), Value::number(0.0))
                .unwrap();
            state
                .heap_mut()
                .set_datum_field(src, field("c"), Value::number(0.0))
                .unwrap();
            let result = execute_in_context(
                &program,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(src), Value::Null),
            );
            assert_eq!(result, Ok(Value::number(15.0)));
        }
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
    fn dm_profile_command_constants_are_byond_bitflags() {
        let source = "/proc/profile_flags()\n\treturn PROFILE_START + PROFILE_REFRESH + PROFILE_STOP + PROFILE_CLEAR + PROFILE_RESTART + PROFILE_AVERAGE\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("BYOND profiling constants should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
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
    fn suffix_array_locals_use_declared_name_and_dynamic_dimensions() {
        let syntax = parse(
            "/proc/one(roomSize)\n\tvar/storage[roomSize]\n\treturn storage.len\n/proc/multi(x, y)\n\tvar/list/grid[x][y]\n\treturn grid[1].len\n",
        ).expect("suffix array source");
        let module = compile_module(&syntax.definitions).expect("suffix arrays compile");
        assert_eq!(
            execute_module(
                &module,
                module.procedure_id("/proc/one").unwrap(),
                &[Value::number(4.0)]
            ),
            Ok(Value::number(4.0)),
        );
        assert_eq!(
            execute_module(
                &module,
                module.procedure_id("/proc/multi").unwrap(),
                &[Value::number(2.0), Value::number(3.0)]
            ),
            Ok(Value::number(3.0)),
        );
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
    fn waitfor_directives_set_procedure_call_scheduling() {
        for (value, waits) in [("FALSE", false), ("TRUE", true), ("0", false), ("1", true)] {
            let syntax = parse(&format!(
                "/proc/scheduled()\n\tset waitfor = {value}\n\treturn 17\n"
            ))
            .expect("source should parse");
            let program = compile_procedure(&syntax.definitions[0])
                .expect("waitfor directive should compile");

            assert_eq!(execute(&program, &[]), Ok(Value::number(17.0)));
            assert_eq!(program.wait_for, waits);
        }
    }

    #[test]
    fn waitfor_false_detaches_at_sleep_and_returns_current_dot_to_caller() {
        let syntax = parse(
            "/proc/c()\n\tsleep(1)\n\tsleep(1)\n\treturn 99\n\n/proc/b()\n\tset waitfor = FALSE\n\t. = 7\n\treturn c()\n\n/proc/a()\n\treturn b() + 1\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("waitfor chain should compile");
        let entry = module.procedure_id("/proc/a").unwrap();
        let mut state = ExecutionState::new();

        assert_eq!(
            execute_module_in_state(&module, entry, &[], &mut state),
            Ok(Value::number(8.0))
        );
        assert_eq!(state.scheduled_task_count(), 1);
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![])
        );
        assert_eq!(state.scheduled_task_count(), 1);
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::number(99.0)])
        );
    }

    #[test]
    fn waitfor_false_without_sleep_returns_normally_and_post_sleep_errors_are_scheduled() {
        let syntax = parse(
            "/proc/plain()\n\tset waitfor = 0\n\treturn 12\n\n/proc/fails_later()\n\tset waitfor = 0\n\t. = 4\n\tsleep(1)\n\tCRASH(\"later\")\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("waitfor procedures compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/plain").unwrap(), &[]),
            Ok(Value::number(12.0))
        );

        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/fails_later").unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::number(4.0))
        );
        let error = advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state)
            .expect_err("errors after detachment belong to the scheduled continuation");
        assert!(error.message.contains("later"));
    }

    #[test]
    fn waitfor_false_preserves_spawned_deletion_and_detached_src_context() {
        let syntax = parse(
            "/proc/run()\n\tset waitfor = FALSE\n\t. = 3\n\tspawn(1)\n\t\tqdel(src)\n\tsleep(2)\n\treturn 9\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("procedure should compile");
        let entry = module.procedure_id("/proc/run").unwrap();
        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/test").unwrap());
        let context = ExecutionContext::new(Value::Datum(datum), Value::Null);

        assert_eq!(
            execute_module_in_context(&module, entry, &[], &mut state, &context),
            Ok(Value::number(3.0))
        );
        assert_eq!(state.scheduled_task_count(), 2);
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state)
            .expect("spawned deletion should run");
        assert!(state.heap().datum(datum).is_err());
        assert_eq!(state.scheduled_task_count(), 1);
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::number(9.0)])
        );
    }

    #[test]
    fn verb_set_directives_are_non_executable_metadata() {
        let syntax = parse(
            "/proc/metadata()\n\tset hidden = TRUE\n\tset category = \"Admin\"\n\tset desc = \"Example\"\n",
        )
        .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("BYOND verb set directives should compile as metadata");
        assert_eq!(execute(&program, &[]), Ok(Value::Null));
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
    fn world_dimension_changes_materialize_and_remove_coordinate_turfs() {
        let source = parse(
            "/world/proc/incrementMaxZ()\n\tworld.maxz++\n/proc/grow()\n\tworld.incrementMaxZ()\n\tworld.maxx = 2\n\tworld.maxy = 2\n\tvar/turf/found = locate(2, 2, 2)\n\treturn istype(found, /turf) + (found.x == 2) + (found.y == 2) + (found.z == 2) + istype(found.loc, /area)\n/proc/shrink()\n\tworld.maxz = 1\n\treturn isnull(locate(2, 2, 2))\n",
        )
        .expect("world geometry source should parse");
        let module = compile_module(&source.definitions).expect("world geometry should compile");
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        for (name, value) in [
            ("maxx", Value::number(1.0)),
            ("maxy", Value::number(1.0)),
            ("maxz", Value::number(1.0)),
            ("area", Value::TypePath(TypePath::parse("/area").unwrap())),
            ("turf", Value::TypePath(TypePath::parse("/turf").unwrap())),
        ] {
            state
                .heap_mut()
                .set_datum_field(world, field(name), value)
                .unwrap();
        }
        state.set_global(field("world"), Value::Datum(world));

        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/grow").unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::number(5.0)),
        );
        assert_eq!(state.world_turfs.len(), 8);
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/shrink").unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::number(1.0)),
        );
        assert_eq!(state.world_turfs.len(), 4);
    }

    #[test]
    fn spatial_contents_and_atom_new_preserve_byond_map_cell_identity() {
        let source = parse(
            "/turf/floor/New(where)\n\tsrc.saw_cell = (src == where) + (src.x == 2) + (src.y == 2) + (src.z == 2)\n/obj/item/New(where)\n\tsrc.saw_location = (src.loc == where) + (src.x == 2) + (src.y == 2) + (src.z == 2)\n/proc/load_cell(area/target_area)\n\tvar/turf/original = locate(2, 2, 2)\n\ttarget_area.contents.Add(original)\n\tvar/turf/replaced = new /turf/floor(original)\n\tvar/obj/item = new /obj/item(original)\n\treturn list(original, replaced, item)\n",
        )
        .expect("spatial construction source should parse");
        let module =
            compile_module(&source.definitions).expect("spatial construction should compile");
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        for (name, value) in [
            ("maxx", Value::number(2.0)),
            ("maxy", Value::number(2.0)),
            ("maxz", Value::number(2.0)),
            (
                "area",
                Value::TypePath(TypePath::parse("/area/default").unwrap()),
            ),
            (
                "turf",
                Value::TypePath(TypePath::parse("/turf/default").unwrap()),
            ),
        ] {
            state
                .heap_mut()
                .set_datum_field(world, field(name), value)
                .unwrap();
        }
        state.set_global(field("world"), Value::Datum(world));
        state.resize_world_geometry(world, (2, 2, 2)).unwrap();
        let original = state.turf_at(2, 2, 2).expect("corner turf");
        let old_area = match state.heap().datum_field(original, &field("loc")).unwrap() {
            Value::Datum(area) => *area,
            value => panic!("expected old area, got {value:?}"),
        };
        let new_area =
            allocate_initialized_datum(&mut state, TypePath::parse("/area/replacement").unwrap())
                .unwrap();

        let result = execute_module_in_state(
            &module,
            module.procedure_id("/proc/load_cell").unwrap(),
            &[Value::Datum(new_area)],
            &mut state,
        )
        .expect("map-shaped spatial operations should execute");
        let Value::List(result) = result else {
            panic!("expected result list")
        };
        let values = state
            .heap()
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert_eq!(values[0], Value::Datum(original));
        assert_eq!(
            values[1],
            Value::Datum(original),
            "new turf preserves cell identity"
        );
        let Value::Datum(item) = values[2] else {
            panic!("expected created item")
        };
        let turf = state.heap().datum(original).unwrap();
        assert_eq!(turf.type_path().as_str(), "/turf/floor");
        assert_eq!(turf.field(&field("loc")), Ok(&Value::Datum(new_area)));
        assert_eq!(turf.field(&field("saw_cell")), Ok(&Value::number(4.0)));
        let item_datum = state.heap().datum(item).unwrap();
        assert_eq!(item_datum.field(&field("loc")), Ok(&Value::Datum(original)));
        assert_eq!(
            item_datum.field(&field("saw_location")),
            Ok(&Value::number(4.0)),
            "movable loc and coordinates exist before New"
        );
        let old_contents = match state
            .heap()
            .datum_field(old_area, &field("contents"))
            .unwrap()
        {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        assert!(
            !state
                .heap()
                .list(old_contents)
                .unwrap()
                .contains(&Value::Datum(original))
        );
        let area_contents = match state
            .heap()
            .datum_field(new_area, &field("contents"))
            .unwrap()
        {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        let area_values = state.heap().list(area_contents).unwrap();
        assert_eq!(
            area_values
                .positions()
                .filter(|(_, value)| value.semantic_eq(&Value::Datum(original)))
                .count(),
            1
        );
        assert_eq!(
            area_values
                .positions()
                .filter(|(_, value)| value.semantic_eq(&Value::Datum(item)))
                .count(),
            1
        );
    }

    #[test]
    fn datum_vars_loc_write_uses_engine_spatial_assignment() {
        let source = parse(
            "/proc/preload_loc(atom/movable/thing, turf/destination)\n\tthing.vars[\"loc\"] = destination\n\treturn thing.loc\n",
        )
        .expect("preloader-shaped vars write should parse");
        let module = compile_module(&source.definitions).expect("vars write should compile");
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        for (name, value) in [
            ("maxx", Value::number(2.0)),
            ("maxy", Value::number(1.0)),
            ("maxz", Value::number(1.0)),
            ("area", Value::TypePath(TypePath::parse("/area").unwrap())),
            ("turf", Value::TypePath(TypePath::parse("/turf").unwrap())),
        ] {
            state
                .heap_mut()
                .set_datum_field(world, field(name), value)
                .unwrap();
        }
        state.set_global(field("world"), Value::Datum(world));
        state.resize_world_geometry(world, (2, 1, 1)).unwrap();
        let old_turf = state.turf_at(1, 1, 1).unwrap();
        let new_turf = state.turf_at(2, 1, 1).unwrap();
        let movable = allocate_initialized_datum(
            &mut state,
            TypePath::parse("/atom/movable/preloaded").unwrap(),
        )
        .unwrap();
        super::builtins::move_movable_to_turf(&mut state, movable, old_turf).unwrap();

        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/preload_loc").unwrap(),
                &[Value::Datum(movable), Value::Datum(new_turf)],
                &mut state,
            ),
            Ok(Value::Datum(new_turf))
        );
        for (turf, expected) in [(old_turf, 0), (new_turf, 1)] {
            let Value::List(contents) = state.heap().datum_field(turf, &field("contents")).unwrap()
            else {
                panic!("turf contents must be a list")
            };
            assert_eq!(
                state
                    .heap()
                    .list(*contents)
                    .unwrap()
                    .positions()
                    .filter(|(_, value)| value.semantic_eq(&Value::Datum(movable)))
                    .count(),
                expected
            );
        }
        let datum = state.heap().datum(movable).unwrap();
        assert_eq!(datum.field(&field("x")), Ok(&Value::number(2.0)));
        assert_eq!(datum.field(&field("y")), Ok(&Value::number(1.0)));
        assert_eq!(datum.field(&field("z")), Ok(&Value::number(1.0)));
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
    fn word_filter_findtext_accepts_regex_needles() {
        let syntax = parse(
            "/proc/run(value)\n\tvar/regex/word = regex(@\"^\\w+$\")\n\treturn findtext(value, word)\n",
        )
        .expect("word-filter regex source should parse");
        let module = compile_module(&syntax.definitions).expect("regex findtext should compile");
        let entry = module.procedure_id("/proc/run").unwrap();
        assert_eq!(
            execute_module(&module, entry, &[Value::text("admin")]),
            Ok(Value::number(1.0))
        );
        assert_eq!(
            execute_module(&module, entry, &[Value::text("admin help")]),
            Ok(Value::number(0.0))
        );
    }

    #[test]
    fn multiline_global_regex_find_advances_and_populates_capture_groups() {
        let syntax = parse(
            "/proc/run(text)\n\tvar/regex/entries = new(@\"^(?!#)(.+?)\\s+=\\s+(.+)\", \"gm\")\n\tvar/result = \"\"\n\twhile(entries.Find(text))\n\t\tresult += \"[entries.group[1]]:[entries.group[2]]|\"\n\treturn result\n",
        )
        .expect("admins regex source should parse");
        let module = compile_module(&syntax.definitions).expect("regex.Find should compile");
        let entry = module.procedure_id("/proc/run").unwrap();
        assert_eq!(
            execute_module(
                &module,
                entry,
                &[Value::text(
                    "# ignored = nope\nAlice = Admin\nBob = Moderator\n"
                )],
            ),
            Ok(Value::text("Alice:Admin|Bob:Moderator|"))
        );
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
    fn matrix_constructor_methods_and_equivalence_use_affine_components() {
        let syntax = parse(
            "/proc/run()\n\tvar/matrix/value = matrix(1, 2, 3, 4, 5, 6)\n\tvalue.Add(matrix(7, 8, 9, 10, 11, 12))\n\tvalue.Subtract(matrix(7, 8, 9, 10, 11, 12))\n\tvalue.Multiply(matrix(7, 8, 9, 10, 11, 12))\n\treturn value ~= matrix(39, 54, 78, 54, 75, 108)\n",
        )
        .expect("matrix source should parse");
        let module = compile_module(&syntax.definitions).expect("matrix source should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
    }

    #[test]
    fn matrix_transform_methods_mutate_the_six_public_fields() {
        let syntax = parse(
            "/proc/run()\n\tvar/matrix/value = matrix(1, 2, 3, 4, 5, 6)\n\tvalue.Translate(2)\n\tvalue.Turn(90)\n\treturn value ~= matrix(4, 5, 8, -1, -2, -5)\n",
        )
        .expect("matrix source should parse");
        let module = compile_module(&syntax.definitions).expect("matrix source should compile");
        let entry = module.procedure_id("/proc/run").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
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
        let towards =
            parse("/proc/towards(source, target)\n\treturn get_step_towards(source, target)\n")
                .unwrap();
        let towards = compile_procedure(&towards.definitions[0]).unwrap();
        let target = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/obj/target").unwrap());
        for (name, value) in [("x", 20.0), ("y", 30.0), ("z", 2.0)] {
            state
                .heap_mut()
                .set_datum_field(target, field(name), Value::number(value))
                .unwrap();
        }
        assert_eq!(
            execute_in_state(
                &towards,
                &[Value::Datum(origin), Value::Datum(target)],
                &mut state,
            ),
            Ok(Value::Datum(north_east))
        );
    }

    #[test]
    fn resource_regex_json_and_headless_ui_natives_follow_byond_contracts() {
        let syntax = parse(
            "/proc/resource(value)\n\treturn fcopy_rsc(value)\n/proc/quote(value)\n\treturn REGEX_QUOTE(value)\n/proc/pretty_flag()\n\treturn JSON_PRETTY_PRINT\n/proc/mask_inverse()\n\treturn MASK_INVERSE\n/proc/floor_value(value, multiple)\n\treturn FLOOR(value, multiple)\n/proc/ui(client)\n\twinset(client, \"main\", \"flash=5\")\n\treturn browse(\"<b>ready</b>\", \"window=status\")\n/proc/window_exists(client, control)\n\treturn winexists(client, control)\n/proc/choose(client)\n\treturn alert(client, \"Continue?\", \"Dream64\", \"Yes\", \"No\")\n/proc/colors()\n\tvar/icon/value = icon()\n\tvalue.MapColors(1,0,0, 0,1,0, 0,0,1, 0,0,0)\n\tvalue.Blend(\"#ffffff\", ICON_SUBTRACT, 2, 3)\n\tvalue.SetIntensity(0.25, 0.5, 0.75)\n\treturn value\n",
        )
        .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();
        let mut state = ExecutionState::new();
        let run = |path: &str, arguments: &[Value], state: &mut ExecutionState| {
            execute_module_in_state(
                &module,
                module.procedure_id(path).unwrap(),
                arguments,
                state,
            )
            .unwrap()
        };
        assert_eq!(
            run("/proc/resource", &[Value::text("icons/a.dmi")], &mut state),
            Value::text("icons/a.dmi")
        );
        assert_eq!(
            run("/proc/quote", &[Value::text("a+b.c?")], &mut state),
            Value::text("a\\+b\\.c\\?")
        );
        assert_eq!(
            run("/proc/pretty_flag", &[], &mut state),
            Value::number(1.0)
        );
        assert_eq!(
            run("/proc/mask_inverse", &[], &mut state),
            Value::number(1.0)
        );
        assert_eq!(
            run(
                "/proc/floor_value",
                &[Value::number(17.0), Value::number(5.0)],
                &mut state,
            ),
            Value::number(15.0)
        );

        let client = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/client").unwrap());
        assert_eq!(
            run("/proc/choose", &[Value::Datum(client)], &mut state),
            Value::text("Yes")
        );
        let browse = run("/proc/ui", &[Value::Datum(client)], &mut state);
        assert!(matches!(browse, Value::List(_)));
        let settings = state
            .heap()
            .datum_field(client, &field("_dream64_winset"))
            .unwrap();
        let Value::List(settings) = settings else {
            panic!("winset state should be a list");
        };
        assert_eq!(
            state
                .heap()
                .list(*settings)
                .unwrap()
                .get_key(&Value::text("main")),
            Ok(&Value::text("flash=5"))
        );
        assert_eq!(
            run(
                "/proc/window_exists",
                &[Value::Datum(client), Value::text("main")],
                &mut state,
            ),
            Value::number(1.0)
        );
        let Value::Datum(icon) = run("/proc/colors", &[], &mut state) else {
            panic!("icon() should return an icon datum");
        };
        let Value::List(matrix) = state
            .heap()
            .datum_field(icon, &field("_dream64_color_matrix"))
            .unwrap()
        else {
            panic!("MapColors should retain the headless matrix");
        };
        assert_eq!(state.heap().list(*matrix).unwrap().len(), 12);
        assert_eq!(
            state.heap().list(*matrix).unwrap().get(1),
            Ok(&Value::number(0.25))
        );
        assert_eq!(
            state.heap().list(*matrix).unwrap().get(5),
            Ok(&Value::number(0.5))
        );
        assert_eq!(
            state.heap().list(*matrix).unwrap().get(9),
            Ok(&Value::number(0.75))
        );
        let Value::List(blends) = state
            .heap()
            .datum_field(icon, &field("_dream64_blends"))
            .unwrap()
        else {
            panic!("Blend should retain its headless composition operation");
        };
        assert_eq!(state.heap().list(*blends).unwrap().len(), 1);
    }

    #[test]
    fn byond_control_freak_constants_are_available_to_dm_code() {
        let syntax =
            parse("/proc/control_flags()\n\treturn CONTROL_FREAK_SKIN | CONTROL_FREAK_MACROS\n")
                .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();
        assert_eq!(
            execute_module(
                &module,
                module.procedure_id("/proc/control_flags").unwrap(),
                &[]
            ),
            Ok(Value::number(3.0))
        );
    }

    #[test]
    fn contextual_icon_new_matches_icon_builtin_constructor_fields() {
        let syntax = parse(
            "/proc/contextual()\n\tvar/icon/value = new /icon(fcopy_rsc(\"icons/title.dmi\"), \"idle\", 4, 2, 1)\n\treturn value\n/proc/direct()\n\treturn icon(fcopy_rsc(\"icons/title.dmi\"), \"idle\", 4, 2, 1)\n/proc/title_shaped()\n\tvar/icon/title\n\ttitle = new /icon(fcopy_rsc(\"icons/runtime/default_title.dmi\"))\n\treturn title\n",
        )
        .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();
        let mut state = ExecutionState::new();
        let run = |path: &str, state: &mut ExecutionState| {
            let value =
                execute_module_in_state(&module, module.procedure_id(path).unwrap(), &[], state)
                    .unwrap();
            let Value::Datum(datum) = value else {
                panic!("icon constructor should return a datum");
            };
            datum
        };
        let contextual = run("/proc/contextual", &mut state);
        let direct = run("/proc/direct", &mut state);
        for name in ["icon", "icon_state", "dir", "frame", "moving"] {
            assert_eq!(
                state.heap().datum_field(contextual, &field(name)),
                state.heap().datum_field(direct, &field(name)),
                "contextual new should preserve the builtin {name} field",
            );
        }
        let title = run("/proc/title_shaped", &mut state);
        assert_eq!(
            state.heap().datum_field(title, &field("icon")),
            Ok(&Value::text("icons/runtime/default_title.dmi")),
        );
    }

    #[test]
    fn icon_geometry_methods_mutate_headless_dimensions_and_dispatch_both_ways() {
        let syntax = parse(
            "/icon/proc/resize_inside()\n\tScale(48, 64)\n\treturn Width() * 100 + Height()\n/proc/resize_outside()\n\tvar/icon/value = icon()\n\tvalue.Scale(20)\n\tvalue.Crop(2, 3, 11, 18)\n\tvalue.Shift(NORTH, 2)\n\tvalue.DrawBox(\"#ffffff\", 1, 1, 2, 2)\n\tvalue.Insert(icon(), \"state\")\n\treturn value.Width() * 100 + value.Height()\n",
        )
        .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();
        let mut state = ExecutionState::new();
        let icon = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/icon").unwrap());
        assert_eq!(
            execute_module_in_context(
                &module,
                module.procedure_id("/icon/proc/resize_inside").unwrap(),
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(icon), Value::Null),
            ),
            Ok(Value::number(4864.0))
        );
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/resize_outside").unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::number(1016.0))
        );
    }

    #[test]
    fn procedure_static_list_persists_for_protected_holder_shape() {
        let syntax = parse(
            "/datum/manager/proc/get_protected(list/list_ref)\n\tvar/static/list/protected_lists\n\tif(list_ref)\n\t\tprotected_lists = list_ref\n\treturn protected_lists\n/datum/manager/proc/update(key, list/value)\n\tvar/list/protected = src.get_protected()\n\tprotected[key] = value\n\treturn protected[key]\n/proc/run()\n\tvar/datum/manager/manager = new /datum/manager\n\tmanager.get_protected(list())\n\treturn manager.update(\"ADMIN\", list(1, 2))\n",
        )
        .unwrap();
        let module = compile_module(&syntax.definitions).unwrap();
        let mut state = ExecutionState::new();
        let result = execute_module_in_state(
            &module,
            module.procedure_id("/proc/run").unwrap(),
            &[],
            &mut state,
        )
        .unwrap();
        let Value::List(value) = result else {
            panic!("procedure-static protected storage should remain a list");
        };
        assert_eq!(state.heap().list(value).unwrap().len(), 2);
    }

    #[test]
    fn implicit_src_engine_methods_resolve_for_icon_and_matrix_procs() {
        let syntax = parse(
            "/icon/proc/colors()\n\tMapColors(1,0,0, 0,1,0, 0,0,1, 0,0,0)\n\tBlend(\"#808080\", ICON_MULTIPLY)\n\tSetIntensity(0.5)\n\treturn src\n/matrix/proc/rotate()\n\tTurn(90)\n\treturn src\n",
        )
        .unwrap();
        let icon_program = compile_procedure(&syntax.definitions[0]).unwrap();
        let matrix_program = compile_procedure(&syntax.definitions[1]).unwrap();
        let mut state = ExecutionState::new();
        let icon = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/icon").unwrap());
        assert_eq!(
            execute_in_context(
                &icon_program,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(icon), Value::Null),
            ),
            Ok(Value::Datum(icon))
        );
        assert!(matches!(
            state
                .heap()
                .datum_field(icon, &field("_dream64_color_matrix")),
            Ok(Value::List(_))
        ));
        assert!(matches!(
            state.heap().datum_field(icon, &field("_dream64_blends")),
            Ok(Value::List(_))
        ));
        let Value::List(intensity) = state
            .heap()
            .datum_field(icon, &field("_dream64_color_matrix"))
            .unwrap()
        else {
            panic!("SetIntensity should lower to an icon color matrix");
        };
        for index in [1, 5, 9] {
            assert_eq!(
                state.heap().list(*intensity).unwrap().get(index),
                Ok(&Value::number(0.5))
            );
        }

        let matrix = allocate_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], state.heap_mut()).unwrap();
        assert_eq!(
            execute_in_context(
                &matrix_program,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(matrix), Value::Null),
            ),
            Ok(Value::Datum(matrix))
        );
        assert_eq!(
            matrix_components(matrix, state.heap()).unwrap(),
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
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
    fn qdel_builtin_removes_a_datum_from_heap() {
        let syntax = parse("/proc/test(v)\n\tqdel(v)\n").expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("qdel call should compile");
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::StandardBuiltin { name, .. } if name == "qdel"))
        );

        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").expect("type path"));
        let result = execute_in_state(&program, &[Value::Datum(datum)], &mut state)
            .expect("qdel should execute");
        assert_eq!(result, Value::Null);
        assert!(state.heap().datum(datum).is_err());
    }

    #[test]
    fn del_builtin_destroys_the_target_list_itself() {
        let syntax = parse("/proc/test(v)\n\tdel(v)\n").expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("del should compile");
        let mut state = ExecutionState::new();
        let list = state.heap_mut().allocate_list();
        execute_in_state(&program, &[Value::List(list)], &mut state)
            .expect("del should destroy a list");
        assert!(state.heap().list(list).is_err());
    }

    #[test]
    fn del_dispatches_effective_hook_before_invalidating_the_datum() {
        let syntax = parse(
            "/proc/run(v)\n\tdel(v)\n\treturn global.calls\n/datum/example/Del()\n\tglobal.calls += 1\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("deletion module should compile");
        let entry = module
            .procedure_id("/proc/run")
            .expect("entry should exist");
        let mut state = ExecutionState::new();
        state.set_global(field("calls"), Value::number(0.0));
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());

        assert_eq!(
            execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state),
            Ok(Value::number(1.0))
        );
        assert!(state.heap().datum(datum).is_err());
    }

    #[test]
    fn del_finalizes_after_hook_failure_and_tolerates_reentrant_del() {
        for body in ["\tCRASH(\"boom\")\n", "\tdel(src)\n"] {
            let syntax = parse(&format!(
                "/proc/run(v)\n\tdel(v)\n/datum/example/Del()\n\tglobal.calls += 1\n{body}"
            ))
            .expect("source should parse");
            let module =
                compile_module(&syntax.definitions).expect("deletion module should compile");
            let entry = module
                .procedure_id("/proc/run")
                .expect("entry should exist");
            let mut state = ExecutionState::new();
            state.set_global(field("calls"), Value::number(0.0));
            let datum = state
                .heap_mut()
                .allocate_datum(TypePath::parse("/datum/example").unwrap());

            let result =
                execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state);
            if body.contains("CRASH") {
                assert!(result.is_err());
            } else {
                assert_eq!(result, Ok(Value::Null));
            }
            assert!(state.heap().datum(datum).is_err());
            assert_eq!(state.global(&field("calls")), Some(&Value::number(1.0)));
        }
    }

    #[test]
    fn project_qdel_procedure_shadows_the_native_fallback() {
        let syntax = parse(
            "/proc/qdel(v)\n\tglobal.calls += 1\n/proc/run(v)\n\tqdel(v)\n\treturn global.calls\n",
        )
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("qdel override should compile");
        let entry = module
            .procedure_id("/proc/run")
            .expect("entry should exist");
        let mut state = ExecutionState::new();
        state.set_global(field("calls"), Value::number(0.0));
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());

        assert_eq!(
            execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state),
            Ok(Value::number(1.0))
        );
        assert!(state.heap().datum(datum).is_ok());
    }

    #[test]
    fn sort_list_builtin_orders_positional_values_with_stable_text_order() {
        let syntax = parse("/proc/test()\n\tvar/list/L = list(2, 10, 1)\n\treturn sort_list(L)\n")
            .expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("sort_list call should compile");
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::StandardBuiltin { name, .. } if name == "sort_list"))
        );

        let mut state = ExecutionState::new();
        let result = execute_in_state(&program, &[], &mut state).expect("sort_list should execute");
        let Value::List(sorted) = result else {
            panic!("sort_list should return a list");
        };
        let items = state
            .heap()
            .list(sorted)
            .expect("sorted list should exist")
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            items,
            vec![Value::number(1.0), Value::number(2.0), Value::number(10.0)]
        );
    }

    #[test]
    fn typecacheof_builtin_returns_descendant_type_map() {
        let syntax =
            parse("/proc/test()\n\treturn typecacheof(/datum)\n").expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("typecacheof call should compile");
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::StandardBuiltin { name, .. } if name == "typecacheof"))
        );

        let mut state = ExecutionState::new();
        let base = TypePath::parse("/datum").expect("type path");
        let child = TypePath::parse("/datum/child").expect("type path");
        let grandchild = TypePath::parse("/datum/child/grandchild").expect("type path");
        state.set_type_paths([base.clone(), child.clone(), grandchild.clone()]);

        let result =
            execute_in_state(&program, &[], &mut state).expect("typecacheof should execute");
        let Value::List(cache) = result else {
            panic!("typecacheof should return a list");
        };
        let cache = state
            .heap()
            .list(cache)
            .expect("type cache list should exist");
        assert_eq!(
            cache.get_key(&Value::TypePath(base)),
            Ok(&Value::number(1.0))
        );
        assert_eq!(
            cache.get_key(&Value::TypePath(child)),
            Ok(&Value::number(1.0))
        );
        assert_eq!(
            cache.get_key(&Value::TypePath(grandchild)),
            Ok(&Value::number(1.0))
        );
    }

    #[test]
    fn typecacheof_unions_a_list_of_base_paths() {
        let syntax =
            parse("/proc/test()\n\treturn typecacheof(list(null, /datum/one, /obj/item))\n")
                .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("typecache list compiles");
        let mut state = ExecutionState::new();
        let one = TypePath::parse("/datum/one").unwrap();
        let one_child = TypePath::parse("/datum/one/child").unwrap();
        let item = TypePath::parse("/obj/item").unwrap();
        let unrelated = TypePath::parse("/mob").unwrap();
        state.set_type_paths([
            one.clone(),
            one_child.clone(),
            item.clone(),
            unrelated.clone(),
        ]);
        let Value::List(cache) = execute_in_state(&program, &[], &mut state).unwrap() else {
            panic!("typecacheof should return a list");
        };
        let cache = state.heap().list(cache).unwrap();
        for included in [one, one_child, item] {
            assert_eq!(
                cache.get_key(&Value::TypePath(included)),
                Ok(&Value::number(1.0))
            );
        }
        assert!(cache.get_key(&Value::TypePath(unrelated)).is_err());
    }

    #[test]
    fn min_and_max_accept_variadic_values_and_single_lists() {
        let syntax = parse("/proc/test()\n\treturn min(8, 3, 5) + max(list(2, 9, 4))\n")
            .expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("extrema compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(12.0)));
    }

    #[test]
    fn image_builtin_constructs_image_datum_with_icon_fields() {
        let syntax = parse("/proc/build()\n\treturn image(null, null, \"state\", 4, 2)\n")
            .expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("image constructor should compile");
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::StandardBuiltin { name, .. } if name == "image"))
        );
        let mut state = ExecutionState::new();
        let result =
            execute_in_state(&program, &[], &mut state).expect("image constructor should execute");
        let Value::Datum(image) = result else {
            panic!("image should return a datum");
        };
        let datum = state.heap().datum(image).expect("image datum should exist");
        assert_eq!(datum.type_path(), &TypePath::parse("/image").unwrap());
        assert_eq!(datum.field(&field("icon")), Ok(&Value::Null));
        assert_eq!(datum.field(&field("loc")), Ok(&Value::Null));
        assert_eq!(datum.field(&field("icon_state")), Ok(&Value::text("state")));
        assert_eq!(datum.field(&field("layer")), Ok(&Value::number(4.0)));
        assert_eq!(datum.field(&field("dir")), Ok(&Value::number(2.0)));
        assert_eq!(datum.field(&field("alpha")), Ok(&Value::number(255.0)));
        assert_eq!(
            datum.field(&field("appearance_flags")),
            Ok(&Value::number(0.0))
        );
        assert_eq!(datum.field(&field("overlays")), Ok(&Value::Null));
    }

    #[test]
    fn typesof_null_is_empty_for_typecache_style_root_lists() {
        let syntax = parse(
            "/proc/build_cache()\n\tvar/list/roots = list(null, /mob)\n\tvar/list/result = list()\n\tfor(var/root in roots)\n\t\tfor(var/path in typesof(root))\n\t\t\tresult += path\n\treturn result\n",
        )
        .expect("typecache-shaped source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("typecache-shaped source compiles");
        let mut state = ExecutionState::new();
        let mob = TypePath::parse("/mob").unwrap();
        let child = TypePath::parse("/mob/living").unwrap();
        state.set_type_paths([mob.clone(), child.clone(), TypePath::parse("/obj").unwrap()]);

        let Value::List(result) = execute_in_state(&program, &[], &mut state)
            .expect("BYOND filters the null typesof selector")
        else {
            panic!("expected expanded type list")
        };
        let values = state
            .heap()
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert_eq!(values, vec![Value::TypePath(mob), Value::TypePath(child)]);
    }

    #[test]
    fn savefile_index_output_and_export_text_support_icon_base64_pipeline() {
        let source = parse(
            "/proc/round_trip()\n\tvar/savefile/cache = new /savefile(\"memory.sav\")\n\tvar/icon/value = icon()\n\tcache[\"dummy\"] << value\n\tvar/exported = cache.ExportText(\"dummy\")\n\tvar/list/partial = splittext(exported, \"{\")\n\treturn list(exported, replacetext(copytext_char(partial[2], 3, -5), \"\\n\", \"\"))\n",
        )
        .expect("savefile round-trip source should parse");
        let module = compile_module(&source.definitions).expect("savefile round-trip compiles");
        let mut state = ExecutionState::new();
        let result = execute_module_in_state(
            &module,
            module.procedure_id("/proc/round_trip").unwrap(),
            &[],
            &mut state,
        )
        .expect("indexed savefile output and ExportText should execute");
        let Value::List(result) = result else {
            panic!("ExportText pipeline should return a list")
        };
        let values = state
            .heap()
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        let Value::Text(exported) = &values[0] else {
            panic!("ExportText should return text")
        };
        assert!(exported.starts_with("dummy = {\""));
        assert!(exported.contains("ZHJlYW02NA=="));
        assert_eq!(values[1], Value::text("ZHJlYW02NA=="));
    }

    #[test]
    fn savefile_cd_dir_eof_and_input_follow_byond_navigation() {
        let source = parse(
            "/proc/savefile_walk()\n\tvar/savefile/cache = new /savefile(\"memory.sav\")\n\tcache.cd = \"/prefs\"\n\tcache[\"volume\"] << 9\n\tcache.cd = \"volume\"\n\tvar/sequential\n\tcache >> sequential\n\tvar/at_entry = cache.eof\n\tcache.cd = \"/prefs\"\n\tvar/keyed\n\tcache[\"volume\"] >> keyed\n\tvar/list/names = cache.dir\n\tcache.cd = \"/missing\"\n\treturn list(sequential, keyed, at_entry, names.len, cache.eof)\n",
        )
        .expect("savefile navigation source should parse");
        let module = compile_module(&source.definitions).expect("savefile navigation compiles");
        let mut state = ExecutionState::new();
        let Value::List(result) = execute_module_in_state(
            &module,
            module.procedure_id("/proc/savefile_walk").unwrap(),
            &[],
            &mut state,
        )
        .expect("savefile navigation and input should execute") else {
            panic!("expected result list")
        };
        let values = state
            .heap()
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            vec![
                Value::number(9.0),
                Value::number(9.0),
                Value::number(0.0),
                Value::number(1.0),
                Value::number(1.0),
            ]
        );
    }

    #[test]
    fn gate44_byond_parser_forms_compile_as_one_language_family() {
        let source = parse(
            "/client/proc/verb_metadata()\n\tset name = \"Example\"\n\tset category = \"Admin\"\n\tset desc = \"Description\"\n\treturn 1\n/proc/inline_bodies(list/items)\n\tvar/total = 0\n\tfor(var/item in items) total += item\n\twhile(total > 10) total--\n\treturn total\n/proc/read_old_save(savefile/file)\n\tvar/value\n\tfile >> value\n\treturn value\n/proc/write_pointer(pointer)\n\t*pointer = 0\n/proc/use_pointer()\n\tvar/x = 4\n\twrite_pointer(&x)\n\treturn x\n/datum/proc/safe_initial(mob/who)\n\treturn initial(who.client?.mouse_override_icon)\n/proc/expanded_min(list/values)\n\treturn min(arglist(values))\n/proc/named_image()\n\treturn image(\"icon\" = 'icon.dmi')\n",
        )
        .expect("Gate44 compatibility source should parse");
        let module =
            compile_module(&source.definitions).expect("Gate44 compatibility forms should compile");
        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/use_pointer").unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::number(0.0)),
            "address-of and dereference preserve an output-parameter alias"
        );
    }

    #[test]
    fn else_for_and_post_conditional_while_retain_indented_bodies() {
        let source = parse(
            "/proc/admin_shape(list/flags, exact)\n\tif(exact)\n\t\treturn 1\n\telse for(var/flag in flags)\n\t\tif(!flag)\n\t\t\tcontinue\n\t\t. += flag\n\treturn .\n/proc/map_shape(text, matcher)\n\tif(isfile(text))\n\t\ttext = file2text(text)\n\telse if(isnull(text))\n\t\treturn\n\tvar/list/bounds = list(1, 1, 1)\n\tif(findtext(text, \"tgm\"))\n\t\t. = 1\n\telse\n\t\t. = 2\n\tvar/stored_index = 1\n\tvar/list/regex_output\n\twhile(matcher.Find(text, stored_index))\n\t\tstored_index = matcher.next\n\t\tregex_output = matcher.group\n\treturn stored_index\n",
        )
        .expect("combined control-flow shapes should parse");
        compile_module(&source.definitions)
            .expect("combined else-for and following while bodies should compile");
    }

    #[test]
    fn empty_while_uses_condition_side_effects_without_consuming_next_sibling() {
        let source = parse(
            "/proc/skip_blanks(list/lines)\n\tvar/leading_blanks = 0\n\twhile(leading_blanks < length(lines) && lines[++leading_blanks] == \"\")\n\tif(leading_blanks > 1)\n\t\treturn leading_blanks\n\treturn 0\n",
        )
        .expect("empty while source should parse");
        let module = compile_module(&source.definitions)
            .expect("BYOND permits a condition-only empty while");
        let mut state = ExecutionState::new();
        let lines = state.heap_mut().allocate_list();
        for value in ["", "", "occupied"] {
            state
                .heap_mut()
                .list_mut(lines)
                .unwrap()
                .add(Value::text(value));
        }
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id("/proc/skip_blanks").unwrap(),
                &[Value::List(lines)],
                &mut state,
            ),
            Ok(Value::number(3.0)),
            "the following if remains a sibling and observes condition mutation"
        );
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

    #[test]
    fn vector_constructor_operators_and_methods_use_three_numeric_components() {
        let source = concat!(
            "/proc/run()\n",
            "\tvar/vector/a = vector(3, 3)\n",
            "\tvar/vector/b = vector(4, 4, 4)\n",
            "\tvar/vector/c = a + b\n",
            "\ta *= b\n",
            "\tvar/vector/i = vector(1, 1).Interpolate(vector(12, 124, 91), 0.5)\n",
            "\tvar/vector/n = vector(3, 4)\n",
            "\tn.Normalize()\n",
            "\treturn c.x + c.z + a.x + i.x + i.y + i.z + n.size\n",
        );
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("vector source should compile");
        let entry = module.procedure_id("/proc/run").expect("run should exist");

        assert_eq!(
            execute_module(&module, entry, &[]),
            Ok(Value::number(138.5))
        );
    }

    #[test]
    fn animate_applies_named_values_and_continues_the_last_sequence_headlessly() {
        let syntax = parse(
            "/proc/run()\n\tanimate(src, alpha = 128, time = 5)\n\tanimate(pixel_x = 12, time = 2)\n\treturn src.alpha + src.pixel_x\n",
        )
        .expect("animate source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("animate should compile");
        let mut state = ExecutionState::new();
        let object = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/obj").expect("obj path"));
        let context = ExecutionContext::new(Value::Datum(object), Value::Null);

        assert_eq!(
            execute_in_context(&program, &[], &mut state, &context),
            Ok(Value::number(140.0))
        );
    }

    #[test]
    fn flick_does_not_mutate_the_persistent_icon_state() {
        let syntax = parse("/proc/run()\n\tflick(\"opening\", src)\n\treturn src.icon_state\n")
            .expect("flick source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("flick should compile");
        let mut state = ExecutionState::new();
        let object = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/obj").expect("obj path"));
        state
            .heap_mut()
            .set_datum_field(object, field("icon_state"), Value::text("closed"))
            .expect("icon state should materialize");
        let context = ExecutionContext::new(Value::Datum(object), Value::Null);

        assert_eq!(
            execute_in_context(&program, &[], &mut state, &context),
            Ok(Value::text("closed"))
        );
    }

    #[test]
    fn filter_preserves_named_properties_on_a_filter_datum() {
        let syntax =
            parse("/proc/run()\n\tvar/f = filter(type = \"blur\", size = 4)\n\treturn f.size\n")
                .expect("filter source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("filter should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(4.0)));
    }

    #[test]
    fn filter_arglist_spreads_associative_entries_as_named_properties() {
        let syntax = parse(
            "/proc/run()\n\tvar/list/arguments = list(\"type\" = \"blur\", \"size\" = 7)\n\tvar/f = filter(arglist(arguments))\n\treturn f.size\n",
        )
        .expect("filter arglist source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("filter arglist should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(7.0)));
    }

    #[test]
    fn deeply_nested_macro_ternaries_remain_one_call_argument() {
        let source = "/proc/run(a, b, t, m, p)\n\treturn helper(((a) ? (b?[\"[-9]\"] ? -9 : (-9) - ((110 - -100) * (((a && (isloc(t)))) ? (t.z ? (m[t.z]) : ((p) ? p[\"[t.plane]\"] : t.plane)) : 0))) : (-9)), 1)\n/proc/helper(value, other)\n\treturn value\n";
        let syntax = parse(source).expect("nested conditional source should parse");
        compile_module(&syntax.definitions)
            .expect("nested conditional must not consume the call delimiter");
    }

    #[test]
    fn safe_index_does_not_expose_named_call_assignment_as_statement_assignment() {
        let source = "/proc/run(mapping, flags)\n\thelper(mapping?[\"key\"], add_appearance_flags = flags)\n\treturn 1\n/proc/helper(value, add_appearance_flags)\n\treturn value\n";
        let syntax = parse(source).expect("safe-index call source should parse");
        compile_module(&syntax.definitions)
            .expect("named argument inside a safe-index call must remain nested");
    }

    #[test]
    fn output_preserves_message_and_control_for_later_client_routing() {
        let syntax = parse(
            "/proc/run()\n\tvar/o = output(\"score: 5\", \"scorepane.output\")\n\treturn o.control\n",
        )
        .expect("output source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("output should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::text("scorepane.output")));
    }

    #[test]
    fn eager_compact_control_and_expression_families_lower_together() {
        let source = concat!(
            "/proc/run(flag)\n",
            "\tvar/i = 0\n",
            "\t++i\n",
            "\twhile(i < 3) i++\n",
            "\tvar/list/values = list(10, 20)\n",
            "\tvar/picked = values[flag ? 1 : 2]\n",
            "\tif(i in 1 to 4)\n",
            "\t\tswitch(picked)\n",
            "\t\t\tif(10) return i + picked\n",
            "\t\t\telse return 99\n",
            "\treturn 0\n",
        );
        let syntax = parse(source).expect("compact eager-family source should parse");
        let module = compile_module(&syntax.definitions)
            .expect("compact control, prefix mutation, range, and ternary index should compile");
        let entry = module.procedure_id("/proc/run").expect("run should exist");
        assert_eq!(
            execute_module(&module, entry, &[Value::number(1.0)]),
            Ok(Value::number(13.0))
        );
        assert_eq!(
            execute_module(&module, entry, &[Value::number(0.0)]),
            Ok(Value::number(99.0))
        );
    }

    #[test]
    fn gate3_prefix_compact_switch_colon_and_optional_call_shapes_compile() {
        let source = concat!(
            "/datum/worker/proc/queue(value)\n\treturn value\n",
            "/proc/run(worker, current_vote, choice)\n",
            "\tvar/count = 0\n",
            "\t++count\n",
            "\tcurrent_vote?.reset()\n",
            "\tvar/result = worker:queue(count)\n",
            "\tswitch(choice)\n",
            "\t\tif(1) return result\n",
            "\t\telse return 0\n",
        );
        let syntax = parse(source).expect("gate3 syntax shapes should parse");
        compile_module(&syntax.definitions)
            .expect("prefix, compact switch, colon call, and optional call should lower");
    }

    #[test]
    fn gate4_macro_expanded_statement_and_expression_shapes_compile() {
        let cases = [
            (
                "verb metadata",
                "/verb/succumb()\n\tset hidden = TRUE\n\treturn 1\n",
            ),
            (
                "increments",
                "/proc/run(target)\n\t++.\n\t++target.AdminProcCallCount\n",
            ),
            (
                "colon ternary",
                "/proc/run(target)\n\treturn target ? target:client : (target:current?:client)\n",
            ),
            (
                "inline for",
                "/proc/run(generated_actions)\n\tif(generated_actions) { for(var/I in generated_actions) qdel(I); generated_actions.Cut(); }\n",
            ),
            (
                "empty switch",
                "/proc/run(x)\n\tswitch(x)\n\t\tif(1)\n\t\tif(2) return 2\n",
            ),
            (
                "input suffix",
                "/proc/run(items)\n\treturn input(null, \"pick\", \"title\", null) as null|anything in items\n",
            ),
        ];
        for (name, source) in cases {
            let syntax = parse(source).unwrap_or_else(|error| panic!("{name}: {error}"));
            compile_module(&syntax.definitions)
                .unwrap_or_else(|error| panic!("{name}: {}", error.message));
        }
    }

    #[test]
    fn global_vars_is_a_live_iterable_namespace_over_global_storage() {
        let source = concat!(
            "/proc/run()\n",
            "\tvar/list/reflection = global.vars\n",
            "\tvar/total = 0\n",
            "\tfor(var/name in reflection)\n",
            "\t\ttotal += reflection[name]\n",
            "\tglobal.counter = 5\n",
            "\tvar/live_read = reflection[\"counter\"]\n",
            "\treflection[\"counter\"] = 7\n",
            "\treturn total * 100 + live_read * 10 + global.counter\n",
        );
        let syntax = parse(source).expect("global.vars source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("global.vars iteration and indexed writes should compile");
        let mut state = ExecutionState::new();
        state.set_global(field("counter"), Value::number(3.0));
        let qualified = FieldName::static_storage("/datum/example/var/static/shared");
        state.set_global(qualified, Value::number(8.0));

        assert_eq!(
            execute_in_state(&program, &[], &mut state),
            Ok(Value::number(1157.0))
        );
        assert_eq!(state.global(&field("counter")), Some(&Value::number(7.0)));
    }

    #[test]
    fn nested_waitfor_false_processing_loop_does_not_block_staged_initializer() {
        let source = concat!(
            "/proc/main()\n\tinitialize()\n\treturn 1\n",
            "/proc/initialize()\n",
            "\tset waitfor = 0\n",
            "\tsleep(1)\n",
            "\tglobal.stage = 1\n",
            "\tstart_processing()\n",
            "\tglobal.stage = 2\n",
            "\tsleep(1)\n",
            "\tglobal.stage = 3\n",
            "/proc/start_processing()\n",
            "\tset waitfor = 0\n",
            "\tglobal.loops += 1\n",
            "\tsleep(1)\n",
            "\tglobal.loops += 10\n",
        );
        let syntax = parse(source).expect("staged waitfor source should parse");
        let module =
            compile_module(&syntax.definitions).expect("staged waitfor source should compile");
        let mut state = ExecutionState::new();
        state.set_global(field("stage"), Value::number(0.0));
        state.set_global(field("loops"), Value::number(0.0));
        let main = module.procedure_id("/proc/main").expect("main exists");

        assert_eq!(
            execute_module_in_state(&module, main, &[], &mut state),
            Ok(Value::number(1.0))
        );
        assert_eq!(state.scheduled_task_count(), 1);
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state)
            .expect("first initialization slice should run");
        assert_eq!(state.global(&field("stage")), Some(&Value::number(2.0)));
        assert_eq!(state.global(&field("loops")), Some(&Value::number(1.0)));
        assert_eq!(state.scheduled_task_count(), 2);
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state)
            .expect("processing and initialization continuations should both run");
        assert_eq!(state.global(&field("stage")), Some(&Value::number(3.0)));
        assert_eq!(state.global(&field("loops")), Some(&Value::number(11.0)));
    }
}

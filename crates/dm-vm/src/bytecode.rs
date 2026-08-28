//! Portable bytecode/IR representation: `Instruction`, `Program`, and
//! `Module`.
//!
//! This module owns the data shapes the compiler produces and the
//! interpreter consumes. It contains no compilation logic itself beyond
//! `Module`'s lazy/deferred procedure compilation bookkeeping, which calls
//! back into the crate root's compiler entry points.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use dm_core::{DmNumberBits, SourceSpan};
use dm_syntax::Definition;
use dm_value::{FieldName, TypePath, Value};

use crate::compact_wordcode::{CompactWordcodeError, CompactWordcodeImage};
use crate::{
    CompileError, FullyEagerCompileErrors, boot_trace_enabled, compile_error,
    compile_procedure_with_resolver_and_fields,
};

/// One instruction in the portable reference bytecode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Instruction {
    /// Pushes `null`.
    PushNull,
    /// Pushes a numeric constant.
    PushNumber(DmNumberBits),
    /// Pushes a text constant.
    PushText(Arc<str>),
    /// Pushes a first-class BYOND file/resource constant.
    PushFile(String),
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
        /// Number of already-evaluated constructor arguments.
        argument_count: u16,
        /// Name of each source constructor argument, or `None` when positional.
        argument_names: Vec<Option<String>>,
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
    /// Constructs BYOND's built-in `/mutable_appearance` datum when a project
    /// has not replaced the call with its own global procedure.
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
        /// Name of each source argument, or `None` for a positional argument.
        ///
        /// Most native procedures consume their arguments positionally. Image
        /// construction is the important exception during startup: BYOND
        /// reorders named `icon_state`, `layer`, and direction arguments before
        /// `/image/New` applies them to the copied appearance.
        argument_names: Vec<Option<String>>,
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
        /// Source argument positions supplied through `arglist()`.
        expanded_indices: Vec<u16>,
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
    /// Reads an initial datum field selected by a runtime text key, as in
    /// `initial(object.vars[name])`.
    InitialDynamicField,
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
    /// Selects from the runtime argument vector produced by `arglist(...)`.
    ///
    /// Unlike a single ordinary list candidate, this first expands the outer
    /// argument list. If expansion yields one list, ordinary `pick(list)`
    /// semantics then select from that nested list.
    PickExpandedArguments,
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
    /// The catalog belongs to `ExecutionState`, allowing the runtime image
    /// to provide the complete object tree without coupling bytecode to its
    /// materialization implementation.
    TypesOf {
        /// Number of selectors whose inclusive type families are concatenated.
        argument_count: u8,
    },
    /// Tests whether a receiver exposes a named procedure.
    HasCall,
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
    /// Implements `local ||= list()` without materializing branch bytecode.
    LogicalOrEmptyListLocal(u16),
    /// Implements `global ||= list()` without materializing branch bytecode.
    LogicalOrEmptyListGlobal(FieldName),
    /// Pops a captured receiver and implements `receiver.field ||= list()`.
    LogicalOrEmptyListField(FieldName),
    /// Pops a captured key and receiver and implements `receiver[key] ||= list()`.
    LogicalOrEmptyListIndex,
    /// Pops a numeric 1-based index and a list handle, then pushes the entry.
    IndexList,
    /// Pops an index/key and reads a list receiver directly from a local slot.
    ///
    /// This avoids materializing and validating the same live list once in
    /// `LoadLocal` and again in `IndexList`.
    IndexLocalList(u16),
    /// Reads a local list's length without a separate `LoadLocal` dispatch.
    ///
    /// Iteration loops execute this operation for every condition check, so
    /// retaining the local slot in the opcode avoids repeated stack traffic
    /// while preserving [`Self::ListLength`] as the semantic implementation.
    ListLengthLocal(u16),
    /// Advances one compiler-generated local-list iteration header.
    ///
    /// The original seven instructions remain in the program for stable jump
    /// targets, but execution enters through this fused header and skips their
    /// redundant stack traffic.
    NextLocalListIteration {
        /// Local containing the immutable iteration snapshot.
        list_slot: u16,
        /// Local containing the current one-based iteration index.
        index_slot: u16,
        /// Local receiving the current iteration value.
        item_slot: u16,
        /// Instruction after the loop when the snapshot is exhausted.
        exit: usize,
    },
    /// Pops a value, index/key, and list handle and updates that list.
    SetListIndex,
    /// Like [`Self::SetListIndex`], but leaves the stored value on the stack.
    SetListIndexKeep,
    /// Pops a value, index/key, and list handle, applies a numeric operation to
    /// the current indexed value and the supplied value, then updates that
    /// list.
    CompoundListIndex(CompoundListIndexOperator),
    /// Compound indexed assignment that leaves the assigned value as the
    /// expression result.
    CompoundListIndexKeep(CompoundListIndexOperator),
    /// Pops a list handle and pushes its deterministic iteration length.
    ListLength,
    /// Converts an atom/world iterable to its live `contents` list. Lists and
    /// null pass through unchanged.
    PrepareIteration,
    /// Tests whether an enumerated value can bind to a typed loop variable.
    IterationTypeFilter(TypePath),
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
    /// Reads a field validated through the receiver's declared DM type.
    ///
    /// BYOND can hold a runtime-incompatible value in a typed variable. The
    /// statically valid slot then reads as null instead of becoming an
    /// undefined dynamic field access.
    LoadDeclaredField(FieldName),
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
    /// Reads `datum.vars[key]` without materializing the complete reflection list.
    LoadDynamicField,
    /// Writes `datum.vars[key]` without materializing the complete reflection list.
    StoreDynamicField,
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
    /// Reorders `value, receiver, key` to `receiver, key, value`.
    ///
    /// Direct indexed assignment evaluates its RHS before its destination in
    /// BYOND, while the list-write instruction consumes destination first.
    PrepareRhsFirstIndexAssignment,
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
        /// Optional BYOND keyword attached to each source argument.
        argument_names: Vec<Option<String>>,
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
        /// Compile-time text selector for ordinary `receiver.method()` calls.
        /// A runtime `call(receiver, selector)()` leaves this unset and pushes
        /// its selector after the receiver instead.
        static_selector: Option<String>,
        /// Number of positional values supplied by the caller.
        argument_count: u16,
        /// Optional BYOND keyword attached to each source argument.
        argument_names: Vec<Option<String>>,
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
        /// Optional names attached to the source argument expressions.
        argument_names: Vec<Option<String>>,
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
pub struct ProcedureId(pub(crate) u32);

impl ProcedureId {
    pub(crate) fn from_index(index: usize) -> Result<Self, CompileError> {
        u32::try_from(index)
            .map(Self)
            .map_err(|_| compile_error("module has more than u32::MAX procedures"))
    }

    /// Returns this module-local portable procedure-table index.
    #[must_use]
    pub const fn index(self) -> usize {
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
    /// Declared parameter names in positional order. Empty entries represent
    /// unnamed varargs slots.
    pub parameter_names: Vec<String>,
    /// OpenDream-compatible command-line conversion for each verb parameter.
    /// Non-verb procedures retain `Unsupported` entries because this metadata
    /// is only consulted by the client command surface.
    pub verb_parameter_types: Vec<VerbParameterType>,
    /// BYOND-facing `set name = ...` command label for a verb body.
    /// `None` uses the procedure selector as the command name.
    pub verb_name: Option<String>,
    /// Number of local slots, including parameters.
    pub local_count: usize,
    /// Portable instructions in execution order.
    pub instructions: Vec<Instruction>,
    /// Source line associated with each instruction for diagnostics/debugging.
    pub source_spans: Vec<SourceSpan>,
}

/// Client-side conversion supported by OpenDream's explicit verb command line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerbParameterType {
    /// `as text`, `as message`, or `as command_text`.
    Text,
    /// `as message`.
    Message,
    /// `as num`.
    Number,
    /// `as color`.
    Color,
    /// `as file`, `as icon`, or `as sound`.
    File,
    /// Atom-type union using obj=1, mob=2, turf=4, and area=8.
    Atom(u8),
    /// `as anything` or an untyped verb argument.
    Anything,
    /// Metadata is absent because this is not a verb procedure.
    Unsupported,
}

/// A deterministic table of compiled procedures and their canonical paths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Module {
    pub(crate) identity: ModuleIdentity,
    pub(crate) procedures: Vec<Arc<Program>>,
    pub(crate) paths: Vec<String>,
    pub(crate) names: HashMap<String, ProcedureId>,
    /// Latest implementation for each canonical path with reopening suffixes removed.
    pub(crate) dynamic_names: HashMap<String, ProcedureId>,
    pub(crate) deferred: Arc<HashMap<ProcedureId, DeferredProcedure>>,
    pub(crate) procedure_types: Vec<TypePath>,
    pub(crate) initializer_call_names: Option<InitializerCallNameIndex>,
    pub(crate) compact_wordcode: CompactWordcodeAttachment,
    pub(crate) semantic_digests: ProcedureSemanticDigestAttachment,
}

static NEXT_MODULE_IDENTITY: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModuleIdentity(pub(crate) u64);

// Module identity is an execution-cache namespace, not part of portable
// bytecode semantics. Independently compiled equivalent modules must retain
// their historical structural equality.
impl PartialEq for ModuleIdentity {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for ModuleIdentity {}

#[derive(Clone, Debug, Default)]
pub(crate) struct CompactWordcodeAttachment(pub(crate) Option<Arc<CompactWordcodeImage>>);

#[derive(Clone, Debug, Default)]
pub(crate) struct ProcedureSemanticDigestAttachment(pub(crate) Option<Arc<[[u8; 32]]>>);

impl PartialEq for ProcedureSemanticDigestAttachment {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for ProcedureSemanticDigestAttachment {}

// Compact wordcode is a rebuildable execution cache, not part of the portable
// module semantics. Its presence must not change structural module equality.
impl PartialEq for CompactWordcodeAttachment {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for CompactWordcodeAttachment {}

pub(crate) fn next_module_identity() -> ModuleIdentity {
    ModuleIdentity(NEXT_MODULE_IDENTITY.fetch_add(1, Ordering::Relaxed))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InitializerCallNameIndex {
    pub(crate) names: Arc<HashMap<String, ProcedureId>>,
    pub(crate) module_names_scanned: usize,
}

/// Immutable call-resolution snapshot shared by parallel initializer lowering.
#[derive(Clone, Debug)]
pub struct InitializerCompileContext {
    pub(crate) names: Arc<HashMap<String, ProcedureId>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeferredProcedure {
    pub(crate) definition: Arc<Definition>,
    pub(crate) targets: Arc<HashMap<String, ProcedureId>>,
    pub(crate) src_fields: Arc<BTreeMap<String, FieldName>>,
    pub(crate) global_fields: Arc<BTreeMap<String, FieldName>>,
    pub(crate) global_types: Arc<BTreeMap<String, TypePath>>,
    pub(crate) preflight_error: Option<CompileError>,
    pub(crate) compiled: Arc<OnceLock<Result<Program, CompileError>>>,
}

impl DeferredProcedure {
    fn compile(&self) -> Result<Program, CompileError> {
        if let Some(error) = &self.preflight_error {
            return Err(error.clone());
        }
        compile_procedure_with_resolver_and_fields(
            self.definition.as_ref(),
            self.targets.as_ref(),
            self.src_fields.as_ref(),
            self.global_fields.as_ref(),
            self.global_types.as_ref(),
        )
    }
}

/// An initializer expression linked as an entry point in a VM module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializerProgram {
    pub(crate) module: Module,
    pub(crate) entry: ProcedureId,
}

/// One ordered per-instance initializer action executed before `New()`.
///
/// Constant actions are retained only when an earlier runtime initializer for
/// the same inherited field could otherwise overwrite a descendant constant
/// override. This preserves BYOND's ancestor-to-descendant default ordering
/// without materializing every scalar default on compact engine turfs.
#[derive(Clone)]
pub enum InstanceInitializer {
    /// Reapply an immutable scalar value at its declared position.
    Constant {
        /// Destination datum field.
        field: FieldName,
        /// Declared scalar value.
        value: Value,
    },
    /// Evaluate a fresh runtime expression and store its result.
    Program {
        /// Destination datum field.
        field: FieldName,
        /// Entry in the shared linked initializer module.
        entry: ProcedureId,
    },
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
    /// Builds, validates, and installs the immutable compact dispatch image.
    ///
    /// # Errors
    ///
    /// Returns an error when this module still contains deferred procedures or
    /// cannot be represented by the bounded compact directory.
    pub fn install_compact_wordcode(&mut self) -> Result<(), CompactWordcodeError> {
        let image = CompactWordcodeImage::build(self)?;
        self.compact_wordcode.0 = Some(Arc::new(image));
        Ok(())
    }

    /// Installs a decoded compact image after exact validation against this
    /// module's authoritative rich instructions.
    ///
    /// # Errors
    ///
    /// Returns an error for any procedure, metadata, range, path, or selector
    /// mismatch.
    pub fn attach_compact_wordcode(
        &mut self,
        image: CompactWordcodeImage,
    ) -> Result<(), CompactWordcodeError> {
        image.validate_against(self)?;
        self.compact_wordcode.0 = Some(Arc::new(image));
        Ok(())
    }

    /// Returns the installed immutable compact dispatch image, when present.
    #[must_use]
    pub fn compact_wordcode(&self) -> Option<&CompactWordcodeImage> {
        self.compact_wordcode.0.as_deref()
    }

    /// Returns the artifact-validated semantic digest for one procedure.
    #[must_use]
    pub fn procedure_semantic_digest(&self, procedure: ProcedureId) -> Option<[u8; 32]> {
        self.semantic_digests
            .0
            .as_ref()?
            .get(procedure.index())
            .copied()
    }

    /// Attaches semantic identities after exact count and body validation.
    pub fn attach_procedure_semantic_digests(
        &mut self,
        digests: Vec<[u8; 32]>,
    ) -> Result<(), String> {
        if digests.len() != self.procedure_count() {
            return Err("procedure semantic digest count does not match module".to_owned());
        }
        for (index, digest) in digests.iter().enumerate() {
            let procedure = ProcedureId(index as u32);
            if self.compute_procedure_semantic_digest(procedure)? != *digest {
                return Err(format!(
                    "procedure semantic digest mismatch at index {index}"
                ));
            }
        }
        self.semantic_digests.0 = Some(Arc::from(digests));
        Ok(())
    }

    pub(crate) fn clear_compact_wordcode(&mut self) {
        self.compact_wordcode.0 = None;
    }

    pub(crate) fn resolve_procedure(&self, procedure: ProcedureId) -> Result<&Program, String> {
        if let Some(deferred) = self.deferred.get(&procedure) {
            let pending = deferred.compiled.get().is_none();
            let started = pending.then(Instant::now);
            if pending && boot_trace_enabled() {
                eprintln!(
                    "boot-vm: deferred-materialize-begin path={}",
                    self.paths
                        .get(procedure.index())
                        .map_or("<missing>", String::as_str),
                );
            }
            return deferred
                .compiled
                .get_or_init(|| deferred.compile())
                .as_ref()
                .map_err(|error| error.message.clone())
                .inspect(|_| {
                    if let Some(started) = started
                        && boot_trace_enabled()
                    {
                        eprintln!(
                            "boot-vm: deferred-materialized path={} elapsed_ms={}",
                            self.paths
                                .get(procedure.index())
                                .map_or("<missing>", String::as_str),
                            started.elapsed().as_millis(),
                        );
                    }
                });
        }
        self.procedures
            .get(procedure.index())
            .map(Arc::as_ref)
            .ok_or_else(|| format!("invalid procedure {}", procedure.index()))
    }

    /// Looks up a procedure by canonical path, such as `/proc/main`.
    #[must_use]
    pub fn procedure_id(&self, path: &str) -> Option<ProcedureId> {
        self.names.get(path).copied()
    }

    /// Looks up the effective implementation used by dynamic dispatch.
    #[must_use]
    pub fn effective_procedure_id(&self, path: &str) -> Option<ProcedureId> {
        self.dynamic_names.get(path).copied()
    }

    /// Returns a compiled procedure by module-local identity.
    #[must_use]
    pub fn procedure(&self, procedure: ProcedureId) -> Option<&Program> {
        self.resolve_procedure(procedure).ok()
    }

    /// Total number of stable procedure identities in this module.
    #[must_use]
    pub fn procedure_count(&self) -> usize {
        self.procedures.len()
    }

    /// Iterates diagnostic procedure paths in stable module identity order.
    pub fn procedure_paths(&self) -> impl Iterator<Item = &str> {
        self.paths.iter().map(String::as_str)
    }

    /// Number of symbolically linked procedure bodies not compiled eagerly.
    #[must_use]
    pub fn deferred_procedure_count(&self) -> usize {
        self.deferred.len()
    }

    /// Returns canonical static-storage globals referenced by executable instructions.
    #[must_use]
    pub fn referenced_static_globals(&self) -> BTreeSet<FieldName> {
        self.procedures
            .iter()
            .flat_map(|program| &program.instructions)
            .filter_map(|instruction| match instruction {
                Instruction::LoadGlobal(field)
                | Instruction::LoadInitialGlobal(field)
                | Instruction::StoreGlobal(field)
                | Instruction::LogicalOrEmptyListGlobal(field) => Some(field),
                Instruction::MutateGlobal { name, .. } => Some(name),
                _ => None,
            })
            .filter(|field| field.as_str().starts_with("__dm_static_"))
            .cloned()
            .collect()
    }

    /// Number of deferred bodies materialized by execution or inspection.
    #[must_use]
    pub fn materialized_deferred_procedure_count(&self) -> usize {
        self.deferred
            .values()
            .filter(|procedure| procedure.compiled.get().is_some())
            .count()
    }

    /// Compiles every deferred body and returns a module containing only eager
    /// procedure programs.
    ///
    /// Deferred bodies are visited in stable procedure identity order. Known
    /// preflight failures are reported before any remaining body is lowered,
    /// and all successfully lowered programs are installed only after every
    /// body succeeds. Clones of this module therefore retain their original
    /// lazy state and do not observe partial materialization through shared
    /// deferred caches.
    ///
    /// # Errors
    ///
    /// Returns the first deferred preflight or lowering failure in stable
    /// procedure identity order, annotated with its canonical procedure path.
    pub fn into_fully_eager(mut self) -> Result<Self, CompileError> {
        if self.deferred.is_empty() {
            return Ok(self);
        }

        let mut deferred_ids = self.deferred.keys().copied().collect::<Vec<_>>();
        deferred_ids.sort_unstable_by_key(|procedure| procedure.index());

        for procedure in &deferred_ids {
            let deferred = self
                .deferred
                .get(procedure)
                .expect("deferred identity came from this module");
            if let Some(error) = &deferred.preflight_error {
                let path = self
                    .paths
                    .get(procedure.index())
                    .map_or("<missing procedure>", String::as_str);
                return Err(compile_error(format!("{path}: {}", error.message)));
            }
        }

        let mut compiled = Vec::with_capacity(deferred_ids.len());
        for procedure in deferred_ids {
            let path = self
                .paths
                .get(procedure.index())
                .map_or("<missing procedure>", String::as_str);
            let deferred = self
                .deferred
                .get(&procedure)
                .expect("deferred identity came from this module");
            let program = match deferred.compiled.get() {
                Some(result) => result.clone(),
                None => deferred.compile(),
            }
            .map_err(|error| compile_error(format!("{path}: {}", error.message)))?;
            if self.procedures.get(procedure.index()).is_none() {
                return Err(compile_error(format!(
                    "{path}: invalid deferred procedure identity {}",
                    procedure.index()
                )));
            }
            compiled.push((procedure, program));
        }

        for (procedure, program) in compiled {
            self.procedures[procedure.index()] = Arc::new(program);
        }
        self.deferred = Arc::new(HashMap::new());
        Ok(self)
    }

    /// Attempts every deferred body in stable procedure identity order while
    /// retaining only a bounded diagnostic sample.
    ///
    /// Successful bodies are installed even when independent bodies fail, and
    /// only failed bodies remain deferred. This makes a single artifact-build
    /// pass expose multiple incompatibilities without changing ordinary lazy
    /// dispatch or [`Self::into_fully_eager`]'s first-error contract.
    ///
    /// # Errors
    ///
    /// Returns the bounded leading diagnostics, total failure count, and
    /// successful materialization count when any deferred body fails.
    pub fn materialize_fully_eager_bounded(
        &mut self,
        diagnostic_limit: usize,
    ) -> Result<(), FullyEagerCompileErrors> {
        if self.deferred.is_empty() {
            return Ok(());
        }

        let mut deferred_ids = self.deferred.keys().copied().collect::<Vec<_>>();
        deferred_ids.sort_unstable_by_key(|procedure| procedure.index());

        let mut successful = Vec::with_capacity(deferred_ids.len());
        let mut failed = Vec::new();
        let mut diagnostics = Vec::with_capacity(diagnostic_limit.min(deferred_ids.len()));
        let mut total_failures = 0usize;

        for procedure in deferred_ids {
            let path = self
                .paths
                .get(procedure.index())
                .map_or("<missing procedure>", String::as_str);
            let result = if self.procedures.get(procedure.index()).is_none() {
                Err(compile_error(format!(
                    "invalid deferred procedure identity {}",
                    procedure.index()
                )))
            } else {
                let deferred = self
                    .deferred
                    .get(&procedure)
                    .expect("deferred identity came from this module");
                match deferred.compiled.get() {
                    Some(result) => result.clone(),
                    None => deferred.compile(),
                }
            };

            match result {
                Ok(program) => successful.push((procedure, program)),
                Err(error) => {
                    total_failures += 1;
                    failed.push(procedure);
                    if diagnostics.len() < diagnostic_limit {
                        diagnostics.push(compile_error(format!("{path}: {}", error.message)));
                    }
                }
            }
        }

        let successful_procedures = successful.len();
        for (procedure, program) in successful {
            self.procedures[procedure.index()] = Arc::new(program);
        }

        if total_failures == 0 {
            self.deferred = Arc::new(HashMap::new());
            return Ok(());
        }

        self.deferred = Arc::new(
            failed
                .into_iter()
                .map(|procedure| {
                    let deferred = self
                        .deferred
                        .get(&procedure)
                        .expect("failed deferred identity came from this module")
                        .clone();
                    (procedure, deferred)
                })
                .collect(),
        );
        Err(FullyEagerCompileErrors {
            diagnostics,
            total_failures,
            successful_procedures,
        })
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

    /// Returns procedure paths exposed through BYOND's `typesof(/owner/proc)` catalog.
    pub fn procedure_type_paths(&self) -> impl Iterator<Item = &TypePath> {
        self.procedure_types.iter()
    }
}

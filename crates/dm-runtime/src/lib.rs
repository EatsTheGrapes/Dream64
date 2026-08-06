//! Deterministic materialization of compiler constants into a runtime heap.
//!
//! This crate is intentionally a boundary between frontend identities and the
//! persistent runtime. Object-tree node IDs are consumed while building the
//! image but are never retained; canonical paths are the durable type keys.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use dm_compiler::{Compilation, CompilerDatabase, CompilerError};
use dm_core::{FileId, SourceSpan};
use dm_globals::{
    ConstantEvaluation, ConstantListEntry, ConstantValue, InitializationStep, StorageClass,
    UnsupportedCategory, UnsupportedConstant, VariableEntry, VariableRegistry, evaluate_constant,
};
use dm_lexer::{TokenKind, lex};
use dm_object_tree::NodeKind;
use dm_value::{DatumDefaults, DatumId, FieldName, TypePath, Value, ValueError, ValueHeap};
use dm_vm::{
    ExecutionContext, ExecutionState, InitializerBinding, compile_initializer,
    execute_module_in_context,
};

/// A successfully materialized global or type-static variable.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeVariable {
    /// Canonical variable path from the object tree.
    pub path: String,
    /// Global or type-static storage lifetime.
    pub storage: StorageClass,
    /// Last constant value assigned in project source order.
    pub value: Value,
    /// Declaration ordinal of the assignment that produced `value`.
    pub ordinal: usize,
}

/// A runtime initializer deliberately left for a later execution phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeInitializerDiagnostic {
    /// Canonical variable path.
    pub variable_path: String,
    /// Storage lifetime that eventually receives the value.
    pub storage: StorageClass,
    /// Expanded declaration ordinal.
    pub ordinal: usize,
    /// Physical source file.
    pub file_id: FileId,
    /// Project-relative source path.
    pub source_path: String,
    /// Complete initializer span in original source bytes.
    pub initializer_span: SourceSpan,
    /// Precise unsupported token span in original source bytes.
    pub blocker_span: SourceSpan,
    /// Conservative reason materialization stopped.
    pub category: UnsupportedCategory,
    /// Runtime phase that rejected the initializer.
    pub phase: InitializerFailurePhase,
    /// Recoverable lowering or execution detail.
    pub message: String,
}

/// Phase that retained an initializer for a later compatibility pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitializerFailurePhase {
    /// Conservative constant evaluation rejected syntax not supported by the VM.
    ConstantEvaluation,
    /// VM expression lowering could not resolve or represent the expression.
    Lowering,
    /// Valid bytecode failed while reading current runtime state.
    Execution,
}

/// Result of conservatively applying one field expression to a live datum.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstantFieldApplication {
    /// The expression was proven constant and stored on the datum.
    Applied,
    /// Runtime evaluation or unsupported syntax is still required.
    Unsupported(UnsupportedConstant),
}

/// Canonical runtime metadata and direct defaults for one object type.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeType {
    path: TypePath,
    parent: Option<TypePath>,
    defaults: DatumDefaults,
}

impl RuntimeType {
    /// Returns the canonical type path.
    #[must_use]
    pub const fn path(&self) -> &TypePath {
        &self.path
    }

    /// Returns the effective canonical parent type, when one exists.
    #[must_use]
    pub const fn parent(&self) -> Option<&TypePath> {
        self.parent.as_ref()
    }

    /// Returns defaults declared directly on this type.
    #[must_use]
    pub const fn defaults(&self) -> &DatumDefaults {
        &self.defaults
    }
}

/// Deterministic materialization counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeImageStats {
    /// Variable declarations inventoried by the frontend.
    pub variables: usize,
    /// Explicit initializer steps attempted.
    pub initializer_steps: usize,
    /// Constant steps successfully converted.
    pub constants_materialized: usize,
    /// Nonconstant initializer steps successfully executed by the VM.
    pub dynamic_initializers_materialized: usize,
    /// Unique global/static slots with a materialized value.
    pub runtime_variables: usize,
    /// Canonical types retained for later allocation.
    pub runtime_types: usize,
    /// Direct type-default layers containing at least one constant field.
    pub default_layers: usize,
    /// Constant list objects allocated, including nested lists.
    pub constant_lists: usize,
    /// Initializers retained for a future runtime phase.
    pub unsupported_initializers: usize,
    /// Datums allocated after image construction.
    pub datums_allocated: usize,
}

/// A deterministic runtime-ready constant image for one compiled project.
pub struct RuntimeImage {
    heap: ValueHeap,
    variables: Vec<RuntimeVariable>,
    types: BTreeMap<TypePath, RuntimeType>,
    diagnostics: Vec<RuntimeInitializerDiagnostic>,
    stats: RuntimeImageStats,
}

struct DynamicInitializerFailure {
    phase: InitializerFailurePhase,
    message: String,
    expanded_span: SourceSpan,
}

struct RuntimeBindingIndex {
    globals: BTreeMap<String, FieldName>,
    instance_fields: BTreeMap<String, BTreeMap<String, FieldName>>,
}

impl RuntimeBindingIndex {
    fn build(registry: &VariableRegistry) -> Result<Self, RuntimeImageError> {
        let mut globals = BTreeMap::new();
        let mut instance_fields = BTreeMap::<String, BTreeMap<String, FieldName>>::new();
        for entry in registry.entries() {
            let field = variable_field(&entry.path)?;
            if entry.storage == StorageClass::Instance {
                if let Some(owner) = &entry.owner {
                    instance_fields
                        .entry(owner.path.clone())
                        .or_default()
                        .insert(field.as_str().to_owned(), field);
                }
            } else {
                globals.insert(field.as_str().to_owned(), field);
            }
        }
        Ok(Self {
            globals,
            instance_fields,
        })
    }
}

impl RuntimeImage {
    /// Compiles and materializes a project without allocating map atoms.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeImageLoadError`] for project loading or an invalid
    /// canonical path produced at the frontend/runtime boundary.
    pub fn load(root_file: impl AsRef<Path>) -> Result<Self, RuntimeImageLoadError> {
        let compilation = CompilerDatabase::new()
            .compile(root_file)
            .map_err(RuntimeImageLoadError::Compiler)?;
        Self::from_compilation(&compilation).map_err(RuntimeImageLoadError::Image)
    }

    /// Materializes one existing frontend snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeImageError`] if a frontend path cannot be represented
    /// by the runtime's canonical path or field-name types.
    pub fn from_compilation(compilation: &Compilation) -> Result<Self, RuntimeImageError> {
        let registry = VariableRegistry::build(compilation);
        let plans = registry.initialization_plans();
        let binding_index = RuntimeBindingIndex::build(&registry)?;
        let mut image = Self {
            heap: ValueHeap::new(),
            variables: Vec::new(),
            types: runtime_types(compilation)?,
            diagnostics: Vec::new(),
            stats: RuntimeImageStats {
                variables: registry.entries().len(),
                initializer_steps: plans.global_steps.len()
                    + plans
                        .type_defaults
                        .iter()
                        .map(|plan| plan.steps.len())
                        .sum::<usize>(),
                ..RuntimeImageStats::default()
            },
        };

        let mut steps = plans
            .global_steps
            .iter()
            .chain(
                plans
                    .type_defaults
                    .iter()
                    .flat_map(|plan| plan.steps.iter()),
            )
            .collect::<Vec<_>>();
        steps.sort_by_key(|step| step.ordinal);
        for step in steps {
            let entry = &registry.entries()[step.entry_index];
            match &step.evaluation {
                ConstantEvaluation::Value(constant) => {
                    let value = image.convert_constant(constant)?;
                    image.apply_step_value(entry, step, value)?;
                    image.stats.constants_materialized += 1;
                }
                ConstantEvaluation::Unsupported(unsupported) => {
                    match image.execute_dynamic_initializer(&binding_index, entry, step) {
                        Ok(value) => {
                            image.apply_step_value(entry, step, value)?;
                            image.stats.dynamic_initializers_materialized += 1;
                        }
                        Err(failure) => image.retain_dynamic_failure(
                            compilation,
                            entry,
                            step,
                            unsupported,
                            failure,
                        )?,
                    }
                }
            }
        }
        image.stats.runtime_variables = image.variables.len();
        image.stats.runtime_types = image.types.len();
        image.stats.default_layers = image
            .types
            .values()
            .filter(|runtime_type| runtime_type.defaults.fields().len() != 0)
            .count();
        image.stats.unsupported_initializers = image.diagnostics.len();
        Ok(image)
    }

    /// Returns the runtime value heap.
    #[must_use]
    pub const fn heap(&self) -> &ValueHeap {
        &self.heap
    }

    /// Returns the runtime value heap for later execution integration.
    #[must_use]
    pub const fn heap_mut(&mut self) -> &mut ValueHeap {
        &mut self.heap
    }

    /// Returns materialized global/static slots in first-encounter order.
    #[must_use]
    pub fn variables(&self) -> &[RuntimeVariable] {
        &self.variables
    }

    /// Looks up a materialized global/static slot by canonical variable path.
    #[must_use]
    pub fn variable(&self, path: &str) -> Option<&RuntimeVariable> {
        self.variables.iter().find(|variable| variable.path == path)
    }

    /// Iterates canonical types in lexical path order.
    pub fn types(&self) -> impl Iterator<Item = (&TypePath, &RuntimeType)> {
        self.types.iter()
    }

    /// Returns retained unsupported initializers in project source order.
    #[must_use]
    pub fn diagnostics(&self) -> &[RuntimeInitializerDiagnostic] {
        &self.diagnostics
    }

    /// Returns deterministic materialization counters.
    #[must_use]
    pub const fn stats(&self) -> &RuntimeImageStats {
        &self.stats
    }

    /// Allocates one datum with all constant ancestor defaults applied.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeImageError::UnknownType`] for absent metadata or
    /// [`RuntimeImageError::InheritanceCycle`] for an invalid retained chain.
    pub fn allocate_datum(&mut self, type_path: &TypePath) -> Result<DatumId, RuntimeImageError> {
        let mut chain = Vec::new();
        let mut current = Some(type_path.clone());
        let mut visited = BTreeSet::new();
        while let Some(path) = current.take() {
            if !visited.insert(path.clone()) {
                return Err(RuntimeImageError::InheritanceCycle(path));
            }
            let runtime_type = self
                .types
                .get(&path)
                .ok_or_else(|| RuntimeImageError::UnknownType(path.clone()))?;
            chain.push(runtime_type.defaults.clone());
            current.clone_from(&runtime_type.parent);
        }
        chain.reverse();
        let datum = self
            .heap
            .allocate_datum_with_defaults(type_path.clone(), &chain);
        self.stats.datums_allocated += 1;
        Ok(datum)
    }

    /// Conservatively evaluates and applies one expression to a live datum field.
    ///
    /// This method executes no DM procedures. Unsupported expressions are
    /// returned to the caller with expression-relative source spans and leave
    /// the existing field value unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeImageError`] when a proven constant cannot be converted
    /// to a runtime value or `datum` is stale.
    pub fn apply_constant_field_expression(
        &mut self,
        datum: DatumId,
        field: FieldName,
        expression: &str,
    ) -> Result<ConstantFieldApplication, RuntimeImageError> {
        let tokens = match lex(expression) {
            Ok(tokens) => tokens
                .into_iter()
                .filter(|token| {
                    !matches!(
                        token.kind,
                        TokenKind::LineStart { .. }
                            | TokenKind::Newline
                            | TokenKind::LineContinuation
                    )
                })
                .collect::<Vec<_>>(),
            Err(error) => {
                return Ok(ConstantFieldApplication::Unsupported(UnsupportedConstant {
                    category: UnsupportedCategory::InvalidSyntax,
                    span: error.span,
                }));
            }
        };
        match evaluate_constant(&tokens) {
            ConstantEvaluation::Value(constant) => {
                let value = self.convert_constant(&constant)?;
                self.heap.set_datum_field(datum, field, value)?;
                Ok(ConstantFieldApplication::Applied)
            }
            ConstantEvaluation::Unsupported(unsupported) => {
                Ok(ConstantFieldApplication::Unsupported(unsupported))
            }
        }
    }

    fn convert_constant(&mut self, constant: &ConstantValue) -> Result<Value, RuntimeImageError> {
        Ok(match constant {
            ConstantValue::Null => Value::Null,
            ConstantValue::Number(number) => Value::Number(*number),
            ConstantValue::Text(text) => Value::text(text.as_str()),
            ConstantValue::TypePath(path) => Value::TypePath(parse_type_path(path)?),
            ConstantValue::List(entries) => {
                let list = self.heap.allocate_list();
                self.stats.constant_lists += 1;
                for entry in entries {
                    match entry {
                        ConstantListEntry::Positional(constant) => {
                            let value = self.convert_constant(constant)?;
                            self.heap.list_mut(list)?.add(value);
                        }
                        ConstantListEntry::Associative { key, value } => {
                            let key = self.convert_constant(key)?;
                            let value = self.convert_constant(value)?;
                            self.heap.list_mut(list)?.set_key(key, value);
                        }
                    }
                }
                Value::List(list)
            }
        })
    }

    fn apply_step_value(
        &mut self,
        entry: &VariableEntry,
        step: &InitializationStep,
        value: Value,
    ) -> Result<(), RuntimeImageError> {
        if step.storage == StorageClass::Instance {
            let owner = entry
                .owner
                .as_ref()
                .ok_or_else(|| RuntimeImageError::MissingOwner(step.path.clone()))?;
            let owner = parse_type_path(&owner.path)?;
            let field = variable_field(&step.path)?;
            let runtime_type = self
                .types
                .get_mut(&owner)
                .ok_or_else(|| RuntimeImageError::UnknownType(owner.clone()))?;
            runtime_type.defaults.set(field, value);
            return Ok(());
        }
        if let Some(variable) = self
            .variables
            .iter_mut()
            .find(|variable| variable.path == step.path)
        {
            variable.value = value;
            variable.ordinal = step.ordinal;
            variable.storage = step.storage;
            return Ok(());
        }
        self.variables.push(RuntimeVariable {
            path: step.path.clone(),
            storage: step.storage,
            value,
            ordinal: step.ordinal,
        });
        Ok(())
    }

    fn execute_dynamic_initializer(
        &mut self,
        binding_index: &RuntimeBindingIndex,
        entry: &VariableEntry,
        step: &InitializationStep,
    ) -> Result<Value, DynamicInitializerFailure> {
        let initializer = entry.initializer.as_ref().ok_or_else(|| {
            DynamicInitializerFailure {
                phase: InitializerFailurePhase::Lowering,
                message: format!("initialization step for {:?} has no syntax", step.path),
                expanded_span: entry.span,
            }
        })?;
        let initializer_span = initializer.expanded_span;
        let bindings = self.initializer_bindings(binding_index, entry)?;
        let program = compile_initializer(&initializer.tokens, &bindings, None).map_err(|error| {
            DynamicInitializerFailure {
                phase: InitializerFailurePhase::Lowering,
                message: error.message,
                expanded_span: initializer_span,
            }
        })?;

        let src = if step.storage == StorageClass::Instance {
            let owner = entry
                .owner
                .as_ref()
                .ok_or_else(|| DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Lowering,
                    message: format!("instance variable {:?} has no owning type", step.path),
                    expanded_span: initializer_span,
                })?;
            let owner =
                TypePath::parse(&owner.path).map_err(|error| DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Lowering,
                    message: error.to_string(),
                    expanded_span: initializer_span,
                })?;
            let layers = self.default_layers(&owner).map_err(|mut failure| {
                failure.expanded_span = initializer_span;
                failure
            })?;
            Some(
                self.heap
                    .allocate_datum_with_defaults(owner, layers.as_slice()),
            )
        } else {
            None
        };

        let mut state = ExecutionState::from_heap(std::mem::take(&mut self.heap));
        for name in binding_index.globals.values() {
            state.set_global(name.clone(), Value::Null);
        }
        for variable in &self.variables {
            if let Ok(name) = variable_field(&variable.path) {
                state.set_global(name, variable.value.clone());
            }
        }
        let context = ExecutionContext::new(src.map_or(Value::Null, Value::Datum), Value::Null);
        let result =
            execute_module_in_context(program.module(), program.entry(), &[], &mut state, &context);
        self.heap = state.into_heap();
        if let Some(src) = src {
            let _ = self.heap.destroy_datum(src);
        }
        match result {
            Ok(Value::Datum(_)) => Err(DynamicInitializerFailure {
                phase: InitializerFailurePhase::Execution,
                message: "datum references require per-instance initialization".to_owned(),
                expanded_span: initializer_span,
            }),
            Ok(value) => Ok(value),
            Err(error) => Err(DynamicInitializerFailure {
                phase: InitializerFailurePhase::Execution,
                message: error.message,
                expanded_span: error.source_span.unwrap_or(initializer_span),
            }),
        }
    }

    fn initializer_bindings(
        &self,
        binding_index: &RuntimeBindingIndex,
        entry: &VariableEntry,
    ) -> Result<BTreeMap<String, InitializerBinding>, DynamicInitializerFailure> {
        let mut bindings = binding_index
            .globals
            .iter()
            .map(|(name, field)| (name.clone(), InitializerBinding::Global(field.clone())))
            .collect::<BTreeMap<_, _>>();
        if entry.storage != StorageClass::Instance {
            return Ok(bindings);
        }

        let Some(owner) = &entry.owner else {
            return Ok(bindings);
        };
        let mut owners = Vec::new();
        let mut current =
            Some(
                TypePath::parse(&owner.path).map_err(|error| DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Lowering,
                    message: error.to_string(),
                    expanded_span: entry.span,
                })?,
            );
        while let Some(path) = current.take() {
            owners.push(path.clone());
            current = self
                .types
                .get(&path)
                .and_then(|runtime_type| runtime_type.parent.clone());
        }
        owners.reverse();
        for owner in owners {
            if let Some(fields) = binding_index.instance_fields.get(owner.as_str()) {
                for (name, field) in fields {
                    bindings.insert(name.clone(), InitializerBinding::SrcField(field.clone()));
                }
            }
        }
        Ok(bindings)
    }

    fn default_layers(
        &self,
        type_path: &TypePath,
    ) -> Result<Vec<DatumDefaults>, DynamicInitializerFailure> {
        let mut layers = Vec::new();
        let mut current = Some(type_path.clone());
        let mut visited = BTreeSet::new();
        while let Some(path) = current.take() {
            if !visited.insert(path.clone()) {
                return Err(DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Execution,
                    message: format!("runtime inheritance cycle at {path}"),
                    expanded_span: SourceSpan::new(0, 0),
                });
            }
            let runtime_type = self
                .types
                .get(&path)
                .ok_or_else(|| DynamicInitializerFailure {
                    phase: InitializerFailurePhase::Execution,
                    message: format!("runtime type {path} is absent"),
                    expanded_span: SourceSpan::new(0, 0),
                })?;
            layers.push(runtime_type.defaults.clone());
            current.clone_from(&runtime_type.parent);
        }
        layers.reverse();
        Ok(layers)
    }

    fn retain_dynamic_failure(
        &mut self,
        compilation: &Compilation,
        entry: &VariableEntry,
        step: &InitializationStep,
        unsupported: &dm_globals::UnsupportedConstant,
        failure: DynamicInitializerFailure,
    ) -> Result<(), RuntimeImageError> {
        let initializer = entry
            .initializer
            .as_ref()
            .ok_or_else(|| RuntimeImageError::MissingInitializer(step.path.clone()))?;
        let file = compilation
            .project()
            .file(entry.file_id)
            .ok_or(RuntimeImageError::MissingSourceFile(entry.file_id))?;
        let blocker_span = compilation
            .original_span(entry.file_id, failure.expanded_span)
            .ok_or(RuntimeImageError::MissingSourceFile(entry.file_id))?;
        self.diagnostics.push(RuntimeInitializerDiagnostic {
            variable_path: step.path.clone(),
            storage: step.storage,
            ordinal: step.ordinal,
            file_id: entry.file_id,
            source_path: file.relative_path.to_string_lossy().into_owned(),
            initializer_span: initializer.original_span,
            blocker_span,
            category: unsupported.category,
            phase: failure.phase,
            message: failure.message,
        });
        Ok(())
    }
}

fn runtime_types(
    compilation: &Compilation,
) -> Result<BTreeMap<TypePath, RuntimeType>, RuntimeImageError> {
    let mut types = BTreeMap::new();
    for node in compilation.code_tree().nodes() {
        if node.kind != NodeKind::Type {
            continue;
        }
        let path = parse_type_path(&node.path.to_string())?;
        let parent = node
            .parent_type
            .and_then(|parent| compilation.code_tree().node(parent))
            .map(|parent| parse_type_path(&parent.path.to_string()))
            .transpose()?;
        types.insert(
            path.clone(),
            RuntimeType {
                defaults: DatumDefaults::new(path.clone()),
                path,
                parent,
            },
        );
    }
    Ok(types)
}

fn parse_type_path(path: &str) -> Result<TypePath, RuntimeImageError> {
    TypePath::parse(path).map_err(RuntimeImageError::Value)
}

fn variable_field(path: &str) -> Result<FieldName, RuntimeImageError> {
    let name = path
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| RuntimeImageError::InvalidVariablePath(path.to_owned()))?;
    FieldName::parse(name).map_err(RuntimeImageError::Value)
}

/// Failure while converting a valid frontend snapshot into runtime storage.
#[derive(Debug)]
pub enum RuntimeImageError {
    /// Runtime canonical-value validation failed.
    Value(ValueError),
    /// A planned variable path had no final field segment.
    InvalidVariablePath(String),
    /// A type referenced by a plan or allocation was absent.
    UnknownType(TypePath),
    /// Retained type metadata contained an inheritance cycle.
    InheritanceCycle(TypePath),
    /// An initialization plan referred to a missing initializer.
    MissingInitializer(String),
    /// An instance-default plan referred to a variable without an owner.
    MissingOwner(String),
    /// An initialization entry referred to an absent project file.
    MissingSourceFile(FileId),
}

impl fmt::Display for RuntimeImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(error) => write!(formatter, "invalid runtime value: {error}"),
            Self::InvalidVariablePath(path) => write!(formatter, "invalid variable path {path:?}"),
            Self::UnknownType(path) => write!(formatter, "runtime type {path} is absent"),
            Self::InheritanceCycle(path) => {
                write!(formatter, "runtime inheritance cycle at {path}")
            }
            Self::MissingInitializer(path) => {
                write!(
                    formatter,
                    "initialization step for {path} has no initializer"
                )
            }
            Self::MissingOwner(path) => {
                write!(formatter, "instance variable {path} has no owning type")
            }
            Self::MissingSourceFile(file) => {
                write!(
                    formatter,
                    "initializer source file {} is absent",
                    file.index()
                )
            }
        }
    }
}

impl std::error::Error for RuntimeImageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Value(error) => Some(error),
            Self::InvalidVariablePath(_)
            | Self::UnknownType(_)
            | Self::InheritanceCycle(_)
            | Self::MissingInitializer(_)
            | Self::MissingOwner(_)
            | Self::MissingSourceFile(_) => None,
        }
    }
}

impl From<ValueError> for RuntimeImageError {
    fn from(error: ValueError) -> Self {
        Self::Value(error)
    }
}

/// Failure while loading or materializing a project.
#[derive(Debug)]
pub enum RuntimeImageLoadError {
    /// Frontend project loading failed.
    Compiler(CompilerError),
    /// Runtime materialization failed.
    Image(RuntimeImageError),
}

impl fmt::Display for RuntimeImageLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiler(error) => write!(formatter, "frontend compilation failed: {error}"),
            Self::Image(error) => write!(formatter, "runtime image failed: {error}"),
        }
    }
}

impl std::error::Error for RuntimeImageLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compiler(error) => Some(error),
            Self::Image(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::CompilerDatabase;
    use dm_globals::{StorageClass, UnsupportedCategory};
    use dm_value::{FieldName, TypePath, Value};

    use super::{ConstantFieldApplication, InitializerFailurePhase, RuntimeImage};

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "dream64-dm-runtime-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("fixture directory should be created");
            Self(path)
        }

        fn write(&self, name: &str, source: &str) {
            fs::write(self.0.join(name), source).expect("fixture source should be written");
        }

        fn image(&self) -> RuntimeImage {
            let compilation = CompilerDatabase::new()
                .compile(self.0.join("world.dme"))
                .expect("fixture should compile");
            RuntimeImage::from_compilation(&compilation).expect("fixture should materialize")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("fixture directory should be removed");
        }
    }

    fn field(name: &str) -> FieldName {
        FieldName::parse(name).expect("test field should be valid")
    }

    fn type_path(path: &str) -> TypePath {
        TypePath::parse(path).expect("test type path should be valid")
    }

    #[test]
    fn applies_global_and_static_overrides_in_project_order() {
        let fixture = Fixture::new();
        fixture.write(
            "world.dme",
            "#include \"first.dm\"\n#include \"second.dm\"\n",
        );
        fixture.write(
            "first.dm",
            "var/global/root = 1\n/datum/example\n\tvar/static/shared = 2\n",
        );
        fixture.write("second.dm", "root = 3\n/datum/example\n\tshared = 4\n");

        let image = fixture.image();
        let root = image
            .variables()
            .iter()
            .find(|variable| variable.path.ends_with("/root"))
            .expect("root should be materialized");
        let shared = image
            .variables()
            .iter()
            .find(|variable| variable.path.ends_with("/shared"))
            .expect("static should be materialized");

        assert_eq!(root.storage, StorageClass::Global);
        assert_eq!(root.value.as_number(), Some(3.0));
        assert_eq!(shared.storage, StorageClass::Static);
        assert_eq!(shared.value.as_number(), Some(4.0));
        assert!(root.ordinal < shared.ordinal);
        assert_eq!(image.stats().constants_materialized, 4);
        assert_eq!(image.stats().runtime_variables, 2);
    }

    #[test]
    fn layers_ancestor_defaults_and_reopen_overrides_deterministically() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/name = \"base\"\n\tvar/health = 10\n/datum/base/child\n\tname = \"child\"\n/datum/base/child\n\tname = \"reopened\"\n\tvar/speed = 2\n",
        );

        let mut image = fixture.image();
        let datum_id = image
            .allocate_datum(&type_path("/datum/base/child"))
            .expect("child datum should allocate");
        let datum = image.heap().datum(datum_id).expect("datum should be live");
        let names = datum
            .fields()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names, ["name", "health", "speed"]);
        assert_eq!(datum.field(&field("name")), Ok(&Value::text("reopened")));
        assert_eq!(
            datum.field(&field("health")).unwrap().as_number(),
            Some(10.0)
        );
        assert_eq!(datum.field(&field("speed")).unwrap().as_number(), Some(2.0));
        assert_eq!(image.stats().default_layers, 2);
    }

    #[test]
    fn converts_nested_associative_lists_and_preserves_default_aliases() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"lists.dm\"\n");
        fixture.write(
            "lists.dm",
            "/datum/holder\n\tvar/items = list(1, \"nested\" = list(2, \"answer\" = 3))\n",
        );

        let mut image = fixture.image();
        let repeated = fixture.image();
        assert_eq!(image.variables, repeated.variables);
        assert_eq!(image.types, repeated.types);
        assert_eq!(image.diagnostics, repeated.diagnostics);
        assert_eq!(image.stats, repeated.stats);

        let holder = type_path("/datum/holder");
        let first = image
            .allocate_datum(&holder)
            .expect("first holder should allocate");
        let second = image
            .allocate_datum(&holder)
            .expect("second holder should allocate");
        let first_items = image
            .heap()
            .datum(first)
            .unwrap()
            .field(&field("items"))
            .unwrap()
            .clone();
        let second_items = image
            .heap()
            .datum(second)
            .unwrap()
            .field(&field("items"))
            .unwrap()
            .clone();
        let (Value::List(first_list), Value::List(second_list)) = (first_items, second_items)
        else {
            panic!("items should be list handles");
        };

        assert_eq!(
            first_list, second_list,
            "defaults use shallow DM value copies"
        );
        let outer = image.heap().list(first_list).unwrap();
        assert_eq!(outer.get(1).unwrap().as_number(), Some(1.0));
        let Value::List(nested_id) = outer.get_key(&Value::text("nested")).unwrap() else {
            panic!("nested key should contain a list handle");
        };
        let nested = image.heap().list(*nested_id).unwrap();
        assert_eq!(nested.get(1).unwrap().as_number(), Some(2.0));
        assert_eq!(
            nested.get_key(&Value::text("answer")).unwrap().as_number(),
            Some(3.0)
        );
        assert_eq!(image.stats().constant_lists, 2);
    }

    #[test]
    fn retains_source_mapped_unsupported_initializers_without_guessing() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "var/global/dynamic = build_value()\n/datum/example\n\tvar/runtime = new /datum\n",
        );

        let mut image = fixture.image();
        let diagnostics = image.diagnostics();

        assert_eq!(diagnostics.len(), 2);
        assert_eq!(diagnostics[0].category, UnsupportedCategory::Call);
        assert_eq!(diagnostics[0].phase, InitializerFailurePhase::Lowering);
        assert!(diagnostics[0].message.contains("unknown procedure"));
        assert_eq!(diagnostics[1].category, UnsupportedCategory::NewExpression);
        assert!(diagnostics[0].ordinal < diagnostics[1].ordinal);
        assert!(
            diagnostics
                .iter()
                .all(|diagnostic| diagnostic.source_path == "vars.dm")
        );
        assert!(diagnostics.iter().all(|diagnostic| {
            !diagnostic.initializer_span.is_empty() && !diagnostic.blocker_span.is_empty()
        }));
        assert!(
            image
                .variables()
                .iter()
                .all(|variable| !variable.path.ends_with("/dynamic"))
        );

        let datum_id = image
            .allocate_datum(&type_path("/datum/example"))
            .expect("datum should still allocate");
        assert!(
            image
                .heap()
                .datum(datum_id)
                .unwrap()
                .field(&field("runtime"))
                .is_err(),
            "unsupported defaults must not be invented"
        );
    }

    #[test]
    fn executes_identifier_dependencies_and_overrides_in_source_order() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "var/global/base = 2\nvar/global/derived = base + 3\nbase = 10\nvar/global/final_value = base + derived\n",
        );

        let image = fixture.image();
        let number = |suffix: &str| {
            image
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with(suffix))
                .and_then(|variable| variable.value.as_number())
        };

        assert_eq!(number("/base"), Some(10.0));
        assert_eq!(number("/derived"), Some(5.0));
        assert_eq!(number("/final_value"), Some(15.0));
        assert_eq!(image.stats().constants_materialized, 2);
        assert_eq!(image.stats().dynamic_initializers_materialized, 2);
        assert!(image.diagnostics().is_empty());
    }

    #[test]
    fn executes_src_field_and_explicit_global_references() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "var/global/offset = 3\n/datum/example\n\tvar/base = 4\n\tvar/combined = base + global.offset\n",
        );

        let mut image = fixture.image();
        assert!(image.diagnostics().is_empty(), "{:?}", image.diagnostics());
        let datum = image
            .allocate_datum(&type_path("/datum/example"))
            .expect("datum should allocate");

        assert_eq!(
            image
                .heap()
                .datum_field(datum, &field("combined"))
                .unwrap()
                .as_number(),
            Some(7.0)
        );
        assert_eq!(image.stats().dynamic_initializers_materialized, 1);
        assert!(image.diagnostics().is_empty());
    }

    #[test]
    fn executes_list_expressions_with_runtime_values() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "var/global/seed = 2\nvar/global/items = list(seed, \"answer\" = seed + 1)\n",
        );

        let image = fixture.image();
        let items = image
            .variables()
            .iter()
            .find(|variable| variable.path.ends_with("/items"))
            .expect("items should materialize");
        let Value::List(items) = items.value else {
            panic!("items should be a runtime list");
        };
        let list = image.heap().list(items).expect("list should remain live");

        assert_eq!(list.get(1).unwrap().as_number(), Some(2.0));
        assert_eq!(
            list.get_key(&Value::text("answer")).unwrap().as_number(),
            Some(3.0)
        );
        assert_eq!(image.stats().dynamic_initializers_materialized, 1);
        assert!(image.diagnostics().is_empty());
    }

    #[test]
    fn applies_only_proven_constant_field_expressions() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write("types.dm", "/datum/example\n\tvar/value = 2\n");
        let mut image = fixture.image();
        let datum = image
            .allocate_datum(&type_path("/datum/example"))
            .expect("datum should allocate");

        assert_eq!(
            image
                .apply_constant_field_expression(datum, field("value"), "3 + 4")
                .expect("constant should apply"),
            ConstantFieldApplication::Applied
        );
        let unsupported = image
            .apply_constant_field_expression(datum, field("value"), "build_value()")
            .expect("unsupported expression should remain recoverable");
        assert!(matches!(
            unsupported,
            ConstantFieldApplication::Unsupported(ref blocker)
                if blocker.category == UnsupportedCategory::Call
        ));
        assert_eq!(
            image
                .heap()
                .datum_field(datum, &field("value"))
                .unwrap()
                .as_number(),
            Some(7.0)
        );
        assert_eq!(image.stats().datums_allocated, 1);
    }
}

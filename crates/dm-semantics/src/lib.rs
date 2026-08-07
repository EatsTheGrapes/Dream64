//! Procedure-level semantic indexing for Dream Maker projects.
//!
//! The registry converts canonical procedure nodes into source-ordered
//! implementation chains. Each implementation records the exact predecessor a
//! future `..()` lowering should invoke: the previous implementation on the
//! same type, or the inherited procedure's effective implementation.
//!
//! Inheritance follows the object tree's resolved hierarchy, including the final
//! source-ordered constant `parent_type = /some/type` assignment on each type.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{BTreeMap, BTreeSet};

use dm_compiler::Compilation;
use dm_core::{FileId, SourceSpan};
use dm_lexer::TokenKind;
use dm_object_tree::{CodePath, NodeId, NodeKind};
use dm_syntax::DefinitionKind;
use dm_value::FieldName;

const STANDARD_BUILTINS: &str = concat!(
    "/proc/isarea(...)\n",
    "\tfor(var/location in args)\n",
    "\t\tif(!istype(location, /area))\n",
    "\t\t\treturn 0\n",
    "\treturn 1\n",
    "/proc/ismob(...)\n",
    "\tfor(var/location in args)\n",
    "\t\tif(!istype(location, /mob))\n",
    "\t\t\treturn 0\n",
    "\treturn 1\n",
    "/proc/isobj(...)\n",
    "\tfor(var/location in args)\n",
    "\t\tif(!istype(location, /obj))\n",
    "\t\t\treturn 0\n",
    "\treturn 1\n",
    "/proc/get_dir(reference, target)\n",
    "\tif(!istype(reference, /atom) || !istype(target, /atom))\n",
    "\t\treturn 0\n",
    "\tvar/direction = 0\n",
    "\tif(target.y > reference.y)\n",
    "\t\tdirection |= 1\n",
    "\telse if(target.y < reference.y)\n",
    "\t\tdirection |= 2\n",
    "\tif(target.x > reference.x)\n",
    "\t\tdirection |= 4\n",
    "\telse if(target.x < reference.x)\n",
    "\t\tdirection |= 8\n",
    "\treturn direction\n",
    "/proc/istext(value)\n",
    "\treturn !isnull(value) && !isnum(value) && !ispath(value) && !islist(value) && !istype(value)\n",
    "/proc/orange(first, second = usr)\n",
    "\tvar/distance\n",
    "\tvar/center\n",
    "\tif(isnum(first))\n",
    "\t\tdistance = first\n",
    "\t\tcenter = second\n",
    "\telse\n",
    "\t\tcenter = first\n",
    "\t\tdistance = second\n",
    "\tvar/output = list()\n",
    "\tfor(var/atom/candidate in range(distance, center))\n",
    "\t\tif(candidate == center || candidate.loc == center)\n",
    "\t\t\tcontinue\n",
    "\t\toutput[length(output) + 1] = candidate\n",
    "\treturn output\n",
    "/proc/min(...)\n",
    "\tvar/list/values = args\n",
    "\tif(length(args) == 1 && islist(args[1]))\n",
    "\t\tvalues = args[1]\n",
    "\tif(!length(values))\n",
    "\t\treturn null\n",
    "\tvar/result = values[1]\n",
    "\tfor(var/value in values)\n",
    "\t\tif(value < result)\n",
    "\t\t\tresult = value\n",
    "\treturn result\n",
    "/proc/max(...)\n",
    "\tvar/list/values = args\n",
    "\tif(length(args) == 1 && islist(args[1]))\n",
    "\t\tvalues = args[1]\n",
    "\tif(!length(values))\n",
    "\t\treturn null\n",
    "\tvar/result = values[1]\n",
    "\tfor(var/value in values)\n",
    "\t\tif(value > result)\n",
    "\t\t\tresult = value\n",
    "\treturn result\n",
);
const STANDARD_BUILTIN_NAMES: [&str; 8] = [
    "isarea", "ismob", "isobj", "get_dir", "istext", "orange", "min", "max",
];

/// Tree-local identity of a canonical procedure.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcedureId(u32);

impl ProcedureId {
    /// Returns this identity's index in [`ProcedureRegistry::procedures`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Tree-local identity of one body in a procedure's override chain.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcedureImplementationId {
    procedure: ProcedureId,
    index: u32,
}

impl ProcedureImplementationId {
    /// Returns the canonical procedure containing this implementation.
    #[must_use]
    pub const fn procedure(self) -> ProcedureId {
        self.procedure
    }

    /// Returns this implementation's source-order index within its procedure.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index as usize
    }
}

/// Syntactic role used to introduce a procedure implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcedureImplementationKind {
    /// A body declared through an explicit `proc` path node.
    Declaration,
    /// A body written directly beneath an owning type.
    Override,
}

/// One source body in a canonical procedure's override chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcedureImplementation {
    /// Tree-local identity of this implementation.
    pub id: ProcedureImplementationId,
    /// Physical source file containing the declaration.
    pub file_id: FileId,
    /// Index into the corresponding syntax file's definition table.
    pub definition_index: usize,
    /// Position in the preprocessor-expanded declaration stream.
    pub ordinal: usize,
    /// Complete source span of the procedure header.
    pub span: SourceSpan,
    /// Whether source used a declaration or override spelling.
    pub kind: ProcedureImplementationKind,
    /// Exact implementation a future `..()` expression should invoke.
    pub parent_target: Option<ProcedureImplementationId>,
}

/// One canonical procedure and all bodies assigned to it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Procedure {
    /// Tree-local canonical procedure identity.
    pub id: ProcedureId,
    /// Corresponding canonical object-tree node.
    pub node: NodeId,
    /// Canonical absolute procedure path.
    pub path: CodePath,
    /// Owning type, or `None` for a global procedure under `/proc`.
    pub owner_type: Option<NodeId>,
    /// Nearest same-name procedure in the default parent hierarchy.
    pub inherited_procedure: Option<ProcedureId>,
    /// Implementations in preprocessor-expanded source order.
    pub implementations: Vec<ProcedureImplementation>,
    /// Body selected when this canonical procedure is dispatched.
    pub effective_target: Option<ProcedureImplementationId>,
}

/// Project-wide registry of canonical procedures and override chains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcedureRegistry {
    procedures: Vec<Procedure>,
    by_node: BTreeMap<NodeId, ProcedureId>,
    by_path: BTreeMap<CodePath, ProcedureId>,
}

/// Executable VM module paired with semantic implementation identities.
#[derive(Debug)]
pub struct ExecutableProcedures {
    module: dm_vm::Module,
    implementations: BTreeMap<ProcedureImplementationId, dm_vm::ProcedureId>,
}

impl ExecutableProcedures {
    /// Returns the compiled VM module.
    #[must_use]
    pub const fn module(&self) -> &dm_vm::Module {
        &self.module
    }

    /// Resolves a semantic implementation to its VM-local identity.
    #[must_use]
    pub fn implementation(
        &self,
        implementation: ProcedureImplementationId,
    ) -> Option<dm_vm::ProcedureId> {
        self.implementations.get(&implementation).copied()
    }
}

impl ProcedureRegistry {
    /// Builds a registry from the compiler's accepted canonical declarations.
    #[must_use]
    pub fn build(compilation: &Compilation) -> Self {
        let tree = compilation.code_tree();
        let procedure_nodes: Vec<_> = tree
            .nodes()
            .iter()
            .filter(|node| node.kind == NodeKind::Procedure)
            .collect();
        let by_node: BTreeMap<_, _> = procedure_nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id, procedure_id(index)))
            .collect();
        let by_path = procedure_nodes
            .iter()
            .map(|node| (node.path.clone(), by_node[&node.id]))
            .collect();

        let mut procedures: Vec<_> = procedure_nodes
            .into_iter()
            .map(|node| {
                let id = by_node[&node.id];
                let implementations = node
                    .declarations
                    .iter()
                    .filter_map(|declaration_id| tree.declaration(*declaration_id))
                    .filter_map(|declaration| {
                        let kind = match declaration.kind {
                            DefinitionKind::Procedure => ProcedureImplementationKind::Declaration,
                            DefinitionKind::ProcedureOverride => {
                                ProcedureImplementationKind::Override
                            }
                            DefinitionKind::Type
                            | DefinitionKind::Verb
                            | DefinitionKind::Variable
                            | DefinitionKind::VariableOverride => return None,
                        };
                        Some((declaration, kind))
                    })
                    .enumerate()
                    .map(|(index, (declaration, kind))| ProcedureImplementation {
                        id: implementation_id(id, index),
                        file_id: declaration.file_id,
                        definition_index: declaration.definition_index,
                        ordinal: declaration.ordinal,
                        span: declaration.span,
                        kind,
                        parent_target: None,
                    })
                    .collect::<Vec<_>>();
                Procedure {
                    id,
                    node: node.id,
                    path: node.path.clone(),
                    owner_type: node.owner_type,
                    inherited_procedure: node
                        .inherited_member
                        .and_then(|parent| by_node.get(&parent).copied()),
                    effective_target: implementations.last().map(|body| body.id),
                    implementations,
                }
            })
            .collect();

        for procedure_index in 0..procedures.len() {
            let inherited_target = procedures[procedure_index]
                .inherited_procedure
                .and_then(|parent| effective_target(&procedures, parent));
            for implementation_index in 0..procedures[procedure_index].implementations.len() {
                let parent_target = if implementation_index == 0 {
                    inherited_target
                } else {
                    Some(implementation_id(
                        procedures[procedure_index].id,
                        implementation_index - 1,
                    ))
                };
                procedures[procedure_index].implementations[implementation_index].parent_target =
                    parent_target;
            }
        }

        Self {
            procedures,
            by_node,
            by_path,
        }
    }

    /// Returns canonical procedures in object-tree node order.
    #[must_use]
    pub fn procedures(&self) -> &[Procedure] {
        &self.procedures
    }

    /// Looks up a canonical procedure by registry identity.
    #[must_use]
    pub fn procedure(&self, id: ProcedureId) -> Option<&Procedure> {
        self.procedures.get(id.index())
    }

    /// Looks up a canonical procedure by object-tree node identity.
    #[must_use]
    pub fn find_node(&self, node: NodeId) -> Option<ProcedureId> {
        self.by_node.get(&node).copied()
    }

    /// Looks up a canonical procedure by absolute code path.
    #[must_use]
    pub fn find_path(&self, path: &CodePath) -> Option<ProcedureId> {
        self.by_path.get(path).copied()
    }

    /// Looks up one procedure implementation by its composite identity.
    #[must_use]
    pub fn implementation(
        &self,
        id: ProcedureImplementationId,
    ) -> Option<&ProcedureImplementation> {
        self.procedure(id.procedure())?
            .implementations
            .get(id.index())
    }

    /// Compiles every registered implementation with its exact resolved
    /// parent-call target.
    ///
    /// # Errors
    ///
    /// Returns [`dm_vm::CompileError`] when a retained source definition is
    /// unavailable or a procedure body is outside the executable VM subset.
    pub fn compile_vm(
        &self,
        compilation: &Compilation,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        self.compile_vm_selected(
            compilation,
            self.procedures.iter().flat_map(|procedure| {
                procedure
                    .implementations
                    .iter()
                    .map(|implementation| implementation.id)
            }),
        )
    }

    /// Compiles selected implementations and their exact `..()` ancestors.
    ///
    /// This lets bounded runtime phases compile only their declared entry
    /// points without unrelated unsupported procedures preventing execution.
    ///
    /// # Errors
    ///
    /// Returns [`dm_vm::CompileError`] when a selected body, or one of its
    /// parent-call targets, is unavailable or outside the executable subset.
    pub fn compile_vm_implementations(
        &self,
        compilation: &Compilation,
        implementations: impl IntoIterator<Item = ProcedureImplementationId>,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let mut selected: BTreeSet<_> = implementations.into_iter().collect();
        let mut pending: Vec<_> = selected.iter().copied().collect();
        while let Some(implementation) = pending.pop() {
            if let Some(parent) = self
                .implementation(implementation)
                .and_then(|body| body.parent_target)
                && selected.insert(parent)
            {
                pending.push(parent);
            }

            let Some(body) = self.implementation(implementation) else {
                continue;
            };
            let Some(definition) = compilation
                .syntax(body.file_id)
                .and_then(|syntax| syntax.definitions.get(body.definition_index))
            else {
                continue;
            };
            for selector in dynamic_call_literal_selectors(definition) {
                for procedure in &self.procedures {
                    if !procedure_matches_dynamic_selector(procedure, &selector) {
                        continue;
                    }
                    if let Some(target) = procedure.effective_target
                        && selected.insert(target)
                    {
                        pending.push(target);
                    }
                }
            }
            for selector in static_call_selectors(definition) {
                if let Some(target) =
                    self.static_call_target(implementation, &selector, compilation)
                    && selected.insert(target)
                {
                    pending.push(target);
                }
            }
        }
        self.compile_vm_selected(compilation, selected)
    }

    /// Compiles each requested body independently, without following calls or
    /// `..()` targets, and retains every lowering result.
    ///
    /// This is intended for fast compatibility inventories. Runtime phases
    /// should use [`Self::compile_vm_implementations`] so their dependency
    /// closure is present in the generated module.
    #[must_use]
    pub fn compile_vm_bodies_independently(
        &self,
        compilation: &Compilation,
        implementations: impl IntoIterator<Item = ProcedureImplementationId>,
    ) -> Vec<(
        ProcedureImplementationId,
        Result<ExecutableProcedures, dm_vm::CompileError>,
    )> {
        let direct_fields = direct_instance_fields(compilation);
        let mut inherited_field_cache = BTreeMap::new();
        implementations
            .into_iter()
            .map(|implementation| {
                (
                    implementation,
                    self.compile_vm_selected_with_fields(
                        compilation,
                        [implementation],
                        &direct_fields,
                        &mut inherited_field_cache,
                        false,
                    ),
                )
            })
            .collect()
    }

    fn compile_vm_selected(
        &self,
        compilation: &Compilation,
        selected: impl IntoIterator<Item = ProcedureImplementationId>,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let direct_fields = direct_instance_fields(compilation);
        let mut inherited_field_cache = BTreeMap::new();
        self.compile_vm_selected_with_fields(
            compilation,
            selected,
            &direct_fields,
            &mut inherited_field_cache,
            true,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn compile_vm_selected_with_fields(
        &self,
        compilation: &Compilation,
        selected: impl IntoIterator<Item = ProcedureImplementationId>,
        direct_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
        inherited_field_cache: &mut BTreeMap<NodeId, BTreeMap<String, FieldName>>,
        include_parent_targets: bool,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let selected: BTreeSet<_> = selected.into_iter().collect();
        let global_fields = declared_global_fields(compilation);
        let mut ordered = Vec::new();
        for procedure in &self.procedures {
            for implementation in &procedure.implementations {
                if !selected.contains(&implementation.id) {
                    continue;
                }
                let definition = compilation
                    .syntax(implementation.file_id)
                    .and_then(|syntax| syntax.definitions.get(implementation.definition_index))
                    .ok_or_else(|| dm_vm::CompileError {
                        message: format!(
                            "missing syntax definition for implementation of {}",
                            procedure.path
                        ),
                    })?;
                ordered.push((procedure, implementation, definition));
            }
        }
        let indices: BTreeMap<_, _> = ordered
            .iter()
            .enumerate()
            .map(|(index, (_, implementation, _))| (implementation.id, index))
            .collect();
        let builtin_syntax =
            dm_syntax::parse(STANDARD_BUILTINS).map_err(|error| dm_vm::CompileError {
                message: format!("failed to parse Dream64 standard location builtins: {error}"),
            })?;
        if builtin_syntax.definitions.len() != STANDARD_BUILTIN_NAMES.len() {
            return Err(dm_vm::CompileError {
                message: "Dream64 standard location builtin catalog is malformed".to_owned(),
            });
        }
        let builtin_indices: BTreeMap<_, _> = STANDARD_BUILTIN_NAMES
            .iter()
            .enumerate()
            .map(|(offset, name)| ((*name).to_owned(), ordered.len() + offset))
            .collect();
        let mut specs: Vec<_> = ordered
            .iter()
            .map(|(procedure, implementation, definition)| {
                let parent = if include_parent_targets {
                    implementation
                        .parent_target
                        .map(|parent| {
                            indices
                                .get(&parent)
                                .copied()
                                .ok_or_else(|| dm_vm::CompileError {
                                    message: format!(
                                        "parent implementation for {} is missing from the VM module",
                                        procedure.path
                                    ),
                                })
                        })
                        .transpose()?
                } else {
                    None
                };
                let selectors = static_call_selectors(definition);
                let mut static_calls: BTreeMap<_, _> = selectors
                    .iter()
                    .filter_map(|selector| {
                        self.static_call_target(implementation.id, selector, compilation)
                            .and_then(|target| indices.get(&target).copied())
                            .map(|target| (selector.clone(), target))
                    })
                    .collect();
                for selector in selectors {
                    if let Some(target) = builtin_indices.get(&selector) {
                        static_calls.entry(selector).or_insert(*target);
                    }
                }
                Ok(dm_vm::ProcedureSpec {
                    path: format!("{}@{}", procedure.path, implementation.ordinal),
                    definition,
                    parent,
                    static_calls,
                    src_fields: procedure
                        .owner_type
                        .map(|owner| {
                            inherited_fields(
                                compilation,
                                owner,
                                direct_fields,
                                inherited_field_cache,
                            )
                        })
                        .unwrap_or_default(),
                    global_fields: global_fields.clone(),
                })
            })
            .collect::<Result<_, dm_vm::CompileError>>()?;
        for (offset, definition) in builtin_syntax.definitions.iter().enumerate() {
            let name = STANDARD_BUILTIN_NAMES[offset];
            specs.push(dm_vm::ProcedureSpec {
                path: format!("/proc/{name}@dream64_builtin"),
                definition,
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: global_fields.clone(),
            });
        }
        let module = dm_vm::compile_module_specs(&specs)?;
        let implementations = ordered
            .iter()
            .enumerate()
            .map(|(index, (_, implementation, _))| {
                Ok((
                    implementation.id,
                    module
                        .procedure_id_at(index)
                        .ok_or_else(|| dm_vm::CompileError {
                            message: format!("compiled procedure spec {index} has no VM identity"),
                        })?,
                ))
            })
            .collect::<Result<_, dm_vm::CompileError>>()?;
        Ok(ExecutableProcedures {
            module,
            implementations,
        })
    }
}

/// Returns bare names for project-wide variables, including globals introduced
/// by macros such as `SUBSYSTEM_DEF(mapping)`. A global has no owning type;
/// static and instance variables remain deliberately excluded. BYOND's built-in
/// `world` singleton is also available as a bare global even though it has no
/// source declaration.
fn declared_global_fields(compilation: &Compilation) -> BTreeMap<String, FieldName> {
    let mut fields: BTreeMap<String, FieldName> = compilation
        .code_tree()
        .nodes()
        .iter()
        .filter(|node| node.kind == NodeKind::Variable && node.owner_type.is_none())
        .filter_map(|node| {
            let name = node.path.segments().last()?;
            FieldName::parse(name)
                .ok()
                .map(|field| (name.clone(), field))
        })
        .collect();
    fields.insert(
        "world".to_owned(),
        FieldName::parse("world").expect("built-in world global name is valid"),
    );
    fields
}

impl ProcedureRegistry {
    fn static_call_target(
        &self,
        implementation: ProcedureImplementationId,
        selector: &str,
        compilation: &Compilation,
    ) -> Option<ProcedureImplementationId> {
        let procedure = self.procedure(implementation.procedure())?;
        let mut owner = procedure.owner_type;
        let tree = compilation.code_tree();
        while let Some(current_owner) = owner {
            if let Some(candidate) = self.procedures.iter().find(|candidate| {
                candidate.owner_type == Some(current_owner)
                    && candidate
                        .path
                        .segments()
                        .last()
                        .is_some_and(|name| name == selector)
            }) {
                return effective_target(&self.procedures, candidate.id);
            }
            owner = tree.node(current_owner).and_then(|node| node.parent_type);
        }
        self.procedures
            .iter()
            .find(|candidate| {
                candidate.owner_type.is_none()
                    && candidate
                        .path
                        .segments()
                        .last()
                        .is_some_and(|name| name == selector)
            })
            .and_then(|candidate| effective_target(&self.procedures, candidate.id))
    }
}

fn dynamic_call_literal_selectors(definition: &dm_syntax::Definition) -> BTreeSet<String> {
    let mut selectors = BTreeSet::new();
    for line in &definition.body {
        let tokens = &line.tokens;
        for call_index in 0..tokens.len().saturating_sub(1) {
            if !matches!(&tokens[call_index].kind, TokenKind::Identifier(name) if name == "call")
                || !matches!(tokens[call_index + 1].kind, TokenKind::Punctuation('('))
            {
                continue;
            }
            let mut depth = 1usize;
            let mut separator = None;
            for (offset, token) in tokens[call_index + 2..].iter().enumerate() {
                match &token.kind {
                    TokenKind::Punctuation('(') => depth += 1,
                    TokenKind::Punctuation(')') => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    TokenKind::Punctuation(',') if depth == 1 => {
                        separator = Some(call_index + 2 + offset);
                        break;
                    }
                    _ => {}
                }
            }
            let selector_index = separator.map_or(call_index + 2, |index| index + 1);
            if let Some(TokenKind::String(selector) | TokenKind::RawString(selector)) =
                tokens.get(selector_index).map(|token| &token.kind)
            {
                selectors.insert(selector.clone());
            }
        }
    }
    selectors
}

fn static_call_selectors(definition: &dm_syntax::Definition) -> BTreeSet<String> {
    definition
        .body
        .iter()
        .flat_map(|line| line.tokens.windows(2))
        .filter_map(|tokens| match (&tokens[0].kind, &tokens[1].kind) {
            (TokenKind::Identifier(name), TokenKind::Punctuation('(')) => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn procedure_matches_dynamic_selector(procedure: &Procedure, selector: &str) -> bool {
    let selector = selector.trim_matches('/');
    let selector = selector.rsplit('/').next().unwrap_or(selector);
    procedure
        .path
        .segments()
        .last()
        .is_some_and(|name| name == selector)
}

fn direct_instance_fields(
    compilation: &Compilation,
) -> BTreeMap<NodeId, BTreeMap<String, FieldName>> {
    let tree = compilation.code_tree();
    let mut fields = BTreeMap::<NodeId, BTreeMap<String, FieldName>>::new();
    for node in tree
        .nodes()
        .iter()
        .filter(|node| node.kind == NodeKind::Variable)
    {
        let Some(owner) = node.owner_type else {
            continue;
        };
        let Some(name) = node.path.segments().last() else {
            continue;
        };
        if let Ok(field) = FieldName::parse(name) {
            fields.entry(owner).or_default().insert(name.clone(), field);
        }
    }
    fields
}

fn inherited_fields(
    compilation: &Compilation,
    owner: NodeId,
    direct_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
    cache: &mut BTreeMap<NodeId, BTreeMap<String, FieldName>>,
) -> BTreeMap<String, FieldName> {
    if let Some(fields) = cache.get(&owner) {
        return fields.clone();
    }
    let tree = compilation.code_tree();
    let mut hierarchy = Vec::new();
    let mut current = Some(owner);
    while let Some(node) = current {
        hierarchy.push(node);
        current = tree.node(node).and_then(|type_node| type_node.parent_type);
    }
    hierarchy.reverse();
    let mut fields = BTreeMap::new();
    for node in hierarchy {
        if let Some(direct) = direct_fields.get(&node) {
            fields.extend(direct.clone());
        }
        standard_instance_fields(tree.node(node).map(|node| &node.path), &mut fields);
    }
    cache.insert(owner, fields.clone());
    fields
}

/// Adds the fields supplied by BYOND's built-in datum and atom hierarchies.
///
/// The object tree deliberately seeds only standard *types*, since their
/// members have no user source declaration. VM lowering still needs the
/// corresponding names, however, so bare reads such as `type`, `loc`, and
/// `dir` lower exactly like declared `src` fields. Keep this catalog at the
/// semantic boundary: atom-only names must not become visible on arbitrary
/// `/datum`s.
fn standard_instance_fields(path: Option<&CodePath>, fields: &mut BTreeMap<String, FieldName>) {
    let Some(path) = path else {
        return;
    };
    let names: &[&str] = match path.to_string().as_str() {
        // Every datum exposes its canonical runtime type through this
        // read-only built-in field. The VM materializes its value from the
        // datum record rather than from a user-declared default.
        "/datum" => &["tag", "type", "parent_type"],
        "/world" => &[
            "system_type",
            "icon_size",
            "tick_lag",
            "fps",
            "timezone",
            "cpu",
            "time",
            "timeofday",
            "realtime",
            "maxx",
            "maxy",
            "maxz",
        ],
        "/atom" => &[
            "alpha",
            "appearance_flags",
            "blend_mode",
            "color",
            "density",
            "desc",
            "dir",
            "icon",
            "icon_state",
            "invisibility",
            "layer",
            "loc",
            "maptext",
            "maptext_height",
            "maptext_width",
            "mouse_opacity",
            "name",
            "opacity",
            "overlays",
            "plane",
            "pixel_w",
            "pixel_z",
            "render_source",
            "render_target",
            "transform",
            "underlays",
            "vis_contents",
            "vis_flags",
            "x",
            "y",
            "z",
        ],
        "/atom/movable" => &[
            "animate_movement",
            "bound_height",
            "bound_width",
            "bound_x",
            "bound_y",
            "glide_size",
            "pixel_x",
            "pixel_y",
            "screen_loc",
            "step_size",
        ],
        _ => return,
    };
    for name in names {
        // All catalog entries are fixed, valid DM identifiers.
        fields.insert(
            (*name).to_owned(),
            FieldName::parse(name).expect("standard field name is valid"),
        );
    }
}

fn effective_target(
    procedures: &[Procedure],
    procedure: ProcedureId,
) -> Option<ProcedureImplementationId> {
    let procedure = procedures.get(procedure.index())?;
    procedure.effective_target.or_else(|| {
        procedure
            .inherited_procedure
            .and_then(|parent| effective_target(procedures, parent))
    })
}

fn procedure_id(index: usize) -> ProcedureId {
    ProcedureId(u32::try_from(index).expect("a registry cannot contain more than u32::MAX procs"))
}

fn implementation_id(
    procedure: ProcedureId,
    implementation_index: usize,
) -> ProcedureImplementationId {
    ProcedureImplementationId {
        procedure,
        index: u32::try_from(implementation_index)
            .expect("a procedure cannot contain more than u32::MAX implementations"),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::{Compilation, CompilerDatabase};
    use dm_value::TypePath;
    use dm_vm::{
        ExecutionContext, ExecutionState, Instruction, RuntimeError, Value, execute_module,
        execute_module_in_context,
    };

    use super::{Procedure, ProcedureImplementationKind, ProcedureRegistry};

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct TestProject {
        root: PathBuf,
    }

    impl TestProject {
        fn compile(source: &str) -> Compilation {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "dream64-dm-semantics-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&root).expect("test project directory should be created");
            let project = Self { root };
            fs::write(project.root.join("world.dme"), "#include \"types.dm\"\n")
                .expect("environment should be written");
            fs::write(project.root.join("types.dm"), source).expect("source should be written");
            CompilerDatabase::new()
                .compile(project.root.join("world.dme"))
                .expect("test project should compile")
        }
    }

    impl Drop for TestProject {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.root) {
                eprintln!(
                    "failed to clean test project {}: {error}",
                    self.root.display()
                );
            }
        }
    }

    fn procedure_by_path<'registry>(
        registry: &'registry ProcedureRegistry,
        path: &str,
    ) -> &'registry Procedure {
        registry
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == path)
            .expect("procedure path should exist")
    }

    fn execute_effective(
        compilation: &Compilation,
        path: &str,
        arguments: &[Value],
    ) -> Result<Value, RuntimeError> {
        let registry = ProcedureRegistry::build(compilation);
        let procedure = procedure_by_path(&registry, path);
        let target = procedure
            .effective_target
            .expect("procedure should have an effective implementation");
        let executable = registry
            .compile_vm(compilation)
            .expect("procedure registry should compile to VM bytecode");
        let entry = executable
            .implementation(target)
            .expect("effective implementation should have a VM identity");
        execute_module(executable.module(), entry, arguments)
    }

    #[test]
    fn indexes_a_base_procedure_with_source_identity() {
        let compilation = TestProject::compile("/datum/base\n\tproc/run()\n\t\treturn 1\n");
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/datum/base/proc/run");

        assert!(procedure.owner_type.is_some());
        assert_eq!(procedure.implementations.len(), 1);
        assert_eq!(
            procedure.implementations[0].kind,
            ProcedureImplementationKind::Declaration
        );
        assert_eq!(procedure.implementations[0].definition_index, 1);
        assert!(!procedure.implementations[0].span.is_empty());
        assert_eq!(procedure.implementations[0].parent_target, None);
        assert_eq!(
            procedure.effective_target,
            Some(procedure.implementations[0].id)
        );
    }

    #[test]
    fn links_a_child_override_to_the_inherited_effective_body() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base/child\n\trun()\n\t\treturn ..()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let base = procedure_by_path(&registry, "/datum/base/proc/run");
        let child = procedure_by_path(&registry, "/datum/base/child/proc/run");

        assert_eq!(child.inherited_procedure, Some(base.id));
        assert_eq!(child.implementations.len(), 1);
        assert_eq!(
            child.implementations[0].parent_target,
            base.effective_target
        );
    }

    #[test]
    fn chains_multiple_reopenings_in_expanded_source_order() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base\n\trun()\n\t\treturn 2\n/datum/base\n\trun()\n\t\treturn 3\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/datum/base/proc/run");

        assert_eq!(procedure.implementations.len(), 3);
        assert!(
            procedure
                .implementations
                .windows(2)
                .all(|pair| pair[0].ordinal < pair[1].ordinal)
        );
        assert_eq!(procedure.implementations[0].parent_target, None);
        assert_eq!(
            procedure.implementations[1].parent_target,
            Some(procedure.implementations[0].id)
        );
        assert_eq!(
            procedure.implementations[2].parent_target,
            Some(procedure.implementations[1].id)
        );
        assert_eq!(
            procedure.effective_target,
            Some(procedure.implementations[2].id)
        );
    }

    #[test]
    fn indexes_a_global_procedure_without_an_owner_or_parent() {
        let compilation = TestProject::compile("/proc/global_run()\n\treturn 1\n");
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/proc/global_run");

        assert_eq!(procedure.owner_type, None);
        assert_eq!(procedure.inherited_procedure, None);
        assert_eq!(procedure.implementations[0].parent_target, None);
    }

    #[test]
    fn follows_a_resolved_explicit_parent_type() {
        let compilation = TestProject::compile(
            "/datum/original\n\tproc/run()\n\t\treturn 1\n/datum/alternate\n\tproc/run()\n\t\treturn 2\n/custom\n\tparent_type = /datum/alternate\n\trun()\n\t\treturn ..()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let original = procedure_by_path(&registry, "/datum/original/proc/run");
        let alternate = procedure_by_path(&registry, "/datum/alternate/proc/run");
        let custom = procedure_by_path(&registry, "/custom/proc/run");

        assert_ne!(custom.inherited_procedure, Some(original.id));
        assert_eq!(custom.inherited_procedure, Some(alternate.id));
        assert_eq!(
            custom.implementations[0].parent_target,
            alternate.effective_target
        );
    }

    #[test]
    fn lowers_bare_inherited_fields_as_src_fields_after_local_resolution() {
        let compilation = TestProject::compile(
            "/datum/base\n\tvar/loading_id = 1\n/datum/base/child\n\tproc/run(loading_id)\n\t\tvar/local = loading_id\n\t\tloading_id = local\n\t\treturn src.loading_id\n/datum/base/child\n\tproc/use_field()\n\t\tloading_id += 1\n\t\treturn loading_id\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/datum/base/child/proc/use_field");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                procedure.implementations.iter().map(|body| body.id),
            )
            .expect("bare inherited field should compile");
        let entry = executable
            .implementation(procedure.effective_target.expect("procedure has a body"))
            .expect("implementation should be present");
        let program = executable
            .module()
            .procedure(entry)
            .expect("program should exist");

        assert!(program.instructions.windows(3).any(|instructions| matches!(
            instructions,
            [Instruction::LoadSrc, Instruction::Duplicate, Instruction::LoadField(field)]
                if field.as_str() == "loading_id"
        )));
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::StoreField(field) if field.as_str() == "loading_id"
        )));
    }

    #[test]
    fn lowers_standard_atom_fields_only_for_their_builtin_hierarchy() {
        let compilation = TestProject::compile(
            "/obj/example\n\tproc/read()\n\t\tloc = src\n\t\tpixel_x += 1\n\t\talpha -= 1\n\t\treturn list(dir, color, desc, blend_mode, alpha, appearance_flags, layer, plane, transform, overlays, underlays, vis_contents, x, y, z)\n/datum/example\n\tproc/read()\n\t\treturn alpha\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let object = procedure_by_path(&registry, "/obj/example/proc/read");
        registry
            .compile_vm_implementations(
                &compilation,
                object.implementations.iter().map(|body| body.id),
            )
            .expect("standard atom fields should compile as src fields");

        let datum = procedure_by_path(&registry, "/datum/example/proc/read");
        let error = registry
            .compile_vm_implementations(
                &compilation,
                datum.implementations.iter().map(|body| body.id),
            )
            .expect_err("atom fields must not become datum locals");
        assert!(
            error.message.contains("unknown local \"alpha\""),
            "unexpected diagnostic: {}",
            error.message
        );
    }

    #[test]
    fn lowers_standard_datum_type_field_for_all_datums() {
        let compilation =
            TestProject::compile("/datum/example\n\tproc/read()\n\t\treturn list(type, tag)\n");
        let registry = ProcedureRegistry::build(&compilation);
        let datum = procedure_by_path(&registry, "/datum/example/proc/read");
        registry
            .compile_vm_implementations(
                &compilation,
                datum.implementations.iter().map(|body| body.id),
            )
            .expect("datum type should compile as its built-in src field");
    }

    #[test]
    fn links_standard_location_predicates_as_variadic_builtins() {
        let compilation = TestProject::compile(concat!(
            "/proc/valid_locations()\n",
            "\treturn isarea(new /area, new /area/station) + isobj(new /obj, new /obj/item) + ismob(new /mob, new /mob/living)\n",
            "/proc/invalid_locations()\n",
            "\treturn isarea(new /turf) + isobj(new /mob) + ismob(new /obj)\n",
        ));

        assert_eq!(
            execute_effective(&compilation, "/proc/valid_locations", &[]),
            Ok(Value::number(3.0))
        );
        assert_eq!(
            execute_effective(&compilation, "/proc/invalid_locations", &[]),
            Ok(Value::number(0.0))
        );
    }

    #[test]
    fn links_direction_text_and_orange_standard_builtins() {
        let compilation = TestProject::compile(concat!(
            "/proc/classify()\n",
            "\treturn istext(\"hello\") + istext(3)\n",
            "/atom/example/proc/neighbors(other)\n",
            "\treturn get_dir(src, other) + length(orange(1, src))\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        registry
            .compile_vm(&compilation)
            .expect("standard direction/text/orange builtins should link");
        assert_eq!(
            execute_effective(&compilation, "/proc/classify", &[]),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn selected_dynamic_literal_call_includes_matching_method_implementation() {
        let compilation = TestProject::compile(
            "/datum/receiver\n\tproc/entry()\n\t\treturn call(src, \"register\")()\n\tproc/register()\n\t\treturn 9\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/datum/receiver/proc/entry");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                entry
                    .implementations
                    .iter()
                    .map(|implementation| implementation.id),
            )
            .expect("literal dynamic method should be included");
        let entry = executable
            .implementation(entry.effective_target.expect("entry has a body"))
            .expect("entry should be linked");
        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/receiver").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(receiver), Value::Null),
            ),
            Ok(Value::number(9.0))
        );
    }

    #[test]
    fn selected_static_call_uses_object_tree_ancestor_not_lexical_path_ancestor() {
        let compilation = TestProject::compile(
            "/datum/proc/RegisterSignals()\n\treturn 42\n/area/centcom/proc/Initialize()\n\treturn RegisterSignals()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/area/centcom/proc/Initialize");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                entry
                    .implementations
                    .iter()
                    .map(|implementation| implementation.id),
            )
            .expect("a call inherited from /datum should be linked into the selected module");
        let entry = executable
            .implementation(entry.effective_target.expect("entry has a body"))
            .expect("entry should be linked");
        assert_eq!(
            execute_module(executable.module(), entry, &[]),
            Ok(Value::number(42.0))
        );
    }

    #[test]
    fn selected_method_includes_and_resolves_direct_helper_calls() {
        let compilation = TestProject::compile(
            "/datum/receiver\n\tproc/entry()\n\t\treturn helper()\n\tproc/helper()\n\t\treturn 9\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/datum/receiver/proc/entry");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                entry
                    .implementations
                    .iter()
                    .map(|implementation| implementation.id),
            )
            .expect("direct helper method should be included");
        let entry = executable
            .implementation(entry.effective_target.expect("entry has a body"))
            .expect("entry should be linked");
        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/receiver").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(receiver), Value::Null),
            ),
            Ok(Value::number(9.0))
        );
    }

    #[test]
    fn executes_an_inherited_override_and_reuses_omitted_arguments() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run(value = 2)\n\t\treturn value + 1\n/datum/base/child\n\trun(value = 2)\n\t\treturn ..() + 10\n",
        );

        assert_eq!(
            execute_effective(&compilation, "/datum/base/child/proc/run", &[]),
            Ok(Value::number(13.0))
        );
        assert_eq!(
            execute_effective(&compilation, "/datum/base/child/proc/run", &[Value::Null]),
            Ok(Value::number(11.0)),
            "explicit null is reused and BYOND arithmetic treats it as numeric zero",
        );
    }

    #[test]
    fn executes_multiple_reopenings_through_the_exact_parent_chain() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base\n\trun()\n\t\treturn ..() + 1\n/datum/base\n\trun()\n\t\treturn ..() + 1\n",
        );

        assert_eq!(
            execute_effective(&compilation, "/datum/base/proc/run", &[]),
            Ok(Value::number(3.0))
        );
    }

    #[test]
    fn executes_parent_target_from_explicit_parent_type_and_explicit_arguments() {
        let compilation = TestProject::compile(
            "/datum/alternate\n\tproc/run(value = 1)\n\t\treturn value * 2\n/custom\n\tparent_type = /datum/alternate\n\trun(value = 3)\n\t\treturn ..(value + 1)\n",
        );

        assert_eq!(
            execute_effective(&compilation, "/custom/proc/run", &[]),
            Ok(Value::number(8.0))
        );
    }

    #[test]
    fn missing_parent_target_is_a_source_mapped_runtime_error() {
        let compilation = TestProject::compile("/proc/orphan()\n\treturn ..()\n");
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/proc/orphan");
        let implementation = procedure.implementations[0];
        assert_eq!(implementation.parent_target, None);
        let expected_span = compilation
            .syntax(implementation.file_id)
            .expect("source syntax should exist")
            .definitions[implementation.definition_index]
            .body[0]
            .span;
        let error = execute_effective(&compilation, "/proc/orphan", &[])
            .expect_err("orphan parent call should fail at runtime");

        assert_eq!(
            error.message,
            "parent procedure call has no resolved target"
        );
        assert_eq!(error.source_span, Some(expected_span));
        assert_eq!(error.call_stack.len(), 1);
    }

    #[test]
    fn parent_failure_preserves_both_source_mapped_frames() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn \"text\" + 1\n/datum/base/child\n\trun()\n\t\treturn ..()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let base = procedure_by_path(&registry, "/datum/base/proc/run");
        let base_implementation = base.implementations[0];
        let expected_span = compilation
            .syntax(base_implementation.file_id)
            .expect("source syntax should exist")
            .definitions[base_implementation.definition_index]
            .body[0]
            .span;
        let error = execute_effective(&compilation, "/datum/base/child/proc/run", &[])
            .expect_err("parent numeric failure should propagate");

        assert_eq!(error.source_span, Some(expected_span));
        assert_eq!(error.call_stack.len(), 2);
        assert!(
            error.call_stack[0]
                .procedure
                .contains("/datum/base/child/proc/run")
        );
        assert!(
            error.call_stack[1]
                .procedure
                .contains("/datum/base/proc/run")
        );
        assert_eq!(error.call_stack[1].source_span, Some(expected_span));
    }
}

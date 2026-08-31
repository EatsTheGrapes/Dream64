//! The project-wide [`ProcedureRegistry`]: canonical procedure indexing
//! (`build` / `build_lazy` / `build_with_stable_ids`), the lazily initialized
//! per-body dependency indexes, the dependency-closure walks, and the
//! `compile_vm*` family that lowers selected implementations into an
//! [`ExecutableProcedures`] artifact.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use dm_compiler::Compilation;
use dm_globals::VariableRegistry;
use dm_object_tree::{CodePath, NodeId, NodeKind};
use dm_syntax::DefinitionKind;
use dm_value::{FieldName, TypePath};

use super::{
    ConstBindings, ExecutableProcedureStats, ExecutableProcedures, NATIVE_PARENT_BUILTINS,
    Procedure, ProcedureClosureStats, ProcedureId, ProcedureImplementation,
    ProcedureImplementationId, ProcedureImplementationKind, ProcedureRegistryBuildStats,
    STANDARD_BUILTINS, compiler_type_predicate, construction_dependencies,
    constructor_targets_by_ancestor, declared_field_types, declared_global_fields,
    declared_global_types, declared_receiver_types, direct_instance_field_types,
    direct_instance_fields, direct_static_fields, dynamic_call_literal_selectors, effective_target,
    expand_proc_pseudo_macro, implementation_id, member_call_dependencies, native_member_index,
    native_parent_index, normalize_upward_paths, procedure_id, referenced_identifiers,
    referenced_inherited_field_types, referenced_inherited_fields, static_call_selectors,
    static_proc_reference_paths, static_procedure_type_families, type_is_descendant_or_same,
    validate_const_assignments,
};

/// Project-wide registry of canonical procedures and override chains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcedureRegistry {
    procedures: Vec<Procedure>,
    by_node: BTreeMap<NodeId, ProcedureId>,
    by_path: BTreeMap<CodePath, ProcedureId>,
    by_owner_name: BTreeMap<(Option<NodeId>, String), ProcedureId>,
    dynamic_targets: BTreeMap<String, Vec<ProcedureImplementationId>>,
    dependencies: OnceLock<ProcedureDependencies>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ProcedureDependencies {
    static_selectors: BTreeMap<ProcedureImplementationId, BTreeSet<String>>,
    dynamic_selectors: BTreeMap<ProcedureImplementationId, BTreeSet<String>>,
    exact_member_targets: BTreeMap<ProcedureImplementationId, BTreeSet<ProcedureImplementationId>>,
    typed_virtual_targets: BTreeMap<ProcedureImplementationId, BTreeSet<ProcedureImplementationId>>,
    build_stats: ProcedureRegistryBuildStats,
}

impl ProcedureRegistry {
    /// Computes a procedure-only semantic digest for persistent executable reuse.
    #[must_use]
    pub fn persistent_semantic_digest(&self, compilation: &Compilation) -> [u8; 32] {
        let mut first = md5::Context::new();
        first.consume(b"dream64-procedure-semantics-v1");
        for procedure in &self.procedures {
            first.consume(procedure.path.to_string().as_bytes());
            for implementation in &procedure.implementations {
                first.consume((implementation.id.index() as u64).to_le_bytes());
                if let Some(definition) = compilation
                    .syntax(implementation.file_id)
                    .and_then(|syntax| syntax.definitions.get(implementation.definition_index))
                {
                    first.consume(format!("{definition:?}").as_bytes());
                }
            }
        }
        let first = first.compute().0;
        let second = md5::compute(first).0;
        let mut digest = [0; 32];
        digest[..16].copy_from_slice(&first);
        digest[16..].copy_from_slice(&second);
        digest
    }

    /// Builds stable procedure identities and dispatch inventory without
    /// analyzing individual bodies. Dependency closure and linking methods
    /// initialize the exact eager dependency indexes on first use.
    #[must_use]
    pub fn build_lazy(compilation: &Compilation) -> Self {
        let tree = compilation.code_tree();
        let procedure_nodes: Vec<_> = tree
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind, NodeKind::Procedure | NodeKind::Verb))
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
                            DefinitionKind::Procedure | DefinitionKind::Verb => {
                                ProcedureImplementationKind::Declaration
                            }
                            DefinitionKind::ProcedureOverride => {
                                ProcedureImplementationKind::Override
                            }
                            _ => return None,
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
                procedures[procedure_index].implementations[implementation_index].parent_target =
                    if implementation_index == 0 {
                        inherited_target
                    } else {
                        Some(implementation_id(
                            procedures[procedure_index].id,
                            implementation_index - 1,
                        ))
                    };
            }
        }
        let by_owner_name = procedures
            .iter()
            .filter_map(|procedure| {
                procedure
                    .path
                    .segments()
                    .last()
                    .map(|name| ((procedure.owner_type, name.clone()), procedure.id))
            })
            .collect();
        let mut dynamic_targets = BTreeMap::<String, Vec<_>>::new();
        for procedure in &procedures {
            if let (Some(name), Some(target)) =
                (procedure.path.segments().last(), procedure.effective_target)
            {
                dynamic_targets
                    .entry(name.clone())
                    .or_default()
                    .push(target);
            }
        }
        Self {
            procedures,
            by_node,
            by_path,
            by_owner_name,
            dynamic_targets,
            dependencies: OnceLock::new(),
        }
    }

    /// Builds a registry from the compiler's accepted canonical declarations.
    #[must_use]
    pub fn build(compilation: &Compilation) -> Self {
        Self::build_with_stable_ids(compilation, &BTreeMap::new())
            .expect("object-tree procedure order is always representable")
    }

    /// Builds a registry in authoritative persistent linker-ID order.
    ///
    /// Persistent IDs may be sparse 64-bit identities. The semantic and VM
    /// layers use bounded dense `u32` indices, so their ordering is the stable
    /// ID ordering rather than the current lexical object-tree ordering.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, or out-of-range stable IDs.
    pub fn build_with_stable_ids(
        compilation: &Compilation,
        stable_ids: &BTreeMap<String, u64>,
    ) -> Result<Self, String> {
        let profile = std::env::var_os("DREAM64_PROFILE_REGISTRY").is_some();
        let build_started = Instant::now();
        let tree = compilation.code_tree();
        let mut procedure_nodes: Vec<_> = tree
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind, NodeKind::Procedure | NodeKind::Verb))
            .collect();
        if !stable_ids.is_empty() {
            let mut assigned = BTreeSet::new();
            for node in &procedure_nodes {
                let path = node.path.to_string();
                let id = stable_ids
                    .get(&path)
                    .copied()
                    .ok_or_else(|| format!("persistent procedure ID is missing for {path}"))?;
                u32::try_from(id)
                    .map_err(|_| format!("persistent procedure ID for {path} exceeds u32"))?;
                if !assigned.insert(id) {
                    return Err(format!("duplicate persistent procedure ID {id}"));
                }
            }
            if stable_ids.len() != procedure_nodes.len() {
                return Err("persistent procedure IDs contain unknown paths".to_owned());
            }
            procedure_nodes.sort_by_key(|node| stable_ids[&node.path.to_string()]);
        }
        let by_node: BTreeMap<_, _> = procedure_nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id, procedure_id(index)))
            .collect();
        let by_path = procedure_nodes
            .iter()
            .map(|node| (node.path.clone(), by_node[&node.id]))
            .collect();
        let by_text_path = procedure_nodes
            .iter()
            .map(|node| (node.path.to_string(), by_node[&node.id]))
            .collect::<BTreeMap<_, _>>();

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
                            DefinitionKind::Procedure | DefinitionKind::Verb => {
                                ProcedureImplementationKind::Declaration
                            }
                            DefinitionKind::ProcedureOverride => {
                                ProcedureImplementationKind::Override
                            }
                            DefinitionKind::Type
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

        let by_owner_name = procedures
            .iter()
            .filter_map(|procedure| {
                procedure
                    .path
                    .segments()
                    .last()
                    .map(|name| ((procedure.owner_type, name.clone()), procedure.id))
            })
            .collect();
        let mut dynamic_targets = BTreeMap::<String, Vec<_>>::new();
        for procedure in &procedures {
            if let (Some(name), Some(target)) =
                (procedure.path.segments().last(), procedure.effective_target)
            {
                dynamic_targets
                    .entry(name.clone())
                    .or_default()
                    .push(target);
            }
        }
        let mut static_selectors = BTreeMap::new();
        let mut dynamic_selectors = BTreeMap::new();
        let mut exact_member_targets = BTreeMap::new();
        let mut typed_virtual_targets = BTreeMap::new();
        let base_index_elapsed = build_started.elapsed();
        let phase = Instant::now();
        let global_types = declared_global_types(compilation);
        let global_types_elapsed = phase.elapsed();
        let phase = Instant::now();
        let field_types = declared_field_types(compilation);
        let field_types_elapsed = phase.elapsed();
        let phase = Instant::now();
        let new_targets_by_ancestor =
            constructor_targets_by_ancestor(compilation, &procedures, &by_owner_name);
        let constructors_index_elapsed = phase.elapsed();
        // Index exact `/owner/proc` families once. Re-scanning a production
        // procedure table for every `typesof(/owner/proc)` made warm registry
        // construction quadratic. Target vectors retain canonical order.
        let mut procedure_family_targets = BTreeMap::<Vec<String>, Vec<_>>::new();
        for candidate in &procedures {
            let segments = candidate.path.segments();
            if segments.len() >= 2
                && segments[segments.len() - 2] == "proc"
                && let Some(target) = candidate.effective_target
            {
                procedure_family_targets
                    .entry(segments[..segments.len() - 1].to_vec())
                    .or_default()
                    .push(target);
            }
        }
        let mut build_stats = ProcedureRegistryBuildStats::default();
        let dependency_started = Instant::now();
        let mut dependency_bodies = 0_usize;
        let mut static_selector_elapsed = Duration::ZERO;
        let mut member_dependency_elapsed = Duration::ZERO;
        let mut static_reference_elapsed = Duration::ZERO;
        let mut family_elapsed = Duration::ZERO;
        let mut dynamic_selector_elapsed = Duration::ZERO;
        let mut construction_elapsed = Duration::ZERO;
        for procedure in &procedures {
            for implementation in &procedure.implementations {
                if let Some(definition) = compilation
                    .syntax(implementation.file_id)
                    .and_then(|syntax| syntax.definitions.get(implementation.definition_index))
                {
                    dependency_bodies += 1;
                    let phase = profile.then(Instant::now);
                    let selectors = static_call_selectors(definition);
                    if let Some(phase) = phase {
                        static_selector_elapsed += phase.elapsed();
                    }
                    static_selectors.insert(implementation.id, selectors);
                    let phase = profile.then(Instant::now);
                    let (mut member_targets, virtual_targets, unresolved_members) =
                        member_call_dependencies(
                            definition,
                            procedure.owner_type,
                            compilation,
                            &global_types,
                            &field_types,
                            &by_owner_name,
                            &dynamic_targets,
                            &procedures,
                        );
                    if let Some(phase) = phase {
                        member_dependency_elapsed += phase.elapsed();
                    }
                    let phase = profile.then(Instant::now);
                    for referenced_path in static_proc_reference_paths(definition, &procedure.path)
                    {
                        build_stats.static_proc_reference_index_lookups += 1;
                        if let Some(target) = by_text_path
                            .get(&referenced_path)
                            .and_then(|procedure| procedures[procedure.index()].effective_target)
                        {
                            member_targets.insert(target);
                        }
                    }
                    if let Some(phase) = phase {
                        static_reference_elapsed += phase.elapsed();
                    }
                    // `typesof(/owner/proc)` materializes procedure paths which
                    // are commonly fed straight into `call(src, path)()`.  The
                    // selector is intentionally runtime data, but the family is
                    // statically bounded: every effective procedure below that
                    // procedure-type path must be present in the symbolic
                    // module.  In particular, tgstation's generated
                    // `InitGlobal*` procedures are discovered this way.
                    let phase = profile.then(Instant::now);
                    for family in static_procedure_type_families(definition) {
                        if let Some(targets) = procedure_family_targets.get(&family) {
                            member_targets.extend(targets.iter().copied());
                        }
                    }
                    if let Some(phase) = phase {
                        family_elapsed += phase.elapsed();
                    }
                    let phase = profile.then(Instant::now);
                    let mut dynamic = dynamic_call_literal_selectors(definition);
                    if let Some(phase) = phase {
                        dynamic_selector_elapsed += phase.elapsed();
                    }
                    let phase = profile.then(Instant::now);
                    let construction = construction_dependencies(
                        definition,
                        compilation,
                        &procedures,
                        &by_owner_name,
                        &new_targets_by_ancestor,
                    );
                    if let Some(phase) = phase {
                        construction_elapsed += phase.elapsed();
                    }
                    member_targets.extend(construction.targets);
                    if construction.unbounded {
                        dynamic.insert("New".to_owned());
                    }
                    dynamic.extend(unresolved_members);
                    dynamic_selectors.insert(implementation.id, dynamic);
                    exact_member_targets.insert(implementation.id, member_targets);
                    typed_virtual_targets.insert(implementation.id, virtual_targets);
                }
            }
        }
        if profile {
            eprintln!(
                "procedure-registry-profile: procedures={} bodies={} total_ms={} base_index_ms={} global_types_ms={} field_types_ms={} constructor_index_ms={} dependencies_ms={} static_selectors_ms={} member_dependencies_ms={} static_references_ms={} procedure_families_ms={} dynamic_selectors_ms={} construction_dependencies_ms={}",
                procedures.len(),
                dependency_bodies,
                build_started.elapsed().as_millis(),
                base_index_elapsed.as_millis(),
                global_types_elapsed.as_millis(),
                field_types_elapsed.as_millis(),
                constructors_index_elapsed.as_millis(),
                dependency_started.elapsed().as_millis(),
                static_selector_elapsed.as_millis(),
                member_dependency_elapsed.as_millis(),
                static_reference_elapsed.as_millis(),
                family_elapsed.as_millis(),
                dynamic_selector_elapsed.as_millis(),
                construction_elapsed.as_millis(),
            );
        }
        Ok(Self {
            procedures,
            by_node,
            by_path,
            by_owner_name,
            dynamic_targets,
            dependencies: OnceLock::from(ProcedureDependencies {
                static_selectors,
                dynamic_selectors,
                exact_member_targets,
                typed_virtual_targets,
                build_stats,
            }),
        })
    }

    /// Returns canonical procedures in object-tree node order.
    #[must_use]
    pub fn procedures(&self) -> &[Procedure] {
        &self.procedures
    }

    /// Returns deterministic procedure-index construction counters.
    #[must_use]
    pub fn build_stats(&self) -> &ProcedureRegistryBuildStats {
        static EMPTY: ProcedureRegistryBuildStats = ProcedureRegistryBuildStats {
            static_proc_reference_index_lookups: 0,
        };
        self.dependencies
            .get()
            .map_or(&EMPTY, |dependencies| &dependencies.build_stats)
    }

    fn dependencies(&self, compilation: &Compilation) -> &ProcedureDependencies {
        self.dependencies.get_or_init(|| {
            ProcedureRegistry::build(compilation)
                .dependencies
                .into_inner()
                .expect("eager registry initializes procedure dependencies")
        })
    }

    /// Whether per-body dependency analysis has run for this registry.
    #[doc(hidden)]
    #[must_use]
    pub fn dependencies_initialized(&self) -> bool {
        self.dependencies.get().is_some()
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
        let selected = self.implementation_closure(compilation, implementations);
        self.compile_vm_selected(compilation, selected)
    }

    /// Links the complete dynamic dispatch symbol graph while eagerly
    /// compiling only statically proven, typed-member, and parent targets.
    /// Genuinely untyped receiver candidates keep stable VM identities and
    /// materialize on their first runtime dispatch.
    pub fn compile_vm_implementations_symbolic_dynamic(
        &self,
        compilation: &Compilation,
        implementations: impl IntoIterator<Item = ProcedureImplementationId>,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let roots = implementations.into_iter().collect::<BTreeSet<_>>();
        let selected = self.implementation_closure(compilation, roots.iter().copied());
        let eager = self.eager_implementation_closure(compilation, roots);
        self.compile_vm_selected_deferred(compilation, selected, &eager)
    }

    /// Links every registered implementation while deferring all procedure
    /// bodies until their first runtime dispatch.
    ///
    /// This is useful for bootstrap expression modules whose entry points are
    /// appended after linking. Those expressions may name any project
    /// procedure, so the complete symbol graph is required, but eagerly
    /// lowering every unrelated body would make image construction scale with
    /// the entire project rather than the procedures actually reached during
    /// initialization.
    ///
    /// # Errors
    ///
    /// Returns [`dm_vm::CompileError`] when retained procedure metadata cannot
    /// be represented by the VM. A body-specific lowering error remains
    /// attached to its deferred symbol and is reported if that body is called.
    pub fn compile_vm_all_symbolic_deferred(
        &self,
        compilation: &Compilation,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let selected = self.procedures.iter().flat_map(|procedure| {
            procedure
                .implementations
                .iter()
                .map(|implementation| implementation.id)
        });
        self.compile_vm_selected_deferred(compilation, selected, &BTreeSet::new())
    }

    /// Links the complete project procedure table while eagerly lowering only
    /// the statically proven closure of the supplied startup roots.
    ///
    /// A BYOND/OpenDream runtime can dynamically invoke any project method
    /// whose selector and receiver are produced at runtime. Keeping every
    /// identity deferred preserves that contract without paying to lower
    /// unrelated UI/admin/gameplay bodies during headless startup.
    ///
    /// # Errors
    ///
    /// Returns an error when symbol metadata or an eagerly required body
    /// cannot be represented. Deferred body errors remain attached until the
    /// corresponding procedure is actually reached.
    pub fn compile_vm_all_symbolic_with_eager_roots(
        &self,
        compilation: &Compilation,
        roots: impl IntoIterator<Item = ProcedureImplementationId>,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let roots = roots.into_iter().collect::<BTreeSet<_>>();
        let eager = self.eager_implementation_closure(compilation, roots);
        let selected = self.procedures.iter().flat_map(|procedure| {
            procedure
                .implementations
                .iter()
                .map(|implementation| implementation.id)
        });
        self.compile_vm_selected_deferred(compilation, selected, &eager)
    }

    /// Links the conservative procedure frontier named by bootstrap
    /// initializer expressions while deferring every selected project body.
    ///
    /// Each selector retains every implementation with that leaf procedure
    /// name. The ordinary semantic closure then adds exact parents, static
    /// callees, typed virtual targets, and dynamic candidates reachable from
    /// those roots. This preserves runtime dispatch without preprocessing
    /// unrelated procedures merely because they exist in the project.
    ///
    /// # Errors
    ///
    /// Returns [`dm_vm::CompileError`] when retained symbol metadata cannot be
    /// represented by the VM. Body-specific failures remain deferred.
    pub fn compile_vm_initializer_frontier_symbolic_deferred<'name>(
        &self,
        compilation: &Compilation,
        selectors: impl IntoIterator<Item = &'name str>,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let roots = selectors
            .into_iter()
            .flat_map(|selector| {
                self.dynamic_targets
                    .get(
                        selector
                            .trim_matches('/')
                            .rsplit('/')
                            .next()
                            .unwrap_or(selector),
                    )
                    .into_iter()
                    .flatten()
                    .copied()
            })
            .collect::<BTreeSet<_>>();
        let selected = self.implementation_closure(compilation, roots);
        self.compile_vm_selected_deferred(compilation, selected, &BTreeSet::new())
    }

    /// Links an initializer frontier with construction targets narrowed to
    /// statically known datum types. Genuinely dynamic construction can opt in
    /// to every `New` implementation without broadening other selectors.
    ///
    /// # Errors
    ///
    /// Returns [`dm_vm::CompileError`] under the same conditions as
    /// [`Self::compile_vm_initializer_frontier_symbolic_deferred`].
    pub fn compile_vm_initializer_typed_frontier_symbolic_deferred<'name, 'type_path>(
        &self,
        compilation: &Compilation,
        selectors: impl IntoIterator<Item = &'name str>,
        constructed_types: impl IntoIterator<Item = &'type_path TypePath>,
        include_dynamic_constructors: bool,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let mut roots = selectors
            .into_iter()
            .flat_map(|selector| {
                self.dynamic_targets
                    .get(
                        selector
                            .trim_matches('/')
                            .rsplit('/')
                            .next()
                            .unwrap_or(selector),
                    )
                    .into_iter()
                    .flatten()
                    .copied()
            })
            .collect::<BTreeSet<_>>();
        if include_dynamic_constructors && let Some(constructors) = self.dynamic_targets.get("New")
        {
            roots.extend(constructors.iter().copied());
        }
        for type_path in constructed_types {
            if let Some(constructor) = self.effective_type_member(compilation, type_path, "New") {
                roots.insert(constructor);
            }
        }
        let selected = self.implementation_closure(compilation, roots);
        self.compile_vm_selected_deferred(compilation, selected, &BTreeSet::new())
    }

    fn effective_type_member(
        &self,
        compilation: &Compilation,
        type_path: &TypePath,
        selector: &str,
    ) -> Option<ProcedureImplementationId> {
        let path = dm_syntax::DefinitionPath::new(
            type_path
                .as_str()
                .trim_matches('/')
                .split('/')
                .map(str::to_owned)
                .collect(),
        );
        let mut current = compilation.code_tree().find(&path);
        while let Some(owner) = current {
            if let Some(procedure) = self
                .by_owner_name
                .get(&(Some(owner), selector.to_owned()))
                .and_then(|procedure| self.procedure(*procedure))
                .and_then(|procedure| procedure.effective_target)
            {
                return Some(procedure);
            }
            current = compilation
                .code_tree()
                .node(owner)
                .and_then(|node| node.parent_type);
        }
        None
    }

    /// Resolves the bodies that a symbolic module must compile eagerly:
    /// roots, exact parents, statically resolved calls, and typed member
    /// targets. Genuinely untyped dynamic candidates are intentionally absent.
    #[must_use]
    pub fn eager_implementation_closure(
        &self,
        compilation: &Compilation,
        implementations: impl IntoIterator<Item = ProcedureImplementationId>,
    ) -> BTreeSet<ProcedureImplementationId> {
        let dependencies = self.dependencies(compilation);
        let mut selected = implementations.into_iter().collect::<BTreeSet<_>>();
        let mut pending = selected.iter().copied().collect::<Vec<_>>();
        while let Some(implementation) = pending.pop() {
            let parent = self
                .implementation(implementation)
                .and_then(|body| body.parent_target);
            let exact = dependencies
                .exact_member_targets
                .get(&implementation)
                .into_iter()
                .flatten()
                .copied();
            let static_targets = dependencies
                .static_selectors
                .get(&implementation)
                .into_iter()
                .flatten()
                .filter_map(|selector| {
                    self.static_call_target(implementation, selector, compilation)
                });
            for target in parent.into_iter().chain(exact).chain(static_targets) {
                if selected.insert(target) {
                    pending.push(target);
                }
            }
        }
        selected
    }

    /// Resolves the complete static, dynamic-literal, and parent-call closure
    /// for a set of procedure implementations without compiling it.
    #[must_use]
    pub fn implementation_closure(
        &self,
        compilation: &Compilation,
        implementations: impl IntoIterator<Item = ProcedureImplementationId>,
    ) -> BTreeSet<ProcedureImplementationId> {
        self.implementation_closure_with_stats(compilation, implementations)
            .0
    }

    /// Resolves a dependency closure and returns indexed-resolution counters.
    #[must_use]
    pub fn implementation_closure_with_stats(
        &self,
        compilation: &Compilation,
        implementations: impl IntoIterator<Item = ProcedureImplementationId>,
    ) -> (BTreeSet<ProcedureImplementationId>, ProcedureClosureStats) {
        let dependencies = self.dependencies(compilation);
        let mut stats = ProcedureClosureStats::default();
        let mut selected: BTreeSet<_> = implementations.into_iter().collect();
        let mut pending: Vec<_> = selected.iter().copied().collect();
        let mut resolved_dynamic_selectors = BTreeSet::new();
        while let Some(implementation) = pending.pop() {
            stats.bodies_visited += 1;
            if let Some(parent) = self
                .implementation(implementation)
                .and_then(|body| body.parent_target)
                && selected.insert(parent)
            {
                pending.push(parent);
            }

            for &target in dependencies
                .exact_member_targets
                .get(&implementation)
                .into_iter()
                .flatten()
            {
                if selected.insert(target) {
                    pending.push(target);
                }
            }

            for &target in dependencies
                .typed_virtual_targets
                .get(&implementation)
                .into_iter()
                .flatten()
            {
                if selected.insert(target) {
                    pending.push(target);
                }
            }

            for selector in dependencies
                .dynamic_selectors
                .get(&implementation)
                .into_iter()
                .flatten()
            {
                let selector = selector.trim_matches('/');
                let selector = selector.rsplit('/').next().unwrap_or(selector);
                if !resolved_dynamic_selectors.insert(selector.to_owned()) {
                    continue;
                }
                stats.dynamic_selectors_resolved += 1;
                if let Some(targets) = self.dynamic_targets.get(selector) {
                    stats.dynamic_candidates_considered += targets.len();
                    for &target in targets {
                        if selected.insert(target) {
                            pending.push(target);
                        }
                    }
                }
            }
            for selector in dependencies
                .static_selectors
                .get(&implementation)
                .into_iter()
                .flatten()
            {
                stats.static_selectors_resolved += 1;
                let resolved = self.static_call_target(implementation, selector, compilation);
                if let Some(target) = resolved {
                    if selected.insert(target) {
                        pending.push(target);
                    }
                }
                // A bare member call is statically name-resolved but remains
                // virtual on src at runtime. Retain compatible descendant
                // overrides as deferred symbols so an inherited body running
                // on a concrete subtype can dispatch to that subtype's method.
                let owner = self
                    .procedure(implementation.procedure())
                    .and_then(|procedure| procedure.owner_type);
                let resolved_owner = resolved
                    .and_then(|target| self.procedure(target.procedure()))
                    .and_then(|procedure| procedure.owner_type);
                if let (Some(owner), Some(resolved_owner)) = (owner, resolved_owner)
                    && type_is_descendant_or_same(compilation, owner, resolved_owner)
                    && let Some(candidates) = self.dynamic_targets.get(selector)
                {
                    for &candidate in candidates {
                        let candidate_owner = self
                            .procedure(candidate.procedure())
                            .and_then(|procedure| procedure.owner_type);
                        if candidate_owner.is_some_and(|candidate_owner| {
                            type_is_descendant_or_same(compilation, candidate_owner, owner)
                        }) && selected.insert(candidate)
                        {
                            pending.push(candidate);
                        }
                    }
                }
            }
        }
        (selected, stats)
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
        let static_registry = VariableRegistry::build(compilation);
        let direct_fields = direct_instance_fields(&static_registry);
        let direct_field_types = direct_instance_field_types(compilation, &static_registry);
        let direct_static_fields = direct_static_fields(&static_registry);
        let global_fields = declared_global_fields(compilation);
        let global_types = declared_global_types(compilation);
        let const_bindings = ConstBindings::build(compilation);
        let mut inherited_field_cache = BTreeMap::new();
        let mut inherited_static_field_cache = BTreeMap::new();
        implementations
            .into_iter()
            .map(|implementation| {
                (
                    implementation,
                    self.compile_vm_selected_with_fields(
                        compilation,
                        [implementation],
                        &direct_fields,
                        &direct_field_types,
                        &mut inherited_field_cache,
                        &direct_static_fields,
                        &mut inherited_static_field_cache,
                        &global_fields,
                        &global_types,
                        &const_bindings,
                        false,
                        None,
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
        let static_registry = VariableRegistry::build(compilation);
        let direct_fields = direct_instance_fields(&static_registry);
        let direct_field_types = direct_instance_field_types(compilation, &static_registry);
        let direct_static_fields = direct_static_fields(&static_registry);
        let global_fields = declared_global_fields(compilation);
        let global_types = declared_global_types(compilation);
        let const_bindings = ConstBindings::build(compilation);
        let mut inherited_field_cache = BTreeMap::new();
        let mut inherited_static_field_cache = BTreeMap::new();
        self.compile_vm_selected_with_fields(
            compilation,
            selected,
            &direct_fields,
            &direct_field_types,
            &mut inherited_field_cache,
            &direct_static_fields,
            &mut inherited_static_field_cache,
            &global_fields,
            &global_types,
            &const_bindings,
            true,
            None,
        )
    }

    fn compile_vm_selected_deferred(
        &self,
        compilation: &Compilation,
        selected: impl IntoIterator<Item = ProcedureImplementationId>,
        eager: &BTreeSet<ProcedureImplementationId>,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let static_registry = VariableRegistry::build(compilation);
        let direct_fields = direct_instance_fields(&static_registry);
        let direct_field_types = direct_instance_field_types(compilation, &static_registry);
        let direct_static_fields = direct_static_fields(&static_registry);
        let global_fields = declared_global_fields(compilation);
        let global_types = declared_global_types(compilation);
        let const_bindings = ConstBindings::build(compilation);
        let mut inherited_field_cache = BTreeMap::new();
        let mut inherited_static_field_cache = BTreeMap::new();
        self.compile_vm_selected_with_fields(
            compilation,
            selected,
            &direct_fields,
            &direct_field_types,
            &mut inherited_field_cache,
            &direct_static_fields,
            &mut inherited_static_field_cache,
            &global_fields,
            &global_types,
            &const_bindings,
            true,
            Some(eager),
        )
    }

    #[allow(clippy::too_many_lines)]
    fn compile_vm_selected_with_fields(
        &self,
        compilation: &Compilation,
        selected: impl IntoIterator<Item = ProcedureImplementationId>,
        direct_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
        direct_field_types: &BTreeMap<NodeId, BTreeMap<String, TypePath>>,
        _inherited_field_cache: &mut BTreeMap<NodeId, BTreeMap<String, FieldName>>,
        direct_static_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
        _inherited_static_field_cache: &mut BTreeMap<NodeId, BTreeMap<String, FieldName>>,
        global_fields: &BTreeMap<String, FieldName>,
        global_types: &BTreeMap<String, TypePath>,
        const_bindings: &ConstBindings,
        include_parent_targets: bool,
        eager_implementations: Option<&BTreeSet<ProcedureImplementationId>>,
    ) -> Result<ExecutableProcedures, dm_vm::CompileError> {
        let dependencies = self.dependencies(compilation);
        let selected: BTreeSet<_> = selected.into_iter().collect();
        let mut ordered = Vec::with_capacity(selected.len());
        let mut deferred_validation_errors =
            BTreeMap::<ProcedureImplementationId, dm_vm::CompileError>::new();
        for implementation_id in selected {
            let procedure = self
                .procedure(implementation_id.procedure())
                .ok_or_else(|| dm_vm::CompileError {
                    message: format!(
                        "missing procedure for selected implementation {:?}",
                        implementation_id
                    ),
                })?;
            let implementation = procedure
                .implementations
                .get(implementation_id.index())
                .ok_or_else(|| dm_vm::CompileError {
                    message: format!(
                        "missing selected implementation of {} at index {}",
                        procedure.path,
                        implementation_id.index()
                    ),
                })?;
            let definition = compilation
                .syntax(implementation.file_id)
                .and_then(|syntax| syntax.definitions.get(implementation.definition_index))
                .ok_or_else(|| dm_vm::CompileError {
                    message: format!(
                        "missing syntax definition for implementation of {}",
                        procedure.path
                    ),
                })?;
            let validation = validate_const_assignments(
                definition,
                procedure.owner_type,
                compilation,
                const_bindings,
                self,
                implementation.id,
            );
            if eager_implementations.is_none_or(|eager| eager.contains(&implementation.id)) {
                validation?;
            } else if let Err(error) = validation {
                deferred_validation_errors.insert(implementation.id, error);
            }
            ordered.push((procedure, implementation, definition));
        }
        let indices: BTreeMap<_, _> = ordered
            .iter()
            .enumerate()
            .map(|(index, (_, implementation, _))| (implementation.id, index))
            .collect();
        let normalized_definitions: Vec<_> = ordered
            .iter()
            .map(|(procedure, _, definition)| {
                let mut normalized = normalize_upward_paths(
                    compilation,
                    procedure.owner_type,
                    definition,
                    &global_types,
                );
                expand_proc_pseudo_macro(&mut normalized, &procedure.path);
                normalized
            })
            .collect();
        let builtin_syntax =
            dm_syntax::parse(STANDARD_BUILTINS).map_err(|error| dm_vm::CompileError {
                message: format!(
                    "failed to parse Dream64 standard location builtins: {}",
                    error
                ),
            })?;
        let native_parent_syntax =
            dm_syntax::parse(NATIVE_PARENT_BUILTINS).map_err(|error| dm_vm::CompileError {
                message: format!("failed to parse Dream64 native parent builtins: {error}"),
            })?;
        let mut builtin_names = Vec::with_capacity(builtin_syntax.definitions.len());
        for definition in &builtin_syntax.definitions {
            let name = definition
                .path
                .segments()
                .last()
                .ok_or_else(|| dm_vm::CompileError {
                    message: "Dream64 standard location builtin with invalid path".to_owned(),
                })?;
            builtin_names.push(name.to_owned());
        }
        let builtin_indices: BTreeMap<_, _> = builtin_names
            .iter()
            .enumerate()
            .map(|(offset, name)| (name.clone(), ordered.len() + offset))
            .collect();
        let native_parent_indices = native_parent_syntax
            .definitions
            .iter()
            .enumerate()
            .map(|(offset, definition)| {
                (
                    definition.path.to_string(),
                    ordered.len() + builtin_syntax.definitions.len() + offset,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let global_binding_index_lookups = Cell::new(0usize);
        let typed_global_index_lookups = Cell::new(0usize);
        let inherited_field_name_lookups = Cell::new(0usize);
        let mut specs: Vec<_> = ordered
            .iter()
            .enumerate()
            .map(|(ordered_index, (procedure, implementation, _))| {
                let definition = &normalized_definitions[ordered_index];
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
                        .or_else(|| {
                            native_parent_index(
                                &procedure.path,
                                procedure.owner_type,
                                compilation,
                                &native_parent_indices,
                            )
                        })
                } else {
                    None
                };
                let selectors = dependencies
                    .static_selectors
                    .get(&implementation.id)
                    .cloned()
                    .unwrap_or_default();
                let mut static_calls: BTreeMap<_, _> = selectors
                    .iter()
                    .filter_map(|selector| {
                        // These spellings are language intrinsics, not
                        // overrideable global procedures. Leaving them out of
                        // the static call table lets the bytecode compiler emit
                        // one TypePredicate instruction instead of a call into
                        // the synthetic builtin inventory. This is especially
                        // important in typed world loops such as ruin placement.
                        if compiler_type_predicate(selector) {
                            return None;
                        }
                        let project_target =
                            self.static_call_target(implementation.id, selector, compilation);
                        let project_member = project_target.filter(|target| {
                            self.procedure(target.procedure())
                                .is_some_and(|procedure| procedure.owner_type.is_some())
                        });
                        let native_member = native_member_index(
                            selector,
                            procedure.owner_type,
                            compilation,
                            &native_parent_indices,
                        );
                        let target = if let Some(project_member) = project_member {
                            // A real project member always overrides an engine-owned method.
                            // Independent body inventories intentionally omit the target
                            // implementation, so preserve that resolution with the inert
                            // procedure instead of falling through to the native member.
                            indices.get(&project_member).copied().or_else(|| {
                                (!include_parent_targets).then_some(ordered.len())
                            })
                        } else {
                            native_member.or_else(|| {
                                project_target.and_then(|target| {
                                    indices.get(&target).copied().or_else(|| {
                                        // Independent body inventories intentionally omit the
                                        // target implementation. Point a semantically resolved
                                        // call at an inert standard procedure so lowering can
                                        // validate the caller without recursively compiling it.
                                        (!include_parent_targets).then_some(ordered.len())
                                    })
                                })
                            })
                        };
                        target.map(|target| (selector.clone(), target))
                    })
                    .collect();
                for selector in selectors {
                    if let Some(target) = builtin_indices.get(&selector) {
                        static_calls.entry(selector).or_insert(*target);
                    }
                }
                let referenced = referenced_identifiers(definition);
                let mut src_fields: BTreeMap<String, FieldName> = procedure
                    .owner_type
                    .map(|owner| {
                        inherited_field_name_lookups.set(
                            inherited_field_name_lookups.get() + referenced.len(),
                        );
                        referenced_inherited_fields(
                            compilation,
                            Some(owner),
                            direct_fields,
                            &referenced,
                            true,
                        )
                    })
                    .unwrap_or_default();
                if procedure.owner_type.is_some() {
                    for builtin in ["type", "parent_type"] {
                        if referenced.contains(builtin) {
                            src_fields.insert(
                                builtin.to_owned(),
                                FieldName::parse(builtin)
                                    .expect("built-in datum field name is valid"),
                            );
                        }
                    }
                }
                global_binding_index_lookups.set(
                    global_binding_index_lookups.get() + referenced.len(),
                );
                let referenced_globals = referenced
                    .iter()
                    .filter_map(|name| {
                        global_fields
                            .get(name)
                            .map(|field| (name.clone(), field.clone()))
                    })
                    .collect();
                let mut referenced_globals: BTreeMap<String, FieldName> = referenced_globals;
                inherited_field_name_lookups.set(
                    inherited_field_name_lookups.get() + referenced.len(),
                );
                for (name, field) in referenced_inherited_fields(
                    compilation,
                    procedure.owner_type,
                    direct_static_fields,
                    &referenced,
                    false,
                ) {
                    if referenced.contains(&name) && !src_fields.contains_key(&name) {
                        // Type statics use a qualified VM slot and shadow a
                        // project global of the same bare name.
                        referenced_globals.insert(name.clone(), field.clone());
                        referenced_globals.insert(format!("src.{name}"), field);
                    }
                }
                for (receiver, path) in declared_receiver_types(definition) {
                    if let Some(type_id) = compilation.code_tree().find(&path) {
                        inherited_field_name_lookups.set(
                            inherited_field_name_lookups.get() + referenced.len(),
                        );
                        for (name, field) in referenced_inherited_fields(
                            compilation,
                            Some(type_id),
                            direct_static_fields,
                            &referenced,
                            false,
                        ) {
                            if referenced.contains(&name) {
                                referenced_globals.insert(format!("{receiver}.{name}"), field);
                            }
                        }
                    }
                }
                // A typed project global is also a statically known receiver.
                // BYOND resolves type-owned `var/global` members through their
                // qualified storage even while the receiver value itself is
                // null during early bootstrap (notably GLOB log fields).
                typed_global_index_lookups
                    .set(typed_global_index_lookups.get() + referenced.len());
                for receiver in &referenced {
                    let Some(path) = global_types.get(receiver) else {
                        continue;
                    };
                    let path = dm_syntax::DefinitionPath::new(
                        path.as_str()
                            .trim_matches('/')
                            .split('/')
                            .map(str::to_owned)
                            .collect(),
                    );
                    if let Some(type_id) = compilation.code_tree().find(&path) {
                        inherited_field_name_lookups.set(
                            inherited_field_name_lookups.get() + referenced.len(),
                        );
                        for (name, field) in referenced_inherited_fields(
                            compilation,
                            Some(type_id),
                            direct_static_fields,
                            &referenced,
                            false,
                        ) {
                            if referenced.contains(&name) {
                                referenced_globals.insert(format!("{receiver}.{name}"), field);
                            }
                        }
                    }
                }
                Ok(dm_vm::ProcedureSpec {
                    path: format!("{}@{}", procedure.path, implementation.ordinal),
                    definition,
                    parent,
                    static_calls,
                    src_fields,
                    global_fields: referenced_globals,
                })
            })
            .collect::<Result<_, dm_vm::CompileError>>()?;
        for definition in &builtin_syntax.definitions {
            specs.push(dm_vm::ProcedureSpec {
                path: format!("{}@dream64_builtin", definition.path),
                definition,
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            });
        }
        for definition in &native_parent_syntax.definitions {
            specs.push(dm_vm::ProcedureSpec {
                path: format!("{}@dream64_native", definition.path),
                definition,
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            });
        }
        let mut spec_global_types = normalized_definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| {
                let referenced = referenced_identifiers(definition);
                typed_global_index_lookups.set(typed_global_index_lookups.get() + referenced.len());
                let mut types = referenced
                    .iter()
                    .filter_map(|name| {
                        global_types
                            .get(name)
                            .map(|path| (name.clone(), path.clone()))
                    })
                    .collect::<BTreeMap<_, _>>();
                if let Some(owner) = ordered[index].0.owner_type {
                    // A bare identifier that resolves to a src field carries
                    // its declared type into unqualified `istype(value)`.
                    // BYOND uses this for guards such as `istype(suit)` where
                    // `suit` is a typed instance field rather than a local.
                    types.extend(referenced_inherited_field_types(
                        compilation,
                        owner,
                        direct_field_types,
                        &referenced,
                    ));
                }
                types
            })
            .collect::<Vec<_>>();
        spec_global_types.resize_with(specs.len(), BTreeMap::new);
        let module = if let Some(eager) = eager_implementations {
            let mut eager_indices = eager
                .iter()
                .filter_map(|implementation| indices.get(implementation).copied())
                .collect::<BTreeSet<_>>();
            eager_indices.extend(ordered.len()..specs.len());
            let deferred_errors = deferred_validation_errors
                .into_iter()
                .filter_map(|(implementation, error)| {
                    indices
                        .get(&implementation)
                        .copied()
                        .map(|index| (index, error))
                })
                .collect::<BTreeMap<_, _>>();
            dm_vm::compile_module_specs_selective_with_errors(
                &specs,
                &spec_global_types,
                &eager_indices,
                &deferred_errors,
            )?
        } else {
            dm_vm::compile_module_specs_with_global_types(&specs, &spec_global_types)?
        };
        let implementations = ordered
            .iter()
            .enumerate()
            .map(|(index, (_, implementation, _))| {
                Ok((
                    implementation.id,
                    module
                        .procedure_id_at(index)
                        .ok_or_else(|| dm_vm::CompileError {
                            message: format!(
                                "compiled procedure spec {} has no VM identity",
                                index
                            ),
                        })?,
                ))
            })
            .collect::<Result<_, dm_vm::CompileError>>()?;
        let stats = ExecutableProcedureStats {
            procedures: specs.len(),
            src_field_bindings: specs.iter().map(|spec| spec.src_fields.len()).sum(),
            global_field_bindings: specs.iter().map(|spec| spec.global_fields.len()).sum(),
            static_registry_builds: 1,
            global_binding_index_lookups: global_binding_index_lookups.get(),
            typed_global_index_lookups: typed_global_index_lookups.get(),
            inherited_field_name_lookups: inherited_field_name_lookups.get(),
        };
        Ok(ExecutableProcedures {
            module,
            implementations,
            stats,
        })
    }
}

impl ProcedureRegistry {
    pub(crate) fn static_call_target(
        &self,
        implementation: ProcedureImplementationId,
        selector: &str,
        compilation: &Compilation,
    ) -> Option<ProcedureImplementationId> {
        let procedure = self.procedure(implementation.procedure())?;
        let mut owner = procedure.owner_type;
        let tree = compilation.code_tree();
        while let Some(current_owner) = owner {
            if let Some(candidate) = self
                .by_owner_name
                .get(&(Some(current_owner), selector.to_owned()))
                .and_then(|id| self.procedure(*id))
            {
                return effective_target(&self.procedures, candidate.id);
            }
            owner = tree.node(current_owner).and_then(|node| node.parent_type);
        }
        self.by_owner_name
            .get(&(None, selector.to_owned()))
            .and_then(|id| self.procedure(*id))
            .and_then(|candidate| effective_target(&self.procedures, candidate.id))
    }
}

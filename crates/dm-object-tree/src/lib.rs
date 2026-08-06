//! Deterministic global object-tree construction for parsed DM declarations.
//!
//! This crate deliberately does not assign meaning to declaration bodies. It
//! canonicalizes the code tree, preserves source ordering, and precomputes the
//! default path-based inheritance relationships consumed by semantic analysis.
//! Constant absolute `parent_type` assignments are resolved after every
//! declaration has been indexed, before member inheritance is attached.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use dm_core::{FileId, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};
use dm_syntax::{Definition, DefinitionKind, DefinitionPath, SyntaxFile};

/// A parsed file paired with its stable project-local identity.
#[derive(Clone, Copy, Debug)]
pub struct SyntaxUnit<'syntax> {
    /// File identity assigned by the project loader.
    pub file_id: FileId,
    /// Declarations parsed from the file.
    pub syntax: &'syntax SyntaxFile,
}

/// One declaration supplied in authoritative preprocessor expansion order.
#[derive(Clone, Copy, Debug)]
pub struct DefinitionUnit<'syntax> {
    /// Physical source file containing the declaration.
    pub file_id: FileId,
    /// Index into the corresponding [`SyntaxFile::definitions`] table.
    pub definition_index: usize,
    /// Parsed declaration at that index.
    pub definition: &'syntax Definition,
}

/// A tree-local identity into a [`CodeTree`]'s lexically ordered node table.
///
/// The same set of paths produces the same identities regardless of file visit
/// order. Adding or removing a path may renumber lexically later nodes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(u32);

impl NodeId {
    /// Returns the node-table index represented by this identity.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A tree-local identity into a [`CodeTree`]'s source-ordered declaration table.
///
/// Adding or removing an earlier declaration may renumber later declarations.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeclarationId(u32);

impl DeclarationId {
    /// Returns the declaration-table index represented by this identity.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// An owned canonical DM path.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CodePath(Vec<String>);

impl CodePath {
    fn from_definition(path: &DefinitionPath) -> Self {
        Self(path.segments().to_vec())
    }

    fn prefix(segments: &[String]) -> Self {
        Self(segments.to_vec())
    }

    /// Returns the path segments without the leading slash.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.0
    }
}

impl fmt::Display for CodePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "/{}", self.0.join("/"))
    }
}

/// The semantic namespace occupied by a code-tree node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeKind {
    /// A datum, atom, or other instantiable/inheritable DM type.
    Type,
    /// A procedure in a type's `proc` namespace.
    Procedure,
    /// A command procedure in a type's `verb` namespace.
    Verb,
    /// A field in a type's `var` namespace.
    Variable,
}

impl NodeKind {
    const fn from_definition(kind: DefinitionKind) -> Self {
        match kind {
            DefinitionKind::Type => Self::Type,
            DefinitionKind::Procedure | DefinitionKind::ProcedureOverride => Self::Procedure,
            DefinitionKind::Verb => Self::Verb,
            DefinitionKind::Variable | DefinitionKind::VariableOverride => Self::Variable,
        }
    }
}

/// One canonical node in the global DM code tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodeNode {
    /// Tree-local identity assigned by canonical lexical path order.
    pub id: NodeId,
    /// Canonical absolute path of this node.
    pub path: CodePath,
    /// Namespace represented by this node.
    pub kind: NodeKind,
    /// Effective parent type after constant `parent_type` assignments are applied.
    pub parent_type: Option<NodeId>,
    /// Type that owns a procedure, verb, or variable.
    pub owner_type: Option<NodeId>,
    /// Nearest same-name member found while walking parent types.
    pub inherited_member: Option<NodeId>,
    /// Effective inheritance children in stable node order.
    pub child_types: Vec<NodeId>,
    /// Source-ordered declarations and overrides attached to this node.
    pub declarations: Vec<DeclarationId>,
    standard: bool,
}

impl CodeNode {
    /// Returns whether this type exists only because a descendant required its path.
    #[must_use]
    pub fn is_implicit(&self) -> bool {
        self.kind == NodeKind::Type && !self.standard && self.declarations.is_empty()
    }

    /// Returns whether this node is supplied by the minimal DM standard prelude.
    #[must_use]
    pub const fn is_standard(&self) -> bool {
        self.standard
    }
}

/// A source declaration retained by the object tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TreeDeclaration {
    /// Tree-local source-order identity.
    pub id: DeclarationId,
    /// Canonical node receiving this declaration.
    pub node: NodeId,
    /// Project-local source file.
    pub file_id: FileId,
    /// Index into the corresponding [`SyntaxFile::definitions`] table.
    pub definition_index: usize,
    /// Global declaration order across all input units.
    pub ordinal: usize,
    /// Original syntax-layer declaration category.
    pub kind: DefinitionKind,
    /// Complete source span of the declaration header.
    pub span: SourceSpan,
}

/// Classification of an inheritance-resolution diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InheritanceDiagnosticKind {
    /// The final assignment is not a constant absolute type path.
    NonConstantParentType,
    /// The constant path does not exist in the completed code tree.
    UnknownParentType,
    /// The constant path exists but names a member rather than a type.
    ParentTargetNotType,
    /// Applying the constant assignment would create an inheritance cycle.
    InheritanceCycle,
}

/// A precise error produced while resolving a type's final `parent_type` value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InheritanceDiagnostic {
    /// Machine-readable failure category.
    pub kind: InheritanceDiagnosticKind,
    /// Type whose parent assignment failed.
    pub owner: CodePath,
    /// Parsed constant target, when the expression was a constant path.
    pub target: Option<CodePath>,
    /// Source declaration containing the final assignment.
    pub declaration: Option<DeclarationId>,
    /// Physical source file containing the assignment.
    pub file_id: FileId,
    /// Complete header span of the assignment.
    pub span: SourceSpan,
    declaration_ordinal: usize,
}

/// Severity of an object-tree construction diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticSeverity {
    /// Informational evidence that does not invalidate the tree.
    Note,
    /// A recoverable condition that should be shown to the developer.
    Warning,
    /// A structural conflict that prevents reliable semantic interpretation.
    Error,
}

/// Classification of an object-tree construction diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticKind {
    /// The same file identity was supplied more than once.
    DuplicateFileUnit,
    /// An explicit declaration repeats one already attached to the same node.
    DuplicateDeclaration,
    /// One canonical path was assigned incompatible semantic namespaces.
    ConflictingNodeKind,
    /// A member path does not contain its required namespace and final name.
    MalformedMemberPath,
}

/// A deterministic diagnostic emitted while constructing the code tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TreeDiagnostic {
    /// Machine-readable diagnostic category.
    pub kind: DiagnosticKind,
    /// Impact of this diagnostic.
    pub severity: DiagnosticSeverity,
    /// Canonical path involved, when the diagnostic concerns a node.
    pub path: Option<CodePath>,
    /// Earlier declaration related to the diagnostic, when available.
    pub previous: Option<DeclarationId>,
    /// Later declaration related to the diagnostic, when available.
    pub current: Option<DeclarationId>,
    /// File identity involved when no declaration has been created.
    pub file_id: Option<FileId>,
    previous_ordinal: Option<usize>,
    current_ordinal: Option<usize>,
    encounter_ordinal: usize,
}

/// A deterministic project-wide DM code tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodeTree {
    nodes: Vec<CodeNode>,
    declarations: Vec<TreeDeclaration>,
    paths: BTreeMap<CodePath, NodeId>,
    inheritance_diagnostics: Vec<InheritanceDiagnostic>,
}

impl CodeTree {
    /// Returns every node in stable canonical path order.
    #[must_use]
    pub fn nodes(&self) -> &[CodeNode] {
        &self.nodes
    }

    /// Returns every declaration in global project/source order.
    #[must_use]
    pub fn declarations(&self) -> &[TreeDeclaration] {
        &self.declarations
    }

    /// Looks up a node by identity.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&CodeNode> {
        self.nodes.get(id.index())
    }

    /// Looks up a declaration by identity.
    #[must_use]
    pub fn declaration(&self, id: DeclarationId) -> Option<&TreeDeclaration> {
        self.declarations.get(id.index())
    }

    /// Finds the node matching a syntax-layer canonical path.
    #[must_use]
    pub fn find(&self, path: &DefinitionPath) -> Option<NodeId> {
        self.paths.get(&CodePath::from_definition(path)).copied()
    }

    /// Returns errors from resolving final constant `parent_type` assignments.
    #[must_use]
    pub fn inheritance_diagnostics(&self) -> &[InheritanceDiagnostic] {
        &self.inheritance_diagnostics
    }
}

/// Result of project-wide code-tree construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildOutput {
    /// Constructed tree, including recoverable conflicting declarations.
    pub tree: CodeTree,
    /// Diagnostics in deterministic input encounter order.
    pub diagnostics: Vec<TreeDiagnostic>,
}

#[derive(Clone, Debug)]
struct PendingNode {
    kind: NodeKind,
    declarations: Vec<PendingDeclaration>,
    standard: bool,
}

#[derive(Clone, Copy, Debug)]
struct PendingDeclaration {
    file_id: FileId,
    definition_index: usize,
    ordinal: usize,
    kind: DefinitionKind,
    span: SourceSpan,
}

#[derive(Clone, Debug)]
struct PendingParentAssignment {
    owner: CodePath,
    target: Option<CodePath>,
    declaration_ordinal: usize,
    file_id: FileId,
    span: SourceSpan,
}

#[derive(Clone, Copy, Debug)]
struct OrderedDefinitionUnit<'syntax> {
    unit: DefinitionUnit<'syntax>,
    encounter_ordinal: usize,
}

/// Builds one global tree from files supplied in project include order.
///
/// Node identities are assigned by canonical lexical path order and therefore
/// remain stable when the same declarations are visited in another file order.
/// Declaration identities intentionally retain the supplied unit order and each
/// syntax file's source order.
#[must_use]
pub fn build(units: &[SyntaxUnit<'_>]) -> BuildOutput {
    let mut diagnostics = Vec::new();
    let mut seen_files = BTreeSet::new();
    let mut definitions = Vec::new();
    let mut next_encounter_ordinal = 0usize;

    for unit in units {
        if !seen_files.insert(unit.file_id) {
            diagnostics.push(TreeDiagnostic {
                kind: DiagnosticKind::DuplicateFileUnit,
                severity: DiagnosticSeverity::Error,
                path: None,
                previous: None,
                current: None,
                file_id: Some(unit.file_id),
                previous_ordinal: None,
                current_ordinal: None,
                encounter_ordinal: next_encounter_ordinal,
            });
            next_encounter_ordinal += 1;
            continue;
        }

        for (definition_index, definition) in unit.syntax.definitions.iter().enumerate() {
            definitions.push(OrderedDefinitionUnit {
                unit: DefinitionUnit {
                    file_id: unit.file_id,
                    definition_index,
                    definition,
                },
                encounter_ordinal: next_encounter_ordinal,
            });
            next_encounter_ordinal += 1;
        }
    }

    build_definition_units(&definitions, diagnostics)
}

/// Builds one global tree from declarations in preprocessor expansion order.
///
/// Unlike [`build`], this entry point permits a file identity to appear in
/// multiple non-contiguous positions. This is required when declarations from
/// an included file are spliced between declarations in the including file.
/// The supplied order becomes [`TreeDeclaration::ordinal`] order exactly.
#[must_use]
pub fn build_definitions(definitions: &[DefinitionUnit<'_>]) -> BuildOutput {
    let definitions: Vec<_> = definitions
        .iter()
        .copied()
        .enumerate()
        .map(|(encounter_ordinal, unit)| OrderedDefinitionUnit {
            unit,
            encounter_ordinal,
        })
        .collect();
    build_definition_units(&definitions, Vec::new())
}

fn build_definition_units(
    definitions: &[OrderedDefinitionUnit<'_>],
    mut diagnostics: Vec<TreeDiagnostic>,
) -> BuildOutput {
    let mut pending = BTreeMap::<CodePath, PendingNode>::new();
    seed_standard_prelude(&mut pending);
    let mut parent_assignments = BTreeMap::<CodePath, PendingParentAssignment>::new();

    for (ordinal, ordered_unit) in definitions.iter().enumerate() {
        let unit = ordered_unit.unit;
        let definition = unit.definition;
        let path = CodePath::from_definition(&definition.path);
        let incoming_kind = NodeKind::from_definition(definition.kind);
        let pending_declaration = PendingDeclaration {
            file_id: unit.file_id,
            definition_index: unit.definition_index,
            ordinal,
            kind: definition.kind,
            span: definition.span,
        };

        if !has_valid_member_namespace(&path, incoming_kind) {
            diagnostics.push(TreeDiagnostic {
                kind: DiagnosticKind::MalformedMemberPath,
                severity: DiagnosticSeverity::Error,
                path: Some(path),
                previous: None,
                current: None,
                file_id: Some(unit.file_id),
                previous_ordinal: None,
                current_ordinal: Some(pending_declaration.ordinal),
                encounter_ordinal: ordered_unit.encounter_ordinal,
            });
            continue;
        }

        ensure_type_ancestors(&mut pending, &path, incoming_kind);
        let pending_node = pending.entry(path.clone()).or_insert_with(|| PendingNode {
            kind: incoming_kind,
            declarations: Vec::new(),
            standard: false,
        });

        if pending_node.kind != incoming_kind {
            let previous_ordinal = pending_node
                .declarations
                .first()
                .map(|declaration| declaration.ordinal);
            diagnostics.push(TreeDiagnostic {
                kind: DiagnosticKind::ConflictingNodeKind,
                severity: DiagnosticSeverity::Error,
                path: Some(path),
                previous: None,
                current: None,
                file_id: Some(unit.file_id),
                previous_ordinal,
                current_ordinal: Some(ordinal),
                encounter_ordinal: ordered_unit.encounter_ordinal,
            });
            continue;
        } else if is_explicit_declaration(definition.kind)
            && pending_node
                .declarations
                .iter()
                .any(|existing| existing.kind == definition.kind)
        {
            let previous_ordinal = pending_node
                .declarations
                .iter()
                .find(|existing| existing.kind == definition.kind)
                .map(|declaration| declaration.ordinal);
            diagnostics.push(TreeDiagnostic {
                kind: DiagnosticKind::DuplicateDeclaration,
                severity: DiagnosticSeverity::Note,
                path: Some(path),
                previous: None,
                current: None,
                file_id: Some(unit.file_id),
                previous_ordinal,
                current_ordinal: Some(ordinal),
                encounter_ordinal: ordered_unit.encounter_ordinal,
            });
        }
        pending_node.declarations.push(pending_declaration);
        if let Some(assignment) = parent_assignment(definition, pending_declaration) {
            parent_assignments.insert(assignment.owner.clone(), assignment);
        }
    }

    finish_tree(pending, diagnostics, parent_assignments)
}

fn parent_assignment(
    definition: &Definition,
    declaration: PendingDeclaration,
) -> Option<PendingParentAssignment> {
    let assignment = definition.header.iter().position(
        |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
    )?;
    if definition
        .header
        .iter()
        .position(|token| token.kind == TokenKind::Punctuation('('))
        .is_some_and(|opening| opening < assignment)
    {
        return None;
    }
    if !matches!(
        definition.header[..assignment]
            .last()
            .map(|token| &token.kind),
        Some(TokenKind::Identifier(identifier)) if identifier == "parent_type"
    ) {
        return None;
    }
    let owner_length = definition.path.segments().len().checked_sub(2)?;
    if owner_length == 0 {
        return None;
    }
    let owner = CodePath::prefix(&definition.path.segments()[..owner_length]);
    let target = parse_constant_type_path(&definition.header[assignment + 1..]);
    Some(PendingParentAssignment {
        owner,
        target,
        declaration_ordinal: declaration.ordinal,
        file_id: declaration.file_id,
        span: declaration.span,
    })
}

fn parse_constant_type_path(tokens: &[SpannedToken]) -> Option<CodePath> {
    if !matches!(
        tokens.first().map(|token| &token.kind),
        Some(TokenKind::Operator(operator)) if operator == "/"
    ) {
        return None;
    }

    let mut segments = Vec::new();
    let mut expect_identifier = true;
    for token in &tokens[1..] {
        match (&token.kind, expect_identifier) {
            (TokenKind::Identifier(identifier), true) => {
                segments.push(identifier.clone());
                expect_identifier = false;
            }
            (TokenKind::Operator(operator), false) if operator == "/" => {
                expect_identifier = true;
            }
            _ => return None,
        }
    }
    (!segments.is_empty() && !expect_identifier).then_some(CodePath(segments))
}

fn is_explicit_declaration(kind: DefinitionKind) -> bool {
    matches!(
        kind,
        DefinitionKind::Procedure | DefinitionKind::Verb | DefinitionKind::Variable
    )
}

fn seed_standard_prelude(pending: &mut BTreeMap<CodePath, PendingNode>) {
    const STANDARD_TYPES: &[&[&str]] = &[
        &["datum"],
        &["atom"],
        &["atom", "movable"],
        &["area"],
        &["turf"],
        &["obj"],
        &["mob"],
        &["world"],
        &["client"],
    ];

    for segments in STANDARD_TYPES {
        pending.insert(
            CodePath(
                segments
                    .iter()
                    .map(|segment| (*segment).to_owned())
                    .collect(),
            ),
            PendingNode {
                kind: NodeKind::Type,
                declarations: Vec::new(),
                standard: true,
            },
        );
    }
}

fn has_valid_member_namespace(path: &CodePath, kind: NodeKind) -> bool {
    let namespace = match kind {
        NodeKind::Type => return true,
        NodeKind::Procedure => "proc",
        NodeKind::Verb => "verb",
        NodeKind::Variable => "var",
    };

    path.0.len() >= 2 && path.0[path.0.len() - 2] == namespace
}

fn ensure_type_ancestors(
    pending: &mut BTreeMap<CodePath, PendingNode>,
    path: &CodePath,
    kind: NodeKind,
) {
    let owner_length = match kind {
        NodeKind::Type => path.0.len().saturating_sub(1),
        NodeKind::Procedure | NodeKind::Verb | NodeKind::Variable => path.0.len() - 2,
    };
    for length in 1..=owner_length {
        pending
            .entry(CodePath::prefix(&path.0[..length]))
            .or_insert_with(|| PendingNode {
                kind: NodeKind::Type,
                declarations: Vec::new(),
                standard: false,
            });
    }
}

fn finish_tree(
    pending: BTreeMap<CodePath, PendingNode>,
    mut diagnostics: Vec<TreeDiagnostic>,
    parent_assignments: BTreeMap<CodePath, PendingParentAssignment>,
) -> BuildOutput {
    let paths: BTreeMap<_, _> = pending
        .keys()
        .cloned()
        .enumerate()
        .map(|(index, path)| (path, node_id(index)))
        .collect();
    let mut declarations = Vec::new();
    let mut nodes = Vec::with_capacity(pending.len());

    for (path, pending_node) in pending {
        let id = paths[&path];
        let mut declaration_ids = Vec::with_capacity(pending_node.declarations.len());
        for declaration in pending_node.declarations {
            declarations.push(TreeDeclaration {
                id: DeclarationId(0),
                node: id,
                file_id: declaration.file_id,
                definition_index: declaration.definition_index,
                ordinal: declaration.ordinal,
                kind: declaration.kind,
                span: declaration.span,
            });
        }
        nodes.push(CodeNode {
            id,
            parent_type: default_parent_type(&path, pending_node.kind, &paths),
            owner_type: owner_type(&path, pending_node.kind, &paths),
            path,
            kind: pending_node.kind,
            inherited_member: None,
            child_types: Vec::new(),
            declarations: std::mem::take(&mut declaration_ids),
            standard: pending_node.standard,
        });
    }

    declarations.sort_by_key(|declaration| declaration.ordinal);
    for (index, declaration) in declarations.iter_mut().enumerate() {
        declaration.id = declaration_id(index);
        nodes[declaration.node.index()]
            .declarations
            .push(declaration.id);
    }
    let inheritance_diagnostics =
        resolve_parent_assignments(&mut nodes, &paths, &declarations, parent_assignments);
    attach_children(&mut nodes);
    attach_inherited_members(&mut nodes, &paths);
    diagnostics.sort_by_key(|diagnostic| diagnostic.encounter_ordinal);
    resolve_diagnostic_declarations(&mut diagnostics, &declarations);

    BuildOutput {
        tree: CodeTree {
            nodes,
            declarations,
            paths,
            inheritance_diagnostics,
        },
        diagnostics,
    }
}

fn node_id(index: usize) -> NodeId {
    NodeId(u32::try_from(index).expect("a code tree cannot contain more than u32::MAX nodes"))
}

fn declaration_id(index: usize) -> DeclarationId {
    DeclarationId(
        u32::try_from(index).expect("a code tree cannot contain more than u32::MAX declarations"),
    )
}

fn default_parent_type(
    path: &CodePath,
    kind: NodeKind,
    paths: &BTreeMap<CodePath, NodeId>,
) -> Option<NodeId> {
    if kind != NodeKind::Type {
        return None;
    }
    let parent = default_parent_path(path)?;
    paths.get(&parent).copied()
}

fn resolve_parent_assignments(
    nodes: &mut [CodeNode],
    paths: &BTreeMap<CodePath, NodeId>,
    declarations: &[TreeDeclaration],
    assignments: BTreeMap<CodePath, PendingParentAssignment>,
) -> Vec<InheritanceDiagnostic> {
    let mut diagnostics = Vec::new();
    let mut applied = BTreeMap::<NodeId, PendingParentAssignment>::new();

    for (owner_path, assignment) in assignments {
        let owner = paths[&owner_path];
        let Some(target_path) = assignment.target.as_ref() else {
            diagnostics.push(inheritance_diagnostic(
                &assignment,
                InheritanceDiagnosticKind::NonConstantParentType,
            ));
            continue;
        };
        let Some(target) = paths.get(target_path).copied() else {
            diagnostics.push(inheritance_diagnostic(
                &assignment,
                InheritanceDiagnosticKind::UnknownParentType,
            ));
            continue;
        };
        if nodes[target.index()].kind != NodeKind::Type {
            diagnostics.push(inheritance_diagnostic(
                &assignment,
                InheritanceDiagnosticKind::ParentTargetNotType,
            ));
            continue;
        }

        nodes[owner.index()].parent_type = Some(target);
        applied.insert(owner, assignment);
    }

    let cycle_nodes = inheritance_cycle_nodes(nodes);
    for (owner, assignment) in applied {
        if !cycle_nodes.contains(&owner) {
            continue;
        }
        diagnostics.push(inheritance_diagnostic(
            &assignment,
            InheritanceDiagnosticKind::InheritanceCycle,
        ));
        nodes[owner.index()].parent_type =
            default_parent_type(&nodes[owner.index()].path, NodeKind::Type, paths);
    }

    let declarations_by_ordinal: BTreeMap<_, _> = declarations
        .iter()
        .map(|declaration| (declaration.ordinal, declaration.id))
        .collect();
    diagnostics.sort_by_key(|diagnostic| diagnostic.declaration_ordinal);
    for diagnostic in &mut diagnostics {
        diagnostic.declaration = declarations_by_ordinal
            .get(&diagnostic.declaration_ordinal)
            .copied();
    }
    diagnostics
}

fn inheritance_diagnostic(
    assignment: &PendingParentAssignment,
    kind: InheritanceDiagnosticKind,
) -> InheritanceDiagnostic {
    InheritanceDiagnostic {
        kind,
        owner: assignment.owner.clone(),
        target: assignment.target.clone(),
        declaration: None,
        file_id: assignment.file_id,
        span: assignment.span,
        declaration_ordinal: assignment.declaration_ordinal,
    }
}

fn inheritance_cycle_nodes(nodes: &[CodeNode]) -> BTreeSet<NodeId> {
    let mut state = vec![0u8; nodes.len()];
    let mut cycle_nodes = BTreeSet::new();

    for start_index in 0..nodes.len() {
        if nodes[start_index].kind != NodeKind::Type || state[start_index] != 0 {
            continue;
        }
        let mut chain = Vec::new();
        let mut current = Some(nodes[start_index].id);
        while let Some(node) = current {
            match state[node.index()] {
                0 => {
                    state[node.index()] = 1;
                    chain.push(node);
                    current = nodes[node.index()].parent_type;
                }
                1 => {
                    if let Some(cycle_start) = chain.iter().position(|candidate| *candidate == node)
                    {
                        cycle_nodes.extend(chain[cycle_start..].iter().copied());
                    }
                    break;
                }
                _ => break,
            }
        }
        for node in chain {
            state[node.index()] = 2;
        }
    }
    cycle_nodes
}

fn default_parent_path(path: &CodePath) -> Option<CodePath> {
    const PRIMITIVE_ROOTS: &[&str] = &[
        "datum", "world", "client", "list", "savefile", "alist", "pixloc", "vector", "callee",
    ];

    if path.0.len() == 1 && PRIMITIVE_ROOTS.contains(&path.0[0].as_str()) {
        return None;
    }
    if path_is(path, &["area"]) || path_is(path, &["turf"]) {
        return Some(CodePath(vec!["atom".to_owned()]));
    }
    if path_is(path, &["obj"]) || path_is(path, &["mob"]) {
        return Some(CodePath(vec!["atom".to_owned(), "movable".to_owned()]));
    }
    if path.0.len() == 1 {
        return Some(CodePath(vec!["datum".to_owned()]));
    }
    Some(CodePath::prefix(&path.0[..path.0.len() - 1]))
}

fn path_is(path: &CodePath, expected: &[&str]) -> bool {
    path.0
        .iter()
        .map(String::as_str)
        .eq(expected.iter().copied())
}

fn owner_type(
    path: &CodePath,
    kind: NodeKind,
    paths: &BTreeMap<CodePath, NodeId>,
) -> Option<NodeId> {
    match kind {
        NodeKind::Type => return None,
        NodeKind::Procedure | NodeKind::Verb | NodeKind::Variable => {}
    }
    let owner_length = path.0.len() - 2;
    (owner_length != 0)
        .then(|| CodePath::prefix(&path.0[..owner_length]))
        .and_then(|owner| paths.get(&owner).copied())
}

fn attach_children(nodes: &mut [CodeNode]) {
    for child_index in 0..nodes.len() {
        let child = nodes[child_index].id;
        if nodes[child_index].kind == NodeKind::Type
            && let Some(parent) = nodes[child_index].parent_type
        {
            nodes[parent.index()].child_types.push(child);
        }
    }
}

fn attach_inherited_members(nodes: &mut [CodeNode], paths: &BTreeMap<CodePath, NodeId>) {
    for node_index in 0..nodes.len() {
        let Some(owner) = nodes[node_index].owner_type else {
            continue;
        };
        let owner_length = nodes[owner.index()].path.0.len();
        let member_suffix = nodes[node_index].path.0[owner_length..].to_vec();
        let mut ancestor = nodes[owner.index()].parent_type;
        while let Some(ancestor_id) = ancestor {
            let mut candidate_segments = nodes[ancestor_id.index()].path.0.clone();
            candidate_segments.extend(member_suffix.iter().cloned());
            if let Some(candidate) = paths.get(&CodePath(candidate_segments))
                && nodes[candidate.index()].kind == nodes[node_index].kind
            {
                nodes[node_index].inherited_member = Some(*candidate);
                break;
            }
            ancestor = nodes[ancestor_id.index()].parent_type;
        }
    }
}

fn resolve_diagnostic_declarations(
    diagnostics: &mut [TreeDiagnostic],
    declarations: &[TreeDeclaration],
) {
    let declarations_by_ordinal: BTreeMap<_, _> = declarations
        .iter()
        .map(|declaration| (declaration.ordinal, declaration.id))
        .collect();
    for diagnostic in diagnostics {
        diagnostic.previous = diagnostic
            .previous_ordinal
            .and_then(|ordinal| declarations_by_ordinal.get(&ordinal))
            .copied();
        diagnostic.current = diagnostic
            .current_ordinal
            .and_then(|ordinal| declarations_by_ordinal.get(&ordinal))
            .copied();
    }
}

#[cfg(test)]
mod tests {
    use dm_core::FileId;
    use dm_syntax::{DefinitionKind, parse};

    use super::{
        DefinitionUnit, DiagnosticKind, DiagnosticSeverity, InheritanceDiagnosticKind, NodeKind,
        SyntaxUnit, build, build_definitions,
    };

    #[test]
    fn assigns_node_ids_by_canonical_path_not_file_visit_order() {
        let datum = parse("/datum/zeta\n").expect("datum source should parse");
        let atom = parse("/atom/alpha\n").expect("atom source should parse");
        let first = build(&[
            SyntaxUnit {
                file_id: FileId::from_index(0),
                syntax: &datum,
            },
            SyntaxUnit {
                file_id: FileId::from_index(1),
                syntax: &atom,
            },
        ]);
        let second = build(&[
            SyntaxUnit {
                file_id: FileId::from_index(1),
                syntax: &atom,
            },
            SyntaxUnit {
                file_id: FileId::from_index(0),
                syntax: &datum,
            },
        ]);

        let first_paths: Vec<_> = first
            .tree
            .nodes()
            .iter()
            .map(|node| (node.id, node.path.to_string()))
            .collect();
        let second_paths: Vec<_> = second
            .tree
            .nodes()
            .iter()
            .map(|node| (node.id, node.path.to_string()))
            .collect();
        assert_eq!(first_paths, second_paths);
        assert!(first_paths.iter().any(|(_, path)| path == "/atom"));
        assert!(first_paths.iter().any(|(_, path)| path == "/atom/alpha"));
    }

    #[test]
    fn builds_implicit_parents_and_member_inheritance() {
        let syntax = parse(
            "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base/child\n\trun()\n\t\treturn 2\n",
        )
        .expect("inheritance source should parse");
        let output = build(&[SyntaxUnit {
            file_id: FileId::from_index(0),
            syntax: &syntax,
        }]);
        let child_path = &syntax.definitions[2].path;
        let override_path = &syntax.definitions[3].path;
        let child = output.tree.find(child_path).expect("child should exist");
        let procedure = output
            .tree
            .find(&syntax.definitions[0].path)
            .expect("base type should exist");
        let override_node = output
            .tree
            .find(override_path)
            .expect("override should exist");
        let base_run = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/datum/base/proc/run")
            .expect("base procedure should exist");

        assert_eq!(
            output.tree.node(child).expect("valid child").parent_type,
            Some(procedure)
        );
        assert_eq!(
            output
                .tree
                .node(override_node)
                .expect("valid override")
                .inherited_member,
            Some(base_run.id)
        );
        let datum = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/datum")
            .expect("standard datum should exist");
        assert!(datum.is_standard());
        assert!(!datum.is_implicit());
    }

    #[test]
    fn applies_standard_and_top_level_default_parents() {
        let syntax =
            parse("/thing\n/thing/child\n/obj/item\n").expect("default-parent source should parse");
        let output = build(&[SyntaxUnit {
            file_id: FileId::from_index(0),
            syntax: &syntax,
        }]);

        let parent_path = |path: &str| {
            let node = output
                .tree
                .nodes()
                .iter()
                .find(|node| node.path.to_string() == path)
                .expect("requested node should exist");
            node.parent_type.map(|parent| {
                output
                    .tree
                    .node(parent)
                    .expect("parent identity should be valid")
                    .path
                    .to_string()
            })
        };

        assert_eq!(parent_path("/thing").as_deref(), Some("/datum"));
        assert_eq!(parent_path("/thing/child").as_deref(), Some("/thing"));
        assert_eq!(parent_path("/atom").as_deref(), Some("/datum"));
        assert_eq!(parent_path("/atom/movable").as_deref(), Some("/atom"));
        assert_eq!(parent_path("/area").as_deref(), Some("/atom"));
        assert_eq!(parent_path("/turf").as_deref(), Some("/atom"));
        assert_eq!(parent_path("/obj").as_deref(), Some("/atom/movable"));
        assert_eq!(parent_path("/mob").as_deref(), Some("/atom/movable"));
        assert_eq!(parent_path("/obj/item").as_deref(), Some("/obj"));
        assert_eq!(parent_path("/world"), None);
        assert_eq!(parent_path("/client"), None);
    }

    #[test]
    fn preserves_declaration_and_override_order() {
        let first =
            parse("/datum/sample/proc/run()\n\treturn 1\n").expect("first source should parse");
        let second =
            parse("/datum/sample\n\trun()\n\t\treturn 2\n").expect("second source should parse");
        let output = build(&[
            SyntaxUnit {
                file_id: FileId::from_index(8),
                syntax: &first,
            },
            SyntaxUnit {
                file_id: FileId::from_index(3),
                syntax: &second,
            },
        ]);
        let procedure = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/datum/sample/proc/run")
            .expect("procedure should exist");
        let declarations: Vec<_> = procedure
            .declarations
            .iter()
            .map(|id| output.tree.declaration(*id).expect("valid declaration"))
            .collect();

        assert_eq!(declarations.len(), 2);
        assert_eq!(declarations[0].file_id, FileId::from_index(8));
        assert_eq!(declarations[0].kind, DefinitionKind::Procedure);
        assert_eq!(declarations[1].file_id, FileId::from_index(3));
        assert_eq!(declarations[1].kind, DefinitionKind::ProcedureOverride);
        assert!(declarations[0].ordinal < declarations[1].ordinal);
    }

    #[test]
    fn accepts_authoritative_include_expansion_order() {
        let outer = parse("/datum/order/var/first\n/datum/order/var/third\n")
            .expect("outer source should parse");
        let included = parse("/datum/order/var/second\n").expect("included source should parse");
        let outer_file = FileId::from_index(0);
        let included_file = FileId::from_index(1);
        let output = build_definitions(&[
            DefinitionUnit {
                file_id: outer_file,
                definition_index: 0,
                definition: &outer.definitions[0],
            },
            DefinitionUnit {
                file_id: included_file,
                definition_index: 0,
                definition: &included.definitions[0],
            },
            DefinitionUnit {
                file_id: outer_file,
                definition_index: 1,
                definition: &outer.definitions[1],
            },
        ]);

        assert!(output.diagnostics.is_empty());
        let order: Vec<_> = output
            .tree
            .declarations()
            .iter()
            .map(|declaration| {
                (
                    declaration.ordinal,
                    declaration.file_id,
                    declaration.definition_index,
                )
            })
            .collect();
        assert_eq!(
            order,
            [
                (0, outer_file, 0),
                (1, included_file, 0),
                (2, outer_file, 1),
            ]
        );
    }

    #[test]
    fn reports_duplicate_explicit_declarations() {
        let syntax = parse("/datum/sample/proc/run()\n/datum/sample/proc/run()\n")
            .expect("duplicate source should parse");
        let output = build(&[SyntaxUnit {
            file_id: FileId::from_index(0),
            syntax: &syntax,
        }]);

        assert_eq!(output.diagnostics.len(), 1);
        assert_eq!(
            output.diagnostics[0].kind,
            DiagnosticKind::DuplicateDeclaration
        );
        assert_eq!(output.diagnostics[0].severity, DiagnosticSeverity::Note);
        assert!(output.diagnostics[0].previous.is_some());
        assert!(output.diagnostics[0].current.is_some());
    }

    #[test]
    fn allows_a_type_to_be_reopened() {
        let syntax =
            parse("/datum/sample\n/datum/sample\n").expect("reopened type source should parse");
        let output = build(&[SyntaxUnit {
            file_id: FileId::from_index(0),
            syntax: &syntax,
        }]);

        assert!(output.diagnostics.is_empty());
        let sample = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/datum/sample")
            .expect("reopened type should exist");
        assert_eq!(sample.declarations.len(), 2);
    }

    #[test]
    fn skips_duplicate_file_units() {
        let procedure = parse("/datum/sample/proc/run()\n").expect("procedure source should parse");
        let file_id = FileId::from_index(4);
        let output = build(&[
            SyntaxUnit {
                file_id,
                syntax: &procedure,
            },
            SyntaxUnit {
                file_id,
                syntax: &procedure,
            },
        ]);

        assert_eq!(output.diagnostics.len(), 1);
        assert_eq!(
            output.diagnostics[0].kind,
            DiagnosticKind::DuplicateFileUnit
        );
        assert_eq!(output.diagnostics[0].severity, DiagnosticSeverity::Error);
        let procedure = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/datum/sample/proc/run")
            .expect("procedure should exist");
        assert_eq!(procedure.declarations.len(), 1);
    }

    #[test]
    fn keeps_diagnostics_in_input_encounter_order() {
        let syntax = parse("/datum/sample/proc/run()\n/datum/sample/proc/run()\n")
            .expect("duplicate source should parse");
        let file_id = FileId::from_index(4);
        let output = build(&[
            SyntaxUnit {
                file_id,
                syntax: &syntax,
            },
            SyntaxUnit {
                file_id,
                syntax: &syntax,
            },
        ]);

        assert_eq!(
            output
                .diagnostics
                .iter()
                .map(|diagnostic| diagnostic.kind)
                .collect::<Vec<_>>(),
            [
                DiagnosticKind::DuplicateDeclaration,
                DiagnosticKind::DuplicateFileUnit,
            ]
        );
    }

    #[test]
    fn rejects_a_conflicting_declaration_from_the_node() {
        let mut type_at_procedure_path =
            parse("/datum/sample/proc/run()\n").expect("type source should parse");
        type_at_procedure_path.definitions[0].kind = DefinitionKind::Type;
        let procedure = parse("/datum/sample/proc/run()\n").expect("procedure source should parse");
        let output = build(&[
            SyntaxUnit {
                file_id: FileId::from_index(0),
                syntax: &type_at_procedure_path,
            },
            SyntaxUnit {
                file_id: FileId::from_index(1),
                syntax: &procedure,
            },
        ]);

        assert_eq!(output.diagnostics.len(), 1);
        assert_eq!(
            output.diagnostics[0].kind,
            DiagnosticKind::ConflictingNodeKind
        );
        assert_eq!(output.diagnostics[0].severity, DiagnosticSeverity::Error);
        assert!(output.diagnostics[0].previous.is_some());
        assert!(output.diagnostics[0].current.is_none());
        let node = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/datum/sample/proc/run")
            .expect("first declaration should retain its node");
        assert_eq!(node.kind, NodeKind::Type);
        assert_eq!(node.declarations.len(), 1);
    }

    #[test]
    fn rejects_a_member_without_its_namespace() {
        let mut malformed = parse("/datum/sample/proc/run()\n").expect("source should parse");
        malformed.definitions[0].path = dm_syntax::DefinitionPath::new(vec![
            "datum".to_owned(),
            "sample".to_owned(),
            "run".to_owned(),
        ]);
        let output = build(&[SyntaxUnit {
            file_id: FileId::from_index(0),
            syntax: &malformed,
        }]);

        assert_eq!(output.diagnostics.len(), 1);
        assert_eq!(
            output.diagnostics[0].kind,
            DiagnosticKind::MalformedMemberPath
        );
        assert_eq!(output.diagnostics[0].severity, DiagnosticSeverity::Error);
        assert_eq!(output.diagnostics[0].file_id, Some(FileId::from_index(0)));
        assert!(
            output
                .tree
                .nodes()
                .iter()
                .all(|node| node.path.to_string() != "/datum/sample/run")
        );
    }

    #[test]
    fn resolves_a_constant_parent_type_and_member_inheritance() {
        let syntax = parse(
            "/datum/alternate\n\tproc/run()\n\t\treturn 1\n/custom\n\tparent_type = /datum/alternate\n\trun()\n\t\treturn ..()\n",
        )
        .expect("reparenting source should parse");
        let output = build(&[SyntaxUnit {
            file_id: FileId::from_index(0),
            syntax: &syntax,
        }]);
        let custom = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/custom")
            .expect("custom type should exist");
        let parent = output
            .tree
            .node(custom.parent_type.expect("custom should have a parent"))
            .expect("parent identity should be valid");
        let parent_run = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/datum/alternate/proc/run")
            .expect("parent procedure should exist");
        let custom_run = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/custom/proc/run")
            .expect("custom procedure should exist");

        assert_eq!(parent.path.to_string(), "/datum/alternate");
        assert_eq!(custom_run.inherited_member, Some(parent_run.id));
        assert!(output.tree.inheritance_diagnostics().is_empty());
    }

    #[test]
    fn final_parent_type_assignment_wins() {
        let syntax = parse(
            "/datum/first\n/datum/second\n/custom\n\tparent_type = /datum/first\n/custom\n\tparent_type = /datum/second\n",
        )
        .expect("override source should parse");
        let output = build(&[SyntaxUnit {
            file_id: FileId::from_index(0),
            syntax: &syntax,
        }]);
        let custom = output
            .tree
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/custom")
            .expect("custom type should exist");
        let parent = output
            .tree
            .node(custom.parent_type.expect("custom should have a parent"))
            .expect("parent identity should be valid");

        assert_eq!(parent.path.to_string(), "/datum/second");
        assert!(output.tree.inheritance_diagnostics().is_empty());
    }

    #[test]
    fn diagnoses_invalid_and_nonconstant_parent_targets() {
        let syntax = parse(
            "/proc/not_a_type()\n/unknown_parent\n\tparent_type = /missing/type\n/member_parent\n\tparent_type = /proc/not_a_type\n/dynamic_parent\n\tparent_type = choose_parent()\n",
        )
        .expect("invalid-parent source should parse");
        let output = build(&[SyntaxUnit {
            file_id: FileId::from_index(0),
            syntax: &syntax,
        }]);
        let diagnostics = output.tree.inheritance_diagnostics();

        assert_eq!(diagnostics.len(), 3);
        assert_eq!(
            diagnostics
                .iter()
                .map(|diagnostic| diagnostic.kind)
                .collect::<Vec<_>>(),
            [
                InheritanceDiagnosticKind::UnknownParentType,
                InheritanceDiagnosticKind::ParentTargetNotType,
                InheritanceDiagnosticKind::NonConstantParentType,
            ]
        );
        assert!(diagnostics.iter().all(|diagnostic| {
            diagnostic.declaration.is_some()
                && diagnostic.file_id == FileId::from_index(0)
                && !diagnostic.span.is_empty()
        }));
    }

    #[test]
    fn diagnoses_and_breaks_parent_type_cycles() {
        let syntax = parse(
            "/datum/first\n\tparent_type = /datum/second\n/datum/second\n\tparent_type = /datum/first\n",
        )
        .expect("cyclic source should parse");
        let output = build(&[SyntaxUnit {
            file_id: FileId::from_index(0),
            syntax: &syntax,
        }]);
        let diagnostics = output.tree.inheritance_diagnostics();

        assert_eq!(diagnostics.len(), 2);
        assert!(diagnostics.iter().all(|diagnostic| {
            diagnostic.kind == InheritanceDiagnosticKind::InheritanceCycle
                && diagnostic.declaration.is_some()
        }));
        for path in ["/datum/first", "/datum/second"] {
            let node = output
                .tree
                .nodes()
                .iter()
                .find(|node| node.path.to_string() == path)
                .expect("cycle type should exist");
            let parent = output
                .tree
                .node(node.parent_type.expect("fallback parent should exist"))
                .expect("parent identity should be valid");
            assert_eq!(parent.path.to_string(), "/datum");
        }
    }
}

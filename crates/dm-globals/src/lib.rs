//! Deterministic inventory of DM variable declarations and initialization work.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use dm_compiler::Compilation;
use dm_core::{FileId, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};
use dm_object_tree::NodeId;
use dm_syntax::{Definition, DefinitionKind};

/// Storage lifetime assigned to a variable node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum StorageClass {
    /// Project-wide state declared through a top-level or `global` variable path.
    Global,
    /// State shared by every instance of one type through the `static` modifier.
    Static,
    /// State stored independently on each datum or atom instance.
    Instance,
}

/// Whether source introduces a variable or assigns an inherited/existing one.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AssignmentKind {
    /// Explicit `var` declaration.
    Declaration,
    /// Value assignment to an existing variable path.
    Override,
}

/// Conservative initialization classification.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InitializerClass {
    /// No initializer syntax was present.
    None,
    /// A single literal, `null`, or absolute type path can be retained directly.
    ConstantSafe,
    /// Evaluation must occur in the runtime initializer phase.
    RequiresRuntime(RuntimeBlocker),
}

/// First syntax shape preventing conservative constant storage.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeBlocker {
    /// A `new` expression allocates runtime state.
    NewExpression,
    /// A procedure or built-in call is present.
    Call,
    /// `list(...)` constructs mutable runtime state.
    ListConstruction,
    /// An identifier may read another variable or constant.
    IdentifierReference,
    /// Multiple operators or values require expression evaluation.
    CompositeExpression,
    /// Syntax is intentionally not classified yet.
    Other,
}

/// One conservative name dependency found in initializer tokens.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializerDependency {
    /// Identifier spelling in first encounter order.
    pub name: String,
}

/// Lossless initializer syntax in the compiler's expanded source view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializerSyntax {
    /// Exact bytes following `=` through the declaration logical line.
    pub text: String,
    /// Tokens after `=`, retaining expanded-source byte spans and spellings.
    pub tokens: Vec<SpannedToken>,
    /// Range in the expanded compiler source.
    pub expanded_span: SourceSpan,
    /// Corresponding range in the original physical source.
    pub original_span: SourceSpan,
    /// Conservative execution classification.
    pub class: InitializerClass,
    /// Identifier dependencies in stable first-use order.
    pub dependencies: Vec<InitializerDependency>,
}

/// Type owning a static or instance variable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariableOwner {
    /// Object-tree identity of the owner.
    pub node: NodeId,
    /// Canonical owner path.
    pub path: String,
}

/// One variable declaration or override in expanded project order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariableEntry {
    /// Global declaration/override ordinal from [`Compilation::declarations`].
    pub ordinal: usize,
    /// Canonical variable node.
    pub node: NodeId,
    /// Canonical variable path.
    pub path: String,
    /// Effective storage lifetime.
    pub storage: StorageClass,
    /// Declaration versus override assignment.
    pub assignment: AssignmentKind,
    /// Owning type, absent for global variables.
    pub owner: Option<VariableOwner>,
    /// Physical source file.
    pub file_id: FileId,
    /// Index into that file's syntax definitions.
    pub definition_index: usize,
    /// Original source range of the declaration header.
    pub span: SourceSpan,
    /// Initializer syntax, when `=` was present.
    pub initializer: Option<InitializerSyntax>,
}

/// Deterministic variable and initialization inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariableRegistry {
    entries: Vec<VariableEntry>,
}

impl VariableRegistry {
    /// Builds a registry without evaluating any initializer.
    #[must_use]
    pub fn build(compilation: &Compilation) -> Self {
        let declared_storage = declared_storage(compilation);
        let entries = compilation
            .declarations()
            .iter()
            .filter_map(|declaration| {
                let syntax = compilation.syntax(declaration.file_id)?;
                let definition = syntax.definitions.get(declaration.definition_index)?;
                if !matches!(
                    definition.kind,
                    DefinitionKind::Variable | DefinitionKind::VariableOverride
                ) {
                    return None;
                }
                let tree_node = compilation.code_tree().node(declaration.node)?;
                let storage = declared_storage
                    .get(&declaration.node)
                    .copied()
                    .unwrap_or_else(|| {
                        classify_storage(definition, tree_node.owner_type.is_none())
                    });
                let owner = tree_node.owner_type.and_then(|owner| {
                    compilation
                        .code_tree()
                        .node(owner)
                        .map(|node| VariableOwner {
                            node: owner,
                            path: node.path.to_string(),
                        })
                });
                Some(VariableEntry {
                    ordinal: declaration.ordinal,
                    node: declaration.node,
                    path: tree_node.path.to_string(),
                    storage,
                    assignment: if definition.kind == DefinitionKind::VariableOverride {
                        AssignmentKind::Override
                    } else {
                        AssignmentKind::Declaration
                    },
                    owner,
                    file_id: declaration.file_id,
                    definition_index: declaration.definition_index,
                    span: declaration.span,
                    initializer: initializer(compilation, declaration.file_id, definition),
                })
            })
            .collect();
        Self { entries }
    }

    /// Returns entries in exact expanded declaration order.
    #[must_use]
    pub fn entries(&self) -> &[VariableEntry] {
        &self.entries
    }

    /// Returns deterministic counts by storage lifetime.
    #[must_use]
    pub fn storage_counts(&self) -> BTreeMap<StorageClass, usize> {
        let mut counts = BTreeMap::new();
        for entry in &self.entries {
            *counts.entry(entry.storage).or_default() += 1;
        }
        counts
    }

    /// Returns deterministic counts of runtime-blocking initializer shapes.
    #[must_use]
    pub fn runtime_blocker_counts(&self) -> BTreeMap<RuntimeBlocker, usize> {
        let mut counts = BTreeMap::new();
        for initializer in self
            .entries
            .iter()
            .filter_map(|entry| entry.initializer.as_ref())
        {
            if let InitializerClass::RequiresRuntime(blocker) = initializer.class {
                *counts.entry(blocker).or_default() += 1;
            }
        }
        counts
    }
}

fn declared_storage(compilation: &Compilation) -> HashMap<NodeId, StorageClass> {
    let mut storage = HashMap::new();
    for declaration in compilation.declarations() {
        let Some(definition) = compilation
            .syntax(declaration.file_id)
            .and_then(|syntax| syntax.definitions.get(declaration.definition_index))
        else {
            continue;
        };
        if definition.kind != DefinitionKind::Variable {
            continue;
        }
        let global_owner = compilation
            .code_tree()
            .node(declaration.node)
            .is_none_or(|node| node.owner_type.is_none());
        storage
            .entry(declaration.node)
            .or_insert_with(|| classify_storage(definition, global_owner));
    }
    storage
}

fn classify_storage(definition: &Definition, has_no_owner: bool) -> StorageClass {
    let identifiers: BTreeSet<_> = definition
        .header
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(name) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    if has_no_owner || identifiers.contains("global") {
        StorageClass::Global
    } else if identifiers.contains("static") {
        StorageClass::Static
    } else {
        StorageClass::Instance
    }
}

fn initializer(
    compilation: &Compilation,
    file_id: FileId,
    definition: &Definition,
) -> Option<InitializerSyntax> {
    let equals = definition.header.iter().position(
        |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
    )?;
    let equals_token = &definition.header[equals];
    let expanded_span = SourceSpan::new(equals_token.span.end, definition.span.end);
    let file = compilation.project().file(file_id)?;
    let source = file.compiler_text().ok()?;
    let text = source
        .get(expanded_span.start..expanded_span.end)?
        .to_owned();
    let tokens = definition.header[equals + 1..].to_vec();
    let class = classify_initializer(&tokens);
    Some(InitializerSyntax {
        text,
        tokens: tokens.clone(),
        expanded_span,
        original_span: file.original_span(expanded_span),
        class,
        dependencies: initializer_dependencies(&tokens),
    })
}

fn classify_initializer(tokens: &[SpannedToken]) -> InitializerClass {
    if tokens.is_empty() {
        return InitializerClass::RequiresRuntime(RuntimeBlocker::Other);
    }
    if tokens.len() == 1 {
        return match &tokens[0].kind {
            TokenKind::Number(_)
            | TokenKind::String(_)
            | TokenKind::RawString(_)
            | TokenKind::TextBlock(_)
            | TokenKind::Resource(_) => InitializerClass::ConstantSafe,
            TokenKind::Identifier(name) if name == "null" => InitializerClass::ConstantSafe,
            TokenKind::Identifier(_) => {
                InitializerClass::RequiresRuntime(RuntimeBlocker::IdentifierReference)
            }
            _ => InitializerClass::RequiresRuntime(RuntimeBlocker::Other),
        };
    }
    if is_absolute_path(tokens) {
        return InitializerClass::ConstantSafe;
    }
    if tokens
        .iter()
        .any(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "new"))
    {
        return InitializerClass::RequiresRuntime(RuntimeBlocker::NewExpression);
    }
    if tokens.windows(2).any(|pair| {
        matches!(&pair[0].kind, TokenKind::Identifier(name) if name == "list")
            && pair[1].kind == TokenKind::Punctuation('(')
    }) {
        return InitializerClass::RequiresRuntime(RuntimeBlocker::ListConstruction);
    }
    if tokens.windows(2).any(|pair| {
        matches!(pair[0].kind, TokenKind::Identifier(_))
            && pair[1].kind == TokenKind::Punctuation('(')
    }) {
        return InitializerClass::RequiresRuntime(RuntimeBlocker::Call);
    }
    if tokens
        .iter()
        .any(|token| matches!(token.kind, TokenKind::Identifier(_)))
    {
        return InitializerClass::RequiresRuntime(RuntimeBlocker::IdentifierReference);
    }
    InitializerClass::RequiresRuntime(RuntimeBlocker::CompositeExpression)
}

fn is_absolute_path(tokens: &[SpannedToken]) -> bool {
    matches!(
        tokens.first().map(|token| &token.kind),
        Some(TokenKind::Operator(operator)) if operator == "/"
    ) && tokens.iter().all(|token| {
        matches!(token.kind, TokenKind::Identifier(_))
            || matches!(&token.kind, TokenKind::Operator(operator) if operator == "/")
    })
}

fn initializer_dependencies(tokens: &[SpannedToken]) -> Vec<InitializerDependency> {
    let ignored = ["new", "null", "TRUE", "FALSE", "list"];
    let mut seen = BTreeSet::new();
    let mut dependencies = Vec::new();
    for name in tokens.iter().filter_map(|token| match &token.kind {
        TokenKind::Identifier(name) => Some(name),
        _ => None,
    }) {
        if ignored.contains(&name.as_str()) || !seen.insert(name.clone()) {
            continue;
        }
        dependencies.push(InitializerDependency { name: name.clone() });
    }
    dependencies
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::CompilerDatabase;

    use super::{AssignmentKind, InitializerClass, RuntimeBlocker, StorageClass, VariableRegistry};

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "dream64-dm-globals-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("fixture directory should be created");
            Self(path)
        }

        fn write(&self, name: &str, source: &str) {
            fs::write(self.0.join(name), source).expect("fixture source should be written");
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("fixture directory should be removed");
        }
    }

    #[test]
    fn inventories_global_static_instance_and_override_order() {
        let fixture = Fixture::new();
        fixture.write(
            "world.dme",
            "#include \"first.dm\"\n#include \"second.dm\"\n",
        );
        fixture.write(
            "first.dm",
            "var/global/root = 1\nvar/plain = \"plain\"\n/datum/example\n\tvar/instance = new /datum\n\tvar/static/shared = 2\n\tinstance = dependency + 1\n",
        );
        fixture.write(
            "second.dm",
            "/datum/example\n\tshared = other_dependency\n\tvar/other = /datum/example\n",
        );
        let compilation = CompilerDatabase::new()
            .compile(fixture.0.join("world.dme"))
            .expect("fixture should compile");
        let registry = VariableRegistry::build(&compilation);
        let entries = registry.entries();

        assert_eq!(entries.len(), 7);
        assert_eq!(entries[0].storage, StorageClass::Global);
        assert_eq!(entries[1].storage, StorageClass::Global);
        assert_eq!(entries[2].storage, StorageClass::Instance);
        assert_eq!(entries[3].storage, StorageClass::Static);
        assert_eq!(entries[4].assignment, AssignmentKind::Override);
        assert_eq!(entries[4].storage, StorageClass::Instance);
        assert_eq!(entries[5].assignment, AssignmentKind::Override);
        assert_eq!(entries[5].storage, StorageClass::Static);
        assert_eq!(entries[6].storage, StorageClass::Instance);
        assert!(
            entries
                .windows(2)
                .all(|pair| pair[0].ordinal < pair[1].ordinal)
        );
        assert_eq!(
            entries[0].initializer.as_ref().unwrap().class,
            InitializerClass::ConstantSafe
        );
        assert_eq!(
            entries[2].initializer.as_ref().unwrap().class,
            InitializerClass::RequiresRuntime(RuntimeBlocker::NewExpression)
        );
        assert_eq!(
            entries[5].initializer.as_ref().unwrap().class,
            InitializerClass::RequiresRuntime(RuntimeBlocker::IdentifierReference)
        );
        assert_eq!(
            entries[6].initializer.as_ref().unwrap().class,
            InitializerClass::ConstantSafe
        );
        assert_eq!(
            entries[4].initializer.as_ref().unwrap().dependencies[0].name,
            "dependency"
        );
    }

    #[test]
    fn retains_exact_initializer_text_and_source_identity() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write("vars.dm", "var/global/value = other + 2\n");
        let compilation = CompilerDatabase::new()
            .compile(fixture.0.join("world.dme"))
            .expect("fixture should compile");
        let registry = VariableRegistry::build(&compilation);
        let initializer = registry.entries()[0].initializer.as_ref().unwrap();

        assert_eq!(initializer.text, " other + 2\n");
        assert_eq!(initializer.tokens.len(), 3);
        assert!(initializer.original_span.start < initializer.original_span.end);
        assert_eq!(
            initializer.class,
            InitializerClass::RequiresRuntime(RuntimeBlocker::IdentifierReference)
        );
    }
}

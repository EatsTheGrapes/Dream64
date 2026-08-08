//! Deterministic inventory of DM variable declarations and initialization work.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use dm_compiler::Compilation;
use dm_core::{FileId, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};
use dm_object_tree::NodeId;
use dm_syntax::{Definition, DefinitionKind, DefinitionPath};

mod constant;

pub use constant::{
    ConstantEvaluation, ConstantListEntry, ConstantValue, ConstantValueShape, UnsupportedCategory,
    UnsupportedConstant, evaluate_constant,
};

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

/// Semantic modifiers retained from a variable's introducing declaration.
///
/// Overrides inherit these flags from the original declaration. This mirrors
/// DM's rule that assigning a subtype default does not redeclare the field.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VariableModifiers {
    /// The field cannot be assigned after its constant initializer is resolved.
    pub constant: bool,
    /// The field is excluded from savefile persistence.
    pub temporary: bool,
}

/// Conservative initialization classification.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InitializerClass {
    /// The typed evaluator proved that no runtime state is read.
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
    /// Typed constant value or the precise reason evaluation stopped.
    pub evaluation: ConstantEvaluation,
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
    /// Effective declaration modifiers, inherited by override entries.
    pub modifiers: VariableModifiers,
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

/// One explicit initializer assignment in an execution plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializationStep {
    /// Index of the corresponding entry in [`VariableRegistry::entries`].
    pub entry_index: usize,
    /// Expanded declaration ordinal used to preserve override order.
    pub ordinal: usize,
    /// Object-tree identity of the variable node.
    pub node: NodeId,
    /// Canonical variable path.
    pub path: String,
    /// Declaration versus later override assignment.
    pub assignment: AssignmentKind,
    /// Storage lifetime receiving the value.
    pub storage: StorageClass,
    /// Constant value or an explicit runtime-evaluation requirement.
    pub evaluation: ConstantEvaluation,
}

/// Ordered instance-default assignments belonging to one object type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeDefaultPlan {
    /// Type receiving the default assignments.
    pub owner: VariableOwner,
    /// Assignments in expanded declaration and override order.
    pub steps: Vec<InitializationStep>,
}

/// Deterministic explicit-initializer plans derived from the registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializationPlans {
    /// Global and type-static assignments in expanded source order.
    pub global_steps: Vec<InitializationStep>,
    /// Instance-default plans in owner first-encounter order.
    pub type_defaults: Vec<TypeDefaultPlan>,
}

impl VariableRegistry {
    /// Builds a registry without evaluating any initializer.
    #[must_use]
    pub fn build(compilation: &Compilation) -> Self {
        let declared_storage = declared_storage(compilation);
        let declared_modifiers = declared_modifiers(compilation);
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
                let initializer =
                    initializer(compilation, declaration.file_id, definition).map(|initializer| {
                        normalize_initializer_paths(compilation, owner.as_ref(), initializer)
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
                    modifiers: effective_modifiers(
                        compilation,
                        &declared_modifiers,
                        declaration.node,
                    )
                    .unwrap_or_else(|| classify_modifiers(definition)),
                    owner,
                    file_id: declaration.file_id,
                    definition_index: declaration.definition_index,
                    span: declaration.span,
                    initializer,
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

    /// Returns deterministic top-level shapes for proven constant values.
    #[must_use]
    pub fn constant_value_counts(&self) -> BTreeMap<ConstantValueShape, usize> {
        let mut counts = BTreeMap::new();
        for evaluation in self.entries.iter().filter_map(|entry| {
            entry
                .initializer
                .as_ref()
                .map(|initializer| &initializer.evaluation)
        }) {
            if let ConstantEvaluation::Value(value) = evaluation {
                *counts.entry(value.shape()).or_default() += 1;
            }
        }
        counts
    }

    /// Returns deterministic counts for exact constant-evaluation blockers.
    #[must_use]
    pub fn unsupported_constant_counts(&self) -> BTreeMap<UnsupportedCategory, usize> {
        let mut counts = BTreeMap::new();
        for evaluation in self.entries.iter().filter_map(|entry| {
            entry
                .initializer
                .as_ref()
                .map(|initializer| &initializer.evaluation)
        }) {
            if let ConstantEvaluation::Unsupported(unsupported) = evaluation {
                *counts.entry(unsupported.category).or_default() += 1;
            }
        }
        counts
    }

    /// Builds ordered plans for every explicit initializer.
    ///
    /// Global and static values execute once in project order. Instance values
    /// become per-owner default assignments, with reopenings and overrides
    /// appended to the same owner plan in exact source order.
    #[must_use]
    pub fn initialization_plans(&self) -> InitializationPlans {
        let mut global_steps = Vec::new();
        let mut type_defaults = Vec::<TypeDefaultPlan>::new();
        let mut owner_plans = HashMap::<NodeId, usize>::new();
        for (entry_index, entry) in self.entries.iter().enumerate() {
            let Some(initializer) = &entry.initializer else {
                continue;
            };
            let step = InitializationStep {
                entry_index,
                ordinal: entry.ordinal,
                node: entry.node,
                path: entry.path.clone(),
                assignment: entry.assignment,
                storage: entry.storage,
                evaluation: initializer.evaluation.clone(),
            };
            if entry.storage != StorageClass::Instance {
                global_steps.push(step);
                continue;
            }
            let Some(owner) = &entry.owner else {
                global_steps.push(step);
                continue;
            };
            let plan_index = *owner_plans.entry(owner.node).or_insert_with(|| {
                let index = type_defaults.len();
                type_defaults.push(TypeDefaultPlan {
                    owner: owner.clone(),
                    steps: Vec::new(),
                });
                index
            });
            type_defaults[plan_index].steps.push(step);
        }
        InitializationPlans {
            global_steps,
            type_defaults,
        }
    }
}

fn normalize_initializer_paths(
    compilation: &Compilation,
    owner: Option<&VariableOwner>,
    mut initializer: InitializerSyntax,
) -> InitializerSyntax {
    let anchor = owner.map(|owner| {
        DefinitionPath::new(
            owner
                .path
                .split('/')
                .filter(|segment| !segment.is_empty())
                .map(str::to_owned)
                .collect(),
        )
    });
    initializer.tokens = compilation
        .code_tree()
        .normalize_upward_paths(anchor.as_ref(), &initializer.tokens);
    initializer.evaluation = evaluate_constant(&initializer.tokens);
    initializer.class = classify_evaluation(&initializer.evaluation);
    initializer.dependencies = initializer_dependencies(&initializer.tokens);
    initializer
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

fn declared_modifiers(compilation: &Compilation) -> HashMap<NodeId, VariableModifiers> {
    let mut modifiers = HashMap::new();
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
        modifiers
            .entry(declaration.node)
            .or_insert_with(|| classify_modifiers(definition));
    }
    modifiers
}

fn effective_modifiers(
    compilation: &Compilation,
    declared: &HashMap<NodeId, VariableModifiers>,
    mut node: NodeId,
) -> Option<VariableModifiers> {
    loop {
        if let Some(modifiers) = declared.get(&node) {
            return Some(*modifiers);
        }
        node = compilation.code_tree().node(node)?.inherited_member?;
    }
}

fn classify_modifiers(definition: &Definition) -> VariableModifiers {
    let identifiers: BTreeSet<_> = definition
        .header
        .iter()
        .take_while(
            |token| !matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
        )
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(name) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    VariableModifiers {
        constant: identifiers.contains("const"),
        temporary: identifiers.contains("tmp"),
    }
}

fn classify_storage(definition: &Definition, has_no_owner: bool) -> StorageClass {
    let identifiers: BTreeSet<_> = definition
        .header
        .iter()
        .take_while(
            |token| !matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
        )
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
    let evaluation = evaluate_constant(&tokens);
    let class = classify_evaluation(&evaluation);
    Some(InitializerSyntax {
        text,
        tokens: tokens.clone(),
        expanded_span,
        original_span: file.original_span(expanded_span),
        class,
        evaluation,
        dependencies: initializer_dependencies(&tokens),
    })
}

const fn classify_evaluation(evaluation: &ConstantEvaluation) -> InitializerClass {
    match evaluation {
        ConstantEvaluation::Value(_) => InitializerClass::ConstantSafe,
        ConstantEvaluation::Unsupported(unsupported) => {
            InitializerClass::RequiresRuntime(match unsupported.category {
                UnsupportedCategory::Identifier => RuntimeBlocker::IdentifierReference,
                UnsupportedCategory::Call => RuntimeBlocker::Call,
                UnsupportedCategory::NewExpression => RuntimeBlocker::NewExpression,
                UnsupportedCategory::UnsupportedOperator
                | UnsupportedCategory::TypeMismatch
                | UnsupportedCategory::DynamicExpression => RuntimeBlocker::CompositeExpression,
                UnsupportedCategory::EmptyExpression
                | UnsupportedCategory::DynamicText
                | UnsupportedCategory::ResourceLiteral
                | UnsupportedCategory::InvalidSyntax
                | UnsupportedCategory::InvalidNumber => RuntimeBlocker::Other,
            })
        }
    }
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
    use dm_lexer::{TokenKind, lex};

    use super::{
        AssignmentKind, ConstantEvaluation, ConstantListEntry, ConstantValue, InitializerClass,
        RuntimeBlocker, StorageClass, UnsupportedCategory, VariableRegistry, evaluate_constant,
    };

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

    fn evaluate(source: &str) -> ConstantEvaluation {
        let tokens: Vec<_> = lex(source)
            .expect("expression should lex")
            .into_iter()
            .filter(|token| !matches!(token.kind, TokenKind::LineStart { .. } | TokenKind::Newline))
            .collect();
        evaluate_constant(&tokens)
    }

    fn evaluate_number(source: &str) -> u32 {
        let ConstantEvaluation::Value(ConstantValue::Number(number)) = evaluate(source) else {
            panic!("{source:?} should evaluate to a number");
        };
        number.bits()
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
    fn retains_const_and_tmp_modifiers_across_subtype_overrides() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "/datum/base\n\tvar/const/answer = 42\n\tvar/tmp/transient = 1\n/datum/base/child\n\tanswer = 42\n\ttransient = 2\n\tvar/plain = global.answer\n",
        );
        let compilation = CompilerDatabase::new()
            .compile(fixture.0.join("world.dme"))
            .expect("fixture should compile");
        let registry = VariableRegistry::build(&compilation);
        let entries = registry.entries();

        assert_eq!(entries.len(), 5);
        assert!(entries[0].modifiers.constant);
        assert!(!entries[0].modifiers.temporary);
        assert!(!entries[1].modifiers.constant);
        assert!(entries[1].modifiers.temporary);
        assert_eq!(entries[2].assignment, AssignmentKind::Override);
        assert!(entries[2].modifiers.constant);
        assert_eq!(entries[3].assignment, AssignmentKind::Override);
        assert!(entries[3].modifiers.temporary);
        assert_eq!(entries[4].modifiers, Default::default());
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

    #[test]
    fn evaluates_binary32_truth_paths_and_ordered_lists() {
        let evaluation =
            evaluate(r#"list(-(1 + 2 * 3), "text", /datum/example, "answer" = !null && TRUE)"#);
        let ConstantEvaluation::Value(ConstantValue::List(entries)) = evaluation else {
            panic!("initializer should evaluate to a constant list");
        };

        assert_eq!(entries.len(), 4);
        let ConstantListEntry::Positional(ConstantValue::Number(number)) = &entries[0] else {
            panic!("first entry should be numeric");
        };
        assert_eq!(number.bits(), (-7.0_f32).to_bits());
        assert_eq!(
            entries[1],
            ConstantListEntry::Positional(ConstantValue::Text("text".to_owned()))
        );
        assert_eq!(
            entries[2],
            ConstantListEntry::Positional(ConstantValue::TypePath("/datum/example".to_owned()))
        );
        let ConstantListEntry::Associative { key, value } = &entries[3] else {
            panic!("last entry should be associative");
        };
        assert_eq!(key, &ConstantValue::Text("answer".to_owned()));
        let ConstantValue::Number(value) = value else {
            panic!("truth expression should produce a number");
        };
        assert_eq!(value.bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn applies_binary32_rounding_and_dm_truth_rules() {
        assert_eq!(evaluate_number("16777216 + 1"), 16_777_216.0_f32.to_bits());
        assert_eq!(evaluate_number("!null"), 1.0_f32.to_bits());
        assert_eq!(evaluate_number("!\"\""), 1.0_f32.to_bits());
        assert_eq!(evaluate_number("!/datum/example"), 0.0_f32.to_bits());
        assert_eq!(evaluate_number("!list()"), 0.0_f32.to_bits());
    }

    #[test]
    fn reports_precise_unsupported_categories_and_spans() {
        let cases = [
            ("dependency", UnsupportedCategory::Identifier),
            ("build_value()", UnsupportedCategory::Call),
            ("new /datum/example", UnsupportedCategory::NewExpression),
            (r#""value [world.time]""#, UnsupportedCategory::DynamicText),
            ("1 ** 2", UnsupportedCategory::UnsupportedOperator),
        ];

        for (source, expected) in cases {
            let ConstantEvaluation::Unsupported(unsupported) = evaluate(source) else {
                panic!("{source:?} should require runtime evaluation");
            };
            assert_eq!(unsupported.category, expected, "source: {source}");
            assert!(!unsupported.span.is_empty(), "source: {source}");
            assert!(unsupported.span.end <= source.len(), "source: {source}");
        }
    }

    #[test]
    fn builds_source_ordered_global_and_type_default_plans() {
        let fixture = Fixture::new();
        fixture.write(
            "world.dme",
            "#include \"first.dm\"\n#include \"second.dm\"\n",
        );
        fixture.write(
            "first.dm",
            "var/global/root = 1\n/datum/example\n\tvar/instance = 2\n\tvar/static/shared = 3\n",
        );
        fixture.write(
            "second.dm",
            "root = 4\n/datum/example\n\tinstance = 5\n\tshared = 6\n",
        );
        let compilation = CompilerDatabase::new()
            .compile(fixture.0.join("world.dme"))
            .expect("fixture should compile");
        let registry = VariableRegistry::build(&compilation);
        let plans = registry.initialization_plans();

        assert_eq!(plans.global_steps.len(), 4);
        assert_eq!(plans.type_defaults.len(), 1);
        assert_eq!(plans.type_defaults[0].owner.path, "/datum/example");
        assert_eq!(plans.type_defaults[0].steps.len(), 2);
        assert!(
            plans
                .global_steps
                .windows(2)
                .all(|steps| steps[0].ordinal < steps[1].ordinal)
        );
        assert!(
            plans.type_defaults[0]
                .steps
                .windows(2)
                .all(|steps| steps[0].ordinal < steps[1].ordinal)
        );
        assert_eq!(plans.global_steps[2].assignment, AssignmentKind::Override);
        assert_eq!(
            plans.type_defaults[0].steps[1].assignment,
            AssignmentKind::Override
        );
    }

    #[test]
    fn storage_modifiers_do_not_scan_initializer_expressions() {
        let fixture = Fixture::new();
        fixture.write("world.dme", "#include \"vars.dm\"\n");
        fixture.write(
            "vars.dm",
            "/datum/example\n\tvar/value = global.offset\n\tvar/static/shared = 2\n",
        );
        let compilation = CompilerDatabase::new()
            .compile(fixture.0.join("world.dme"))
            .expect("fixture should compile");
        let registry = VariableRegistry::build(&compilation);

        assert_eq!(registry.entries()[0].storage, StorageClass::Instance);
        assert_eq!(registry.entries()[1].storage, StorageClass::Static);
    }
}

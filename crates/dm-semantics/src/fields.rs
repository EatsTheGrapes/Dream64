//! Static and instance field resolution: extracting a type's directly declared
//! `var` / `var/static` slots and their datum types from the variable registry,
//! resolving a referenced name through owner ancestry, and the catalog of
//! engine-owned fields BYOND supplies for built-in datum/atom types.

use std::collections::{BTreeMap, BTreeSet};

use dm_compiler::Compilation;
use dm_globals::{StorageClass, VariableRegistry};
use dm_lexer::{SpannedToken, TokenKind};
use dm_object_tree::{CodePath, NodeId};
use dm_value::{FieldName, TypePath};

use super::declared_type_path;

pub(crate) fn direct_static_fields(
    registry: &VariableRegistry,
) -> BTreeMap<NodeId, BTreeMap<String, FieldName>> {
    let mut fields = BTreeMap::<NodeId, BTreeMap<String, FieldName>>::new();
    for entry in registry
        .entries()
        .iter()
        // DM spells type-owned shared storage both `var/static` and
        // `var/global`.  The registry preserves that spelling as distinct
        // storage classes, but both lower to the same owner-qualified VM slot.
        .filter(|entry| matches!(entry.storage, StorageClass::Static | StorageClass::Global))
    {
        let Some(owner) = entry.owner.as_ref().map(|owner| owner.node) else {
            continue;
        };
        let Some(name) = entry.path.rsplit('/').next() else {
            continue;
        };
        fields
            .entry(owner)
            .or_default()
            .insert(name.to_owned(), FieldName::static_storage(&entry.path));
    }
    fields
}

#[cfg(test)]
fn inherited_static_fields(
    compilation: &Compilation,
    owner: Option<NodeId>,
    direct_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
    cache: &mut BTreeMap<NodeId, BTreeMap<String, FieldName>>,
) -> BTreeMap<String, FieldName> {
    let Some(owner) = owner else {
        return BTreeMap::new();
    };
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
    }
    cache.insert(owner, fields.clone());
    fields
}

pub(crate) fn referenced_inherited_fields(
    compilation: &Compilation,
    owner: Option<NodeId>,
    direct_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
    referenced: &BTreeSet<String>,
    include_standard: bool,
) -> BTreeMap<String, FieldName> {
    let Some(mut current) = owner else {
        return BTreeMap::new();
    };
    let tree = compilation.code_tree();
    let mut unresolved = referenced.clone();
    let mut fields = BTreeMap::new();
    while !unresolved.is_empty() {
        let mut available = direct_fields.get(&current).cloned().unwrap_or_default();
        if include_standard {
            standard_instance_fields(tree.node(current).map(|node| &node.path), &mut available);
        }
        if !available.is_empty() {
            let resolved = unresolved
                .iter()
                .filter_map(|name| {
                    available
                        .get(name)
                        .map(|field| (name.clone(), field.clone()))
                })
                .collect::<Vec<_>>();
            for (name, field) in resolved {
                unresolved.remove(&name);
                fields.insert(name, field);
            }
        }
        let Some(parent) = tree.node(current).and_then(|node| node.parent_type) else {
            break;
        };
        current = parent;
    }
    fields
}

/// Resolves the DM scope-operator form `Type::name` appearing anywhere in a
/// procedure body to the qualified static-storage slot of `name`.
///
/// `Type::name` reads or writes the *shared* `static` / `global` var `name`
/// declared on the literal type path `Type` (or any of its ancestors) through
/// its type-owned slot — the same `__dm_static_<hex>` global that
/// `direct_static_fields` assigns. This is distinct from typed *instance*
/// access (`obj.name`): the receiver here is a compile-time type literal, not a
/// value.
///
/// Every resolved pair is returned keyed `"<canonical type path>::<name>"`. The
/// `::` separator cannot collide with the `"<receiver>.<name>"` keys the
/// instance-static binding pass emits, so both sets of entries can share one
/// procedure's `global_fields` map. A `Type::name` where `name` is a plain
/// (non-shared) instance var resolves to nothing and keeps its `initial()`
/// lowering.
pub(crate) fn scope_operator_static_fields(
    compilation: &Compilation,
    definition: &dm_syntax::Definition,
    direct_static_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
) -> BTreeMap<String, FieldName> {
    let mut resolved = BTreeMap::new();
    collect_scope_operator_statics(
        compilation,
        &definition.header,
        direct_static_fields,
        &mut resolved,
    );
    for line in &definition.body {
        collect_scope_operator_statics(
            compilation,
            &line.tokens,
            direct_static_fields,
            &mut resolved,
        );
    }
    resolved
}

fn collect_scope_operator_statics(
    compilation: &Compilation,
    tokens: &[SpannedToken],
    direct_static_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
    resolved: &mut BTreeMap<String, FieldName>,
) {
    let mut index = 0;
    while index < tokens.len() {
        if !matches!(&tokens[index].kind, TokenKind::Operator(operator) if operator == "/") {
            index += 1;
            continue;
        }
        // A leading `/` only starts a type-path literal when the preceding
        // token cannot itself yield a value; otherwise this is division.
        if index > 0 && token_yields_value(&tokens[index - 1]) {
            index += 1;
            continue;
        }
        let mut cursor = index;
        let mut segments = Vec::new();
        while matches!(tokens.get(cursor).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
        {
            let Some(TokenKind::Identifier(segment)) =
                tokens.get(cursor + 1).map(|token| &token.kind)
            else {
                break;
            };
            segments.push(segment.clone());
            cursor += 2;
        }
        if segments.is_empty() {
            index += 1;
            continue;
        }
        if let (Some(TokenKind::Operator(operator)), Some(TokenKind::Identifier(name))) = (
            tokens.get(cursor).map(|token| &token.kind),
            tokens.get(cursor + 1).map(|token| &token.kind),
        ) && operator == "::"
            && let Some(field) =
                inherited_static_slot(compilation, &segments, name, direct_static_fields)
        {
            resolved.insert(format!("/{}::{name}", segments.join("/")), field);
        }
        index = cursor.max(index + 1);
    }
}

fn token_yields_value(token: &SpannedToken) -> bool {
    matches!(
        &token.kind,
        TokenKind::Identifier(_)
            | TokenKind::Number(_)
            | TokenKind::String(_)
            | TokenKind::RawString(_)
            | TokenKind::TextBlock(_)
            | TokenKind::Resource(_)
            | TokenKind::Punctuation(')' | ']')
    )
}

fn inherited_static_slot(
    compilation: &Compilation,
    segments: &[String],
    name: &str,
    direct_static_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
) -> Option<FieldName> {
    let tree = compilation.code_tree();
    let mut current = Some(tree.find(&dm_syntax::DefinitionPath::new(segments.to_vec()))?);
    while let Some(node) = current {
        if let Some(field) = direct_static_fields
            .get(&node)
            .and_then(|fields| fields.get(name))
        {
            return Some(field.clone());
        }
        current = tree.node(node).and_then(|type_node| type_node.parent_type);
    }
    None
}

pub(crate) fn direct_instance_fields(
    registry: &VariableRegistry,
) -> BTreeMap<NodeId, BTreeMap<String, FieldName>> {
    let mut fields = BTreeMap::<NodeId, BTreeMap<String, FieldName>>::new();
    for entry in registry
        .entries()
        .iter()
        .filter(|entry| entry.storage == StorageClass::Instance)
    {
        let Some(owner) = entry.owner.as_ref().map(|owner| owner.node) else {
            continue;
        };
        let Some(name) = entry.path.rsplit('/').next() else {
            continue;
        };
        if let Ok(field) = FieldName::parse(name) {
            fields
                .entry(owner)
                .or_default()
                .insert(name.to_owned(), field);
        }
    }
    fields
}

pub(crate) fn direct_instance_field_types(
    compilation: &Compilation,
    registry: &VariableRegistry,
) -> BTreeMap<NodeId, BTreeMap<String, TypePath>> {
    let mut types = BTreeMap::<NodeId, BTreeMap<String, TypePath>>::new();
    for entry in registry
        .entries()
        .iter()
        .filter(|entry| entry.storage == StorageClass::Instance)
    {
        let Some(owner) = entry.owner.as_ref().map(|owner| owner.node) else {
            continue;
        };
        let Some(name) = entry.path.rsplit('/').next() else {
            continue;
        };
        let Some(definition) = compilation
            .syntax(entry.file_id)
            .and_then(|syntax| syntax.definitions.get(entry.definition_index))
        else {
            continue;
        };
        let Some(path) = declared_type_path(&definition.header, name) else {
            continue;
        };
        let Ok(path) = TypePath::parse(&path.to_string()) else {
            continue;
        };
        types
            .entry(owner)
            .or_default()
            .insert(name.to_owned(), path);
    }
    types
}

pub(crate) fn referenced_inherited_field_types(
    compilation: &Compilation,
    owner: NodeId,
    direct_types: &BTreeMap<NodeId, BTreeMap<String, TypePath>>,
    referenced: &BTreeSet<String>,
) -> BTreeMap<String, TypePath> {
    let tree = compilation.code_tree();
    let mut current = Some(owner);
    let mut unresolved = referenced.clone();
    let mut types = BTreeMap::new();
    while let Some(node) = current {
        if let Some(available) = direct_types.get(&node) {
            let resolved = unresolved
                .iter()
                .filter_map(|name| available.get(name).map(|path| (name.clone(), path.clone())))
                .collect::<Vec<_>>();
            for (name, path) in resolved {
                unresolved.remove(&name);
                types.insert(name, path);
            }
        }
        if unresolved.is_empty() {
            break;
        }
        current = tree.node(node).and_then(|type_node| type_node.parent_type);
    }
    types
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
    let path = path.to_string();
    let names = standard_instance_field_names(&path);
    if names.is_empty() {
        return;
    }
    for name in names {
        // All catalog entries are fixed, valid DM identifiers.
        fields.insert(
            (*name).to_owned(),
            FieldName::parse(name).expect("standard field name is valid"),
        );
    }
}

/// Returns the engine-owned fields declared directly by a built-in DM type.
///
/// This is public so runtime materialization tests can enforce that the
/// semantic catalog and concrete datum defaults never drift apart.
#[doc(hidden)]
#[must_use]
pub fn standard_instance_field_names(path: &str) -> &'static [&'static str] {
    match path {
        // Every datum exposes its canonical runtime type through this
        // read-only built-in field. The VM materializes its value from the
        // datum record rather than from a user-declared default.
        "/datum" => &["datum_flags", "tag", "type", "parent_type", "vars"],
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
            "params",
            "log",
            "name",
            "hub",
            "hub_password",
            "internet_address",
            "address",
            "status",
            "port",
            "area",
            "mob",
            "turf",
            "byond_version",
            "byond_build",
            "cache_lifespan",
            "executor",
            "game_state",
            "host",
            "loop_checks",
            "map_format",
            "map_cpu",
            "movement_mode",
            "process",
            "reachable",
            "sleep_offline",
            "tick_usage",
            "url",
            "version",
            "view",
            "visibility",
        ],
        "/atom" => &[
            "alpha",
            "appearance",
            "appearance_flags",
            "blend_mode",
            "color",
            "contents",
            "density",
            "desc",
            "dir",
            "gender",
            "filters",
            "icon",
            "icon_state",
            "invisibility",
            "layer",
            "loc",
            "luminosity",
            "maptext",
            "maptext_height",
            "maptext_width",
            "maptext_x",
            "maptext_y",
            "mouse_opacity",
            "mouse_over_pointer",
            "name",
            "opacity",
            "overlays",
            "particles",
            "plane",
            "pixel_x",
            "pixel_y",
            "pixel_w",
            "pixel_z",
            "render_source",
            "render_target",
            "suffix",
            "text",
            "transform",
            "underlays",
            "vis_contents",
            "vis_locs",
            "vis_flags",
            "verbs",
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
            "locs",
            "screen_loc",
            "step_x",
            "step_y",
            "step_size",
        ],
        "/mob" => &[
            "ckey",
            "client",
            "eye",
            "key",
            "perspective",
            "see_in_dark",
            "see_infrared",
            "see_invisible",
            "sight",
        ],
        "/client" => &[
            "address",
            "ckey",
            "computer_id",
            "connection",
            "control_freak",
            "dir",
            "gender",
            "byond_build",
            "byond_version",
            "key",
            "eye",
            "fps",
            "images",
            "inactivity",
            "mob",
            "mouse_pointer_icon",
            "perspective",
            "pixel_w",
            "pixel_x",
            "pixel_y",
            "pixel_z",
            "screen",
            "statobj",
            "verbs",
            "view",
        ],
        "/matrix" => &["a", "b", "c", "d", "e", "f"],
        // BYOND exposes the state of the most recent regex operation as
        // ordinary fields. Map readers in tg-derived projects use `next`
        // directly to advance a global regex sweep.
        "/regex" => &["text", "flags", "match", "index", "group", "next"],
        // `/sound` is an engine value with fields supplied by BYOND even when
        // no project declaration exists. OpenDream exposes the core fields
        // through DreamObjectSound; BYOND also exposes constructor controls.
        "/sound" => &[
            "file",
            "repeat",
            "wait",
            "channel",
            "volume",
            "frequency",
            "pan",
            "offset",
        ],
        "/particles" => &[
            "color",
            "width",
            "height",
            "count",
            "spawning",
            "bound1",
            "bound2",
            "gravity",
            "gradient",
            "color_change",
            "transform",
            "icon",
            "icon_state",
            "lifespan",
            "fadein",
            "fade",
            "position",
            "velocity",
            "scale",
            "grow",
            "rotation",
            "spin",
            "friction",
            "drift",
        ],
        // `/image` is an engine-owned appearance datum rather than an atom,
        // but BYOND exposes the same mutable appearance surface used by
        // overlays. These fields exist without user declarations.
        "/image" => &[
            "alpha",
            "appearance",
            "appearance_flags",
            "blend_mode",
            "color",
            "dir",
            "icon",
            "icon_state",
            "layer",
            "loc",
            "name",
            "overlays",
            "plane",
            "pixel_x",
            "pixel_y",
            "pixel_w",
            "pixel_z",
            "transform",
            "underlays",
            "vis_contents",
        ],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::{Compilation, CompilerDatabase};
    use dm_globals::VariableRegistry;

    use super::{direct_static_fields, inherited_static_fields};
    use crate::{Procedure, ProcedureRegistry};

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct TestProject {
        root: std::path::PathBuf,
    }

    impl TestProject {
        fn compile(source: &str) -> Compilation {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "dream64-dm-semantics-fields-{}-{}",
                ordinal,
                std::process::id()
            ));
            std::fs::create_dir(&root).expect("test project directory should be created");
            let project = Self { root };
            std::fs::write(project.root.join("world.dme"), "#include \"types.dm\"\n")
                .expect("environment should be written");
            std::fs::write(project.root.join("types.dm"), source)
                .expect("source should be written");
            CompilerDatabase::new()
                .compile(project.root.join("world.dme"))
                .expect("test project should compile")
        }
    }

    impl Drop for TestProject {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
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

    #[test]
    fn indexed_static_fields_preserve_inheritance_shadowing_and_build_once() {
        let compilation = TestProject::compile(
            "/datum/base\n\tvar/static/shared = 1\n\tproc/read_base()\n\t\treturn shared\n/datum/child\n\tparent_type = /datum/base\n\tvar/static/shared = 2\n\tproc/read_child()\n\t\treturn shared\n/datum/reader/proc/read_receiver(datum/base/value)\n\treturn value.shared\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let base = procedure_by_path(&registry, "/datum/base/proc/read_base");
        let child = procedure_by_path(&registry, "/datum/child/proc/read_child");
        let receiver = procedure_by_path(&registry, "/datum/reader/proc/read_receiver");
        let targets = [base, child, receiver]
            .into_iter()
            .map(|procedure| procedure.effective_target.expect("procedure body"));
        let executable = registry
            .compile_vm_implementations(&compilation, targets)
            .expect("inherited and receiver statics should compile");

        assert_eq!(executable.stats().static_registry_builds, 1);
        assert_eq!(
            executable.stats().global_field_bindings,
            5,
            "typed slash-parameter receiver contributes its qualified static binding",
        );

        let variables = VariableRegistry::build(&compilation);
        let direct = direct_static_fields(&variables);
        let mut cache = BTreeMap::new();
        let child_node = compilation
            .code_tree()
            .find(&dm_syntax::DefinitionPath::new(vec![
                "datum".to_owned(),
                "child".to_owned(),
            ]))
            .expect("child type");
        let inherited =
            inherited_static_fields(&compilation, Some(child_node), &direct, &mut cache);
        assert_eq!(
            inherited.get("shared"),
            Some(&dm_value::FieldName::static_storage(
                "/datum/child/var/shared"
            )),
            "child static must shadow its inherited static"
        );
    }

    #[test]
    fn owner_static_binds_in_new_assignment_and_increment() {
        let compilation = TestProject::compile(
            "/datum/conversation\n\tvar/static/uid = 0\n\tvar/id\n/datum/conversation/New()\n\tid = uid\n\tuid++\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let variables = VariableRegistry::build(&compilation);
        let direct = direct_static_fields(&variables);
        assert!(
            direct.values().any(|fields| fields.contains_key("uid")),
            "registry entries: {:?}",
            variables.entries()
        );
        let new = procedure_by_path(&registry, "/datum/conversation/proc/New");
        let mut cache = BTreeMap::new();
        let inherited = inherited_static_fields(&compilation, new.owner_type, &direct, &mut cache);
        assert!(
            inherited.contains_key("uid"),
            "owner={:?} inherited={inherited:?}",
            new.owner_type
        );
        registry
            .compile_vm_implementations(&compilation, [new.effective_target.expect("New body")])
            .expect("owner static should bind as a qualified global slot");
    }
}

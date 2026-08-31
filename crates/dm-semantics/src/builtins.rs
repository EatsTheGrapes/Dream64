//! Engine-provided procedure sources and compiler-intrinsic predicates.
//!
//! These blobs are parsed as ordinary DM and appended to every compiled module
//! so headless projects retain callable engine hooks (`istype`, `New`, `Del`,
//! `world.Profile`, ...). `compiler_type_predicate` names the spellings the
//! bytecode compiler lowers to a single instruction instead of a builtin call.

use std::collections::BTreeMap;

use dm_compiler::Compilation;
use dm_object_tree::{CodePath, NodeId};

use super::type_is_descendant_or_same;

pub(crate) const STANDARD_BUILTINS: &str = concat!(
    // Headless servers have no authenticated BYOND membership session.
    // Preserve the engine query as a callable predicate with a false result.
    "/client/proc/IsByondMember()\n\treturn 0\n",
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

// Engine-owned methods that user DM may override and reach through `..()`.
// These are kept distinct from global standard procedures so they never enter
// bare-call resolution. Headless profiling control has no observable profile
// payload yet, but BYOND's start/stop/restart calls legitimately return null.
pub(crate) const NATIVE_PARENT_BUILTINS: &str = concat!(
    // Every allocatable DM object has engine-owned terminal constructor and
    // destructor hooks. User code may call `..()` even when no source-level
    // `/datum/New` or `/datum/Del` declaration exists.
    "/datum/New(...)\n\treturn null\n",
    "/datum/Del()\n\treturn null\n",
    "/datum/Topic(href, list/href_list)\n\treturn null\n",
    // BYOND's terminal client Click implementation dispatches the addressed
    // atom after project overrides finish their rate-limit/signal handling.
    // TG/Monkestation explicitly rely on `..()` here for every HUD button.
    "/client/Click(atom/object, atom/location, control, params)\n",
    "\tif(object)\n",
    "\t\treturn object.Click(location, control, params)\n",
    "\treturn null\n",
    // BYOND owns the terminal movable Bump implementation. Its ordinary
    // obstacle path has no action and returns null; retaining the entry is
    // nevertheless required so a source override can legally call `..()`.
    "/atom/movable/Bump(atom/Obstacle)\n\treturn null\n",
    // `/generator.Rand()` and the `/icon` manipulation family are methods on
    // engine-owned datum types. Source helpers may invoke them as bare member
    // calls from an override even though no project definition exists.
    "/generator/Rand()\n\treturn _dream64_generator_rand(src)\n",
    "/icon/SwapColor(old_rgb, new_rgb)\n\treturn _dream64_icon_swap_color(src, old_rgb, new_rgb)\n",
    "/world/Profile(command, type, format)\n\treturn _dream64_world_profile(command, type, format)\n",
    "/world/GetConfig(config_set, param)\n\treturn _dream64_world_get_config(config_set, param)\n",
    "/world/SetConfig(config_set, param, value)\n\treturn _dream64_world_set_config(config_set, param, value)\n",
    "/world/OpenPort(port)\n\treturn _dream64_world_open_port(src, port)\n",
    "/world/IsBanned(key, address, computer_id, type)\n\treturn 0\n",
    "/world/Error(exception)\n\treturn null\n",
);

pub(crate) fn native_parent_index(
    path: &CodePath,
    owner_type: Option<NodeId>,
    compilation: &Compilation,
    native_parent_indices: &BTreeMap<String, usize>,
) -> Option<usize> {
    let path = path.to_string();
    native_parent_indices.get(&path).copied().or_else(|| {
        let terminal = if path.ends_with("/proc/New") {
            "/datum/proc/New"
        } else if path.ends_with("/proc/Del") {
            "/datum/proc/Del"
        } else if path.ends_with("/proc/Bump")
            && owner_type.is_some_and(|owner| {
                compilation
                    .code_tree()
                    .find(&dm_syntax::DefinitionPath::new(vec![
                        "atom".to_owned(),
                        "movable".to_owned(),
                    ]))
                    .is_some_and(|movable| type_is_descendant_or_same(compilation, owner, movable))
            })
        {
            "/atom/movable/proc/Bump"
        } else {
            return None;
        };
        native_parent_indices.get(terminal).copied()
    })
}

pub(crate) fn native_member_index(
    selector: &str,
    mut owner_type: Option<NodeId>,
    compilation: &Compilation,
    native_indices: &BTreeMap<String, usize>,
) -> Option<usize> {
    while let Some(owner) = owner_type {
        let node = compilation.code_tree().node(owner)?;
        let path = format!("{}/proc/{selector}", node.path);
        if let Some(index) = native_indices.get(&path) {
            return Some(*index);
        }
        owner_type = node.parent_type;
    }
    None
}

pub(crate) fn compiler_type_predicate(selector: &str) -> bool {
    matches!(
        selector,
        "isnull"
            | "isnum"
            | "ispath"
            | "islist"
            | "ismovable"
            | "isturf"
            | "isloc"
            | "isicon"
            | "istype"
    )
}

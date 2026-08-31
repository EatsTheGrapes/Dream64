//! Semantic assignment checks: rejecting an override that changes its
//! inherited scalar/datum return type, and the declared-type / scalar
//! compatibility rules applied to `var` initializers and reassignments.

use dm_compiler::Compilation;
use dm_object_tree::NodeId;

use super::{
    ScalarConstraint, ScalarType, effective_datum_return, effective_scalar_return,
    procedure_return_type_node, procedure_scalar_return,
};

pub(crate) fn validate_override_return_signature(
    compilation: &Compilation,
    node: NodeId,
) -> Result<(), dm_vm::CompileError> {
    let tree_node = compilation
        .code_tree()
        .node(node)
        .expect("procedure node should exist");
    let Some(parent) = tree_node.inherited_member else {
        return Ok(());
    };
    let direct_scalar = tree_node.declarations.iter().rev().find_map(|declaration| {
        let declaration = compilation.code_tree().declaration(*declaration)?;
        let definition = compilation
            .syntax(declaration.file_id)?
            .definitions
            .get(declaration.definition_index)?;
        procedure_scalar_return(&definition.header)
    });
    if let (Some(child), Some(parent)) =
        (direct_scalar, effective_scalar_return(compilation, parent))
        && (child.kind != parent.kind || child.allows_null != parent.allows_null)
    {
        return Err(dm_vm::CompileError {
            message: "procedure override changes its inherited scalar return type".to_owned(),
        });
    }
    let direct_datum = tree_node.declarations.iter().rev().find_map(|declaration| {
        let declaration = compilation.code_tree().declaration(*declaration)?;
        let definition = compilation
            .syntax(declaration.file_id)?
            .definitions
            .get(declaration.definition_index)?;
        procedure_return_type_node(compilation, &definition.header)
    });
    if let (Some(child), Some(parent)) = (direct_datum, effective_datum_return(compilation, parent))
    {
        validate_type_assignment(compilation, "return", parent, child)?;
    }
    Ok(())
}

pub(crate) fn validate_type_assignment(
    compilation: &Compilation,
    name: &str,
    expected: NodeId,
    actual: NodeId,
) -> Result<(), dm_vm::CompileError> {
    let tree = compilation.code_tree();
    let mut current = Some(actual);
    while let Some(node) = current {
        if node == expected {
            return Ok(());
        }
        current = tree.node(node).and_then(|node| node.parent_type);
    }
    // DreamMaker's path annotations permit a value declared as a base path to
    // flow into a variable declared as one of its subpaths. The runtime value
    // can still be an instance of that subtype. Unrelated type branches remain
    // a compile-time mismatch.
    let mut current = Some(expected);
    while let Some(node) = current {
        if node == actual {
            return Ok(());
        }
        current = tree.node(node).and_then(|node| node.parent_type);
    }
    // A datum path on a DM local/parameter is a static hint used for member
    // lookup and inferred `new`; it is not a Rust-style assignment barrier.
    // BYOND accepts values from unrelated branches here (including values
    // supplied through `as mob|obj|turf` call sites) and leaves runtime
    // predicates/casts to the program. Keep the stricter check for declared
    // procedure return contracts, which Dream64 uses to narrow call results.
    if name != "return" {
        return Ok(());
    }
    let expected = tree
        .node(expected)
        .map(|node| node.path.to_string())
        .unwrap_or_else(|| "<unknown>".to_owned());
    let actual = tree
        .node(actual)
        .map(|node| node.path.to_string())
        .unwrap_or_else(|| "<unknown>".to_owned());
    Err(dm_vm::CompileError {
        message: format!(
            "cannot assign {actual} to typed variable `{name}` declared as {expected}"
        ),
    })
}

pub(crate) fn validate_scalar_assignment(
    name: &str,
    expected: ScalarConstraint,
    actual: ScalarType,
) -> Result<(), dm_vm::CompileError> {
    if expected.kind == ScalarType::Dynamic
        || actual == expected.kind
        || (actual == ScalarType::Null && expected.allows_null)
    {
        return Ok(());
    }
    Err(dm_vm::CompileError {
        message: format!(
            "cannot assign {actual:?} to typed variable `{name}` declared as {:?}",
            expected.kind
        ),
    })
}

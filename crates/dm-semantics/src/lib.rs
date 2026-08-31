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

mod builtins;
mod const_eval;
mod deps;
mod executable;
mod fields;
mod ids;
mod ir;
mod path_qualify;
mod registry;
mod scalar_infer;
mod selectors;
mod type_resolve;
mod validate;

use builtins::{
    NATIVE_PARENT_BUILTINS, STANDARD_BUILTINS, compiler_type_predicate, native_member_index,
    native_parent_index,
};
use const_eval::{ConstBindings, inferred_assignment_type, validate_const_assignments};
use deps::{
    construction_dependencies, constructor_targets_by_ancestor, dynamic_call_literal_selectors,
    member_call_dependencies, static_proc_reference_paths, static_procedure_type_families,
    type_is_descendant_or_same,
};
use fields::{
    direct_instance_field_types, direct_instance_fields, direct_static_fields,
    referenced_inherited_field_types, referenced_inherited_fields, scope_operator_static_fields,
};
use ids::{implementation_id, procedure_id};
use ir::effective_target;
use path_qualify::{
    expand_proc_pseudo_macro, normalize_upward_paths, top_level_simple_assignment,
    type_node_from_tokens,
};
use scalar_infer::{
    ScalarConstraint, ScalarType, condition_is_known_truthy, effective_datum_return,
    effective_scalar_return, expression_is_proven_truthy, find_member_node, matching_closing,
    parenthesized_receiver_method, procedure_scalar_return, proven_literal_scalar_type,
    proven_receiver_type, proven_scalar_type, receiver_member_expression, scalar_constraint,
    statically_called_procedure, top_level_ternary,
};
use selectors::{
    collect_text_member_call_selectors, referenced_identifiers, static_call_selectors,
};
use type_resolve::{
    assigned_receiver_field, declared_field_types, declared_global_fields, declared_global_types,
    declared_receiver_types, declared_type_node, declared_type_path,
    grouped_local_declaration_names, inherited_declared_field_type, is_assignment_operator,
    is_known_declared_type, parameter_declaration_name, procedure_return_type_node,
    procedure_return_type_path, proven_datum_expression_type, validate_declared_type_exists,
};
use validate::{
    validate_override_return_signature, validate_scalar_assignment, validate_type_assignment,
};

pub use executable::{ExecutableProcedureStats, ExecutableProcedures};
pub use fields::standard_instance_field_names;
pub use ids::{ProcedureId, ProcedureImplementationId};
pub use ir::{
    Procedure, ProcedureClosureStats, ProcedureImplementation, ProcedureImplementationKind,
    ProcedureRegistryBuildStats,
};
pub use registry::ProcedureRegistry;

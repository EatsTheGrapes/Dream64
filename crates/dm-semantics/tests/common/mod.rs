//! Shared harness for the dm-semantics integration test suites.
//!
//! Each `tests/<subject>.rs` file pulls this in with `use common::*;`; the
//! allow attributes cover the re-exports and helpers a given subject file does
//! not happen to use.
#![allow(dead_code, unused_imports)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

pub use std::collections::{BTreeMap, BTreeSet};

pub use dm_compiler::{Compilation, CompilerDatabase};
pub use dm_value::{FieldName, TypePath};
pub use dm_vm::{
    ExecutionContext, ExecutionState, Instruction, RuntimeError, Value, execute_module,
    execute_module_in_context, execute_module_in_state,
};

pub use dm_semantics::{
    ExecutableProcedures, Procedure, ProcedureImplementationKind, ProcedureRegistry,
};

static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

pub struct TestProject {
    root: PathBuf,
}

impl TestProject {
    pub fn compile(source: &str) -> Compilation {
        let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "dream64-dm-semantics-{}-{}",
            ordinal,
            std::process::id()
        ));
        std::fs::create_dir(&root).expect("test project directory should be created");
        let project = Self { root };
        std::fs::write(project.root.join("world.dme"), "#include \"types.dm\"\n")
            .expect("environment should be written");
        std::fs::write(project.root.join("types.dm"), source).expect("source should be written");
        CompilerDatabase::new()
            .compile(project.root.join("world.dme"))
            .expect("test project should compile")
    }
}

impl Drop for TestProject {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.root) {
            eprintln!(
                "failed to clean test project {}: {error}",
                self.root.display()
            );
        }
    }
}

pub fn procedure_by_path<'registry>(
    registry: &'registry ProcedureRegistry,
    path: &str,
) -> &'registry Procedure {
    registry
        .procedures()
        .iter()
        .find(|procedure| procedure.path.to_string() == path)
        .expect("procedure path should exist")
}

pub fn execute_effective(
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

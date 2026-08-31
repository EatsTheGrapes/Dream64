//! The linked-executable artifact: [`ExecutableProcedures`] pairs a compiled
//! `dm_vm::Module` with the semantic implementation identities that produced it,
//! plus the length-prefixed binary codec for the Dream64 compiled-artifact file.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};

use super::{ProcedureId, ProcedureImplementationId};

/// Executable VM module paired with semantic implementation identities.
#[derive(Debug)]
pub struct ExecutableProcedures {
    pub(crate) module: dm_vm::Module,
    pub(crate) implementations: BTreeMap<ProcedureImplementationId, dm_vm::ProcedureId>,
    pub(crate) stats: ExecutableProcedureStats,
}

/// Deterministic module-building counters for semantic procedure compilation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutableProcedureStats {
    /// Procedure specifications compiled into the module.
    pub procedures: usize,
    /// Referenced `src` field bindings copied into procedure specifications.
    pub src_field_bindings: usize,
    /// Referenced global bindings copied into procedure specifications.
    pub global_field_bindings: usize,
    /// Full variable-registry passes used to index static fields.
    pub static_registry_builds: usize,
    /// Referenced identifiers probed against the global-field index.
    pub global_binding_index_lookups: usize,
    /// Referenced identifiers probed against the typed-global index.
    pub typed_global_index_lookups: usize,
    /// Referenced field names resolved through owner ancestry without
    /// materializing whole inherited-field maps per procedure owner.
    pub inherited_field_name_lookups: usize,
}

impl ExecutableProcedures {
    /// Materializes every deferred project body and returns an artifact-ready
    /// executable with no retained source definitions.
    ///
    /// # Errors
    ///
    /// Returns the first deferred preflight or lowering failure in stable
    /// procedure order.
    pub fn into_fully_eager(mut self) -> Result<Self, dm_vm::CompileError> {
        self.module = self.module.into_fully_eager()?;
        Ok(self)
    }

    /// Attempts every deferred project body and returns bounded aggregate
    /// diagnostics suitable for a complete compiled-artifact build.
    ///
    /// Ordinary lazy dispatch and [`Self::into_fully_eager`] retain their
    /// existing first-error behavior.
    ///
    /// # Errors
    ///
    /// Returns all failures counted across the pass with only the requested
    /// leading diagnostic sample retained.
    pub fn into_fully_eager_bounded(
        mut self,
        diagnostic_limit: usize,
    ) -> Result<Self, dm_vm::FullyEagerCompileErrors> {
        self.module
            .materialize_fully_eager_bounded(diagnostic_limit)?;
        Ok(self)
    }

    /// Encodes this fully eager executable and its semantic implementation
    /// mapping for a Dream64 compiled artifact.
    ///
    /// # Errors
    ///
    /// Returns an error when the module remains deferred or cannot be encoded.
    #[doc(hidden)]
    pub fn encode_compiled_artifact(&self) -> Result<Vec<u8>, String> {
        let segments = self.encode_compiled_artifact_segments()?;
        let total_length = segments.iter().map(Vec::len).sum();
        let mut output = Vec::with_capacity(total_length);
        for segment in segments {
            output.extend_from_slice(&segment);
        }
        debug_assert_eq!(output.len(), total_length);
        Ok(output)
    }

    /// Encodes the executable artifact as ordered byte segments without
    /// copying its portable module into a second contiguous allocation.
    #[doc(hidden)]
    pub fn encode_compiled_artifact_segments(&self) -> Result<Vec<Vec<u8>>, String> {
        let module = self
            .module
            .encode_portable()
            .map_err(|error| error.to_string())?;
        let mut header = EXECUTABLE_ARTIFACT_MAGIC.to_vec();
        executable_artifact_write_len(&mut header, module.len());
        let mut tail = Vec::new();
        executable_artifact_write_len(&mut tail, self.implementations.len());
        for (implementation, procedure) in &self.implementations {
            executable_artifact_write_u32(&mut tail, implementation.procedure.0);
            executable_artifact_write_u32(&mut tail, implementation.index);
            let path = self.module.procedure_path(*procedure).ok_or_else(|| {
                format!(
                    "implementation {}:{} references a missing VM procedure",
                    implementation.procedure.index(),
                    implementation.index()
                )
            })?;
            executable_artifact_write_string(&mut tail, path);
        }
        for value in [
            self.stats.procedures,
            self.stats.src_field_bindings,
            self.stats.global_field_bindings,
            self.stats.static_registry_builds,
            self.stats.global_binding_index_lookups,
            self.stats.typed_global_index_lookups,
            self.stats.inherited_field_name_lookups,
        ] {
            executable_artifact_write_len(&mut tail, value);
        }
        Ok(vec![header, module, tail])
    }

    /// Decodes a fully eager executable and reconstructs its semantic mapping.
    ///
    /// # Errors
    ///
    /// Returns an error for corrupt, truncated, oversized, or internally
    /// inconsistent payloads.
    #[doc(hidden)]
    pub fn decode_compiled_artifact(bytes: &[u8]) -> Result<Self, String> {
        let mut input = Cursor::new(bytes);
        let mut magic = vec![0; EXECUTABLE_ARTIFACT_MAGIC.len()];
        input
            .read_exact(&mut magic)
            .map_err(|_| "compiled executable is truncated before its header".to_owned())?;
        if magic != EXECUTABLE_ARTIFACT_MAGIC {
            return Err("compiled executable has an unsupported header".to_owned());
        }
        let module_bytes = executable_artifact_read_bytes(&mut input, "module")?;
        let module =
            dm_vm::Module::decode_portable(module_bytes).map_err(|error| error.to_string())?;
        let mapping_count = executable_artifact_read_count(&mut input, "implementation count")?;
        let mut implementations = BTreeMap::new();
        for _ in 0..mapping_count {
            let semantic = ProcedureImplementationId {
                procedure: ProcedureId(executable_artifact_read_u32(
                    &mut input,
                    "semantic procedure identity",
                )?),
                index: executable_artifact_read_u32(&mut input, "semantic implementation index")?,
            };
            let path = executable_artifact_read_string(&mut input)?;
            let procedure = module.procedure_id(&path).ok_or_else(|| {
                format!("compiled executable mapping references unknown path {path}")
            })?;
            if implementations.insert(semantic, procedure).is_some() {
                return Err(format!(
                    "compiled executable repeats semantic implementation {}:{}",
                    semantic.procedure.index(),
                    semantic.index()
                ));
            }
        }
        let stats = ExecutableProcedureStats {
            procedures: executable_artifact_read_len(&mut input, "procedure statistic")?,
            src_field_bindings: executable_artifact_read_len(&mut input, "src binding statistic")?,
            global_field_bindings: executable_artifact_read_len(
                &mut input,
                "global binding statistic",
            )?,
            static_registry_builds: executable_artifact_read_len(
                &mut input,
                "registry build statistic",
            )?,
            global_binding_index_lookups: executable_artifact_read_len(
                &mut input,
                "global lookup statistic",
            )?,
            typed_global_index_lookups: executable_artifact_read_len(
                &mut input,
                "typed global lookup statistic",
            )?,
            inherited_field_name_lookups: executable_artifact_read_len(
                &mut input,
                "inherited field lookup statistic",
            )?,
        };
        if input.position() != bytes.len() as u64 {
            return Err("compiled executable contains trailing bytes".to_owned());
        }
        Ok(Self {
            module,
            implementations,
            stats,
        })
    }

    /// Returns the compiled VM module.
    #[must_use]
    pub const fn module(&self) -> &dm_vm::Module {
        &self.module
    }

    /// Returns the linked VM module for append-only runtime initializer entries.
    ///
    /// Appending preserves all existing procedure identities and deferred
    /// project bodies while allowing map expressions to be linked after world
    /// allocation.
    pub const fn module_mut(&mut self) -> &mut dm_vm::Module {
        &mut self.module
    }

    /// Resolves a semantic implementation to its VM-local identity.
    #[must_use]
    pub fn implementation(
        &self,
        implementation: ProcedureImplementationId,
    ) -> Option<dm_vm::ProcedureId> {
        self.implementations.get(&implementation).copied()
    }

    /// Returns deterministic module-building counters.
    #[must_use]
    pub const fn stats(&self) -> &ExecutableProcedureStats {
        &self.stats
    }
}

const EXECUTABLE_ARTIFACT_MAGIC: &[u8] = b"DREAM64-EXECUTABLE\0\x01";
const MAX_EXECUTABLE_ARTIFACT_ITEMS: usize = 16_777_216;
const MAX_EXECUTABLE_ARTIFACT_STRING_BYTES: usize = 268_435_456;

fn executable_artifact_write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn executable_artifact_write_len(output: &mut Vec<u8>, value: usize) {
    output.extend_from_slice(&(value as u64).to_le_bytes());
}

fn executable_artifact_write_bytes(output: &mut Vec<u8>, bytes: &[u8]) {
    executable_artifact_write_len(output, bytes.len());
    output.extend_from_slice(bytes);
}

fn executable_artifact_write_string(output: &mut Vec<u8>, value: &str) {
    executable_artifact_write_bytes(output, value.as_bytes());
}

fn executable_artifact_read_u32(input: &mut Cursor<&[u8]>, what: &str) -> Result<u32, String> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| format!("compiled executable is truncated while reading {what}"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn executable_artifact_read_len(input: &mut Cursor<&[u8]>, what: &str) -> Result<usize, String> {
    let mut bytes = [0; 8];
    input
        .read_exact(&mut bytes)
        .map_err(|_| format!("compiled executable is truncated while reading {what}"))?;
    usize::try_from(u64::from_le_bytes(bytes))
        .map_err(|_| format!("compiled executable {what} exceeds this platform"))
}

fn executable_artifact_read_count(input: &mut Cursor<&[u8]>, what: &str) -> Result<usize, String> {
    let count = executable_artifact_read_len(input, what)?;
    if count > MAX_EXECUTABLE_ARTIFACT_ITEMS {
        return Err(format!(
            "compiled executable {what} exceeds the limit of {MAX_EXECUTABLE_ARTIFACT_ITEMS}"
        ));
    }
    Ok(count)
}

fn executable_artifact_read_bytes<'artifact>(
    input: &mut Cursor<&'artifact [u8]>,
    what: &str,
) -> Result<&'artifact [u8], String> {
    let length = executable_artifact_read_len(input, &format!("{what} length"))?;
    let start = usize::try_from(input.position())
        .map_err(|_| format!("compiled executable {what} offset exceeds this platform"))?;
    let end = start
        .checked_add(length)
        .filter(|end| *end <= input.get_ref().len())
        .ok_or_else(|| format!("compiled executable {what} is truncated"))?;
    input.set_position(end as u64);
    Ok(&input.get_ref()[start..end])
}

fn executable_artifact_read_string(input: &mut Cursor<&[u8]>) -> Result<String, String> {
    let bytes = executable_artifact_read_bytes(input, "string")?;
    if bytes.len() > MAX_EXECUTABLE_ARTIFACT_STRING_BYTES {
        return Err(format!(
            "compiled executable string exceeds the limit of {MAX_EXECUTABLE_ARTIFACT_STRING_BYTES} bytes"
        ));
    }
    String::from_utf8(bytes.to_vec())
        .map_err(|_| "compiled executable contains non-UTF-8 text".to_owned())
}

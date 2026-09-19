//! Pre-lifecycle state serialization for compile-time deterministic boot.
//!
//! Captures the ExecutionState, atom bindings, and event list produced by the
//! deterministic compilation pipeline (WorldAllocation,
//! materialize_world_map_state, apply_dynamic_map_overrides) so the server can
//! skip all deterministic work and jump straight to lifecycle hook execution.

use std::io::{self, Read, Write};

use dm_value::DatumId;
use dm_vm::ExecutionState;
use serde::{Deserialize, Serialize};

use crate::LifecycleKind;
use crate::initialization_plan::{EventSubject, InitializationEvent, InitializationPlan};

const PRE_LIFECYCLE_MAGIC: &[u8; 8] = b"D64PRELF";
const PRE_LIFECYCLE_VERSION: u32 = 1;

type DatumHandle = (u32, u32);

fn datum_handle(id: DatumId) -> DatumHandle {
    (id.index(), id.generation())
}

fn datum_from_handle(handle: DatumHandle) -> DatumId {
    DatumId::from_parts(handle.0, handle.1)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum SerializableSubject {
    Globals,
    World,
    MapAtom(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum SerializableKind {
    Genesis,
    New,
    Initialize,
    LateInitialize,
    Destroy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum SerializableEvent {
    Globals,
    Lifecycle {
        subject: SerializableSubject,
        kind: SerializableKind,
        type_index: usize,
    },
}

impl From<InitializationEvent> for SerializableEvent {
    fn from(event: InitializationEvent) -> Self {
        match event {
            InitializationEvent::Globals => SerializableEvent::Globals,
            InitializationEvent::Lifecycle {
                subject,
                kind,
                type_index,
            } => SerializableEvent::Lifecycle {
                subject: match subject {
                    EventSubject::Globals => SerializableSubject::Globals,
                    EventSubject::World => SerializableSubject::World,
                    EventSubject::MapAtom(i) => SerializableSubject::MapAtom(i),
                },
                kind: match kind {
                    LifecycleKind::Genesis => SerializableKind::Genesis,
                    LifecycleKind::New => SerializableKind::New,
                    LifecycleKind::Initialize => SerializableKind::Initialize,
                    LifecycleKind::LateInitialize => SerializableKind::LateInitialize,
                    LifecycleKind::Destroy => SerializableKind::Destroy,
                },
                type_index,
            },
        }
    }
}

impl From<SerializableEvent> for InitializationEvent {
    fn from(event: SerializableEvent) -> Self {
        match event {
            SerializableEvent::Globals => InitializationEvent::Globals,
            SerializableEvent::Lifecycle {
                subject,
                kind,
                type_index,
            } => InitializationEvent::Lifecycle {
                subject: match subject {
                    SerializableSubject::Globals => EventSubject::Globals,
                    SerializableSubject::World => EventSubject::World,
                    SerializableSubject::MapAtom(i) => EventSubject::MapAtom(i),
                },
                kind: match kind {
                    SerializableKind::Genesis => LifecycleKind::Genesis,
                    SerializableKind::New => LifecycleKind::New,
                    SerializableKind::Initialize => LifecycleKind::Initialize,
                    SerializableKind::LateInitialize => LifecycleKind::LateInitialize,
                    SerializableKind::Destroy => LifecycleKind::Destroy,
                },
                type_index,
            },
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PreLifecycleMetadata {
    world: Option<DatumHandle>,
    atom_bindings: Vec<Option<DatumHandle>>,
    events: Vec<SerializableEvent>,
    map_atom_type_paths: Vec<String>,
}

/// Decoded pre-lifecycle state ready for boot-time hook execution.
pub struct PreLifecycleState {
    /// `/world` datum, if allocated.
    pub world: Option<DatumId>,
    /// Atom bindings: `PlannedAtom` index → live datum.
    pub atom_bindings: Vec<Option<DatumId>>,
    /// Lifecycle events in execution order.
    pub events: Vec<InitializationEvent>,
    /// Type paths for map atoms (for error diagnostics).
    pub map_atom_type_paths: Vec<String>,
}

/// Encodes the pre-lifecycle state into a self-contained byte buffer.
///
/// Format:
/// ```text
/// [8 bytes: magic "D64PRELF"]
/// [4 bytes: version]
/// [4 bytes: metadata_len]
/// [metadata_len bytes: bincode-encoded PreLifecycleMetadata]
/// [remaining: ExecutionState ready-world snapshot]
/// ```
pub fn encode_pre_lifecycle_state(
    plan: &InitializationPlan,
    bindings: &[Option<DatumId>],
    world: Option<DatumId>,
    state: &ExecutionState,
) -> Result<Vec<u8>, String> {
    let metadata = PreLifecycleMetadata {
        world: world.map(datum_handle),
        atom_bindings: bindings.iter().map(|b| b.map(datum_handle)).collect(),
        events: plan
            .events
            .iter()
            .copied()
            .map(SerializableEvent::from)
            .collect(),
        map_atom_type_paths: plan
            .map_atoms
            .iter()
            .map(|atom| atom.type_path.to_string())
            .collect(),
    };
    let metadata_bytes = bincode::serialize(&metadata)
        .map_err(|error| format!("pre-lifecycle metadata serialization: {error}"))?;

    let mut snapshot_bytes = Vec::new();
    state
        .write_ready_world_snapshot_to(&mut snapshot_bytes)
        .map_err(|error| format!("pre-lifecycle ready-world snapshot: {error}"))?;

    let total = 8 + 4 + 4 + metadata_bytes.len() + snapshot_bytes.len();
    let mut buffer = Vec::with_capacity(total);
    buffer
        .write_all(PRE_LIFECYCLE_MAGIC)
        .map_err(|error| format!("pre-lifecycle magic write: {error}"))?;
    buffer
        .write_all(&PRE_LIFECYCLE_VERSION.to_le_bytes())
        .map_err(|error| format!("pre-lifecycle version write: {error}"))?;
    buffer
        .write_all(&(metadata_bytes.len() as u32).to_le_bytes())
        .map_err(|error| format!("pre-lifecycle metadata length write: {error}"))?;
    buffer
        .write_all(&metadata_bytes)
        .map_err(|error| format!("pre-lifecycle metadata write: {error}"))?;
    buffer
        .write_all(&snapshot_bytes)
        .map_err(|error| format!("pre-lifecycle snapshot write: {error}"))?;
    Ok(buffer)
}

/// Decodes the pre-lifecycle metadata and restores the ready-world snapshot
/// into an existing `ExecutionState` that already has its runtime catalog
/// (type metadata) installed.
///
/// The caller must first restore the runtime catalog from section 6 before
/// calling this function so the heap references resolve correctly.
pub fn decode_pre_lifecycle_state(
    data: &[u8],
    state: &mut ExecutionState,
    module: &dm_vm::Module,
) -> Result<PreLifecycleState, String> {
    let mut cursor = io::Cursor::new(data);
    let mut magic = [0u8; 8];
    cursor
        .read_exact(&mut magic)
        .map_err(|error| format!("pre-lifecycle magic read: {error}"))?;
    if magic != *PRE_LIFECYCLE_MAGIC {
        return Err("pre-lifecycle section has invalid magic".to_owned());
    }
    let mut version = [0u8; 4];
    cursor
        .read_exact(&mut version)
        .map_err(|error| format!("pre-lifecycle version read: {error}"))?;
    let version = u32::from_le_bytes(version);
    if version != PRE_LIFECYCLE_VERSION {
        return Err(format!(
            "unsupported pre-lifecycle version {version}; expected {PRE_LIFECYCLE_VERSION}"
        ));
    }
    let mut meta_len = [0u8; 4];
    cursor
        .read_exact(&mut meta_len)
        .map_err(|error| format!("pre-lifecycle metadata length read: {error}"))?;
    let meta_len = u32::from_le_bytes(meta_len) as usize;
    let mut meta_bytes = vec![0u8; meta_len];
    cursor
        .read_exact(&mut meta_bytes)
        .map_err(|error| format!("pre-lifecycle metadata read: {error}"))?;
    let metadata: PreLifecycleMetadata = bincode::deserialize(&meta_bytes)
        .map_err(|error| format!("pre-lifecycle metadata deserialization: {error}"))?;

    state
        .restore_ready_world_snapshot_from(&mut cursor, module)
        .map_err(|error| format!("pre-lifecycle ready-world restore: {error}"))?;

    Ok(PreLifecycleState {
        world: metadata.world.map(datum_from_handle),
        atom_bindings: metadata
            .atom_bindings
            .into_iter()
            .map(|b| b.map(datum_from_handle))
            .collect(),
        events: metadata
            .events
            .into_iter()
            .map(SerializableEvent::into)
            .collect(),
        map_atom_type_paths: metadata.map_atom_type_paths,
    })
}

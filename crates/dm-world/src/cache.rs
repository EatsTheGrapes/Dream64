use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dm_compiler::Compilation;
use dm_core::SourceSpan;
use dm_map::{MapValue, MapValueKind, MapVariableAssignment};
use dm_object_tree::{NodeId, NodeKind};

use super::{
    AtomCategory, CellTemplate, InitializerResolution, PlannedCell, PlannedInitializer,
    WorldCoordinate, WorldDiagnostic, WorldDiagnosticKind, WorldPlan, build_plan, plan_stats,
};

const MAGIC: &[u8; 16] = b"DREAM64-MAPPLAN!";
const SCHEMA: u16 = 1;
const HEADER: usize = 80;
const MAX_CACHE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_ITEMS: usize = 4_000_000;
const MAX_STRING: usize = 16 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Timing and disposition of one persistent map-plan lookup.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MapPlanCacheStats {
    /// Whether a validated plan was decoded without parsing the map.
    pub hit: bool,
    /// Time spent validating and decoding the cache candidate.
    pub lookup_elapsed: Duration,
    /// Time spent parsing and building after a miss.
    pub build_elapsed: Duration,
    /// Encoded cache bytes installed after a miss.
    pub written_bytes: u64,
}

/// A map-plan cache I/O or codec failure.
#[derive(Debug)]
pub struct MapPlanCacheError(String);

impl std::fmt::Display for MapPlanCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for MapPlanCacheError {}

/// Loads a validated immutable world plan or rebuilds and atomically caches it.
///
/// Cache corruption and identity mismatches are ordinary misses. Errors are
/// returned only when the source map cannot be parsed or the replacement file
/// cannot be installed.
pub fn load_or_build_cached_plan(
    cache_path: impl AsRef<Path>,
    map_bytes: &[u8],
    project_fingerprint: [u8; 16],
    runtime_abi: [u8; 16],
    compilation: &Compilation,
) -> Result<(WorldPlan, MapPlanCacheStats), MapPlanCacheError> {
    let cache_path = cache_path.as_ref();
    let map_fingerprint = digest(map_bytes);
    let lookup = Instant::now();
    if let Ok(bytes) = bounded_read(cache_path)
        && let Ok(plan) = decode(
            &bytes,
            map_fingerprint,
            project_fingerprint,
            runtime_abi,
            compilation,
        )
    {
        return Ok((
            plan,
            MapPlanCacheStats {
                hit: true,
                lookup_elapsed: lookup.elapsed(),
                ..Default::default()
            },
        ));
    }
    let lookup_elapsed = lookup.elapsed();
    let build = Instant::now();
    let source = std::str::from_utf8(map_bytes)
        .map_err(|error| MapPlanCacheError(format!("map is not UTF-8: {error}")))?;
    let map = dm_map::parse(source).map_err(|error| MapPlanCacheError(error.to_string()))?;
    let plan = build_plan(&map, compilation);
    let build_elapsed = build.elapsed();
    let bytes = encode(&plan, map_fingerprint, project_fingerprint, runtime_abi)?;
    write_atomic(cache_path, &bytes)?;
    Ok((
        plan,
        MapPlanCacheStats {
            hit: false,
            lookup_elapsed,
            build_elapsed,
            written_bytes: bytes.len() as u64,
        },
    ))
}

fn digest(bytes: &[u8]) -> [u8; 16] {
    md5::compute(bytes).0
}

fn bounded_read(path: &Path) -> Result<Vec<u8>, MapPlanCacheError> {
    let file = File::open(path).map_err(io_error)?;
    let length = file.metadata().map_err(io_error)?.len();
    if length > MAX_CACHE_BYTES {
        return Err(MapPlanCacheError(
            "map-plan cache exceeds size limit".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(MAX_CACHE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() as u64 != length {
        return Err(MapPlanCacheError(
            "map-plan cache length changed while reading".into(),
        ));
    }
    Ok(bytes)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), MapPlanCacheError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io_error)?;
    }
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = temporary_path(path, sequence);
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io_error)?;
        file.write_all(bytes).map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        drop(file);
        fs::rename(&temporary, path).map_err(io_error)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn temporary_path(path: &Path, sequence: u64) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp.{}.{}", std::process::id(), sequence));
    path.with_file_name(name)
}

fn encode(
    plan: &WorldPlan,
    map: [u8; 16],
    project: [u8; 16],
    abi: [u8; 16],
) -> Result<Vec<u8>, MapPlanCacheError> {
    let mut payload = Vec::new();
    put_len(&mut payload, plan.templates.len())?;
    for template in plan.templates.values() {
        put_template(&mut payload, template)?;
    }
    put_len(&mut payload, plan.cells.len())?;
    for cell in &plan.cells {
        put_cell(&mut payload, cell)?;
    }
    put_len(&mut payload, plan.diagnostics.len())?;
    for diagnostic in &plan.diagnostics {
        put_diagnostic(&mut payload, diagnostic)?;
    }
    let mut out = Vec::with_capacity(HEADER + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&SCHEMA.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&map);
    out.extend_from_slice(&project);
    out.extend_from_slice(&abi);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

fn decode(
    bytes: &[u8],
    map: [u8; 16],
    project: [u8; 16],
    abi: [u8; 16],
    compilation: &Compilation,
) -> Result<WorldPlan, String> {
    if bytes.len() < HEADER || &bytes[..16] != MAGIC {
        return Err("bad header".into());
    }
    if u16::from_le_bytes(bytes[16..18].try_into().unwrap()) != SCHEMA {
        return Err("bad schema".into());
    }
    if bytes[20..36] != map || bytes[36..52] != project || bytes[52..68] != abi {
        return Err("identity mismatch".into());
    }
    let length = u64::from_le_bytes(bytes[68..76].try_into().unwrap()) as usize;
    let crc = u32::from_le_bytes(bytes[76..80].try_into().unwrap());
    if bytes.len() != 80usize.checked_add(length).ok_or("length overflow")? {
        return Err("bad length".into());
    }
    let payload = &bytes[80..];
    if crc32fast::hash(payload) != crc {
        return Err("checksum mismatch".into());
    }
    let mut input = Cursor::new(payload);
    let count = get_len(&mut input)?;
    let mut templates = BTreeMap::new();
    for _ in 0..count {
        let template = get_template(&mut input, compilation)?;
        templates.insert(template.key.clone(), template);
    }
    let count = get_len(&mut input)?;
    let mut cells = Vec::with_capacity(count);
    for _ in 0..count {
        cells.push(get_cell(&mut input)?);
    }
    let count = get_len(&mut input)?;
    let mut diagnostics = Vec::with_capacity(count);
    for _ in 0..count {
        diagnostics.push(get_diagnostic(&mut input)?);
    }
    if input.position() as usize != payload.len() {
        return Err("trailing payload".into());
    }
    let stats = plan_stats(&templates, &cells, diagnostics.len());
    Ok(WorldPlan {
        templates,
        cells,
        diagnostics,
        stats,
    })
}

fn put_template(out: &mut Vec<u8>, value: &CellTemplate) -> Result<(), MapPlanCacheError> {
    put_str(out, &value.key)?;
    put_span(out, value.span)?;
    put_len(out, value.initializers.len())?;
    for x in &value.initializers {
        put_initializer(out, x)?;
    }
    out.push(value.has_area as u8);
    out.push(value.has_turf as u8);
    Ok(())
}
fn get_template(
    input: &mut Cursor<&[u8]>,
    compilation: &Compilation,
) -> Result<CellTemplate, String> {
    let key = get_str(input)?;
    let span = get_span(input)?;
    let n = get_len(input)?;
    let mut initializers = Vec::with_capacity(n);
    for _ in 0..n {
        initializers.push(get_initializer(input, compilation)?);
    }
    Ok(CellTemplate {
        key,
        span,
        initializers,
        has_area: get_bool(input)?,
        has_turf: get_bool(input)?,
    })
}
fn put_initializer(out: &mut Vec<u8>, v: &PlannedInitializer) -> Result<(), MapPlanCacheError> {
    put_str(out, &v.path)?;
    put_span(out, v.span)?;
    put_len(out, v.variables.len())?;
    for x in &v.variables {
        put_assignment(out, x)?;
    }
    match v.resolution {
        InitializerResolution::Resolved { node, category } => {
            out.push(0);
            put_len(out, node.index())?;
            out.push(category_tag(category));
        }
        InitializerResolution::Unknown => out.push(1),
        InitializerResolution::NonType { node, kind } => {
            out.push(2);
            put_len(out, node.index())?;
            out.push(kind_tag(kind));
        }
    }
    Ok(())
}
fn get_initializer(
    input: &mut Cursor<&[u8]>,
    compilation: &Compilation,
) -> Result<PlannedInitializer, String> {
    let path = get_str(input)?;
    let span = get_span(input)?;
    let n = get_len(input)?;
    let mut variables = Vec::with_capacity(n);
    for _ in 0..n {
        variables.push(get_assignment(input)?);
    }
    let resolution = match get_u8(input)? {
        0 => {
            let node = NodeId::from_index(get_len(input)?);
            let category = get_category(input)?;
            validate_node(compilation, node, &path, NodeKind::Type)?;
            InitializerResolution::Resolved { node, category }
        }
        1 => InitializerResolution::Unknown,
        2 => {
            let node = NodeId::from_index(get_len(input)?);
            let kind = get_kind(input)?;
            validate_node(compilation, node, &path, kind)?;
            InitializerResolution::NonType { node, kind }
        }
        _ => return Err("bad resolution".into()),
    };
    Ok(PlannedInitializer {
        path,
        span,
        variables,
        resolution,
    })
}
fn validate_node(c: &Compilation, n: NodeId, path: &str, kind: NodeKind) -> Result<(), String> {
    let x = c.code_tree().node(n).ok_or("node out of range")?;
    if x.path.to_string() != path || x.kind != kind {
        return Err("node identity mismatch".into());
    }
    Ok(())
}
fn put_assignment(out: &mut Vec<u8>, v: &MapVariableAssignment) -> Result<(), MapPlanCacheError> {
    put_str(out, &v.name)?;
    put_span(out, v.name_span)?;
    out.push(value_kind_tag(v.value.kind));
    put_str(out, &v.value.raw)?;
    put_span(out, v.value.span)?;
    put_str(out, &v.raw)?;
    put_span(out, v.span)
}
fn get_assignment(i: &mut Cursor<&[u8]>) -> Result<MapVariableAssignment, String> {
    let name = get_str(i)?;
    let name_span = get_span(i)?;
    let kind = get_value_kind(i)?;
    let raw_value = get_str(i)?;
    let value_span = get_span(i)?;
    let raw = get_str(i)?;
    let span = get_span(i)?;
    Ok(MapVariableAssignment {
        name,
        name_span,
        value: MapValue {
            kind,
            raw: raw_value,
            span: value_span,
        },
        raw,
        span,
    })
}
fn put_cell(o: &mut Vec<u8>, v: &PlannedCell) -> Result<(), MapPlanCacheError> {
    put_coord(o, v.coordinate);
    put_str(o, &v.key)?;
    put_span(o, v.block_span)?;
    put_opt_span(o, v.template_span)
}
fn get_cell(i: &mut Cursor<&[u8]>) -> Result<PlannedCell, String> {
    Ok(PlannedCell {
        coordinate: get_coord(i)?,
        key: get_str(i)?,
        block_span: get_span(i)?,
        template_span: get_opt_span(i)?,
    })
}
fn put_diagnostic(o: &mut Vec<u8>, v: &WorldDiagnostic) -> Result<(), MapPlanCacheError> {
    o.push(diag_tag(v.kind));
    put_str(o, &v.message)?;
    put_span(o, v.span)?;
    put_opt_span(o, v.previous_span)?;
    put_opt_coord(o, v.coordinate);
    put_opt_str(o, v.path.as_deref())
}
fn get_diagnostic(i: &mut Cursor<&[u8]>) -> Result<WorldDiagnostic, String> {
    Ok(WorldDiagnostic {
        kind: get_diag(i)?,
        message: get_str(i)?,
        span: get_span(i)?,
        previous_span: get_opt_span(i)?,
        coordinate: get_opt_coord(i)?,
        path: get_opt_str(i)?,
    })
}

fn put_len(o: &mut Vec<u8>, v: usize) -> Result<(), MapPlanCacheError> {
    let v = u32::try_from(v).map_err(|_| MapPlanCacheError("cache value exceeds u32".into()))?;
    o.extend_from_slice(&v.to_le_bytes());
    Ok(())
}
fn get_len(i: &mut Cursor<&[u8]>) -> Result<usize, String> {
    let n = get_u32(i)? as usize;
    if n > MAX_ITEMS {
        Err("item count exceeds limit".into())
    } else {
        Ok(n)
    }
}
fn put_str(o: &mut Vec<u8>, v: &str) -> Result<(), MapPlanCacheError> {
    put_len(o, v.len())?;
    o.extend_from_slice(v.as_bytes());
    Ok(())
}
fn get_str(i: &mut Cursor<&[u8]>) -> Result<String, String> {
    let n = get_len(i)?;
    if n > MAX_STRING {
        return Err("string exceeds limit".into());
    }
    let mut b = vec![0; n];
    i.read_exact(&mut b).map_err(|_| "truncated string")?;
    String::from_utf8(b).map_err(|_| "invalid UTF-8".into())
}
fn put_span(o: &mut Vec<u8>, v: SourceSpan) -> Result<(), MapPlanCacheError> {
    put_len(o, v.start)?;
    put_len(o, v.end)
}
fn get_span(i: &mut Cursor<&[u8]>) -> Result<SourceSpan, String> {
    let a = get_len(i)?;
    let b = get_len(i)?;
    if a > b {
        return Err("inverted span".into());
    }
    Ok(SourceSpan::new(a, b))
}
fn put_opt_span(o: &mut Vec<u8>, v: Option<SourceSpan>) -> Result<(), MapPlanCacheError> {
    o.push(v.is_some() as u8);
    if let Some(x) = v {
        put_span(o, x)?
    }
    Ok(())
}
fn get_opt_span(i: &mut Cursor<&[u8]>) -> Result<Option<SourceSpan>, String> {
    if get_bool(i)? {
        Ok(Some(get_span(i)?))
    } else {
        Ok(None)
    }
}
fn put_coord(o: &mut Vec<u8>, v: WorldCoordinate) {
    for x in [v.x, v.y, v.z] {
        o.extend_from_slice(&x.to_le_bytes())
    }
}
fn get_coord(i: &mut Cursor<&[u8]>) -> Result<WorldCoordinate, String> {
    Ok(WorldCoordinate {
        x: get_i32(i)?,
        y: get_i32(i)?,
        z: get_i32(i)?,
    })
}
fn put_opt_coord(o: &mut Vec<u8>, v: Option<WorldCoordinate>) {
    o.push(v.is_some() as u8);
    if let Some(x) = v {
        put_coord(o, x)
    }
}
fn get_opt_coord(i: &mut Cursor<&[u8]>) -> Result<Option<WorldCoordinate>, String> {
    if get_bool(i)? {
        Ok(Some(get_coord(i)?))
    } else {
        Ok(None)
    }
}
fn put_opt_str(o: &mut Vec<u8>, v: Option<&str>) -> Result<(), MapPlanCacheError> {
    o.push(v.is_some() as u8);
    if let Some(x) = v {
        put_str(o, x)?
    }
    Ok(())
}
fn get_opt_str(i: &mut Cursor<&[u8]>) -> Result<Option<String>, String> {
    if get_bool(i)? {
        Ok(Some(get_str(i)?))
    } else {
        Ok(None)
    }
}
fn get_u8(i: &mut Cursor<&[u8]>) -> Result<u8, String> {
    let mut b = [0];
    i.read_exact(&mut b).map_err(|_| "truncated byte")?;
    Ok(b[0])
}
fn get_bool(i: &mut Cursor<&[u8]>) -> Result<bool, String> {
    match get_u8(i)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err("bad bool".into()),
    }
}
fn get_u32(i: &mut Cursor<&[u8]>) -> Result<u32, String> {
    let mut b = [0; 4];
    i.read_exact(&mut b).map_err(|_| "truncated u32")?;
    Ok(u32::from_le_bytes(b))
}
fn get_i32(i: &mut Cursor<&[u8]>) -> Result<i32, String> {
    let mut b = [0; 4];
    i.read_exact(&mut b).map_err(|_| "truncated i32")?;
    Ok(i32::from_le_bytes(b))
}
fn category_tag(v: AtomCategory) -> u8 {
    match v {
        AtomCategory::Area => 0,
        AtomCategory::Turf => 1,
        AtomCategory::Movable => 2,
        AtomCategory::OtherType => 3,
    }
}
fn get_category(i: &mut Cursor<&[u8]>) -> Result<AtomCategory, String> {
    match get_u8(i)? {
        0 => Ok(AtomCategory::Area),
        1 => Ok(AtomCategory::Turf),
        2 => Ok(AtomCategory::Movable),
        3 => Ok(AtomCategory::OtherType),
        _ => Err("bad category".into()),
    }
}
fn kind_tag(v: NodeKind) -> u8 {
    match v {
        NodeKind::Type => 0,
        NodeKind::Procedure => 1,
        NodeKind::Verb => 2,
        NodeKind::Variable => 3,
    }
}
fn get_kind(i: &mut Cursor<&[u8]>) -> Result<NodeKind, String> {
    match get_u8(i)? {
        0 => Ok(NodeKind::Type),
        1 => Ok(NodeKind::Procedure),
        2 => Ok(NodeKind::Verb),
        3 => Ok(NodeKind::Variable),
        _ => Err("bad node kind".into()),
    }
}
fn value_kind_tag(v: MapValueKind) -> u8 {
    match v {
        MapValueKind::Text => 0,
        MapValueKind::Resource => 1,
        MapValueKind::List => 2,
        MapValueKind::Path => 3,
        MapValueKind::Number => 4,
        MapValueKind::Null => 5,
        MapValueKind::Identifier => 6,
        MapValueKind::Expression => 7,
    }
}
fn get_value_kind(i: &mut Cursor<&[u8]>) -> Result<MapValueKind, String> {
    match get_u8(i)? {
        0 => Ok(MapValueKind::Text),
        1 => Ok(MapValueKind::Resource),
        2 => Ok(MapValueKind::List),
        3 => Ok(MapValueKind::Path),
        4 => Ok(MapValueKind::Number),
        5 => Ok(MapValueKind::Null),
        6 => Ok(MapValueKind::Identifier),
        7 => Ok(MapValueKind::Expression),
        _ => Err("bad value kind".into()),
    }
}
fn diag_tag(v: WorldDiagnosticKind) -> u8 {
    match v {
        WorldDiagnosticKind::UnknownTypePath => 0,
        WorldDiagnosticKind::PathNotType => 1,
        WorldDiagnosticKind::DuplicateCoordinate => 2,
        WorldDiagnosticKind::CoordinateOverflow => 3,
        WorldDiagnosticKind::MissingKeyDefinition => 4,
        WorldDiagnosticKind::MissingArea => 5,
        WorldDiagnosticKind::MissingTurf => 6,
    }
}
fn get_diag(i: &mut Cursor<&[u8]>) -> Result<WorldDiagnosticKind, String> {
    match get_u8(i)? {
        0 => Ok(WorldDiagnosticKind::UnknownTypePath),
        1 => Ok(WorldDiagnosticKind::PathNotType),
        2 => Ok(WorldDiagnosticKind::DuplicateCoordinate),
        3 => Ok(WorldDiagnosticKind::CoordinateOverflow),
        4 => Ok(WorldDiagnosticKind::MissingKeyDefinition),
        5 => Ok(WorldDiagnosticKind::MissingArea),
        6 => Ok(WorldDiagnosticKind::MissingTurf),
        _ => Err("bad diagnostic kind".into()),
    }
}
fn io_error(error: io::Error) -> MapPlanCacheError {
    MapPlanCacheError(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::CompilerDatabase;

    use super::*;

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> (Self, Compilation) {
            let root = std::env::temp_dir().join(format!(
                "dream64-map-plan-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            fs::write(root.join("world.dme"), "#include \"types.dm\"\n").unwrap();
            fs::write(root.join("types.dm"), "/area/test\n/turf/test\n/obj/test\n").unwrap();
            let compilation = CompilerDatabase::new()
                .compile(root.join("world.dme"))
                .unwrap();
            (Self(root), compilation)
        }
        fn cache(&self) -> PathBuf {
            self.0.join("world.dmm.d64map")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const MAP_A: &[u8] =
        b"\"a\" = (/obj/test{name = \"cached\"},/turf/test,/area/test)\n(1,1,1) = {\"\na\n\"}\n";
    const MAP_B: &[u8] = b"\"a\" = (/turf/test,/area/test)\n(2,1,1) = {\"\na\n\"}\n";

    #[test]
    fn roundtrip_hit_preserves_the_complete_plan() {
        let (fixture, compilation) = Fixture::new();
        let (cold, cold_stats) =
            load_or_build_cached_plan(fixture.cache(), MAP_A, [1; 16], [2; 16], &compilation)
                .unwrap();
        assert!(!cold_stats.hit);
        assert!(cold_stats.written_bytes > 80);
        let (warm, warm_stats) =
            load_or_build_cached_plan(fixture.cache(), MAP_A, [1; 16], [2; 16], &compilation)
                .unwrap();
        assert!(warm_stats.hit);
        assert_eq!(warm, cold);
        assert_eq!(warm_stats.build_elapsed, Duration::ZERO);
    }

    #[test]
    fn map_project_and_abi_changes_each_invalidate() {
        let (fixture, compilation) = Fixture::new();
        let path = fixture.cache();
        assert!(
            !load_or_build_cached_plan(&path, MAP_A, [1; 16], [2; 16], &compilation)
                .unwrap()
                .1
                .hit
        );
        assert!(
            !load_or_build_cached_plan(&path, MAP_B, [1; 16], [2; 16], &compilation)
                .unwrap()
                .1
                .hit
        );
        assert!(
            !load_or_build_cached_plan(&path, MAP_B, [3; 16], [2; 16], &compilation)
                .unwrap()
                .1
                .hit
        );
        assert!(
            !load_or_build_cached_plan(&path, MAP_B, [3; 16], [4; 16], &compilation)
                .unwrap()
                .1
                .hit
        );
        assert!(
            load_or_build_cached_plan(&path, MAP_B, [3; 16], [4; 16], &compilation)
                .unwrap()
                .1
                .hit
        );
    }

    #[test]
    fn corruption_is_a_bounded_recoverable_miss() {
        let (fixture, compilation) = Fixture::new();
        let path = fixture.cache();
        load_or_build_cached_plan(&path, MAP_A, [1; 16], [2; 16], &compilation).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[79] ^= 0xff;
        fs::write(&path, bytes).unwrap();
        let (rebuilt, stats) =
            load_or_build_cached_plan(&path, MAP_A, [1; 16], [2; 16], &compilation).unwrap();
        assert!(!stats.hit);
        assert_eq!(rebuilt.cells().len(), 1);
        assert!(
            load_or_build_cached_plan(&path, MAP_A, [1; 16], [2; 16], &compilation)
                .unwrap()
                .1
                .hit
        );
    }
}

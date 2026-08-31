//! Pure deterministic planning for Monkestation-style TGM columns.
//!
//! Inputs contain only owned strings/scalars: no live VM heap object crosses a
//! worker boundary. The owner thread remains responsible for ordered DM heap
//! commits and constructors.

use std::{collections::BTreeSet, sync::Arc};

// A 65,025-cell station-sized plan is still faster sequentially because the
// coordinate work is tiny relative to scoped thread setup and result merging.
// Reserve worker fan-out for substantially larger batches until the planner
// shares the persistent runtime worker pool.
const PARALLEL_CELL_THRESHOLD: usize = 262_144;

/// Immutable owner-thread snapshot of one TGM column.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GridSet {
    /// Relative column X coordinate.
    pub x: i32,
    /// Relative top Y coordinate.
    pub y: i32,
    /// Embedded Z coordinate.
    pub z: i32,
    /// Model keys from top to bottom.
    pub lines: Arc<[Arc<str>]>,
}

/// Immutable scalar/cache-key snapshot needed by the pure planner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Config {
    /// Destination X offset.
    pub x_offset: i32,
    /// Destination Y offset.
    pub y_offset: i32,
    /// Destination Z offset.
    pub z_offset: i32,
    /// Whether destination world bounds crop cells.
    pub crop_map: bool,
    /// Whether turf changes are suppressed.
    pub no_changeturf: bool,
    /// Inclusive relative lower X bound.
    pub x_lower: i32,
    /// Inclusive relative upper X bound.
    pub x_upper: i32,
    /// Inclusive relative lower Y bound.
    pub y_lower: i32,
    /// Inclusive relative upper Y bound.
    pub y_upper: i32,
    /// Optional inclusive embedded lower Z bound.
    pub z_lower: Option<i32>,
    /// Optional inclusive embedded upper Z bound.
    pub z_upper: Option<i32>,
    /// Current world maximum X.
    pub world_max_x: i32,
    /// Current world maximum Y.
    pub world_max_y: i32,
    /// Current world maximum Z.
    pub world_max_z: i32,
    /// Model key representing default space.
    /// Model key for the canonical default-space cell, when the map defines
    /// one. Maps such as Lavaland can contain no such model; in that case no
    /// cell may be elided by the `no_changeturf` fast path.
    pub space_key: Option<Arc<str>>,
    /// Keys present in the owner-thread model cache.
    pub model_keys: Arc<BTreeSet<Arc<str>>>,
}

/// Exact position in TGM source traversal order.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Ordinal {
    /// Zero-based grid position.
    pub grid: usize,
    /// Zero-based line position within the grid.
    pub line: usize,
}

/// One validated cell for later owner-thread commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cell {
    /// Stable source position.
    pub ordinal: Ordinal,
    /// Destination X coordinate.
    pub x: i32,
    /// Destination Y coordinate.
    pub y: i32,
    /// Destination Z coordinate.
    pub z: i32,
    /// Validated model-cache key.
    pub model_key: Arc<str>,
    /// Whether `AfterChange` must be suppressed for this cell.
    pub no_afterchange: bool,
}

/// Bounds contribution from cells that will actually be committed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bounds {
    /// Minimum committed X.
    pub min_x: i32,
    /// Minimum committed Y.
    pub min_y: i32,
    /// Minimum committed Z.
    pub min_z: i32,
    /// Maximum committed X.
    pub max_x: i32,
    /// Maximum committed Y.
    pub max_y: i32,
    /// Maximum committed Z.
    pub max_z: i32,
}

/// Deterministic equivalent of `_tgm_load`'s undefined-model failure context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MissingModel {
    /// Stable source position.
    pub ordinal: Ordinal,
    /// Missing model-cache key.
    pub model_key: Arc<str>,
    /// Destination X coordinate.
    pub x: i32,
    /// Destination Y coordinate.
    pub y: i32,
    /// Destination Z coordinate.
    pub z: i32,
}

/// One owner-thread action in exact `_tgm_load` source traversal order.
///
/// Planning never performs these actions. In particular, [`Self::Cell`] must
/// still invoke the ordinary DM `build_coordinate` procedure so constructors
/// and hooks remain observable. [`Self::SafepointOnly`] represents a skipped
/// default-space cell, whose `MAPLOADING_CHECK_TICK` remains observable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommitEvent {
    /// Commit one cell, then execute the corresponding scheduler safepoint.
    Cell(Cell),
    /// Execute a safepoint without constructing a cell.
    SafepointOnly(Ordinal),
    /// Raise the canonical undefined-model failure before its safepoint.
    MissingModel(MissingModel),
}

/// Pure worker result, already merged in source order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Plan {
    /// Cells ready for owner-thread commit.
    pub cells: Vec<Cell>,
    /// Aggregate bounds for those cells.
    pub bounds: Option<Bounds>,
    /// Missing keys in stable source order.
    pub missing_models: Vec<MissingModel>,
    /// Ordered commit/yield/error stream for a resumable owner-thread commit.
    pub events: Vec<CommitEvent>,
}

/// Resumable position in a [`Plan`]'s ordered owner-thread commit stream.
///
/// The cursor advances only after the caller acknowledges a completed event.
/// A yield or error therefore retains the exact event for retry/restoration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CommitCursor {
    next: usize,
}

impl CommitCursor {
    /// Returns the next event without consuming it.
    #[must_use]
    pub fn peek<'a>(&self, plan: &'a Plan) -> Option<&'a CommitEvent> {
        plan.events.get(self.next)
    }

    /// Acknowledges that the currently-peeked event completed, and advances.
    pub fn acknowledge(&mut self, plan: &Plan) -> bool {
        if self.next >= plan.events.len() {
            return false;
        }
        self.next += 1;
        true
    }

    /// Returns whether every planned event has completed.
    #[must_use]
    pub fn is_complete(&self, plan: &Plan) -> bool {
        self.next == plan.events.len()
    }
}

/// Uses available CPU parallelism for large snapshots and sequential planning
/// below the crossover threshold.
pub fn prepare(grids: &[GridSet], config: &Config) -> Plan {
    let count = grids.iter().map(|grid| grid.lines.len()).sum::<usize>();
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    if count < PARALLEL_CELL_THRESHOLD {
        sequential(grids, config, 0)
    } else {
        with_workers(grids, config, workers)
    }
}

fn with_workers(grids: &[GridSet], config: &Config, workers: usize) -> Plan {
    let workers = workers.max(1).min(grids.len().max(1));
    if workers == 1 || grids.is_empty() {
        return sequential(grids, config, 0);
    }
    let geometry = Geometry::new(grids, config);
    let chunk = grids.len().div_ceil(workers);
    let mut indexed = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for start in (0..grids.len()).step_by(chunk) {
            let end = (start + chunk).min(grids.len());
            handles.push((
                start,
                scope.spawn(move || plan_range(&grids[start..end], config, start, geometry)),
            ));
        }
        handles
            .into_iter()
            .map(|(index, handle)| (index, handle.join().expect("TGM planner worker panicked")))
            .collect::<Vec<_>>()
    });
    // Indexed stable merge, independent of worker completion order.
    indexed.sort_by_key(|(index, _)| *index);
    merge(indexed.into_iter().map(|(_, plan)| plan))
}

fn sequential(grids: &[GridSet], c: &Config, base: usize) -> Plan {
    if grids.is_empty() {
        return Plan::default();
    }
    plan_range(grids, c, base, Geometry::new(grids, c))
}

#[derive(Clone, Copy)]
struct Geometry {
    skip_start: usize,
    skip_end: usize,
    line_count: usize,
    highest_y: i32,
    final_x: i32,
    lowest_x: i32,
    xr: i32,
    z_shift: i32,
}
impl Geometry {
    fn new(grids: &[GridSet], c: &Config) -> Self {
        let xr = c.x_offset - 1;
        let yr = c.y_offset - 1;
        let relative_y = grids[0].y;
        let skip_start =
            (relative_y - (c.world_max_y - yr).min(c.y_upper).min(relative_y)).max(0) as usize;
        let line_count = grids[0].lines.len();
        let lowest_y = relative_y - line_count.saturating_sub(1) as i32;
        let skip_end = (c.y_lower.max(1 - yr) - lowest_y).max(0) as usize;
        let highest_y = relative_y + yr - skip_start as i32;
        let x_limit = if c.crop_map {
            c.x_upper.min(c.world_max_x)
        } else {
            c.x_upper
        };
        let final_x = (grids.last().expect("nonempty").x + xr).min(x_limit);
        let lowest_x = c.x_lower.max(1 - xr);
        let z_shift = c.z_offset - 1 - c.z_lower.map_or(0, |lower| lower - 1);
        Self {
            skip_start,
            skip_end,
            line_count,
            highest_y,
            final_x,
            lowest_x,
            xr,
            z_shift,
        }
    }
}

fn plan_range(grids: &[GridSet], c: &Config, base: usize, geometry: Geometry) -> Plan {
    let Geometry {
        skip_start,
        skip_end,
        line_count,
        highest_y,
        final_x,
        lowest_x,
        xr,
        z_shift,
    } = geometry;
    let mut plan = Plan::default();
    for (local, grid) in grids.iter().enumerate() {
        if c.z_lower.is_some_and(|bound| grid.z < bound)
            || c.z_upper.is_some_and(|bound| grid.z > bound)
        {
            continue;
        }
        let x = grid.x + xr;
        // Monkestation intentionally compares the lower threshold to relative x.
        if final_x < x || lowest_x > grid.x {
            continue;
        }
        let z = grid.z + z_shift;
        let no_afterchange = c.no_changeturf || z > c.world_max_z;
        let start = skip_start.min(grid.lines.len());
        let end = line_count.saturating_sub(skip_end).min(grid.lines.len());
        if start >= end {
            continue;
        }
        for line in start..end {
            let y = highest_y - (line - start) as i32;
            let key = Arc::clone(&grid.lines[line]);
            if no_afterchange
                && c.space_key
                    .as_ref()
                    .is_some_and(|space| key.as_ref() == space.as_ref())
            {
                plan.events.push(CommitEvent::SafepointOnly(Ordinal {
                    grid: base + local,
                    line,
                }));
                continue;
            }
            let ordinal = Ordinal {
                grid: base + local,
                line,
            };
            if !c.model_keys.contains(&key) {
                let missing = MissingModel {
                    ordinal,
                    model_key: key,
                    x,
                    y,
                    z,
                };
                plan.missing_models.push(missing.clone());
                plan.events.push(CommitEvent::MissingModel(missing));
                continue;
            }
            let cell = Cell {
                ordinal,
                x,
                y,
                z,
                model_key: key,
                no_afterchange,
            };
            plan.cells.push(cell.clone());
            plan.events.push(CommitEvent::Cell(cell));
            extend(&mut plan.bounds, x, y, z);
        }
    }
    plan
}

fn extend(slot: &mut Option<Bounds>, x: i32, y: i32, z: i32) {
    match slot {
        Some(b) => {
            b.min_x = b.min_x.min(x);
            b.min_y = b.min_y.min(y);
            b.min_z = b.min_z.min(z);
            b.max_x = b.max_x.max(x);
            b.max_y = b.max_y.max(y);
            b.max_z = b.max_z.max(z);
        }
        None => {
            *slot = Some(Bounds {
                min_x: x,
                min_y: y,
                min_z: z,
                max_x: x,
                max_y: y,
                max_z: z,
            });
        }
    }
}

fn merge(plans: impl IntoIterator<Item = Plan>) -> Plan {
    let mut out = Plan::default();
    for mut plan in plans {
        out.cells.append(&mut plan.cells);
        out.missing_models.append(&mut plan.missing_models);
        out.events.append(&mut plan.events);
        if let Some(b) = plan.bounds {
            extend(&mut out.bounds, b.min_x, b.min_y, b.min_z);
            extend(&mut out.bounds, b.max_x, b.max_y, b.max_z);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    fn s(value: &str) -> Arc<str> {
        Arc::from(value)
    }
    fn fixture(seed: u64) -> (Vec<GridSet>, Config) {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as u32
        };
        let grids = (0..24)
            .map(|i| GridSet {
                x: i + 1,
                y: 32,
                z: 1 + i % 4,
                lines: (0..32)
                    .map(|_| match next() % 5 {
                        0 => s("missing"),
                        1 => s("b"),
                        2 => s("space"),
                        _ => s("a"),
                    })
                    .collect::<Vec<_>>()
                    .into(),
            })
            .collect();
        let keys = [s("a"), s("b"), s("space")].into_iter().collect();
        let config = Config {
            x_offset: (next() % 4 + 1) as i32,
            y_offset: (next() % 4 + 1) as i32,
            z_offset: (next() % 3 + 1) as i32,
            crop_map: next() % 2 == 0,
            no_changeturf: next() % 2 == 0,
            x_lower: (next() % 5 + 1) as i32,
            x_upper: (next() % 8 + 20) as i32,
            y_lower: (next() % 5 + 1) as i32,
            y_upper: (next() % 8 + 24) as i32,
            z_lower: (next() % 2 == 0).then_some(2),
            z_upper: (next() % 2 == 0).then_some(3),
            world_max_x: 22,
            world_max_y: 28,
            world_max_z: 2,
            space_key: Some(s("space")),
            model_keys: Arc::new(keys),
        };
        (grids, config)
    }
    #[test]
    fn randomized_parallel_equals_sequential_across_all_tgm_controls() {
        for seed in 0..128 {
            let (g, c) = fixture(seed);
            assert_eq!(
                sequential(&g, &c, 0),
                with_workers(&g, &c, 7),
                "seed {seed}"
            );
        }
    }
    #[test]
    fn output_and_diagnostics_are_source_ordered() {
        let (g, mut c) = fixture(9);
        c.no_changeturf = true;
        let p = with_workers(&g, &c, 5);
        assert!(p.cells.windows(2).all(|v| v[0].ordinal < v[1].ordinal));
        assert!(
            p.missing_models
                .windows(2)
                .all(|v| v[0].ordinal < v[1].ordinal)
        );
        assert!(p.cells.iter().all(|v| v.model_key.as_ref() != "space"));
        assert!(
            p.missing_models
                .iter()
                .all(|v| v.model_key.as_ref() == "missing")
        );
    }

    #[test]
    fn commit_events_preserve_cells_space_safepoints_and_errors_in_source_order() {
        let grids = vec![GridSet {
            x: 1,
            y: 4,
            z: 1,
            lines: vec![s("a"), s("space"), s("missing"), s("b")].into(),
        }];
        let config = Config {
            x_offset: 1,
            y_offset: 1,
            z_offset: 1,
            crop_map: false,
            no_changeturf: true,
            x_lower: 1,
            x_upper: 1,
            y_lower: 1,
            y_upper: 4,
            z_lower: None,
            z_upper: None,
            world_max_x: 1,
            world_max_y: 4,
            world_max_z: 1,
            space_key: Some(s("space")),
            model_keys: Arc::new([s("a"), s("b"), s("space")].into_iter().collect()),
        };
        let plan = prepare(&grids, &config);
        assert!(matches!(plan.events[0], CommitEvent::Cell(_)));
        assert!(matches!(
            plan.events[1],
            CommitEvent::SafepointOnly(Ordinal { grid: 0, line: 1 })
        ));
        assert!(matches!(
            plan.events[2],
            CommitEvent::MissingModel(MissingModel {
                ordinal: Ordinal { grid: 0, line: 2 },
                ..
            })
        ));
        assert!(matches!(plan.events[3], CommitEvent::Cell(_)));
    }

    #[test]
    fn commit_cursor_does_not_consume_a_yielded_or_failed_event() {
        let (grids, mut config) = fixture(9);
        config.no_changeturf = true;
        let plan = prepare(&grids, &config);
        let mut cursor = CommitCursor::default();
        let first = cursor.peek(&plan).cloned().expect("plan has work");
        assert_eq!(cursor.peek(&plan), Some(&first));
        assert!(cursor.acknowledge(&plan));
        assert_ne!(cursor.peek(&plan), Some(&first));

        while cursor
            .peek(&plan)
            .is_some_and(|event| !matches!(event, CommitEvent::MissingModel(_)))
        {
            assert!(cursor.acknowledge(&plan));
        }
        let failed = cursor
            .peek(&plan)
            .cloned()
            .expect("fixture has missing model");
        assert!(matches!(failed, CommitEvent::MissingModel(_)));
        assert_eq!(cursor.peek(&plan), Some(&failed));
    }

    #[test]
    #[ignore = "release-only production-sized planner benchmark"]
    fn benchmark_255x255_sequential_vs_available_parallel() {
        const SIDE: usize = 255;
        const ROUNDS: usize = 20;
        let model = s("floor");
        let lines: Arc<[Arc<str>]> = (0..SIDE)
            .map(|_| Arc::clone(&model))
            .collect::<Vec<_>>()
            .into();
        let grids = (0..SIDE)
            .map(|x| GridSet {
                x: x as i32 + 1,
                y: SIDE as i32,
                z: 1,
                lines: Arc::clone(&lines),
            })
            .collect::<Vec<_>>();
        let config = Config {
            x_offset: 1,
            y_offset: 1,
            z_offset: 1,
            crop_map: false,
            no_changeturf: false,
            x_lower: 1,
            x_upper: SIDE as i32,
            y_lower: 1,
            y_upper: SIDE as i32,
            z_lower: None,
            z_upper: None,
            world_max_x: SIDE as i32,
            world_max_y: SIDE as i32,
            world_max_z: 1,
            space_key: Some(s("space")),
            model_keys: Arc::new([Arc::clone(&model)].into_iter().collect()),
        };
        let workers = std::thread::available_parallelism().map_or(1, usize::from);
        let expected = sequential(&grids, &config, 0);
        assert_eq!(expected.cells.len(), SIDE * SIDE);
        assert_eq!(expected, with_workers(&grids, &config, workers));

        let sequential_start = Instant::now();
        for _ in 0..ROUNDS {
            std::hint::black_box(sequential(&grids, &config, 0));
        }
        let sequential_elapsed = sequential_start.elapsed();
        let parallel_start = Instant::now();
        for _ in 0..ROUNDS {
            let plan = with_workers(&grids, &config, workers);
            assert_eq!(plan, expected);
            std::hint::black_box(plan);
        }
        let parallel_elapsed = parallel_start.elapsed();
        eprintln!(
            "TGM planner 255x255={} cells x{ROUNDS}, workers={workers}: sequential={sequential_elapsed:?} parallel={parallel_elapsed:?} speedup={:.2}x",
            SIDE * SIDE,
            sequential_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64(),
        );
    }
}

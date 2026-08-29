//! Deterministic native worker lanes for heap-independent batch calculations.
//!
//! The DM thread owns snapshot construction and result commit. Workers only
//! borrow immutable, pointer-free Rust values and results are restored to input
//! order before returning to the VM.

use std::sync::Arc;

/// One gas entry already resolved from both mixture lists on the DM thread.
#[derive(Clone, Debug, PartialEq)]
pub struct AtmosGasSample {
    /// Stable gas identifier in DM union-iteration order.
    pub id: Arc<str>,
    /// Selected moles/archive value from the cached mixture.
    pub cached: f32,
    /// Selected moles/archive value from the sample mixture.
    pub sample: f32,
}

/// Heap-independent input for `/datum/gas_mixture/compare`.
#[derive(Clone, Debug, PartialEq)]
pub struct AtmosCompareSnapshot {
    /// Union of gas entries, preserving DM list order.
    pub gases: Vec<AtmosGasSample>,
    /// Temperature selected from the cached mixture.
    pub cached_temperature: f32,
    /// Temperature selected from the sample mixture.
    pub sample_temperature: f32,
    /// Minimum absolute mole difference.
    pub minimum_moles_delta: f32,
    /// Minimum relative mole difference.
    pub minimum_air_ratio: f32,
    /// Minimum temperature difference.
    pub minimum_temperature_delta: f32,
}

/// Pure result of one atmos comparison job.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AtmosCompareResult {
    /// Mixtures remain compatible for group processing.
    Compatible,
    /// A gas exceeded both movement thresholds.
    Gas(Arc<str>),
    /// Temperature exceeded the suspension threshold.
    Temperature,
}

/// Computes one gas comparison without access to the VM or live heap.
#[must_use]
pub fn compare_atmos_snapshot(snapshot: &AtmosCompareSnapshot) -> AtmosCompareResult {
    let mut moles_sum = 0.0_f32;
    for gas in &snapshot.gases {
        let delta = (gas.cached - gas.sample).abs();
        if delta > snapshot.minimum_moles_delta && delta > gas.cached * snapshot.minimum_air_ratio {
            return AtmosCompareResult::Gas(Arc::clone(&gas.id));
        }
        moles_sum += gas.cached;
    }
    if moles_sum > snapshot.minimum_moles_delta
        && (snapshot.cached_temperature - snapshot.sample_temperature).abs()
            > snapshot.minimum_temperature_delta
    {
        AtmosCompareResult::Temperature
    } else {
        AtmosCompareResult::Compatible
    }
}

/// Executes immutable jobs concurrently and restores results to input order.
///
/// `workers` is capped by the job count. A single worker uses the same closure
/// and ordering path without creating threads, which provides a serial oracle.
pub fn deterministic_parallel_map<T, R, F>(jobs: &[T], workers: usize, operation: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    if jobs.is_empty() {
        return Vec::new();
    }
    let worker_count = workers.max(1).min(jobs.len());
    if worker_count == 1 {
        return jobs.iter().map(&operation).collect();
    }
    let chunk_size = jobs.len().div_ceil(worker_count);
    let mut chunks = std::thread::scope(|scope| {
        jobs.chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let operation = &operation;
                scope.spawn(move || (chunk_index, chunk.iter().map(operation).collect::<Vec<_>>()))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("native worker lane must not panic"))
            .collect::<Vec<_>>()
    });
    chunks.sort_by_key(|(index, _)| *index);
    chunks
        .into_iter()
        .flat_map(|(_, results)| results)
        .collect()
}

/// Runs a batch of atmos comparisons on immutable snapshots.
pub fn compare_atmos_batch(
    jobs: &[AtmosCompareSnapshot],
    workers: usize,
) -> Vec<AtmosCompareResult> {
    deterministic_parallel_map(jobs, workers, compare_atmos_snapshot)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::thread::ThreadId;

    use super::*;

    fn snapshot(id: usize) -> AtmosCompareSnapshot {
        AtmosCompareSnapshot {
            gases: vec![AtmosGasSample {
                id: Arc::from(format!("gas-{id}")),
                cached: id as f32,
                sample: id as f32 + if id % 3 == 0 { 5.0 } else { 0.0 },
            }],
            cached_temperature: 300.0,
            sample_temperature: if id % 5 == 0 { 400.0 } else { 300.0 },
            minimum_moles_delta: 1.0,
            minimum_air_ratio: 0.1,
            minimum_temperature_delta: 20.0,
        }
    }

    #[test]
    fn atmos_batch_matches_serial_and_preserves_input_order() {
        let jobs = (0..257).map(snapshot).collect::<Vec<_>>();
        let serial = compare_atmos_batch(&jobs, 1);
        let parallel = compare_atmos_batch(&jobs, 4);
        assert_eq!(parallel, serial);
        assert!(matches!(parallel[3], AtmosCompareResult::Gas(ref gas) if &**gas == "gas-3"));
        assert_eq!(parallel[5], AtmosCompareResult::Temperature);
    }

    #[test]
    fn worker_lane_uses_multiple_threads_without_sharing_mutable_jobs() {
        let jobs = (0..128_u32).collect::<Vec<_>>();
        let threads = Mutex::new(HashSet::<ThreadId>::new());
        let output = deterministic_parallel_map(&jobs, 4, |value| {
            threads.lock().unwrap().insert(std::thread::current().id());
            value * value
        });
        assert_eq!(
            output,
            jobs.iter().map(|value| value * value).collect::<Vec<_>>()
        );
        assert!(threads.into_inner().unwrap().len() > 1);
    }
}

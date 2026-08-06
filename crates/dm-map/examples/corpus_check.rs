use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Default)]
struct CorpusStats {
    parsed_files: usize,
    bytes: u64,
    keys: usize,
    atoms: usize,
    assignments: usize,
    blocks: usize,
    cells: usize,
    value_kinds: BTreeMap<dm_map::MapValueKind, usize>,
}

fn main() -> ExitCode {
    let roots: Vec<_> = env::args_os().skip(1).map(PathBuf::from).collect();
    if roots.is_empty() {
        eprintln!("usage: cargo run -p dm-map --example corpus_check -- <file-or-directory>...");
        return ExitCode::from(2);
    }

    let mut files = Vec::new();
    let mut failures = Vec::new();
    for root in roots {
        if let Err(error) = collect_maps(&root, &mut files) {
            failures.push(format!("{}: {error}", root.display()));
        }
    }
    files.sort();
    files.dedup();

    let mut stats = CorpusStats::default();
    for path in &files {
        if let Err(error) = check_map(path, &mut stats) {
            failures.push(format!("{}: {error}", path.display()));
        }
    }

    println!(
        "files={} parsed={} failures={}",
        files.len(),
        stats.parsed_files,
        failures.len()
    );
    println!(
        "bytes={} keys={} atoms={} assignments={} blocks={} cells={}",
        stats.bytes, stats.keys, stats.atoms, stats.assignments, stats.blocks, stats.cells
    );
    print!("value-kinds:");
    for (kind, count) in &stats.value_kinds {
        print!(" {}={count}", kind.label());
    }
    println!();
    for failure in &failures {
        eprintln!("{failure}");
    }
    if failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn collect_maps(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if path.is_file() {
        if has_dmm_extension(path) {
            files.push(path.to_owned());
        }
        return Ok(());
    }

    let mut children = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
    children.sort_by_key(fs::DirEntry::path);
    for child in children {
        let file_type = child.file_type()?;
        if file_type.is_dir() {
            collect_maps(&child.path(), files)?;
        } else if file_type.is_file() && has_dmm_extension(&child.path()) {
            files.push(child.path());
        }
    }
    Ok(())
}

fn has_dmm_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("dmm"))
}

fn check_map(path: &Path, stats: &mut CorpusStats) -> Result<(), String> {
    let source = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let map = dm_map::parse(&source).map_err(|error| error.to_string())?;
    stats.parsed_files += 1;
    stats.bytes += u64::try_from(source.len()).map_err(|error| error.to_string())?;
    stats.keys += map.keys.len();
    stats.atoms += map.keys.values().map(|key| key.atoms.len()).sum::<usize>();
    stats.assignments += map
        .keys
        .values()
        .flat_map(|key| &key.atoms)
        .map(|atom| atom.variable_assignments.len())
        .sum::<usize>();
    for assignment in map
        .keys
        .values()
        .flat_map(|key| &key.atoms)
        .flat_map(|atom| &atom.variable_assignments)
    {
        *stats.value_kinds.entry(assignment.value.kind).or_default() += 1;
    }
    stats.blocks += map.blocks.len();
    stats.cells += map
        .blocks
        .iter()
        .flat_map(|block| &block.rows)
        .map(Vec::len)
        .sum::<usize>();
    Ok(())
}

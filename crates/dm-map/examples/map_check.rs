use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1) else {
        eprintln!("usage: cargo run -p dm-map --example map_check -- <map.dmm>");
        return ExitCode::from(2);
    };
    let source = match fs::read_to_string(&path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("{}: {error}", path.to_string_lossy());
            return ExitCode::from(2);
        }
    };
    match dm_map::parse(&source) {
        Ok(map) => {
            let atoms: usize = map.keys.values().map(|key| key.atoms.len()).sum();
            let cells: usize = map
                .blocks
                .iter()
                .flat_map(|block| &block.rows)
                .map(Vec::len)
                .sum();
            println!(
                "keys={} atoms={} blocks={} cells={} key_width={}",
                map.keys.len(),
                atoms,
                map.blocks.len(),
                cells,
                map.key_width
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{}: {error}", path.to_string_lossy());
            ExitCode::FAILURE
        }
    }
}

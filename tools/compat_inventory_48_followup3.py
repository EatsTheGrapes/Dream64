from pathlib import Path

p = Path("crates/dm-runtime/src/lib.rs")
text = p.read_text()
old = "        assert_eq!(state.project_root(), Some(fixture.root.as_path()));\n"
new = "        assert_eq!(state.project_root(), Some(fixture.0.as_path()));\n"
if text.count(old) != 1:
    raise SystemExit(f"runtime fixture assertion anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

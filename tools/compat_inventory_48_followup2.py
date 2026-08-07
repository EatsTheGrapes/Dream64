from pathlib import Path

p = Path("crates/dm-runtime/src/lib.rs")
text = p.read_text()
old = "            while let Some(candidate) = current {\n"
new = "            while let Some(candidate) = current.take() {\n"
if text.count(old) != 1:
    raise SystemExit(f"runtime parent traversal anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

from pathlib import Path

p = Path("crates/dm-vm/src/builtins.rs")
text = p.read_text()
old = "    let offsets = text\n        .char_indices()\n"
new = "    let mut offsets = text\n        .char_indices()\n"
if text.count(old) != 1:
    raise SystemExit(f"findtext offsets anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

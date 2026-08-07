from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = "use dm_value::{DatumId, FieldName, ListId, TypePath, ValueError, ValueHeap};\n"
new = "use dm_value::{FieldName, ListId, TypePath, ValueError, ValueHeap};\n"
if text.count(old) != 1:
    raise SystemExit(f"DatumId import anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

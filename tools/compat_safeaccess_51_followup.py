from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = '("bump".to_owned(), ProcedureId(2))'
new = '("bump".to_owned(), super::ProcedureId(2))'
if text.count(old) != 1:
    raise SystemExit(f"ProcedureId test anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

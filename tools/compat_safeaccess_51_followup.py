from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = 'if matches!(self.current_operator(), Some("." | ":" | "?." | "?:")) {'
new = 'if matches!(self.current_operator(), Some("." | "?." | "?:")) {'
if text.count(old) != 1:
    raise SystemExit(f"member access parser anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

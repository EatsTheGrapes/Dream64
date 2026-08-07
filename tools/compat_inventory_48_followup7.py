from pathlib import Path

p = Path("crates/dm-lifecycle/tests/headless_boot.rs")
text = p.read_text()
old = '    assert!(error.message.contains("numeric operation received"));\n'
new = '    assert!(error.message.contains("addition requires compatible DM values"));\n'
if text.count(old) != 1:
    raise SystemExit(f"headless arithmetic assertion expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

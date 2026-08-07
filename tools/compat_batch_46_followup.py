from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = '''            "/proc/middle()\\n\\treturn copytext_char(\"AéB\", 2, 3)\\n/proc/tail()\\n\\treturn copytext_char(\"Hi there\", -5)\\n",\n'''
# The first staged patch produced literal, unescaped quotes in the Rust source.
broken = '''            "/proc/middle()\\n\\treturn copytext_char("AéB", 2, 3)\\n/proc/tail()\\n\\treturn copytext_char("Hi there", -5)\\n",\n'''
if text.count(broken) != 1:
    raise SystemExit(f"expected one broken copytext fixture, found {text.count(broken)}")
text = text.replace(broken, old, 1)
p.write_text(text)

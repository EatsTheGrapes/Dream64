from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
broken = r'''            "/proc/middle()\n\treturn copytext_char("AéB", 2, 3)\n/proc/tail()\n\treturn copytext_char("Hi there", -5)\n",
'''
fixed = r'''            "/proc/middle()\n\treturn copytext_char(\"AéB\", 2, 3)\n/proc/tail()\n\treturn copytext_char(\"Hi there\", -5)\n",
'''
if text.count(broken) != 1:
    raise SystemExit(f"expected one broken copytext fixture, found {text.count(broken)}")
text = text.replace(broken, fixed, 1)
p.write_text(text)

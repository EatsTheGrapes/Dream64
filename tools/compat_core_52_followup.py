from pathlib import Path

# Native builtin lint cleanup.
p = Path("crates/dm-vm/src/builtins.rs")
text = p.read_text()
old = '''use std::fs;
use std::path::PathBuf;
'''
new = '''use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
'''
if text.count(old) != 1:
    raise SystemExit("fmt Write import anchor missing")
text = text.replace(old, new, 1)

old = '''        "cmptext" | "cmptextEx" | "sorttext" | "sorttextEx" | "sortText" => (0, usize::MAX),
'''
new = '''        "cmptext" | "cmptextEx" | "sorttext" | "sorttextEx" | "sortText" | "addtext" => {
            (0, usize::MAX)
        }
'''
if text.count(old) != 1:
    raise SystemExit("variadic arity merge anchor missing")
text = text.replace(old, new, 1)
old = '''        "addtext" => (0, usize::MAX),
'''
if text.count(old) != 1:
    raise SystemExit("standalone addtext arity anchor missing")
text = text.replace(old, "", 1)

old = '''            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(char::from(byte))
            }
            b' ' => output.push('+'),
            _ => output.push_str(&format!("%{byte:02X}")),
'''
new = '''            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(char::from(byte));
            }
            b' ' => output.push('+'),
            _ => write!(&mut output, "%{byte:02X}").expect("writing to a String cannot fail"),
'''
if text.count(old) != 1:
    raise SystemExit("form encoding lint anchor missing")
text = text.replace(old, new, 1)
p.write_text(text)

# VM comparison lint cleanup.
p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
replacements = {
    "comparison.is_some_and(|value| value.is_lt())": "comparison.is_some_and(std::cmp::Ordering::is_lt)",
    "comparison.is_some_and(|value| value.is_le())": "comparison.is_some_and(std::cmp::Ordering::is_le)",
    "comparison.is_some_and(|value| value.is_gt())": "comparison.is_some_and(std::cmp::Ordering::is_gt)",
    "comparison.is_some_and(|value| value.is_ge())": "comparison.is_some_and(std::cmp::Ordering::is_ge)",
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"comparison lint anchor expected once: {old}")
    text = text.replace(old, new, 1)
p.write_text(text)

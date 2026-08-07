from pathlib import Path

path = Path("crates/dm-runtime/src/lib.rs")
text = path.read_text()
old = '        assert_eq!(names, ["name", "health", "speed"]);\n'
new = '        assert_eq!(names, ["name", "health", "speed", "tag"]);\n'
if text.count(old) != 1:
    raise SystemExit(f"expected one runtime field-order assertion, found {text.count(old)}")
path.write_text(text.replace(old, new, 1))

project = Path("crates/dm-project/src/lib.rs")
text = project.read_text()
old = '                output.push_str(&format!("{file_macro:?}"));\n'
new = '                let file_literal = format!("{file_macro:?}");\n                output.push_str(&file_literal);\n'
if text.count(old) != 1:
    raise SystemExit(f"expected one file-macro string append, found {text.count(old)}")
project.write_text(text.replace(old, new, 1))

from pathlib import Path

p = Path("crates/dm-project/src/lib.rs")
text = p.read_text()

text = text.replace(
    "    fn append_expanded_source(\n",
    "    #[allow(clippy::too_many_lines)]\n    fn append_expanded_source(\n",
    1,
)

old_direct = '''                        source.as_bytes()[..invocation.start]\n                            .iter()\n                            .filter(|byte| **byte == b'\\n')\n                            .count()\n                            .saturating_add(1)\n                            .to_string()\n'''
if text.count(old_direct) != 1:
    raise SystemExit(f"expected direct __LINE__ byte count once, found {text.count(old_direct)}")
text = text.replace(old_direct, "                        source_line_number(source, invocation.start).to_string()\n", 1)

old_nested = '''                    let line_macro = source.as_bytes()[..invocation.start]\n                        .iter()\n                        .filter(|byte| **byte == b'\\n')\n                        .count()\n                        .saturating_add(1);\n'''
if text.count(old_nested) != 1:
    raise SystemExit(f"expected nested __LINE__ byte count once, found {text.count(old_nested)}")
text = text.replace(old_nested, "                    let line_macro = source_line_number(source, invocation.start);\n", 1)

anchor = "const MAX_MACRO_EXPANSION_DEPTH: usize = 64;\n"
helper = '''fn source_line_number(source: &str, offset: usize) -> usize {\n    source[..offset.min(source.len())]\n        .matches('\\n')\n        .count()\n        .saturating_add(1)\n}\n\n'''
if helper not in text:
    if text.count(anchor) != 1:
        raise SystemExit(f"expected macro depth anchor once, found {text.count(anchor)}")
    text = text.replace(anchor, helper + anchor, 1)

needle = "fn substitute_function_macro(\n"
if "#[allow(clippy::too_many_arguments)]\nfn substitute_function_macro(\n" not in text:
    if text.count(needle) != 1:
        raise SystemExit(f"expected substitute function once, found {text.count(needle)}")
    text = text.replace(needle, "#[allow(clippy::too_many_arguments)]\n" + needle, 1)

p.write_text(text)

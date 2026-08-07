from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
replacements = [
    (
        "fn compile_assignment_statement(\n",
        "#[allow(clippy::too_many_lines)]\nfn compile_assignment_statement(\n",
        "assignment statement lint",
    ),
    (
        "fn emit_assignment_expression(\n",
        "#[allow(clippy::too_many_lines)]\nfn emit_assignment_expression(\n",
        "assignment expression lint",
    ),
    (
        "    #[test]\n    fn null_conditional_field_index_and_call_short_circuit_without_rhs_evaluation() {\n",
        "    #[test]\n    #[allow(clippy::vec_init_then_push)]\n    fn null_conditional_field_index_and_call_short_circuit_without_rhs_evaluation() {\n",
        "safe access regression lint",
    ),
]
for old, new, label in replacements:
    if text.count(old) != 1:
        raise SystemExit(f"{label} anchor expected once, found {text.count(old)}")
    text = text.replace(old, new, 1)
p.write_text(text)

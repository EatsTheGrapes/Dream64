from pathlib import Path

p = Path("crates/dm-vm/src/builtins.rs")
text = p.read_text()
old = '''        CompoundAssignmentOperator::Multiply
        | CompoundAssignmentOperator::Divide
        | CompoundAssignmentOperator::Remainder
        | CompoundAssignmentOperator::ShiftLeft
'''
new = '''        CompoundAssignmentOperator::Multiply
        | CompoundAssignmentOperator::Divide
        | CompoundAssignmentOperator::Remainder
        | CompoundAssignmentOperator::FractionalRemainder
        | CompoundAssignmentOperator::ShiftLeft
'''
if text.count(old) != 1:
    raise SystemExit(f"list compound rejection anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

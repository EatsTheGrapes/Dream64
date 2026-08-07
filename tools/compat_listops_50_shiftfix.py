from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = '''        CompoundAssignmentOperator::ShiftLeft => shift_binary(left, right, true),
        CompoundAssignmentOperator::ShiftRight => shift_binary(left, right, false),
'''
new = '''        CompoundAssignmentOperator::ShiftLeft => {
            bitwise_shift(left, right, |left, right| left << right)
        }
        CompoundAssignmentOperator::ShiftRight => {
            bitwise_shift(left, right, |left, right| left >> right)
        }
'''
if text.count(old) != 1:
    raise SystemExit(f"compound shift anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

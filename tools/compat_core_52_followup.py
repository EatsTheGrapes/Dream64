from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = '''            Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual
            | Instruction::Compare => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let comparison = compare_values(&left, &right)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let result = match instruction {
                    Instruction::Less => comparison.is_some_and(|value| value.is_lt()),
                    Instruction::LessEqual => comparison.is_some_and(|value| value.is_le()),
                    Instruction::Greater => comparison.is_some_and(|value| value.is_gt()),
                    Instruction::GreaterEqual => comparison.is_some_and(|value| value.is_ge()),
                    Instruction::Compare => {
                        let value = comparison.map_or(0.0, |value| match value {
                            std::cmp::Ordering::Less => -1.0,
                            std::cmp::Ordering::Equal => 0.0,
                            std::cmp::Ordering::Greater => 1.0,
                        });
                        frames[frame_index].stack.push(Value::number(value));
                        continue;
                    }
                    _ => unreachable!(),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
'''
new = '''            Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let comparison = compare_values(&left, &right)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let result = match instruction {
                    Instruction::Less => comparison.is_some_and(|value| value.is_lt()),
                    Instruction::LessEqual => comparison.is_some_and(|value| value.is_le()),
                    Instruction::Greater => comparison.is_some_and(|value| value.is_gt()),
                    Instruction::GreaterEqual => comparison.is_some_and(|value| value.is_ge()),
                    _ => unreachable!(),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::Compare => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let comparison = compare_values(&left, &right)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = comparison.map_or(0.0, |value| match value {
                    std::cmp::Ordering::Less => -1.0,
                    std::cmp::Ordering::Equal => 0.0,
                    std::cmp::Ordering::Greater => 1.0,
                });
                frames[frame_index].stack.push(Value::number(value));
            }
'''
if text.count(old) != 1:
    raise SystemExit(f"comparison interpreter anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

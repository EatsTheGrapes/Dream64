from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()

old = '''                let value = match read_list_value(&state.heap, list, &key) {
                    Ok(value) => value.clone(),
                    Err(error) => {
                        return Err(execution_error(module, &frames, error.to_string()));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::SetListIndex => {
'''
new = '''                let value = match read_list_value(&state.heap, list, &key) {
                    Ok(value) => value.clone(),
                    // BYOND associative lookup returns null for an absent key.
                    // Lazy-list idioms such as `lists[target] ||= list()` rely
                    // on this before inserting the new association.
                    Err(ValueError::MissingKey) => Value::Null,
                    Err(error) => {
                        return Err(execution_error(module, &frames, error.to_string()));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::SetListIndex => {
'''
if text.count(old) != 1:
    raise SystemExit(f"IndexList read anchor expected once, found {text.count(old)}")
text = text.replace(old, new, 1)

old = '''            "/datum/example/proc/run()\\n\\tvar/local\\n\\tlocal ||= 3\\n\\tvar/list/values = list()\\n\\tvalues[1] ||= 4\\n\\tsrc.flag ||= 5\\n\\treturn local + values[1] + src.flag\\n",
'''
new = '''            "/datum/example/proc/run()\\n\\tvar/local\\n\\tlocal ||= 3\\n\\tvar/list/values = list()\\n\\tvalues[\\\"entry\\\"] ||= 4\\n\\tsrc.flag ||= 5\\n\\treturn local + values[\\\"entry\\\"] + src.flag\\n",
'''
if text.count(old) != 1:
    raise SystemExit(f"logical assignment fixture expected once, found {text.count(old)}")
text = text.replace(old, new, 1)

p.write_text(text)

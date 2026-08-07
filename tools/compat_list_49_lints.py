from pathlib import Path

p = Path("crates/dm-vm/src/builtins.rs")
text = p.read_text()
old = '''    let mut index = start.min(
        state
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .len()
            + 1,
    );
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for value in values {
        target
            .insert(index, value)
            .map_err(|error| error.to_string())?;
        index += 1;
    }
'''
new = '''    let index = start.min(
        state
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .len()
            + 1,
    );
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for (offset, value) in values.into_iter().enumerate() {
        target
            .insert(index + offset, value)
            .map_err(|error| error.to_string())?;
    }
'''
if text.count(old) != 1:
    raise SystemExit(f"splice counter anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = '''                            Value::Null => 0,
                            _ => 0,
'''
new = '''                            _ => 0,
'''
if text.count(old) != 1:
    raise SystemExit(f"list len duplicate arm expected once, found {text.count(old)}")
text = text.replace(old, new, 1)
old = '''fn datum_receiver(value: &Value, operation: &str) -> Result<DatumId, String> {
    match value {
        Value::Datum(datum) => Ok(*datum),
        Value::Null => Err(format!("{operation} received null")),
        _ => Err(format!("{operation} requires a datum, received {value}")),
    }
}

'''
if text.count(old) != 1:
    raise SystemExit(f"unused datum_receiver anchor expected once, found {text.count(old)}")
text = text.replace(old, "", 1)
p.write_text(text)

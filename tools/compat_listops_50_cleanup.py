from pathlib import Path

p = Path("crates/dm-vm/src/builtins.rs")
text = p.read_text()
old = '''    list.positions()
        .map(|(_, key)| {
            let associated = list.get_key(key).ok().cloned();
            ListOperatorEntry {
                key: key.clone(),
                associated,
            }
        })
        .collect::<Vec<_>>()
        .pipe(Ok)
}

trait Pipe: Sized {
    fn pipe<T>(self, function: impl FnOnce(Self) -> T) -> T {
        function(self)
    }
}
impl<T> Pipe for T {}
'''
new = '''    Ok(list
        .positions()
        .map(|(_, key)| {
            let associated = list.get_key(key).ok().cloned();
            ListOperatorEntry {
                key: key.clone(),
                associated,
            }
        })
        .collect())
}
'''
if text.count(old) != 1:
    raise SystemExit(f"list snapshot pipe anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = '            Ok(Value::number(23.0))\n'
new = '            Ok(Value::number(24.0))\n'
if text.count(old) != 1:
    raise SystemExit(f"binary operator expected result anchor expected once, found {text.count(old)}")
text = text.replace(old, new, 1)
old = '            Ok(Value::number(16.0))\n'
new = '            Ok(Value::number(17.0))\n'
if text.count(old) != 1:
    raise SystemExit(f"compound operator expected result anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

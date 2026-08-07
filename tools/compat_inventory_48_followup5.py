from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = '''                let value = match (left, right) {
                    (Value::Number(left), Value::Number(right)) => {
                        Value::number(left.to_f32() + right.to_f32())
                    }
                    (Value::Text(left), Value::Text(right)) => {
                        Value::text(format!("{left}{right}"))
                    }
                    (left, right) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!(
                                "addition requires two numbers or two text values, received {left} and {right}"
                            ),
                        ));
                    }
                };
'''
new = '''                let value = match (left, right) {
                    (Value::Number(left), Value::Number(right)) => {
                        Value::number(left.to_f32() + right.to_f32())
                    }
                    (Value::Null, Value::Number(right)) => Value::number(right.to_f32()),
                    (Value::Number(left), Value::Null) => Value::number(left.to_f32()),
                    (Value::Null, Value::Null) => Value::number(0.0),
                    (Value::Text(left), Value::Text(right)) => {
                        Value::text(format!("{left}{right}"))
                    }
                    (left, right) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!(
                                "addition requires compatible DM values, received {left} and {right}"
                            ),
                        ));
                    }
                };
'''
if text.count(old) != 1:
    raise SystemExit(f"formatted add execution anchor expected once, found {text.count(old)}")
text = text.replace(old, new, 1)
old = '''fn pop_number(stack: &mut Vec<Value>) -> Result<f32, String> {
    let value = pop(stack)?;
    value
        .as_number()
        .ok_or_else(|| format!("numeric operation received {value}"))
}
'''
new = '''fn pop_number(stack: &mut Vec<Value>) -> Result<f32, String> {
    let value = pop(stack)?;
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        value => Err(format!("numeric operation received {value}")),
    }
}
'''
if text.count(old) != 1:
    raise SystemExit(f"pop_number anchor expected once, found {text.count(old)}")
text = text.replace(old, new, 1)
p.write_text(text)

p = Path("crates/dm-semantics/src/lib.rs")
text = p.read_text()
old = '''        let error = execute_effective(&compilation, "/datum/base/child/proc/run", &[Value::Null])
            .expect_err("explicit null should be reused rather than defaulted");
        assert_eq!(error.message, "numeric operation received null");
'''
new = '''        assert_eq!(
            execute_effective(&compilation, "/datum/base/child/proc/run", &[Value::Null]),
            Ok(Value::number(11.0)),
            "explicit null is reused and BYOND arithmetic treats it as numeric zero",
        );
'''
if text.count(old) != 1:
    raise SystemExit(f"semantic null arithmetic test anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()

old = '''                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("field read requires a datum, received {value}"),
                        ));
                    }
'''
new = '''                    Value::Null => {
                        return Err(execution_error(module, &frames, "field read received null"));
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("field read requires a datum, received {value}"),
                        ));
                    }
'''
if text.count(old) != 1:
    raise SystemExit(f"field diagnostic anchor expected once, found {text.count(old)}")
text = text.replace(old, new, 1)

old = '        assert!(error.message.contains("numeric operation received"));\n'
new = '        assert!(error.message.contains("addition requires compatible DM values"));\n'
if text.count(old) != 1:
    raise SystemExit(f"callee error assertion expected once, found {text.count(old)}")
text = text.replace(old, new, 1)

old = '''        let error = execute(
            &program,
            &[Value::number(10.0), Value::Null, Value::number(1.0)],
        )
        .expect_err("explicit null should suppress the second parameter default");
        assert_eq!(error.message, "numeric operation received null");
        assert_eq!(error.source_span, Some(syntax.definitions[0].body[0].span));
'''
new = '''        assert_eq!(
            execute(
                &program,
                &[Value::number(10.0), Value::Null, Value::number(1.0)],
            ),
            Ok(Value::number(11.0)),
            "explicit null suppresses the default and participates in arithmetic as numeric zero",
        );
'''
if text.count(old) != 1:
    raise SystemExit(f"parameter default null test expected once, found {text.count(old)}")
text = text.replace(old, new, 1)

p.write_text(text)

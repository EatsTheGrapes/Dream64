from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()

# Statement-level assignment detection must recognize the documented %%= form.
old = '''                        | "%="
                        | "&="
                        | "|="
'''
new = '''                        | "%="
                        | "%%="
                        | "&="
                        | "|="
'''
if text.count(old) != 1:
    raise SystemExit(f"top-level %%= anchor expected once, found {text.count(old)}")
text = text.replace(old, new, 1)

# Remove old invalid signed-input expectations from tests. BYOND documents
# bitwise operands as the 24-bit unsigned range 0..2^24-1.
old = '''        let source = "/proc/probe()\\n\\treturn (-1 & 6) + (7 ^ 3 | 8) + (9.9 & 3)\\n";
'''
new = '''        let source = "/proc/probe()\\n\\treturn (0xFFFFFF & 6) + (7 ^ 3 | 8) + (9.9 & 3)\\n";
'''
if text.count(old) != 1:
    raise SystemExit("bitwise positive-range fixture anchor missing")
text = text.replace(old, new, 1)
text = text.replace(
    "        // -1 & 6 = 6, (7 ^ 3) | 8 = 12, and 9.9 truncates to 9 before\\n",
    "        // 0xFFFFFF & 6 = 6, (7 ^ 3) | 8 = 12, and 9.9 truncates to 9 before\\n",
    1,
)

old = '''        assert_eq!(execute(&program, &[]), Ok(Value::number(-11.0)));
'''
new = '''        // ~9 and ~0 are 24-bit complements. Their binary32 sum rounds to
        // the nearest representable value at this magnitude.
        assert_eq!(execute(&program, &[]), Ok(Value::number(33_554_420.0)));
'''
if text.count(old) != 1:
    raise SystemExit("bitwise complement expectation anchor missing")
text = text.replace(old, new, 1)

old = '''        let source = "/proc/probe(items)\\n\\tvar/value = 3 << 2\\n\\tvalue >>= 1\\n\\titems[1] <<= value\\n\\treturn (-8 >> 2) + items[1] + (1 << 33)\\n";
'''
new = '''        let source = "/proc/probe(items)\\n\\tvar/value = 3 << 2\\n\\tvalue >>= 1\\n\\titems[1] <<= value\\n\\treturn (8 >> 2) + items[1] + (1 << 33)\\n";
'''
if text.count(old) != 1:
    raise SystemExit("shift positive-range fixture anchor missing")
text = text.replace(old, new, 1)
text = text.replace(
    '''        // value is (3 << 2) >> 1 = 6; item becomes 1 << 6 = 64. Right
        // BYOND shifts are limited to the low 24 bits; counts >=24 yield zero.
''',
    '''        // value is (3 << 2) >> 1 = 6; item becomes 1 << 6 = 64.
        // 8 >> 2 is 2, and counts >=24 shift every effective bit away.
''',
    1,
)
old = '''            Ok(Value::number(62.0))
'''
new = '''            Ok(Value::number(66.0))
'''
if text.count(old) != 1:
    raise SystemExit("shift expectation anchor missing")
text = text.replace(old, new, 1)

# Exercise the proactive pure standard-proc additions through normal DM source.
test_anchor = '''    #[test]
    fn documented_operator_semantics_cover_short_circuit_modulo_compare_and_equivalence() {
'''
test = r'''    #[test]
    fn documented_pure_standard_procs_cover_sort_params_and_number_text() {
        let source = parse(
            "/proc/probe()\n\tvar/list/p = params2list(\"a=one+two&b=%26\")\n\tif(p[\"a\"] != \"one two\" || p[\"b\"] != \"&\")\n\t\treturn 0\n\tif(list2params(p) != \"a=one+two&b=%26\")\n\t\treturn 0\n\tif(lentext(\"abc\") != 3)\n\t\treturn 0\n\tif(sorttext(\"A\", \"b\") != 1 || sorttextEx(\"a\", \"B\") != -1)\n\t\treturn 0\n\tif(num2text(11, 2, 16) != \"0b\")\n\t\treturn 0\n\treturn 1\n",
        )
        .expect("pure standard-proc source should parse");
        let module = compile_module(&source.definitions).expect("pure standard procs should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
            Ok(Value::number(1.0))
        );
    }

'''
if text.count(test_anchor) != 1:
    raise SystemExit("pure standard-proc test insertion anchor missing")
text = text.replace(test_anchor, test + test_anchor, 1)

p.write_text(text)

from pathlib import Path

p = Path("crates/dm-vm/src/builtins.rs")
text = p.read_text()

intro = '''//! Native implementations of documented BYOND global procedures.
//!
//! These routines are deliberately runtime primitives rather than injected DM
//! source when their behavior depends on host state, type metadata, or precise
//! text-indexing semantics.

'''
replacement = intro + '''#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::unnecessary_wraps,
    reason = "DM uses binary32 numbers for integer/index boundaries and native builtin dispatch shares a Result ABI"
)]

'''
if text.count(intro) != 1:
    raise SystemExit(f"builtin module intro expected once, found {text.count(intro)}")
text = text.replace(intro, replacement, 1)

old = '''        "findtext"
        | "findtextEx"
        | "findtext_char"
        | "findtextEx_char"
        | "findlasttext"
        | "findlasttextEx"
        | "findlasttext_char"
        | "findlasttextEx_char" => (2, 4),
        "splittext" | "splittext_char" => (2, 5),
        "jointext" => (2, 4),
'''
new = '''        "findtext"
        | "findtextEx"
        | "findtext_char"
        | "findtextEx_char"
        | "findlasttext"
        | "findlasttextEx"
        | "findlasttext_char"
        | "findlasttextEx_char"
        | "jointext" => (2, 4),
        "splittext" | "splittext_char" => (2, 5),
'''
if text.count(old) != 1:
    raise SystemExit(f"arity duplicate match anchor expected once, found {text.count(old)}")
text = text.replace(old, new, 1)

old = '''    let explicit = match path {
        "/obj" | "/mob" => Some("/atom/movable"),
        "/area" | "/turf" => Some("/atom"),
        "/atom/movable" => Some("/atom"),
        "/atom" => Some("/datum"),
        "/datum" | "/world" | "/list" | "/client" => None,
        _ => None,
    };
'''
new = '''    let explicit = match path {
        "/obj" | "/mob" => Some("/atom/movable"),
        "/area" | "/turf" | "/atom/movable" => Some("/atom"),
        "/atom" => Some("/datum"),
        _ => None,
    };
'''
if text.count(old) != 1:
    raise SystemExit(f"fallback parent lint anchor expected once, found {text.count(old)}")
text = text.replace(old, new, 1)

old = '''fn turn(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let direction = number(&arguments[0], "turn direction")?.trunc() as i32;
    let angle = number(&arguments[1], "turn angle")?;
    let steps = (angle / 45.0).trunc() as i32;
    if steps == 0 {
        return Ok(Value::number(direction as f32));
    }
    const DIRECTIONS: [i32; 8] = [1, 9, 8, 10, 2, 6, 4, 5];
'''
new = '''fn turn(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    const DIRECTIONS: [i32; 8] = [1, 9, 8, 10, 2, 6, 4, 5];
    let direction = number(&arguments[0], "turn direction")?.trunc() as i32;
    let angle = number(&arguments[1], "turn angle")?;
    let steps = (angle / 45.0).trunc() as i32;
    if steps == 0 {
        return Ok(Value::number(direction as f32));
    }
'''
if text.count(old) != 1:
    raise SystemExit(f"turn const anchor expected once, found {text.count(old)}")
text = text.replace(old, new, 1)
p.write_text(text)

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
text = text.replace(
    "    /// Replaces the runtime type-parent catalog used by subtype and parent_type lookups.\n",
    "    /// Replaces the runtime type-parent catalog used by subtype and `parent_type` lookups.\n",
    1,
)
text = text.replace(
    "    /// Sets the project root used by BYOND filesystem procedures such as fexists().\n",
    "    /// Sets the project root used by BYOND filesystem procedures such as `fexists()`.\n",
    1,
)
p.write_text(text)

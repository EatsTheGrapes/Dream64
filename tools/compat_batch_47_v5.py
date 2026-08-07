from pathlib import Path

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
old = '''        "SOUTHEAST" => Some(6.0),
        "WEST" => Some(8.0),
        "NORTHWEST" => Some(9.0),
        "SOUTHWEST" => Some(10.0),
        "UP" => Some(16.0),
        "DOWN" => Some(32.0),
        // Appearance flags are BYOND bitflags. Keep the complete contiguous
        // built-in flag family here rather than teaching project code about
        // individual flags as each one is encountered.
        // These make an overlay/image ignore the corresponding value
        // inherited from its parent.
        "RESET_TRANSFORM" => Some(8.0),
        "RESET_COLOR" => Some(16.0),
        "RESET_ALPHA" => Some(32.0),
'''
new = '''        "SOUTHEAST" => Some(6.0),
        "WEST" | "RESET_TRANSFORM" => Some(8.0),
        "NORTHWEST" => Some(9.0),
        "SOUTHWEST" => Some(10.0),
        "UP" | "RESET_COLOR" => Some(16.0),
        "DOWN" | "RESET_ALPHA" => Some(32.0),
        // Appearance flags are BYOND bitflags. Keep the complete contiguous
        // built-in flag family here rather than teaching project code about
        // individual flags as each one is encountered.
        // These make an overlay/image ignore the corresponding value
        // inherited from its parent.
'''
if text.count(old) != 1:
    raise SystemExit(f"constant lint anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

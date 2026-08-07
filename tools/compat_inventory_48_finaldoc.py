from pathlib import Path

p = Path("crates/dm-runtime/src/lib.rs")
text = p.read_text()
old = '''    /// Type-static values are intentionally excluded until their type-qualified
    /// storage identities are represented in the VM global namespace.
    #[must_use]
    pub fn take_execution_state(&mut self) -> ExecutionState {
'''
new = '''    /// Type-static values are intentionally excluded until their type-qualified
    /// storage identities are represented in the VM global namespace.
    ///
    /// # Panics
    ///
    /// Panics only if an engine-defined built-in field name is internally invalid;
    /// every such spelling is a fixed canonical DM identifier.
    #[must_use]
    pub fn take_execution_state(&mut self) -> ExecutionState {
'''
if text.count(old) != 1:
    raise SystemExit(f"take_execution_state docs anchor expected once, found {text.count(old)}")
p.write_text(text.replace(old, new, 1))

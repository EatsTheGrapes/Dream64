from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}\n---OLD---\n{old[:900]}")
    return text.replace(old, new, 1)


# Durable ordered-list primitives used by the native /list proc surface.
p = Path("crates/dm-value/src/lib.rs")
text = p.read_text()
anchor = '''    /// Reads an associative value by semantic key equality.
'''
helpers = '''    /// Inserts a positional value at a 1-based boundary and returns that index.
    ///
    /// # Errors
    ///
    /// Returns an index error when `index` is zero or greater than `len + 1`.
    pub fn insert(&mut self, index: usize, value: Value) -> Result<usize, ValueError> {
        checked_boundary(index, self.order.len())?;
        let position = self.positional.len();
        self.positional.push(value);
        self.order.insert(index - 1, ListOrder::Positional(position));
        Ok(index)
    }

    /// Creates a shallow copy of the half-open 1-based range `[start, end)`.
    ///
    /// Associative values remain associated with copied keys.
    ///
    /// # Errors
    ///
    /// Returns an index error for boundaries outside `1..=len + 1`, or
    /// [`ValueError::CorruptListStorage`] if an associative order entry lost
    /// its value.
    pub fn copy_range(&self, start: usize, end: usize) -> Result<Self, ValueError> {
        checked_boundary(start, self.order.len())?;
        checked_boundary(end, self.order.len())?;
        let mut copy = Self::default();
        if end <= start {
            return Ok(copy);
        }
        for entry in &self.order[start - 1..end - 1] {
            match entry {
                ListOrder::Positional(position) => {
                    copy.add(self.positional[*position].clone());
                }
                ListOrder::Associative(key) => {
                    copy.set_key(key.clone(), self.get_key(key)?.clone());
                }
            }
        }
        Ok(copy)
    }

    /// Removes the half-open 1-based range `[start, end)` and returns the
    /// number of removed iteration entries.
    ///
    /// # Errors
    ///
    /// Returns an index error for boundaries outside `1..=len + 1`.
    pub fn cut_range(&mut self, start: usize, end: usize) -> Result<usize, ValueError> {
        checked_boundary(start, self.order.len())?;
        checked_boundary(end, self.order.len())?;
        if end <= start {
            return Ok(0);
        }
        let count = end - start;
        for _ in 0..count {
            self.remove(start)?;
        }
        Ok(count)
    }

    /// Finds the first iteration position semantically equal to `value` in
    /// the half-open 1-based range `[start, end)`, returning zero when absent.
    ///
    /// # Errors
    ///
    /// Returns an index error for boundaries outside `1..=len + 1`.
    pub fn find_position(
        &self,
        value: &Value,
        start: usize,
        end: usize,
    ) -> Result<usize, ValueError> {
        checked_boundary(start, self.order.len())?;
        checked_boundary(end, self.order.len())?;
        if end <= start {
            return Ok(0);
        }
        for index in start..end {
            if self.get(index)?.semantic_eq(value) {
                return Ok(index);
            }
        }
        Ok(0)
    }

    /// Removes the last occurrence equal to `value`, matching BYOND list
    /// subtraction/`Remove()` ordering.
    pub fn remove_last(&mut self, value: &Value) -> Option<Value> {
        let index = (1..=self.len())
            .rev()
            .find(|index| self.get(*index).is_ok_and(|candidate| candidate.semantic_eq(value)))?;
        self.remove(index).ok()
    }

    /// Swaps two 1-based iteration positions while keeping associative values
    /// attached to their keys.
    ///
    /// # Errors
    ///
    /// Returns an index error when either position is outside the list.
    pub fn swap(&mut self, first: usize, second: usize) -> Result<(), ValueError> {
        let first = checked_index(first, self.order.len())?;
        let second = checked_index(second, self.order.len())?;
        self.order.swap(first, second);
        Ok(())
    }

    /// Resizes the list, appending positional `null` values when growing and
    /// cutting the tail when shrinking.
    ///
    /// # Errors
    ///
    /// Returns a storage error only if an existing associative entry is
    /// internally inconsistent while shrinking.
    pub fn resize(&mut self, new_len: usize) -> Result<(), ValueError> {
        while self.len() < new_len {
            self.add(Value::Null);
        }
        if self.len() > new_len {
            let end = self.len() + 1;
            self.cut_range(new_len + 1, end)?;
        }
        Ok(())
    }

'''
text = replace_once(text, anchor, helpers + anchor, "DmList helper insertion")
text = replace_once(
    text,
    '''fn checked_index(index: usize, len: usize) -> Result<usize, ValueError> {
''',
    '''fn checked_boundary(index: usize, len: usize) -> Result<usize, ValueError> {
    if index == 0 {
        return Err(ValueError::IndexZero);
    }
    if index > len.saturating_add(1) {
        return Err(ValueError::IndexOutOfBounds { index, len });
    }
    Ok(index - 1)
}

fn checked_index(index: usize, len: usize) -> Result<usize, ValueError> {
''',
    "checked boundary helper",
)

test_anchor = '''    #[test]
    fn associative_updates_preserve_deterministic_insertion_order() {
'''
test = '''    #[test]
    fn range_insert_swap_and_resize_preserve_order_and_associations() {
        let mut list = DmList::default();
        list.add(text("a"));
        list.set_key(text("key"), Value::number(9.0));
        list.add(text("b"));

        let copy = list.copy_range(2, 4).unwrap();
        assert_eq!(copy.len(), 2);
        assert!(copy.get(1).unwrap().semantic_eq(&text("key")));
        assert!(
            copy.get_key(&text("key"))
                .unwrap()
                .semantic_eq(&Value::number(9.0))
        );

        list.insert(2, text("x")).unwrap();
        assert_eq!(list.find_position(&text("b"), 1, list.len() + 1), Ok(4));
        list.swap(1, 4).unwrap();
        assert!(list.get(1).unwrap().semantic_eq(&text("b")));
        assert!(list.get(4).unwrap().semantic_eq(&text("a")));
        assert!(list.remove_last(&text("x")).is_some());
        list.resize(5).unwrap();
        assert_eq!(list.len(), 5);
        assert!(list.get(5).unwrap().semantic_eq(&Value::Null));
        list.resize(2).unwrap();
        assert_eq!(list.len(), 2);
    }

'''
text = replace_once(text, test_anchor, test + test_anchor, "DmList helper regression")
p.write_text(text)

# Native list procs live beside the documented global builtins so stringification
# and DM-number boundary helpers remain shared.
p = Path("crates/dm-vm/src/builtins.rs")
text = p.read_text()
text = replace_once(
    text,
    "use dm_value::{FieldName, TypePath, Value};\n",
    "use dm_value::{FieldName, ListId, TypePath, Value};\n",
    "list builtin imports",
)

anchor = '''fn resolved_file_path(
'''
list_builtins = r'''pub(super) fn execute_list_method(
    name: &str,
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Option<Result<Value, String>> {
    Some(match name {
        "Add" => list_add(list, arguments, state),
        "Copy" => list_copy(list, arguments, state),
        "Cut" => list_cut(list, arguments, state),
        "Find" => list_find(list, arguments, state),
        "Insert" => list_insert(list, arguments, state),
        "Join" => list_join(list, arguments, state),
        "Remove" => list_remove(list, arguments, state, false),
        "RemoveAll" => list_remove(list, arguments, state, true),
        "Splice" => list_splice(list, arguments, state),
        "Swap" => list_swap(list, arguments, state),
        _ => return None,
    })
}

fn list_integer(value: Option<&Value>, default: i64, context: &str) -> Result<i64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) if number.to_f32().is_finite() => {
            Ok(number.to_f32().trunc() as i64)
        }
        Some(value) => Err(format!("{context} requires a numeric index, received {value}")),
    }
}

fn list_boundary(value: i64, len: usize, zero_is_end: bool) -> Result<usize, String> {
    let limit = i64::try_from(len).unwrap_or(i64::MAX - 1).saturating_add(1);
    let value = if value == 0 {
        if zero_is_end { limit } else { 1 }
    } else {
        value
    };
    if value < 1 || value > limit {
        return Err(format!("list index {value} is outside 1 through {limit}"));
    }
    usize::try_from(value).map_err(|error| format!("list index is not representable: {error}"))
}

fn splice_boundary(value: i64, len: usize, zero_is_end: bool) -> usize {
    let limit = i64::try_from(len).unwrap_or(i64::MAX - 1).saturating_add(1);
    let value = if value == 0 && zero_is_end {
        limit
    } else if value < 0 {
        limit.saturating_add(value)
    } else {
        value
    };
    usize::try_from(value.clamp(1, limit)).unwrap_or(usize::MAX)
}

fn flattened_list_arguments(arguments: &[Value], state: &ExecutionState) -> Result<Vec<Value>, String> {
    let mut values = Vec::new();
    for argument in arguments {
        if let Value::List(list) = argument {
            let snapshot = state
                .heap
                .list(*list)
                .map_err(|error| error.to_string())?
                .positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>();
            values.extend(snapshot);
        } else {
            values.push(argument.clone());
        }
    }
    Ok(values)
}

fn list_add(list: ListId, arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if arguments.is_empty() {
        return Err("list.Add requires at least one item".to_owned());
    }
    let values = flattened_list_arguments(arguments, state)?;
    let target = state.heap.list_mut(list).map_err(|error| error.to_string())?;
    for value in values {
        target.add(value);
    }
    Ok(Value::Null)
}

fn list_copy(list: ListId, arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if arguments.len() > 2 {
        return Err("list.Copy accepts Start and End only".to_owned());
    }
    let source = state.heap.list(list).map_err(|error| error.to_string())?;
    let len = source.len();
    let start = list_boundary(list_integer(arguments.first(), 1, "list.Copy Start")?, len, false)?;
    let end = list_boundary(list_integer(arguments.get(1), 0, "list.Copy End")?, len, true)?;
    let copy = source.copy_range(start, end).map_err(|error| error.to_string())?;
    let result = state.heap.allocate_list();
    *state.heap.list_mut(result).map_err(|error| error.to_string())? = copy;
    Ok(Value::List(result))
}

fn list_cut(list: ListId, arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if arguments.len() > 2 {
        return Err("list.Cut accepts Start and End only".to_owned());
    }
    let len = state.heap.list(list).map_err(|error| error.to_string())?.len();
    let raw_start = list_integer(arguments.first(), 1, "list.Cut Start")?;
    if raw_start < 0 {
        return Err("list.Cut Start cannot be negative".to_owned());
    }
    let start = list_boundary(raw_start.min(i64::try_from(len + 1).unwrap_or(i64::MAX)), len, false)?;
    let raw_end = list_integer(arguments.get(1), 0, "list.Cut End")?;
    if raw_end < 0 {
        return Err("list.Cut End cannot be negative".to_owned());
    }
    let end = if raw_end == 0 || raw_end > i64::try_from(len + 1).unwrap_or(i64::MAX) {
        len + 1
    } else {
        list_boundary(raw_end, len, true)?
    };
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .cut_range(start, end)
        .map_err(|error| error.to_string())?;
    Ok(Value::Null)
}

fn list_find(list: ListId, arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    if arguments.is_empty() || arguments.len() > 3 {
        return Err("list.Find requires Elem and optional Start/End".to_owned());
    }
    let source = state.heap.list(list).map_err(|error| error.to_string())?;
    let len = source.len();
    let raw_start = list_integer(arguments.get(1), 1, "list.Find Start").unwrap_or(1).max(1);
    let start = usize::try_from(raw_start)
        .unwrap_or(usize::MAX)
        .min(len.saturating_add(1));
    let raw_end = list_integer(arguments.get(2), 0, "list.Find End").unwrap_or(0);
    let end = if raw_end <= 0 || raw_end > i64::try_from(len + 1).unwrap_or(i64::MAX) {
        len + 1
    } else {
        usize::try_from(raw_end).unwrap_or(len + 1)
    };
    let found = source
        .find_position(&arguments[0], start.max(1), end.max(1))
        .map_err(|error| error.to_string())?;
    Ok(Value::number(found as f32))
}

fn list_insert(list: ListId, arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if arguments.len() < 2 {
        return Err("list.Insert requires Index and at least one item".to_owned());
    }
    let len = state.heap.list(list).map_err(|error| error.to_string())?.len();
    let raw = list_integer(arguments.first(), 0, "list.Insert Index")?;
    let mut index = if raw <= 0 {
        len + 1
    } else {
        usize::try_from(raw).map_err(|error| format!("list.Insert index is invalid: {error}"))?
    };
    if index > len + 1 {
        return Err(format!("list.Insert index {index} exceeds {}", len + 1));
    }
    let values = flattened_list_arguments(&arguments[1..], state)?;
    let target = state.heap.list_mut(list).map_err(|error| error.to_string())?;
    for value in values {
        target.insert(index, value).map_err(|error| error.to_string())?;
        index += 1;
    }
    Ok(Value::number(index as f32))
}

fn list_join(list: ListId, arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    if arguments.is_empty() || arguments.len() > 3 {
        return Err("list.Join requires Glue and optional Start/End".to_owned());
    }
    let glue = runtime_text(&arguments[0], state, "list.Join Glue")?;
    let source = state.heap.list(list).map_err(|error| error.to_string())?;
    let len = source.len();
    let limit = i64::try_from(len).unwrap_or(i64::MAX - 1).saturating_add(1);
    let mut start = list_integer(arguments.get(1), 1, "list.Join Start").unwrap_or(1);
    let mut end = list_integer(arguments.get(2), 0, "list.Join End").unwrap_or(0);
    if end <= 0 {
        end = end.saturating_add(limit);
    }
    if start < 0 {
        start = start.saturating_add(limit);
    }
    if start == 0 || start >= end {
        return Ok(Value::text(""));
    }
    let start = usize::try_from(start.max(1)).unwrap_or(usize::MAX).min(len + 1);
    let end = usize::try_from(end.max(1)).unwrap_or(usize::MAX).min(len + 1);
    let mut values = Vec::new();
    for index in start..end {
        values.push(runtime_text(
            source.get(index).map_err(|error| error.to_string())?,
            state,
            "list.Join item",
        )?);
    }
    Ok(Value::text(values.join(&glue)))
}

fn list_remove_once(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<usize, String> {
    let mut removed = 0usize;
    for argument in arguments {
        if matches!(argument, Value::List(candidate) if *candidate == list) {
            let len = state.heap.list(list).map_err(|error| error.to_string())?.len();
            state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .resize(0)
                .map_err(|error| error.to_string())?;
            removed += len;
            break;
        }
        let values = flattened_list_arguments(std::slice::from_ref(argument), state)?;
        for value in values {
            if state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .remove_last(&value)
                .is_some()
            {
                removed += 1;
            }
        }
    }
    Ok(removed)
}

fn list_remove(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
    all: bool,
) -> Result<Value, String> {
    if arguments.is_empty() {
        return Err(if all {
            "list.RemoveAll requires at least one item"
        } else {
            "list.Remove requires at least one item"
        }
        .to_owned());
    }
    if all {
        let mut total = 0usize;
        loop {
            let removed = list_remove_once(list, arguments, state)?;
            total += removed;
            if removed == 0 {
                break;
            }
        }
        Ok(Value::number(total as f32))
    } else {
        Ok(Value::number(f32::from(list_remove_once(list, arguments, state)? > 0)))
    }
}

fn list_splice(list: ListId, arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if arguments.len() > 2 && arguments.len() < 3 {
        return Err("invalid list.Splice arguments".to_owned());
    }
    let len = state.heap.list(list).map_err(|error| error.to_string())?.len();
    let mut start = splice_boundary(list_integer(arguments.first(), 1, "list.Splice Start")?, len, false);
    let mut end = splice_boundary(list_integer(arguments.get(1), 0, "list.Splice End")?, len, true);
    if end < start {
        std::mem::swap(&mut start, &mut end);
    }
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .cut_range(start, end)
        .map_err(|error| error.to_string())?;
    if arguments.len() <= 2 {
        return Ok(Value::Null);
    }
    let values = flattened_list_arguments(&arguments[2..], state)?;
    let mut index = start.min(
        state
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .len()
            + 1,
    );
    let target = state.heap.list_mut(list).map_err(|error| error.to_string())?;
    for value in values {
        target.insert(index, value).map_err(|error| error.to_string())?;
        index += 1;
    }
    Ok(Value::Null)
}

fn list_swap(list: ListId, arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if arguments.len() != 2 {
        return Err("list.Swap requires exactly two indices".to_owned());
    }
    let first = list_integer(arguments.first(), 0, "list.Swap Index1")?;
    let second = list_integer(arguments.get(1), 0, "list.Swap Index2")?;
    let first = usize::try_from(first).map_err(|_| "list.Swap Index1 is invalid".to_owned())?;
    let second = usize::try_from(second).map_err(|_| "list.Swap Index2 is invalid".to_owned())?;
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .swap(first, second)
        .map_err(|error| error.to_string())?;
    Ok(Value::Null)
}

'''
text = replace_once(text, anchor, list_builtins + anchor, "native list builtin insertion")
p.write_text(text)

# VM dispatch for /list methods and built-in .len.
p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
text = replace_once(
    text,
    "use builtins::{execute_standard_builtin, is_subtype, standard_builtin_arity};\n",
    "use builtins::{execute_list_method, execute_standard_builtin, is_subtype, standard_builtin_arity};\n",
    "list method VM import",
)

old = '''                    Value::TypePath(path) if name.as_str() == "parent_type" => state
                        .type_parent(&path)
                        .cloned()
                        .map_or(Value::Null, Value::TypePath),
                    Value::Datum(datum) => {
'''
new = '''                    Value::TypePath(path) if name.as_str() == "parent_type" => state
                        .type_parent(&path)
                        .cloned()
                        .map_or(Value::Null, Value::TypePath),
                    Value::List(list) if name.as_str() == "len" => {
                        let len = state.heap.list(list).map_err(|error| {
                            execution_error(module, &frames, error.to_string())
                        })?.len();
                        Value::number(len.to_string().parse::<f32>().map_err(|error| {
                            execution_error(
                                module,
                                &frames,
                                format!("list length cannot be represented as binary32: {error}"),
                            )
                        })?)
                    }
                    Value::Datum(datum) => {
'''
text = replace_once(text, old, new, "list len field read")

old = '''                let datum = match datum_receiver(&receiver, "field write") {
                    Ok(datum) => datum,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if let Err(error) = state
                    .heap
                    .set_datum_field(datum, name.clone(), value.clone())
                {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
                if keep {
                    frames[frame_index].stack.push(value);
                }
'''
new = '''                match receiver {
                    Value::Datum(datum) => {
                        if let Err(error) = state
                            .heap
                            .set_datum_field(datum, name.clone(), value.clone())
                        {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    }
                    Value::List(list) if name.as_str() == "len" => {
                        let new_len = match &value {
                            Value::Number(number) if number.to_f32().is_finite() => number
                                .to_f32()
                                .trunc()
                                .max(0.0)
                                .to_string()
                                .parse::<usize>()
                                .unwrap_or(usize::MAX),
                            Value::Null => 0,
                            _ => 0,
                        };
                        if let Err(error) = state
                            .heap
                            .list_mut(list)
                            .and_then(|values| values.resize(new_len))
                        {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    }
                    Value::Null => {
                        return Err(execution_error(module, &frames, "field write received null"));
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("field write requires a datum or list.len, received {value}"),
                        ));
                    }
                }
                if keep {
                    frames[frame_index].stack.push(value);
                }
'''
text = replace_once(text, old, new, "list len field write")

old = '''            Instruction::CallDynamic { argument_count } => {
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let count = runtime_argument_count(&mut frames[frame_index].stack, argument_count)
'''
new = '''            Instruction::CallDynamic { argument_count } => {
                let count = runtime_argument_count(&mut frames[frame_index].stack, argument_count)
'''
text = replace_once(text, old, new, "native dynamic call depth relocation")

old = '''                let caller_context = frame_context(&frames[frame_index]);
                let (target, context) =
                    dynamic_call_target(module, state, &receiver, &selector, &caller_context)
                        .map_err(|message| execution_error(module, &frames, message))?;
                let Some(target_program) = module.procedure(target) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid dynamic call target {}", target.index()),
                    ));
                };
                frames.push(make_frame(target, target_program, &arguments, &context));
                continue;
'''
new = '''                if let Value::List(list) = receiver {
                    let Value::Text(method) = selector else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list procedure selector must be text, received {selector}"),
                        ));
                    };
                    let Some(result) = execute_list_method(&method, list, &arguments, state) else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("unknown /list procedure {method:?}"),
                        ));
                    };
                    let result = result
                        .map_err(|message| execution_error(module, &frames, message))?;
                    frames[frame_index].stack.push(result);
                } else {
                    if frames.len() >= limits.max_call_depth {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("maximum call depth {} exceeded", limits.max_call_depth),
                        ));
                    }
                    let caller_context = frame_context(&frames[frame_index]);
                    let (target, context) =
                        dynamic_call_target(module, state, &receiver, &selector, &caller_context)
                            .map_err(|message| execution_error(module, &frames, message))?;
                    let Some(target_program) = module.procedure(target) else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("invalid dynamic call target {}", target.index()),
                        ));
                    };
                    frames.push(make_frame(target, target_program, &arguments, &context));
                    continue;
                }
'''
text = replace_once(text, old, new, "native list dynamic dispatch")

# End-to-end regressions exercise every documented list proc and writable len.
test_anchor = '''    #[test]
    fn documented_native_builtins_cover_text_math_and_type_helpers() {
'''
tests = r'''    #[test]
    fn documented_list_methods_and_len_execute_natively() {
        let source = parse(
            "/proc/run()\n\tvar/list/values = list(\"a\", \"b\", \"c\")\n\tvalues.Add(list(\"d\", \"e\"))\n\tvar/list/copied = values.Copy(2, 5)\n\tvalues.Cut(2, 3)\n\tvar/found = values.Find(\"d\")\n\tvar/next_index = values.Insert(2, list(\"x\", \"y\"))\n\tvalues.Splice(-1, 0, \"z\")\n\tvalues.Swap(1, 6)\n\tvalues.len = 7\n\tvar/removed = values.Remove(\"d\")\n\tvar/removed_all = values.RemoveAll(\"x\")\n\treturn copied.len + (copied[1] == \"b\") + (copied[3] == \"d\") + found + next_index + removed + removed_all + values.len + (values[1] == \"z\") + (values[2] == \"y\")\n",
        )
        .expect("list method source should parse");
        let module = compile_module(&source.definitions).expect("list methods should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
            Ok(Value::number(21.0))
        );
    }

    #[test]
    fn list_copy_and_swap_keep_associative_values_attached_to_keys() {
        let source = parse(
            "/proc/run()\n\tvar/list/values = list(\"red\" = 1, \"blue\" = 2, \"green\" = 3)\n\tvar/list/copied = values.Copy()\n\tvalues.Swap(1, 3)\n\treturn (values[1] == \"green\") + (values[\"green\"] == 3) + (copied[1] == \"red\") + (copied[\"red\"] == 1)\n",
        )
        .expect("associative list method source should parse");
        let module = compile_module(&source.definitions).expect("associative list methods should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
            Ok(Value::number(4.0))
        );
    }

'''
text = replace_once(text, test_anchor, tests + test_anchor, "native list method regressions")
p.write_text(text)

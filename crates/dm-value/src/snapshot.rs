use std::io::{self, Read, Write};

use super::{
    HeapSnapshotDatum, HeapSnapshotListEntry, HeapSnapshotSlot, HeapSnapshotValue, Value,
    ValueHeap, ValueHeapSnapshot, restore_list_entries, restore_snapshot_datum,
};

const MAGIC: &[u8; 8] = b"D64HEAP\0";
const VERSION: u32 = 1;
const MAX_SLOTS: usize = 100_000_000;
const MAX_ENTRIES: usize = 100_000_000;
const MAX_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_VALUE_DEPTH: usize = 64;

impl ValueHeap {
    /// Streams this heap as a ready-world section without first cloning the
    /// complete live object graph.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the destination fails or a record length cannot
    /// be represented by the format.
    pub fn write_snapshot_to(&self, writer: &mut impl Write) -> io::Result<()> {
        writer.write_all(MAGIC)?;
        write_u32(writer, VERSION)?;
        write_len(writer, self.datums.slot_len())?;
        for index in 0..self.datums.slot_len() {
            let slot = self
                .datums
                .slot(index)
                .expect("datum snapshot index addresses an allocated slot");
            write_u32(writer, slot.generation)?;
            writer.write_all(&[u8::from(slot.value.is_some())])?;
            if let Some(datum) = &slot.value {
                write_string(writer, datum.type_path().as_str())?;
                write_len(writer, datum.field_len())?;
                for (name, value) in datum.fields() {
                    write_string(writer, name.as_str())?;
                    write_runtime_value(writer, value)?;
                }
            }
        }
        write_u32_vec(writer, &self.datums.free)?;
        write_len(writer, self.lists.slot_len())?;
        for index in 0..self.lists.slot_len() {
            let slot = self
                .lists
                .slot(index)
                .expect("list snapshot index addresses an allocated slot");
            write_u32(writer, slot.generation)?;
            writer.write_all(&[u8::from(slot.value.is_some())])?;
            if let Some(list) = &slot.value {
                write_runtime_list(writer, list)?;
            }
        }
        write_u32_vec(writer, &self.lists.free)
    }

    /// Loads a streamed ready-world heap section and rebuilds derived indexes.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidData`] for malformed records or violated
    /// arena invariants.
    pub fn read_snapshot_from(reader: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0_u8; MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return invalid_data("ready-world heap magic does not match");
        }
        let version = read_u32(reader)?;
        if version != VERSION {
            return invalid_data(format!(
                "unsupported ready-world heap version {version}; expected {VERSION}"
            ));
        }

        let mut heap = Self::default();
        let datum_slots = read_len(reader, MAX_SLOTS)?;
        for _ in 0..datum_slots {
            let generation = read_u32(reader)?;
            let value = match read_u8(reader)? {
                0 => None,
                1 => Some(
                    restore_snapshot_datum(read_datum(reader)?)
                        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
                ),
                tag => return invalid_data(format!("invalid datum-slot presence tag {tag}")),
            };
            heap.datums.push_snapshot_slot(generation, value);
        }
        let datum_free = read_u32_vec(reader, MAX_SLOTS)?;
        heap.datums
            .install_snapshot_free(datum_free, "datum")
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

        let list_slots = read_len(reader, MAX_SLOTS)?;
        for _ in 0..list_slots {
            let generation = read_u32(reader)?;
            let value = match read_u8(reader)? {
                0 => None,
                1 => Some(
                    restore_list_entries(read_list(reader)?)
                        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
                ),
                tag => return invalid_data(format!("invalid list-slot presence tag {tag}")),
            };
            heap.lists.push_snapshot_slot(generation, value);
        }
        let list_free = read_u32_vec(reader, MAX_SLOTS)?;
        heap.lists
            .install_snapshot_free(list_free, "list")
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        heap.compact_restored_datum_layouts();
        Ok(heap)
    }
}

impl ValueHeapSnapshot {
    /// Writes the pointer-free heap section in Dream64's versioned binary format.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the destination fails or a collection length
    /// cannot be represented by the format.
    pub fn write_to(&self, writer: &mut impl Write) -> io::Result<()> {
        writer.write_all(MAGIC)?;
        write_u32(writer, VERSION)?;
        write_slots(writer, &self.datums, write_datum)?;
        write_u32_vec(writer, &self.datum_free)?;
        write_slots(writer, &self.lists, |writer, entries| {
            write_list(writer, entries)
        })?;
        write_u32_vec(writer, &self.list_free)
    }

    /// Reads one pointer-free heap section without allocating unbounded records.
    ///
    /// Arena/free-list consistency is validated when the snapshot is passed to
    /// [`super::ValueHeap::from_snapshot`].
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidData`] for an unknown version, invalid
    /// value tag, excessive length/depth, or malformed UTF-8.
    pub fn read_from(reader: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0_u8; MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return invalid_data("ready-world heap magic does not match");
        }
        let version = read_u32(reader)?;
        if version != VERSION {
            return invalid_data(format!(
                "unsupported ready-world heap version {version}; expected {VERSION}"
            ));
        }
        Ok(Self {
            datums: read_slots(reader, read_datum)?,
            datum_free: read_u32_vec(reader, MAX_SLOTS)?,
            lists: read_slots(reader, read_list)?,
            list_free: read_u32_vec(reader, MAX_SLOTS)?,
        })
    }
}

impl HeapSnapshotValue {
    /// Writes one pointer-free value record.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the destination fails.
    pub fn write_to(&self, writer: &mut impl Write) -> io::Result<()> {
        write_value(writer, self)
    }

    /// Reads one bounded pointer-free value record.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidData`] for malformed tags, excessive
    /// nesting, or invalid string data.
    pub fn read_from(reader: &mut impl Read) -> io::Result<Self> {
        read_value(reader, 0)
    }
}

fn write_datum(writer: &mut impl Write, datum: &HeapSnapshotDatum) -> io::Result<()> {
    write_string(writer, &datum.type_path)?;
    write_len(writer, datum.fields.len())?;
    for (name, value) in &datum.fields {
        write_string(writer, name)?;
        write_value(writer, value)?;
    }
    Ok(())
}

fn read_datum(reader: &mut impl Read) -> io::Result<HeapSnapshotDatum> {
    let type_path = read_string(reader)?;
    let len = read_len(reader, MAX_ENTRIES)?;
    let mut fields = Vec::with_capacity(len);
    for _ in 0..len {
        fields.push((read_string(reader)?, read_value(reader, 0)?));
    }
    Ok(HeapSnapshotDatum { type_path, fields })
}

fn write_list(writer: &mut impl Write, entries: &[HeapSnapshotListEntry]) -> io::Result<()> {
    write_len(writer, entries.len())?;
    for entry in entries {
        match entry {
            HeapSnapshotListEntry::Positional(value) => {
                writer.write_all(&[0])?;
                write_value(writer, value)?;
            }
            HeapSnapshotListEntry::Associative(key, value) => {
                writer.write_all(&[1])?;
                write_value(writer, key)?;
                write_value(writer, value)?;
            }
        }
    }
    Ok(())
}

fn read_list(reader: &mut impl Read) -> io::Result<Vec<HeapSnapshotListEntry>> {
    let len = read_len(reader, MAX_ENTRIES)?;
    let mut entries = Vec::with_capacity(len);
    for _ in 0..len {
        entries.push(match read_u8(reader)? {
            0 => HeapSnapshotListEntry::Positional(read_value(reader, 0)?),
            1 => HeapSnapshotListEntry::Associative(read_value(reader, 0)?, read_value(reader, 0)?),
            tag => return invalid_data(format!("invalid list-entry tag {tag}")),
        });
    }
    Ok(entries)
}

fn write_value(writer: &mut impl Write, value: &HeapSnapshotValue) -> io::Result<()> {
    match value {
        HeapSnapshotValue::Null => writer.write_all(&[0]),
        HeapSnapshotValue::Number(bits) => {
            writer.write_all(&[1])?;
            write_u32(writer, *bits)
        }
        HeapSnapshotValue::Text(text) => {
            writer.write_all(&[2])?;
            write_string(writer, text)
        }
        HeapSnapshotValue::File(path) => {
            writer.write_all(&[3])?;
            write_string(writer, path)
        }
        HeapSnapshotValue::TypePath(path) => {
            writer.write_all(&[4])?;
            write_string(writer, path)
        }
        HeapSnapshotValue::ModifiedTypePath { base, overrides } => {
            writer.write_all(&[5])?;
            write_string(writer, base)?;
            write_len(writer, overrides.len())?;
            for (name, value) in overrides {
                write_string(writer, name)?;
                write_value(writer, value)?;
            }
            Ok(())
        }
        HeapSnapshotValue::Datum { index, generation } => {
            writer.write_all(&[6])?;
            write_u32(writer, *index)?;
            write_u32(writer, *generation)
        }
        HeapSnapshotValue::List { index, generation } => {
            writer.write_all(&[7])?;
            write_u32(writer, *index)?;
            write_u32(writer, *generation)
        }
    }
}

fn write_runtime_value(writer: &mut impl Write, value: &Value) -> io::Result<()> {
    match value {
        Value::Null => writer.write_all(&[0]),
        Value::Number(number) => {
            writer.write_all(&[1])?;
            write_u32(writer, number.bits())
        }
        Value::Text(text) => {
            writer.write_all(&[2])?;
            write_string(writer, text)
        }
        Value::File(path) => {
            writer.write_all(&[3])?;
            write_string(writer, path)
        }
        Value::TypePath(path) => {
            writer.write_all(&[4])?;
            write_string(writer, path.as_str())
        }
        Value::ModifiedTypePath(path) => {
            writer.write_all(&[5])?;
            write_string(writer, path.base().as_str())?;
            write_len(writer, path.overrides().len())?;
            for (name, value) in path.overrides() {
                write_string(writer, name.as_str())?;
                write_runtime_value(writer, value)?;
            }
            Ok(())
        }
        Value::Datum(datum) => {
            writer.write_all(&[6])?;
            write_u32(writer, datum.index())?;
            write_u32(writer, datum.generation())
        }
        Value::List(list) => {
            writer.write_all(&[7])?;
            write_u32(writer, list.index())?;
            write_u32(writer, list.generation())
        }
    }
}

fn write_runtime_list(writer: &mut impl Write, list: &super::DmList) -> io::Result<()> {
    let Some(storage) = list.storage.as_deref() else {
        return write_len(writer, 0);
    };
    let entries = &storage.order[storage.prefix_head..];
    write_len(writer, entries.len())?;
    for entry in entries {
        if let Some(index) = entry.positional_index() {
            writer.write_all(&[0])?;
            write_runtime_value(writer, &storage.positional[index])?;
        } else {
            writer.write_all(&[1])?;
            let index = entry
                .associative_index()
                .expect("live list order entry is valid");
            let (key, value) = &storage.associative[index];
            write_runtime_value(writer, key)?;
            write_runtime_value(writer, value)?;
        }
    }
    Ok(())
}

fn read_value(reader: &mut impl Read, depth: usize) -> io::Result<HeapSnapshotValue> {
    if depth > MAX_VALUE_DEPTH {
        return invalid_data("ready-world value nesting exceeds limit");
    }
    Ok(match read_u8(reader)? {
        0 => HeapSnapshotValue::Null,
        1 => HeapSnapshotValue::Number(read_u32(reader)?),
        2 => HeapSnapshotValue::Text(read_string(reader)?),
        3 => HeapSnapshotValue::File(read_string(reader)?),
        4 => HeapSnapshotValue::TypePath(read_string(reader)?),
        5 => {
            let base = read_string(reader)?;
            let len = read_len(reader, MAX_ENTRIES)?;
            let mut overrides = Vec::with_capacity(len);
            for _ in 0..len {
                overrides.push((read_string(reader)?, read_value(reader, depth + 1)?));
            }
            HeapSnapshotValue::ModifiedTypePath { base, overrides }
        }
        6 => HeapSnapshotValue::Datum {
            index: read_u32(reader)?,
            generation: read_u32(reader)?,
        },
        7 => HeapSnapshotValue::List {
            index: read_u32(reader)?,
            generation: read_u32(reader)?,
        },
        tag => return invalid_data(format!("invalid heap-value tag {tag}")),
    })
}

fn write_slots<T, W: Write>(
    writer: &mut W,
    slots: &[HeapSnapshotSlot<T>],
    mut write_value: impl FnMut(&mut W, &T) -> io::Result<()>,
) -> io::Result<()> {
    write_len(writer, slots.len())?;
    for slot in slots {
        write_u32(writer, slot.generation)?;
        writer.write_all(&[u8::from(slot.value.is_some())])?;
        if let Some(value) = &slot.value {
            write_value(writer, value)?;
        }
    }
    Ok(())
}

fn read_slots<T, R: Read>(
    reader: &mut R,
    mut read_value: impl FnMut(&mut R) -> io::Result<T>,
) -> io::Result<Vec<HeapSnapshotSlot<T>>> {
    let len = read_len(reader, MAX_SLOTS)?;
    let mut slots = Vec::with_capacity(len);
    for _ in 0..len {
        let generation = read_u32(reader)?;
        let value = match read_u8(reader)? {
            0 => None,
            1 => Some(read_value(reader)?),
            tag => return invalid_data(format!("invalid arena-presence tag {tag}")),
        };
        slots.push(HeapSnapshotSlot { generation, value });
    }
    Ok(slots)
}

fn write_u32_vec(writer: &mut impl Write, values: &[u32]) -> io::Result<()> {
    write_len(writer, values.len())?;
    for value in values {
        write_u32(writer, *value)?;
    }
    Ok(())
}

fn read_u32_vec(reader: &mut impl Read, max: usize) -> io::Result<Vec<u32>> {
    let len = read_len(reader, max)?;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(read_u32(reader)?);
    }
    Ok(values)
}

fn write_string(writer: &mut impl Write, value: &str) -> io::Result<()> {
    write_len(writer, value.len())?;
    writer.write_all(value.as_bytes())
}

fn read_string(reader: &mut impl Read) -> io::Result<String> {
    let len = read_len(reader, MAX_STRING_BYTES)?;
    let mut bytes = vec![0; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn write_len(writer: &mut impl Write, len: usize) -> io::Result<()> {
    let len = u64::try_from(len)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "length exceeds u64"))?;
    writer.write_all(&len.to_le_bytes())
}

fn read_len(reader: &mut impl Read, max: usize) -> io::Result<usize> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    let value = u64::from_le_bytes(bytes);
    let value = usize::try_from(value)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "length exceeds usize"))?;
    if value > max {
        return invalid_data(format!("record length {value} exceeds limit {max}"));
    }
    Ok(value)
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut byte = [0];
    reader.read_exact(&mut byte)?;
    Ok(byte[0])
}

fn invalid_data<T>(message: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_heap_section_roundtrips_and_rejects_version_mismatch() {
        let snapshot = ValueHeapSnapshot {
            datums: vec![HeapSnapshotSlot {
                generation: 2,
                value: Some(HeapSnapshotDatum {
                    type_path: "/datum/example".into(),
                    fields: vec![("answer".into(), HeapSnapshotValue::Number(42_f32.to_bits()))],
                }),
            }],
            datum_free: Vec::new(),
            lists: vec![HeapSnapshotSlot {
                generation: 0,
                value: Some(vec![HeapSnapshotListEntry::Associative(
                    HeapSnapshotValue::Text("key".into()),
                    HeapSnapshotValue::Datum {
                        index: 0,
                        generation: 2,
                    },
                )]),
            }],
            list_free: Vec::new(),
        };
        let mut encoded = Vec::new();
        snapshot.write_to(&mut encoded).unwrap();
        assert_eq!(
            ValueHeapSnapshot::read_from(&mut encoded.as_slice()).unwrap(),
            snapshot
        );

        encoded[MAGIC.len()] = 99;
        let error = ValueHeapSnapshot::read_from(&mut encoded.as_slice()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn streaming_heap_writer_roundtrips_without_graph_clone() {
        let mut heap = ValueHeap::new();
        let datum = heap.allocate_datum(super::super::TypePath::parse("/datum/example").unwrap());
        let list = heap.allocate_list();
        heap.list_mut(list).unwrap().add(Value::Datum(datum));
        heap.set_datum_field(
            datum,
            super::super::FieldName::parse("items").unwrap(),
            Value::List(list),
        )
        .unwrap();

        let expected = heap.snapshot();
        let mut encoded = Vec::new();
        heap.write_snapshot_to(&mut encoded).unwrap();
        let restored = ValueHeap::read_snapshot_from(&mut encoded.as_slice()).unwrap();
        assert_eq!(restored.snapshot(), expected);
    }
}

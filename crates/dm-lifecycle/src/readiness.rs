use dm_compiler::Compilation;
use dm_runtime::RuntimeImage;
use dm_value::{FieldName, TypePath, Value};
use dm_vm::ExecutionState;

/// Headless readiness probe for boot validation.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadlessReadinessProbe {
    /// Owner-qualified VM slot when the readiness marker is a type static.
    pub qualified_storage: Option<FieldName>,
    /// Runtime global containing the marker or its root datum.
    pub global: FieldName,
    /// Datum fields followed from the global value, in order.
    pub fields: Vec<FieldName>,
    /// Value which denotes completed startup.
    pub expected: Value,
}

impl HeadlessReadinessProbe {
    /// Encodes a bounded portable boot-readiness manifest.
    pub fn encode_portable_manifest(&self) -> Result<Vec<u8>, String> {
        const MAGIC: &[u8; 8] = b"D64BOOT\0";
        let mut payload = Vec::new();
        put_manifest_field(&mut payload, self.qualified_storage.as_ref())?;
        put_manifest_string(&mut payload, self.global.as_str())?;
        put_manifest_len(&mut payload, self.fields.len())?;
        for field in &self.fields {
            put_manifest_string(&mut payload, field.as_str())?;
        }
        match &self.expected {
            Value::Null => payload.push(0),
            Value::Number(number) => {
                payload.push(1);
                payload.extend_from_slice(&number.bits().to_le_bytes());
            }
            Value::Text(value) => {
                payload.push(2);
                put_manifest_string(&mut payload, value)?;
            }
            Value::TypePath(value) => {
                payload.push(3);
                put_manifest_string(&mut payload, value.as_str())?;
            }
            _ => return Err("boot readiness expected value is not portable".to_owned()),
        }
        let mut bytes = Vec::with_capacity(22 + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    /// Decodes a bounded portable boot-readiness manifest.
    pub fn decode_portable_manifest(bytes: &[u8]) -> Result<Self, String> {
        const MAGIC: &[u8; 8] = b"D64BOOT\0";
        const MAX: usize = 1024 * 1024;
        if bytes.len() < 22
            || &bytes[..8] != MAGIC
            || u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != 1
        {
            return Err("unsupported boot manifest header".to_owned());
        }
        let length = usize::try_from(u64::from_le_bytes(bytes[10..18].try_into().unwrap()))
            .map_err(|_| "boot manifest length overflow")?;
        if length > MAX
            || bytes.len()
                != 22usize
                    .checked_add(length)
                    .ok_or("boot manifest length overflow")?
        {
            return Err("invalid boot manifest length".to_owned());
        }
        let payload = &bytes[22..];
        if crc32fast::hash(payload) != u32::from_le_bytes(bytes[18..22].try_into().unwrap()) {
            return Err("boot manifest checksum mismatch".to_owned());
        }
        let mut input = std::io::Cursor::new(payload);
        let qualified_storage = get_manifest_field(&mut input)?;
        let global = FieldName::parse(&get_manifest_string(&mut input)?)
            .map_err(|error| error.to_string())?;
        let count = get_manifest_len(&mut input)?;
        if count > 1024 {
            return Err("boot manifest field chain exceeds limit".to_owned());
        }
        let mut fields = Vec::with_capacity(count);
        for _ in 0..count {
            fields.push(
                FieldName::parse(&get_manifest_string(&mut input)?)
                    .map_err(|error| error.to_string())?,
            );
        }
        let expected = match get_manifest_u8(&mut input)? {
            0 => Value::Null,
            1 => {
                let mut bits = [0; 4];
                std::io::Read::read_exact(&mut input, &mut bits)
                    .map_err(|error| error.to_string())?;
                Value::number(f32::from_bits(u32::from_le_bytes(bits)))
            }
            2 => Value::text(get_manifest_string(&mut input)?),
            3 => Value::TypePath(
                TypePath::parse(&get_manifest_string(&mut input)?)
                    .map_err(|error| error.to_string())?,
            ),
            _ => return Err("invalid boot manifest expected-value tag".to_owned()),
        };
        if input.position() as usize != payload.len() {
            return Err("boot manifest has trailing bytes".to_owned());
        }
        Ok(Self {
            qualified_storage,
            global,
            fields,
            expected,
        })
    }
}

/// Derives the project's portable lobby-readiness contract from compiler macros.
#[must_use]
pub fn derive_lobby_readiness(
    compilation: &Compilation,
    runtime: &RuntimeImage,
) -> Option<HeadlessReadinessProbe> {
    let has_ticker_type = runtime
        .types()
        .any(|(path, _)| path.as_str() == "/datum/controller/subsystem/ticker");
    let expected = compilation
        .project()
        .object_macro("GAME_STATE_PREGAME")?
        .trim()
        .parse::<f32>()
        .ok()?;
    has_ticker_type.then(|| HeadlessReadinessProbe {
        qualified_storage: None,
        global: FieldName::parse("SSticker").expect("DM global identifier is valid"),
        fields: vec![FieldName::parse("current_state").expect("DM field identifier is valid")],
        expected: Value::number(expected),
    })
}

/// Checks if a codebase-owned lifecycle marker currently matches.
///
/// Hosts use this at generation-activation boundaries as well as during the
/// initial scheduler drain. Keeping the comparison here ensures restored and
/// cold worlds follow the same datum/static-storage semantics.
#[must_use]
pub fn readiness_probe_matches(state: &ExecutionState, probe: &HeadlessReadinessProbe) -> bool {
    let storage = probe.qualified_storage.as_ref().unwrap_or(&probe.global);
    let Some(mut value) = state.global(storage).cloned() else {
        return false;
    };
    for field in &probe.fields {
        let Value::Datum(datum) = value else {
            return false;
        };
        let Ok(next) = state.heap().datum_field(datum, field) else {
            return false;
        };
        value = next.clone();
    }
    value == probe.expected
}

fn put_manifest_len(output: &mut Vec<u8>, value: usize) -> Result<(), String> {
    output.extend_from_slice(
        &u32::try_from(value)
            .map_err(|_| "boot manifest item count exceeds u32")?
            .to_le_bytes(),
    );
    Ok(())
}

fn put_manifest_string(output: &mut Vec<u8>, value: &str) -> Result<(), String> {
    if value.len() > 1024 * 1024 {
        return Err("boot manifest string exceeds limit".to_owned());
    }
    put_manifest_len(output, value.len())?;
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_manifest_field(output: &mut Vec<u8>, value: Option<&FieldName>) -> Result<(), String> {
    output.push(u8::from(value.is_some()));
    if let Some(value) = value {
        put_manifest_string(output, value.as_str())?;
    }
    Ok(())
}

fn get_manifest_u8(input: &mut std::io::Cursor<&[u8]>) -> Result<u8, String> {
    let mut value = [0];
    std::io::Read::read_exact(input, &mut value).map_err(|error| error.to_string())?;
    Ok(value[0])
}

fn get_manifest_len(input: &mut std::io::Cursor<&[u8]>) -> Result<usize, String> {
    let mut value = [0; 4];
    std::io::Read::read_exact(input, &mut value).map_err(|error| error.to_string())?;
    Ok(u32::from_le_bytes(value) as usize)
}

fn get_manifest_string(input: &mut std::io::Cursor<&[u8]>) -> Result<String, String> {
    let length = get_manifest_len(input)?;
    if length > 1024 * 1024 {
        return Err("boot manifest string exceeds limit".to_owned());
    }
    let mut value = vec![0; length];
    std::io::Read::read_exact(input, &mut value).map_err(|error| error.to_string())?;
    String::from_utf8(value).map_err(|error| error.to_string())
}

fn get_manifest_field(input: &mut std::io::Cursor<&[u8]>) -> Result<Option<FieldName>, String> {
    match get_manifest_u8(input)? {
        0 => Ok(None),
        1 => FieldName::parse(&get_manifest_string(input)?)
            .map(Some)
            .map_err(|error| error.to_string()),
        _ => Err("invalid boot manifest optional-field tag".to_owned()),
    }
}

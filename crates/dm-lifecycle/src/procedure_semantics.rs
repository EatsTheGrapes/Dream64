use crc32fast::hash;
use dm_vm::Module;

const PROCEDURE_SEMANTICS_MAGIC: &[u8; 8] = b"D64PSEM\0";
const PROCEDURE_SEMANTICS_VERSION: u16 = 1;
const MAX_PROCEDURE_SEMANTICS_BYTES: u64 = 256 * 1024 * 1024;

/// Builds a portable semantic-identity directory for every eager procedure.
pub fn encode_procedure_semantics(module: &Module) -> Result<Vec<u8>, String> {
    if module.deferred_procedure_count() != 0 || module.procedure_count() > 1_000_000 {
        return Err(
            "procedure semantic directory requires a bounded fully eager module".to_owned(),
        );
    }
    let mut payload = Vec::new();
    payload.extend_from_slice(&(module.procedure_count() as u32).to_le_bytes());
    let digests = module.compute_all_procedure_semantic_digests()?;
    for (path, digest) in module.procedure_paths().zip(digests) {
        if path.len() > 64 * 1024 * 1024 {
            return Err("procedure semantic path exceeds its limit".to_owned());
        }
        payload.extend_from_slice(&(path.len() as u32).to_le_bytes());
        payload.extend_from_slice(path.as_bytes());
        payload.extend_from_slice(&digest);
    }
    if payload.len() as u64 > MAX_PROCEDURE_SEMANTICS_BYTES {
        return Err("procedure semantic directory exceeds its limit".to_owned());
    }
    let mut encoded = Vec::with_capacity(22 + payload.len());
    encoded.extend_from_slice(PROCEDURE_SEMANTICS_MAGIC);
    encoded.extend_from_slice(&PROCEDURE_SEMANTICS_VERSION.to_le_bytes());
    encoded.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    encoded.extend_from_slice(&hash(&payload).to_le_bytes());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

/// Validates and attaches an artifact-emitted semantic directory to a module.
pub fn decode_and_attach_procedure_semantics(
    bytes: &[u8],
    module: &mut Module,
) -> Result<(), String> {
    if bytes.len() < 22 || &bytes[..8] != PROCEDURE_SEMANTICS_MAGIC {
        return Err("invalid procedure semantic directory header".to_owned());
    }
    if u16::from_le_bytes([bytes[8], bytes[9]]) != PROCEDURE_SEMANTICS_VERSION {
        return Err("unsupported procedure semantic directory version".to_owned());
    }
    let length = u64::from_le_bytes(bytes[10..18].try_into().unwrap());
    if length > MAX_PROCEDURE_SEMANTICS_BYTES || length as usize != bytes.len() - 22 {
        return Err("invalid procedure semantic directory length".to_owned());
    }
    let payload = &bytes[22..];
    if hash(payload) != u32::from_le_bytes(bytes[18..22].try_into().unwrap()) {
        return Err("procedure semantic directory checksum mismatch".to_owned());
    }
    let mut cursor = 0usize;
    let take = |cursor: &mut usize, count: usize| -> Result<&[u8], String> {
        let end = cursor
            .checked_add(count)
            .ok_or("procedure semantic offset overflow")?;
        let value = payload
            .get(*cursor..end)
            .ok_or("truncated procedure semantic directory")?;
        *cursor = end;
        Ok(value)
    };
    let count = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
    if count != module.procedure_count() || count > 1_000_000 {
        return Err("procedure semantic count does not match module".to_owned());
    }
    let expected_paths = module.procedure_paths().collect::<Vec<_>>();
    let mut digests = Vec::with_capacity(count);
    for expected in expected_paths {
        let path_len = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
        let path = std::str::from_utf8(take(&mut cursor, path_len)?)
            .map_err(|_| "procedure semantic path is not UTF-8")?;
        if path != expected {
            return Err("procedure semantic path table does not match module".to_owned());
        }
        digests.push(take(&mut cursor, 32)?.try_into().unwrap());
    }
    if cursor != payload.len() {
        return Err("trailing procedure semantic directory bytes".to_owned());
    }
    module.attach_procedure_semantic_digests(digests)
}

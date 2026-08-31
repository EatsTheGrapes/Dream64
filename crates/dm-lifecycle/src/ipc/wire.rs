//! The loopback IPC text wire protocol: the `Command` set that local clients
//! send, the length-prefixed frame reader/writer, and the parser that turns a
//! received frame into a `Command`.

use std::io::{self, Read as _, Write as _};
use std::net::TcpStream;
use std::sync::mpsc::SyncSender;

use dm_vm::{LocalClientPromptResponse, LocalMovementDirection};

use super::{MAX_RESOURCE_CHUNK_BYTES, unhex};

const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Command {
    Ping,
    Attach,
    MapSnapshot {
        session: String,
    },
    ScreenSnapshot {
        session: String,
    },
    Move {
        session: String,
        direction: LocalMovementDirection,
    },
    Resource {
        session: String,
        path: String,
    },
    ResourceChunk {
        session: String,
        path: String,
        offset: u64,
        length: u32,
    },
    UiEvents {
        session: String,
    },
    UiAck {
        session: String,
        sequence: u64,
    },
    SkinReady {
        session: String,
    },
    ResourcesReady {
        session: String,
    },
    InputReady {
        session: String,
    },
    ScreenPointer {
        session: String,
        index: u32,
        generation: u32,
        event: dm_vm::LocalScreenPointerEvent,
        location: String,
        params: String,
    },
    MapPointer {
        session: String,
        index: u32,
        generation: u32,
        x: i32,
        y: i32,
        z: i32,
        control: String,
        params: String,
    },
    BrowserTopic {
        session: String,
        topic: String,
    },
    ClientCommand {
        session: String,
        command: String,
    },
    PromptResponse {
        session: String,
        id: u64,
        response: LocalClientPromptResponse,
    },
}

pub(super) struct Request {
    pub(super) command: Command,
    pub(super) response: SyncSender<String>,
}

pub(super) fn parse_command(frame: &[u8]) -> Result<Command, String> {
    let mut p = std::str::from_utf8(frame)
        .map_err(|_| "frame is not UTF-8".to_owned())?
        .split_ascii_whitespace();
    match p.next() {
        Some("ping") if p.next().is_none() => Ok(Command::Ping),
        Some("attach") if p.next().is_none() => Ok(Command::Attach),
        Some("map_snapshot") => {
            let session = p
                .next()
                .ok_or("map_snapshot session is missing")?
                .to_owned();
            if p.next().is_some() {
                Err("map_snapshot has trailing arguments".into())
            } else {
                Ok(Command::MapSnapshot { session })
            }
        }
        Some("screen_snapshot") => {
            let session = p
                .next()
                .ok_or("screen_snapshot session is missing")?
                .to_owned();
            if p.next().is_some() {
                Err("screen_snapshot has trailing arguments".into())
            } else {
                Ok(Command::ScreenSnapshot { session })
            }
        }
        Some("move") => {
            let session = p.next().ok_or("move session is missing")?.to_owned();
            let direction = match p.next() {
                Some("north") => LocalMovementDirection::North,
                Some("south") => LocalMovementDirection::South,
                Some("east") => LocalMovementDirection::East,
                Some("west") => LocalMovementDirection::West,
                _ => return Err("move direction is invalid".into()),
            };
            if p.next().is_some() {
                Err("move has trailing arguments".into())
            } else {
                Ok(Command::Move { session, direction })
            }
        }
        Some("resource") => {
            let session = p.next().ok_or("resource session is missing")?.to_owned();
            let path = p.next().ok_or("resource path is missing")?;
            if p.next().is_some() {
                return Err("resource has trailing arguments".into());
            }
            let path = String::from_utf8(unhex(path)?)
                .map_err(|_| "resource path is not UTF-8".to_owned())?;
            Ok(Command::Resource { session, path })
        }
        Some("resource_chunk") => {
            let session = p
                .next()
                .ok_or("resource_chunk session is missing")?
                .to_owned();
            let path = p.next().ok_or("resource_chunk path is missing")?;
            let offset = p
                .next()
                .ok_or("resource_chunk offset is missing")?
                .parse::<u64>()
                .map_err(|_| "resource_chunk offset is invalid".to_owned())?;
            let length = p
                .next()
                .ok_or("resource_chunk length is missing")?
                .parse::<u32>()
                .map_err(|_| "resource_chunk length is invalid".to_owned())?;
            if p.next().is_some() {
                return Err("resource_chunk has trailing arguments".into());
            }
            if length == 0 || length > MAX_RESOURCE_CHUNK_BYTES {
                return Err("resource_chunk length is out of range".into());
            }
            let path = String::from_utf8(unhex(path)?)
                .map_err(|_| "resource_chunk path is not UTF-8".to_owned())?;
            Ok(Command::ResourceChunk {
                session,
                path,
                offset,
                length,
            })
        }
        Some("ui_events") => {
            let session = p.next().ok_or("ui_events session is missing")?.to_owned();
            if p.next().is_some() {
                Err("ui_events has trailing arguments".into())
            } else {
                Ok(Command::UiEvents { session })
            }
        }
        Some("ui_ack") => {
            let session = p.next().ok_or("ui_ack session is missing")?.to_owned();
            let sequence = p
                .next()
                .ok_or("ui_ack sequence is missing")?
                .parse()
                .map_err(|_| "ui_ack sequence is invalid")?;
            if p.next().is_some() {
                Err("ui_ack has trailing arguments".into())
            } else {
                Ok(Command::UiAck { session, sequence })
            }
        }
        Some(command @ ("skin_ready" | "resources_ready" | "input_ready")) => {
            let session = p.next().ok_or("readiness session is missing")?.to_owned();
            if p.next().is_some() {
                return Err("readiness command has trailing arguments".into());
            }
            Ok(match command {
                "skin_ready" => Command::SkinReady { session },
                "resources_ready" => Command::ResourcesReady { session },
                "input_ready" => Command::InputReady { session },
                _ => unreachable!(),
            })
        }
        Some("screen_pointer") => {
            let session = p
                .next()
                .ok_or("screen_pointer session is missing")?
                .to_owned();
            let target = p.next().ok_or("screen_pointer target is missing")?;
            let (index, generation) = target
                .split_once(':')
                .ok_or("screen_pointer target is invalid")?;
            let index =
                u32::from_str_radix(index, 16).map_err(|_| "screen_pointer index is invalid")?;
            let generation = u32::from_str_radix(generation, 16)
                .map_err(|_| "screen_pointer generation is invalid")?;
            let event = match p.next() {
                Some("entered") => dm_vm::LocalScreenPointerEvent::Entered,
                Some("exited") => dm_vm::LocalScreenPointerEvent::Exited,
                Some("click") => dm_vm::LocalScreenPointerEvent::Click,
                _ => return Err("screen_pointer event is invalid".into()),
            };
            let decode_text = |value: &str| -> Result<String, String> {
                let bytes = if value == "-" {
                    Vec::new()
                } else {
                    unhex(value)?
                };
                String::from_utf8(bytes).map_err(|_| "screen_pointer field is not UTF-8".into())
            };
            let location = decode_text(p.next().ok_or("screen_pointer location is missing")?)?;
            let params = decode_text(p.next().ok_or("screen_pointer params are missing")?)?;
            if p.next().is_some() {
                return Err("screen_pointer has trailing arguments".into());
            }
            Ok(Command::ScreenPointer {
                session,
                index,
                generation,
                event,
                location,
                params,
            })
        }
        Some("map_pointer") => {
            let session = p.next().ok_or("map_pointer session is missing")?.to_owned();
            let target = p.next().ok_or("map_pointer target is missing")?;
            let (index, generation) = target
                .split_once(':')
                .ok_or("map_pointer target is invalid")?;
            let index =
                u32::from_str_radix(index, 16).map_err(|_| "map_pointer index is invalid")?;
            let generation = u32::from_str_radix(generation, 16)
                .map_err(|_| "map_pointer generation is invalid")?;
            let x = p
                .next()
                .ok_or("map_pointer x is missing")?
                .parse()
                .map_err(|_| "map_pointer x is invalid")?;
            let y = p
                .next()
                .ok_or("map_pointer y is missing")?
                .parse()
                .map_err(|_| "map_pointer y is invalid")?;
            let z = p
                .next()
                .ok_or("map_pointer z is missing")?
                .parse()
                .map_err(|_| "map_pointer z is invalid")?;
            let decode_text = |value: &str| -> Result<String, String> {
                let bytes = if value == "-" {
                    Vec::new()
                } else {
                    unhex(value)?
                };
                String::from_utf8(bytes).map_err(|_| "map_pointer field is not UTF-8".into())
            };
            let control = decode_text(p.next().ok_or("map_pointer control is missing")?)?;
            let params = decode_text(p.next().ok_or("map_pointer params are missing")?)?;
            if p.next().is_some() {
                return Err("map_pointer has trailing arguments".into());
            }
            Ok(Command::MapPointer {
                session,
                index,
                generation,
                x,
                y,
                z,
                control,
                params,
            })
        }
        Some("browser_topic") => {
            let session = p
                .next()
                .ok_or("browser_topic session is missing")?
                .to_owned();
            let topic = p.next().ok_or("browser_topic payload is missing")?;
            if p.next().is_some() {
                return Err("browser_topic has trailing arguments".into());
            }
            let topic = String::from_utf8(unhex(topic)?)
                .map_err(|_| "browser_topic payload is not UTF-8".to_owned())?;
            Ok(Command::BrowserTopic { session, topic })
        }
        Some("client_command") => {
            let session = p
                .next()
                .ok_or("client_command session is missing")?
                .to_owned();
            let command = p.next().ok_or("client_command payload is missing")?;
            if p.next().is_some() {
                return Err("client_command has trailing arguments".into());
            }
            let command = String::from_utf8(unhex(command)?)
                .map_err(|_| "client_command payload is not UTF-8".to_owned())?;
            Ok(Command::ClientCommand { session, command })
        }
        Some("prompt_response") => {
            let session = p
                .next()
                .ok_or("prompt_response session is missing")?
                .to_owned();
            let id = p
                .next()
                .ok_or("prompt_response id is missing")?
                .parse()
                .map_err(|_| "prompt_response id is invalid")?;
            let kind = p.next().ok_or("prompt_response kind is missing")?;
            let payload = p.next().ok_or("prompt_response payload is missing")?;
            if p.next().is_some() {
                return Err("prompt_response has trailing arguments".into());
            }
            let response = match kind {
                "null" if payload == "-" => LocalClientPromptResponse::Null,
                "text" => LocalClientPromptResponse::Text(
                    String::from_utf8(unhex(payload)?)
                        .map_err(|_| "prompt_response text is not UTF-8")?,
                ),
                "number" => LocalClientPromptResponse::Number(
                    payload
                        .parse()
                        .map_err(|_| "prompt_response number is invalid")?,
                ),
                "choice" => LocalClientPromptResponse::Choice(
                    payload
                        .parse()
                        .map_err(|_| "prompt_response choice is invalid")?,
                ),
                _ => return Err("prompt_response kind is invalid".into()),
            };
            Ok(Command::PromptResponse {
                session,
                id,
                response,
            })
        }
        Some(_) => Err("unknown command".into()),
        None => Err("empty command".into()),
    }
}

pub(super) fn read_frame(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut h = [0; 4];
    stream.read_exact(&mut h)?;
    let len = u32::from_be_bytes(h) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IPC frame is too large",
        ));
    }
    let mut p = vec![0; len];
    stream.read_exact(&mut p)?;
    Ok(p)
}

pub(super) fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "IPC frame is too large"))?;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_session_commands() {
        assert_eq!(parse_command(b"attach"), Ok(Command::Attach));
        assert_eq!(
            parse_command(b"map_snapshot s1"),
            Ok(Command::MapSnapshot {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"resource s1 69636f6e732f746573742e646d69"),
            Ok(Command::Resource {
                session: "s1".into(),
                path: "icons/test.dmi".into(),
            })
        );
        assert_eq!(
            parse_command(b"resource_chunk s1 69636f6e732f746573742e646d69 4294967296 262144"),
            Ok(Command::ResourceChunk {
                session: "s1".into(),
                path: "icons/test.dmi".into(),
                offset: 4_294_967_296,
                length: 262_144,
            })
        );
        assert!(parse_command(b"resource_chunk s1 61 0 262145").is_err());
        assert_eq!(
            parse_command(b"ui_events s1"),
            Ok(Command::UiEvents {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"ui_ack s1 47"),
            Ok(Command::UiAck {
                session: "s1".into(),
                sequence: 47,
            })
        );
        assert_eq!(
            parse_command(b"skin_ready s1"),
            Ok(Command::SkinReady {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"resources_ready s1"),
            Ok(Command::ResourcesReady {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"input_ready s1"),
            Ok(Command::InputReady {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"move s1 east"),
            Ok(Command::Move {
                session: "s1".into(),
                direction: LocalMovementDirection::East
            })
        );
        assert_eq!(
            parse_command(b"screen_pointer s1 a:2 click - 6d6f7573652d783d31"),
            Ok(Command::ScreenPointer {
                session: "s1".into(),
                index: 10,
                generation: 2,
                event: dm_vm::LocalScreenPointerEvent::Click,
                location: String::new(),
                params: "mouse-x=1".into(),
            })
        );
        assert_eq!(
            parse_command(b"map_pointer s1 a:2 5 7 1 - 6c6566743d31"),
            Ok(Command::MapPointer {
                session: "s1".into(),
                index: 10,
                generation: 2,
                x: 5,
                y: 7,
                z: 1,
                control: String::new(),
                params: "left=1".into(),
            })
        );
        assert_eq!(
            parse_command(b"browser_topic s1 62796f6e643a2f2f3f616374696f6e3d7265616479"),
            Ok(Command::BrowserTopic {
                session: "s1".into(),
                topic: "byond://?action=ready".into(),
            })
        );
        assert_eq!(
            parse_command(b"client_command s1 726566726573682d7467756920226c6f626279206e6f7722"),
            Ok(Command::ClientCommand {
                session: "s1".into(),
                command: "refresh-tgui \"lobby now\"".into(),
            })
        );
        assert_eq!(
            parse_command(b"prompt_response s1 9 text 68656c6c6f"),
            Ok(Command::PromptResponse {
                session: "s1".into(),
                id: 9,
                response: LocalClientPromptResponse::Text("hello".into()),
            })
        );
        assert_eq!(
            parse_command(b"prompt_response s1 10 choice 2"),
            Ok(Command::PromptResponse {
                session: "s1".into(),
                id: 10,
                response: LocalClientPromptResponse::Choice(2),
            })
        );
    }
}

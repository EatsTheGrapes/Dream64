use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_CLIENT_GENERATION: AtomicU64 = AtomicU64::new(1);

#[repr(u16)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ClientReadiness {
    Transport = 1 << 0,
    LocalResources = 1 << 1,
    Skin = 1 << 2,
    ResourceManifest = 1 << 3,
    ResourcePayload = 1 << 4,
    MapRenderer = 1 << 5,
    WebViewEnvironment = 1 << 6,
    WebViewDocument = 1 << 7,
}

#[derive(Clone, Debug)]
pub(crate) struct GenerationTagged<T> {
    generation: u64,
    command: T,
}

#[derive(Clone, Debug)]
pub(crate) struct UiOwnerCommandQueue<T> {
    generation: u64,
    capacity: usize,
    commands: VecDeque<GenerationTagged<T>>,
    stale_dropped: u64,
    overflow_rejected: u64,
}

impl<T> UiOwnerCommandQueue<T> {
    pub(crate) fn new(generation: u64, capacity: usize) -> Self {
        Self {
            generation,
            capacity,
            commands: VecDeque::with_capacity(capacity),
            stale_dropped: 0,
            overflow_rejected: 0,
        }
    }

    pub(crate) fn advance_generation(&mut self, generation: u64) {
        self.generation = generation;
        let before = self.commands.len();
        self.commands
            .retain(|command| command.generation == generation);
        self.stale_dropped = self
            .stale_dropped
            .saturating_add((before - self.commands.len()) as u64);
    }

    pub(crate) fn push(&mut self, generation: u64, command: T) -> Result<(), T> {
        if generation != self.generation {
            self.stale_dropped = self.stale_dropped.saturating_add(1);
            return Err(command);
        }
        if self.commands.len() == self.capacity {
            self.overflow_rejected = self.overflow_rejected.saturating_add(1);
            return Err(command);
        }
        self.commands.push_back(GenerationTagged {
            generation,
            command,
        });
        Ok(())
    }

    pub(crate) fn pop_current(&mut self) -> Option<T> {
        while let Some(command) = self.commands.pop_front() {
            if command.generation == self.generation {
                return Some(command.command);
            }
            self.stale_dropped = self.stale_dropped.saturating_add(1);
        }
        None
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ClientReadinessCoordinator {
    pub(crate) generation: u64,
    reached: u16,
    pub(crate) webview_required: bool,
    pub(crate) audio_available: bool,
    pub(crate) resources_sent: bool,
    pub(crate) interactive_sent: bool,
}

impl ClientReadinessCoordinator {
    pub(crate) fn new() -> Self {
        Self {
            generation: NEXT_CLIENT_GENERATION.fetch_add(1, Ordering::Relaxed),
            reached: 0,
            webview_required: false,
            audio_available: false,
            resources_sent: false,
            interactive_sent: false,
        }
    }

    pub(crate) fn rearm(&mut self) {
        *self = Self::new();
    }

    pub(crate) fn mark(
        &mut self,
        generation: u64,
        readiness: ClientReadiness,
    ) -> Result<(), &'static str> {
        if generation != self.generation {
            return Err("stale-client-generation");
        }
        self.reached |= readiness as u16;
        Ok(())
    }

    pub(crate) const fn has(&self, readiness: ClientReadiness) -> bool {
        self.reached & readiness as u16 != 0
    }

    pub(crate) fn resources_ready(&self) -> bool {
        self.has(ClientReadiness::Transport)
            && self.has(ClientReadiness::LocalResources)
            && self.has(ClientReadiness::Skin)
            && self.has(ClientReadiness::ResourceManifest)
            && self.has(ClientReadiness::ResourcePayload)
    }

    pub(crate) fn interactive_ready(&self) -> bool {
        self.resources_ready()
            && self.has(ClientReadiness::MapRenderer)
            && (!self.webview_required
                || (self.has(ClientReadiness::WebViewEnvironment)
                    && self.has(ClientReadiness::WebViewDocument)))
    }
}

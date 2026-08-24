//! Loss-aware parsing for BYOND-compatible Dream Maker interface (`.dmf`) files.
//!
//! The initial parser recognizes `window`, `menu`, and `macro` sections, their
//! `elem` children, and ordered key/value properties. It preserves raw values,
//! decoded quoted values, comments, and byte-accurate source spans while
//! recovering from malformed lines to produce actionable diagnostics.
//!
//! This stage does not yet interpret control-specific property schemas, resolve
//! references between controls/windows/menus/macros, expand preprocessing
//! directives, or join multiline/continued property values. Those concerns are
//! deliberately left for validation and lowering passes once reference fixtures
//! establish their exact compatibility behavior.

#![cfg_attr(not(test), deny(missing_docs))]

use dm_core::SourceSpan;

/// A parsed DMF document in deterministic source order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Document {
    /// Top-level windows, menus, and macro sets in encounter order.
    pub sections: Vec<Section>,
    /// Standalone comments in encounter order.
    pub comments: Vec<Comment>,
    /// Recoverable parser diagnostics in encounter order.
    pub diagnostics: Vec<Diagnostic>,
    /// Original source length in bytes.
    pub source_len: usize,
}

/// A typed top-level DMF definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Section {
    /// A client window containing controls.
    Window(Window),
    /// A menu containing menu entries.
    Menu(Menu),
    /// A macro set containing key bindings.
    MacroSet(MacroSet),
}

impl Section {
    /// Returns the section's identifier.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Window(section) => &section.name,
            Self::Menu(section) => &section.name,
            Self::MacroSet(section) => &section.name,
        }
    }

    /// Returns the complete byte range occupied by the section.
    #[must_use]
    pub const fn span(&self) -> SourceSpan {
        match self {
            Self::Window(section) => section.span,
            Self::Menu(section) => section.span,
            Self::MacroSet(section) => section.span,
        }
    }
}

/// A DMF window and its controls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Window {
    /// Window identifier from the quoted header.
    pub name: String,
    /// Complete source range of the window block.
    pub span: SourceSpan,
    /// Controls in source order.
    pub controls: Vec<Control>,
}

/// A control belonging to a [`Window`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Control {
    /// Optional control identifier; anonymous `elem` entries are valid DMF.
    pub id: Option<String>,
    /// Complete source range of the element block.
    pub span: SourceSpan,
    /// Properties in source order.
    pub properties: Vec<Property>,
}

/// A DMF menu and its entries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Menu {
    /// Menu identifier from the quoted header.
    pub name: String,
    /// Complete source range of the menu block.
    pub span: SourceSpan,
    /// Entries in source order.
    pub entries: Vec<MenuEntry>,
}

/// One element in a [`Menu`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MenuEntry {
    /// Optional entry identifier; menu separators and actions may be anonymous.
    pub id: Option<String>,
    /// Complete source range of the element block.
    pub span: SourceSpan,
    /// Properties in source order.
    pub properties: Vec<Property>,
}

/// A named group of client keyboard macros.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MacroSet {
    /// Macro-set identifier from the quoted header.
    pub name: String,
    /// Complete source range of the macro block.
    pub span: SourceSpan,
    /// Macro bindings in source order.
    pub macros: Vec<Macro>,
}

/// One keyboard binding in a [`MacroSet`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Macro {
    /// Optional binding identifier.
    pub id: Option<String>,
    /// Complete source range of the element block.
    pub span: SourceSpan,
    /// Properties in source order.
    pub properties: Vec<Property>,
}

/// One ordered `key = value` assignment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Property {
    /// Property name exactly as spelled, excluding surrounding whitespace.
    pub key: String,
    /// Complete source range of the property line.
    pub span: SourceSpan,
    /// Byte range containing the property name.
    pub key_span: SourceSpan,
    /// Loss-aware property value.
    pub value: PropertyValue,
}

/// A property value retaining both source spelling and decoded content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PropertyValue {
    /// Exact trimmed source spelling, including quotes when present.
    pub raw: String,
    /// Content with surrounding quotes removed and basic escapes decoded.
    pub decoded: String,
    /// Lexical representation used by the value.
    pub kind: ValueKind,
    /// Byte range containing the trimmed raw value.
    pub span: SourceSpan,
}

/// Lexical form of a DMF property value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueKind {
    /// An unquoted token or compound value such as `10,20` or `640x480`.
    Bare,
    /// A double-quoted text value.
    Quoted,
    /// A single-quoted resource path.
    Resource,
}

/// A retained full-line DMF comment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Comment {
    /// Exact comment spelling excluding indentation.
    pub raw: String,
    /// Byte range of the comment text.
    pub span: SourceSpan,
}

/// Impact of a parser diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticSeverity {
    /// Parsing recovered, but the source is ambiguous or suspicious.
    Warning,
    /// The line could not be represented according to the supported grammar.
    Error,
}

/// Machine-readable parser diagnostic category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticKind {
    /// A section or element header was malformed.
    MalformedHeader,
    /// An element appeared before any window, menu, or macro section.
    ElementOutsideSection,
    /// A property appeared before any element in its section.
    PropertyOutsideElement,
    /// A property name or assignment was malformed.
    MalformedProperty,
    /// A quoted name or value did not terminate on its physical line.
    UnterminatedQuote,
    /// Non-comment source followed an otherwise complete quoted value.
    TrailingCharacters,
    /// The line is not part of the supported DMF grammar.
    UnknownStatement,
    /// A property name repeats within one element.
    DuplicateProperty,
    /// A section or element identifier repeats in the same scope.
    DuplicateIdentifier,
}

/// An actionable, byte-located parser problem.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    /// Machine-readable category.
    pub kind: DiagnosticKind,
    /// Whether the problem invalidates the line.
    pub severity: DiagnosticSeverity,
    /// Human-readable explanation.
    pub message: String,
    /// Relevant source byte range.
    pub span: SourceSpan,
}

/// Parses a DMF source document while retaining recoverable source information.
#[must_use]
pub fn parse(source: &str) -> Document {
    Parser::new(source).run()
}

/// A resolved, client-facing view of the window controls in a DMF document.
///
/// The tree deliberately retains property spelling and source order.  Runtime
/// UI state belongs in the client protocol layer; this type describes only the
/// immutable skin definition from which that state is initialized.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ControlTree {
    /// Named windows in source order.
    pub windows: Vec<WindowNode>,
    /// Named elements from macro and menu sections in source order.
    pub auxiliary: Vec<WindowNode>,
    /// Diagnostics produced while resolving controls.
    pub diagnostics: Vec<ControlTreeDiagnostic>,
}

impl ControlTree {
    /// Lowers the window sections of a parsed DMF document into control nodes.
    ///
    /// Anonymous controls remain present but cannot be addressed by client
    /// commands such as `winset`; duplicate identifiers are retained in source
    /// order and reported so a compatibility policy can select one later.
    #[must_use]
    pub fn from_document(document: &Document) -> Self {
        let mut tree = Self::default();
        for section in &document.sections {
            let Section::Window(window) = section else {
                let (kind, name, span, elements): (&str, &str, SourceSpan, Vec<_>) = match section {
                    Section::MacroSet(macros) => (
                        "macro",
                        &macros.name,
                        macros.span,
                        macros
                            .macros
                            .iter()
                            .map(|entry| (entry.id.clone(), entry.span, entry.properties.clone()))
                            .collect(),
                    ),
                    Section::Menu(menu) => (
                        "menu",
                        &menu.name,
                        menu.span,
                        menu.entries
                            .iter()
                            .map(|entry| (entry.id.clone(), entry.span, entry.properties.clone()))
                            .collect(),
                    ),
                    Section::Window(_) => unreachable!(),
                };
                tree.auxiliary.push(WindowNode {
                    id: format!("{kind}:{name}"),
                    span,
                    controls: elements
                        .into_iter()
                        .map(|(id, span, properties)| ControlNode {
                            id,
                            span,
                            control_type: ControlType::Unknown,
                            properties,
                        })
                        .collect(),
                });
                continue;
            };
            let mut node = WindowNode {
                id: window.name.clone(),
                span: window.span,
                controls: Vec::with_capacity(window.controls.len()),
            };
            for control in &window.controls {
                if let Some(id) = &control.id
                    && node
                        .controls
                        .iter()
                        .any(|existing: &ControlNode| existing.id.as_deref() == Some(id))
                {
                    tree.diagnostics.push(ControlTreeDiagnostic {
                        kind: ControlTreeDiagnosticKind::DuplicateControlId,
                        message: format!(
                            "duplicate control identifier {id:?} in window {:?}",
                            window.name
                        ),
                        span: control.span,
                    });
                }
                let control_type = control
                    .properties
                    .iter()
                    .rev()
                    .find(|property| property.key.eq_ignore_ascii_case("type"))
                    .map_or(ControlType::Unknown, |property| {
                        ControlType::from_dmf(&property.value.decoded)
                    });
                node.controls.push(ControlNode {
                    id: control.id.clone(),
                    span: control.span,
                    control_type,
                    properties: control.properties.clone(),
                });
            }
            tree.windows.push(node);
        }
        tree
    }

    /// Resolves an addressable control by window and control identifier.
    #[must_use]
    pub fn control(&self, window_id: &str, control_id: &str) -> Option<&ControlNode> {
        self.windows
            .iter()
            .find(|window| window.id == window_id)?
            .controls
            .iter()
            .find(|control| control.id.as_deref() == Some(control_id))
    }

    fn addressable_control(&self, namespace: &str, control_id: &str) -> Option<&ControlNode> {
        self.control(namespace, control_id).or_else(|| {
            self.auxiliary
                .iter()
                .find(|section| {
                    section.id == namespace
                        || section.id.strip_prefix("macro:") == Some(namespace)
                        || section.id.strip_prefix("menu:") == Some(namespace)
                })?
                .controls
                .iter()
                .find(|control| control.id.as_deref() == Some(control_id))
        })
    }
}

/// One named DMF window in a [`ControlTree`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowNode {
    /// Window identifier.
    pub id: String,
    /// Source range of the window definition.
    pub span: SourceSpan,
    /// Controls in source order.
    pub controls: Vec<ControlNode>,
}

/// One DMF control in a [`WindowNode`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlNode {
    /// Optional control identifier.
    pub id: Option<String>,
    /// Source range of the control definition.
    pub span: SourceSpan,
    /// Recognized type from the final `type` property, if any.
    pub control_type: ControlType,
    /// Ordered, loss-aware control properties.
    pub properties: Vec<Property>,
}

impl ControlNode {
    /// Returns the final case-insensitive value of a DMF property.
    #[must_use]
    pub fn property(&self, name: &str) -> Option<&str> {
        self.properties
            .iter()
            .rev()
            .find(|property| property.key.eq_ignore_ascii_case(name))
            .map(|property| property.value.decoded.as_str())
    }

    /// Resolves a pixel rectangle from the supported `pos` and `size` forms.
    #[must_use]
    pub fn pixel_rect(&self) -> Option<PixelRect> {
        let (x, y) = parse_pair(self.property("pos")?)?;
        let (width, height) = parse_size(self.property("size")?)?;
        Some(PixelRect {
            x,
            y,
            width,
            height,
        })
    }
}

/// A DMF control rectangle in client pixels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelRect {
    /// Left coordinate.
    pub x: u32,
    /// Top coordinate.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

fn parse_pair(value: &str) -> Option<(u32, u32)> {
    let (left, right) = value.split_once(',')?;
    Some((left.trim().parse().ok()?, right.trim().parse().ok()?))
}

fn parse_size(value: &str) -> Option<(u32, u32)> {
    let normalized = value.to_ascii_lowercase();
    let (width, height) = normalized.split_once('x')?;
    Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
}

/// The compatibility-relevant kind of a DMF control.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlType {
    /// A top-level client window.
    Main,
    /// A game map rendering surface.
    Map,
    /// An embedded browser surface.
    Browser,
    /// A text input control.
    Input,
    /// A text output control.
    Output,
    /// A static text label.
    Label,
    /// An interactive button.
    Button,
    /// A control type not yet modeled by this compatibility slice.
    Unknown,
}

impl ControlType {
    fn from_dmf(value: &str) -> Self {
        match value.to_ascii_uppercase().as_str() {
            "MAIN" => Self::Main,
            "MAP" => Self::Map,
            "BROWSER" => Self::Browser,
            "INPUT" => Self::Input,
            "OUTPUT" => Self::Output,
            "LABEL" => Self::Label,
            "BUTTON" => Self::Button,
            _ => Self::Unknown,
        }
    }
}

/// A non-fatal issue found while lowering a [`ControlTree`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlTreeDiagnostic {
    /// Machine-readable classification of the issue.
    pub kind: ControlTreeDiagnosticKind,
    /// Human-readable explanation.
    pub message: String,
    /// Relevant source range.
    pub span: SourceSpan,
}

/// Categories of control-tree lowering issues.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlTreeDiagnosticKind {
    /// Two controls in a window use the same addressable identifier.
    DuplicateControlId,
}

/// Mutable, per-client UI state initialized from a [`ControlTree`].
///
/// This is deliberately headless: it models the observable control properties
/// and commands that a future native shell will render, rather than embedding a
/// particular toolkit in the compatibility layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UiState {
    tree: ControlTree,
    overrides: Vec<ControlOverride>,
    cloned_windows: Vec<ClonedWindow>,
}

impl UiState {
    /// Creates empty runtime overrides for a parsed skin.
    #[must_use]
    pub fn new(tree: ControlTree) -> Self {
        Self {
            tree,
            overrides: Vec::new(),
            cloned_windows: Vec::new(),
        }
    }

    /// Returns the immutable skin definition backing this state.
    #[must_use]
    pub const fn tree(&self) -> &ControlTree {
        &self.tree
    }

    /// Applies one authoritative server-to-client UI command.
    ///
    /// # Errors
    ///
    /// Returns an error when the command references an invalid control or has
    /// malformed parameters.
    pub fn apply(&mut self, command: UiCommand) -> Result<UiCommandReply, UiStateError> {
        match command {
            UiCommand::WinSet {
                control,
                parameters,
            } => {
                self.winset(&control, &parameters)?;
                Ok(UiCommandReply::Applied)
            }
            UiCommand::WinGet { control, property } => {
                Ok(UiCommandReply::Property(self.winget(&control, &property)?))
            }
            UiCommand::WinShow { control, visible } => {
                self.winshow(&control, visible)?;
                Ok(UiCommandReply::Applied)
            }
            UiCommand::WinExists { control } => {
                Ok(UiCommandReply::Exists(self.winexists(&control)))
            }
            UiCommand::WinClone {
                source,
                destination,
            } => {
                self.winclone(&source, &destination)?;
                Ok(UiCommandReply::Applied)
            }
        }
    }

    /// Applies semicolon-separated `property=value` assignments to a control.
    ///
    /// Values may be quoted and are stored without their surrounding quotes.
    /// An unknown control leaves the state unchanged and returns an error.
    ///
    /// # Errors
    ///
    /// Returns an error when the control cannot be resolved or an assignment is malformed.
    pub fn winset(&mut self, control: &str, parameters: &str) -> Result<(), UiStateError> {
        let assignments = parse_assignments(parameters)?;
        let (window_id, control_id) = match self.resolve_control(control) {
            Ok(resolved) => resolved,
            Err(UiStateError::UnknownControl(_)) => {
                let parent = assignments
                    .iter()
                    .find(|(property, _)| property.eq_ignore_ascii_case("parent"))
                    .map(|(_, value)| value.as_str())
                    .filter(|parent| !parent.is_empty())
                    .ok_or_else(|| UiStateError::UnknownControl(control.to_owned()))?;
                let parent_window = self
                    .tree
                    .windows
                    .iter()
                    .chain(self.tree.auxiliary.iter())
                    .find(|window| {
                        window.id == parent
                            || window.id.strip_prefix("macro:") == Some(parent)
                            || window.id.strip_prefix("menu:") == Some(parent)
                    })
                    .map(|window| window.id.clone())
                    .or_else(|| self.resolve_control(parent).ok().map(|(window, _)| window))
                    .ok_or_else(|| UiStateError::UnknownControl(parent.to_owned()))?;
                let (window_id, control_id) = control.split_once('.').map_or_else(
                    || (parent_window.clone(), control.to_owned()),
                    |(window, child)| {
                        let window = self
                            .tree
                            .auxiliary
                            .iter()
                            .find(|section| {
                                section.id == window
                                    || section.id.strip_prefix("macro:") == Some(window)
                                    || section.id.strip_prefix("menu:") == Some(window)
                            })
                            .map_or_else(|| window.to_owned(), |section| section.id.clone());
                        (window, child.to_owned())
                    },
                );
                if window_id != parent_window || control_id.is_empty() {
                    return Err(UiStateError::UnknownControl(control.to_owned()));
                }
                (window_id, control_id)
            }
            Err(error) => return Err(error),
        };
        let overrides = self.overrides_mut(&window_id, &control_id);
        for (property, value) in assignments {
            if let Some(existing) = overrides
                .properties
                .iter_mut()
                .find(|existing| existing.name.eq_ignore_ascii_case(&property))
            {
                existing.value = value;
            } else {
                overrides.properties.push(RuntimeProperty {
                    name: property,
                    value,
                });
            }
        }
        Ok(())
    }

    /// Returns a control property, preferring a runtime override over the skin.
    ///
    /// # Errors
    ///
    /// Returns an error when the control cannot be resolved.
    pub fn winget(&self, control: &str, property: &str) -> Result<String, UiStateError> {
        if control.contains(';') {
            return control
                .split(';')
                .map(str::trim)
                .filter(|control| !control.is_empty())
                .map(|control| {
                    self.winget(control, property)
                        .map(|value| format!("{control}.{property}={value}"))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|entries| entries.join(";"));
        }
        if let Some(namespace) = control.strip_suffix(".*") {
            let section = self
                .tree
                .windows
                .iter()
                .chain(self.tree.auxiliary.iter())
                .find(|section| {
                    section.id == namespace
                        || section.id.strip_prefix("macro:") == Some(namespace)
                        || section.id.strip_prefix("menu:") == Some(namespace)
                })
                .ok_or_else(|| UiStateError::UnknownControl(control.to_owned()))?;
            let mut ids = section
                .controls
                .iter()
                .filter_map(|node| node.id.clone())
                .collect::<Vec<_>>();
            ids.extend(
                self.overrides
                    .iter()
                    .filter(|override_| override_.window_id == section.id)
                    .map(|override_| override_.control_id.clone()),
            );
            ids.sort();
            ids.dedup();
            return ids
                .into_iter()
                .map(|id| {
                    let address = format!("{namespace}.{id}");
                    self.winget(&address, property)
                        .map(|value| format!("{address}.{property}={value}"))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|entries| entries.join(";"));
        }
        let (window_id, control_id) = self.resolve_control(control)?;
        if let Some(value) = self
            .overrides
            .iter()
            .find(|override_| {
                override_.window_id == window_id && override_.control_id == control_id
            })
            .and_then(|override_| {
                override_
                    .properties
                    .iter()
                    .rev()
                    .find(|entry| entry.name.eq_ignore_ascii_case(property))
            })
        {
            return Ok(value.value.clone());
        }
        let value = self
            .tree
            .addressable_control(
                self.source_window_id(&window_id),
                self.source_control_id(&window_id, &control_id),
            )
            .and_then(|node| {
                node.properties
                    .iter()
                    .rev()
                    .find(|entry| entry.key.eq_ignore_ascii_case(property))
            })
            .map_or_else(String::new, |entry| entry.value.decoded.clone());
        Ok(value)
    }

    /// Returns every declared and overridden property for a control.
    ///
    /// Runtime values replace skin defaults case-insensitively. This is the
    /// object form returned by BYOND's browser-side `winget(id, "*")` API.
    ///
    /// # Errors
    ///
    /// Returns an error when the control cannot be resolved.
    pub fn winget_all(
        &self,
        control: &str,
    ) -> Result<std::collections::BTreeMap<String, String>, UiStateError> {
        let (window_id, control_id) = self.resolve_control(control)?;
        let mut properties = std::collections::BTreeMap::<String, (String, String)>::new();
        if let Some(node) = self.tree.addressable_control(
            self.source_window_id(&window_id),
            self.source_control_id(&window_id, &control_id),
        ) {
            for property in &node.properties {
                properties.insert(
                    property.key.to_ascii_lowercase(),
                    (property.key.clone(), property.value.decoded.clone()),
                );
            }
        }
        if let Some(override_) = self.overrides.iter().find(|override_| {
            override_.window_id == window_id && override_.control_id == control_id
        }) {
            for property in &override_.properties {
                properties.insert(
                    property.name.to_ascii_lowercase(),
                    (property.name.clone(), property.value.clone()),
                );
            }
        }
        Ok(properties
            .into_values()
            .collect::<std::collections::BTreeMap<_, _>>())
    }

    /// Sets a control's `is-visible` runtime property.
    ///
    /// # Errors
    ///
    /// Returns an error when the control cannot be resolved.
    pub fn winshow(&mut self, control: &str, visible: bool) -> Result<(), UiStateError> {
        self.winset(
            control,
            if visible {
                "is-visible=true"
            } else {
                "is-visible=false"
            },
        )
    }

    /// Returns whether a named control is present in the loaded skin.
    #[must_use]
    pub fn winexists(&self, control: &str) -> bool {
        self.resolve_control(control).is_ok()
    }

    /// Returns the BYOND control-type name for a named control, or an empty
    /// string when the control is absent.
    #[must_use]
    pub fn winexists_type(&self, control: &str) -> String {
        let Ok((window_id, control_id)) = self.resolve_control(control) else {
            return String::new();
        };
        self.tree
            .addressable_control(
                self.source_window_id(&window_id),
                self.source_control_id(&window_id, &control_id),
            )
            .map(|control| match control.control_type {
                ControlType::Main => "MAIN",
                ControlType::Map => "MAP",
                ControlType::Browser => "BROWSER",
                ControlType::Input => "INPUT",
                ControlType::Output => "OUTPUT",
                ControlType::Label => "LABEL",
                ControlType::Button => "BUTTON",
                ControlType::Unknown => "UNKNOWN",
            })
            .unwrap_or_default()
            .to_owned()
    }

    /// Returns addressable control identifiers in a window/menu/macro section,
    /// including controls created at runtime with `winset(parent=...)`.
    /// Source order and runtime insertion order are preserved.
    pub fn section_control_ids(&self, namespace: &str) -> Result<Vec<String>, UiStateError> {
        let section = self
            .tree
            .windows
            .iter()
            .chain(self.tree.auxiliary.iter())
            .find(|section| {
                section.id == namespace
                    || section.id.strip_prefix("macro:") == Some(namespace)
                    || section.id.strip_prefix("menu:") == Some(namespace)
            })
            .ok_or_else(|| UiStateError::UnknownControl(namespace.to_owned()))?;
        let mut ids = section
            .controls
            .iter()
            .filter_map(|control| control.id.clone())
            .collect::<Vec<_>>();
        for override_ in self
            .overrides
            .iter()
            .filter(|override_| override_.window_id == section.id)
        {
            if !ids.contains(&override_.control_id) {
                ids.push(override_.control_id.clone());
            }
        }
        Ok(ids)
    }

    /// Copies runtime overrides from one control to another.
    ///
    /// # Errors
    ///
    /// Returns an error when either control cannot be resolved.
    pub fn winclone(&mut self, source: &str, destination: &str) -> Result<(), UiStateError> {
        if let Some(source_section) = self
            .tree
            .windows
            .iter()
            .chain(self.tree.auxiliary.iter())
            .find(|section| section.id == source)
        {
            if destination.is_empty() {
                return Err(UiStateError::UnknownControl(destination.to_owned()));
            }
            let source_id = source_section.id.clone();
            self.cloned_windows
                .retain(|clone| clone.destination != destination);
            self.cloned_windows.push(ClonedWindow {
                source: source_id.clone(),
                destination: destination.to_owned(),
            });
            let copied = self
                .overrides
                .iter()
                .filter(|override_| override_.window_id == source_id)
                .cloned()
                .collect::<Vec<_>>();
            for mut override_ in copied {
                override_.window_id = destination.to_owned();
                if override_.control_id == source_id {
                    override_.control_id = destination.to_owned();
                }
                let target = self.overrides_mut(&override_.window_id, &override_.control_id);
                target.properties = override_.properties;
            }
            return Ok(());
        }
        let (source_window, source_control) = self.resolve_control(source)?;
        let (destination_window, destination_control) = self.resolve_control(destination)?;
        let properties = self
            .overrides
            .iter()
            .find(|override_| {
                override_.window_id == source_window && override_.control_id == source_control
            })
            .map_or_else(Vec::new, |override_| override_.properties.clone());
        let destination = self.overrides_mut(&destination_window, &destination_control);
        destination.properties = properties;
        Ok(())
    }

    fn resolve_control(&self, address: &str) -> Result<(String, String), UiStateError> {
        if let Some(selector) = address.strip_prefix(':') {
            let expected = ControlType::from_dmf(selector);
            if expected != ControlType::Unknown {
                let mut matches = self
                    .tree
                    .windows
                    .iter()
                    .chain(self.tree.auxiliary.iter())
                    .flat_map(|window| {
                        window.controls.iter().filter_map(move |control| {
                            (control.control_type == expected).then_some((window, control))
                        })
                    })
                    .collect::<Vec<_>>();
                matches.sort_by_key(|(window, control)| {
                    let is_default = control.property("is-default").is_some_and(|value| {
                        matches!(
                            value.trim().to_ascii_lowercase().as_str(),
                            "true" | "yes" | "1"
                        )
                    });
                    (
                        !is_default,
                        window.id.as_str(),
                        control.id.as_deref().unwrap_or(""),
                    )
                });
                if let Some((window, control)) = matches.first()
                    && let Some(control_id) = control.id.as_deref()
                {
                    return Ok((window.id.clone(), control_id.to_owned()));
                }
            }
        }
        let Some((window_id, control_id)) = address.split_once('.') else {
            let mut matches: Vec<_> = self
                .tree
                .windows
                .iter()
                .chain(self.tree.auxiliary.iter())
                .filter_map(|window| {
                    window
                        .controls
                        .iter()
                        .any(|control| control.id.as_deref() == Some(address))
                        .then_some((window.id.as_str(), address))
                })
                .collect();
            matches.extend(
                self.overrides
                    .iter()
                    .filter(|override_| override_.control_id == address)
                    .map(|override_| (override_.window_id.as_str(), address)),
            );
            matches.extend(self.cloned_windows.iter().filter_map(|clone| {
                let source_control = if address == clone.destination {
                    clone.source.as_str()
                } else {
                    address
                };
                self.tree
                    .addressable_control(&clone.source, source_control)
                    .is_some()
                    .then_some((clone.destination.as_str(), address))
            }));
            matches.sort_unstable();
            matches.dedup();
            return match matches.as_slice() {
                [(window_id, control_id)] => {
                    Ok(((*window_id).to_owned(), (*control_id).to_owned()))
                }
                [] => Err(UiStateError::UnknownControl(address.to_owned())),
                _ => Err(UiStateError::AmbiguousControl(address.to_owned())),
            };
        };
        if let Some(clone) = self
            .cloned_windows
            .iter()
            .find(|clone| clone.destination == window_id)
        {
            let source_control = if control_id == window_id {
                clone.source.as_str()
            } else {
                control_id
            };
            if self
                .tree
                .addressable_control(&clone.source, source_control)
                .is_some()
            {
                return Ok((window_id.to_owned(), control_id.to_owned()));
            }
        }
        if let Some(override_) = self.overrides.iter().find(|override_| {
            (override_.window_id == window_id
                || override_.window_id.strip_prefix("macro:") == Some(window_id)
                || override_.window_id.strip_prefix("menu:") == Some(window_id))
                && override_.control_id == control_id
        }) {
            return Ok((override_.window_id.clone(), control_id.to_owned()));
        }
        if self
            .tree
            .addressable_control(window_id, control_id)
            .is_some()
        {
            Ok((window_id.to_owned(), control_id.to_owned()))
        } else {
            Err(UiStateError::UnknownControl(address.to_owned()))
        }
    }

    fn overrides_mut(&mut self, window_id: &str, control_id: &str) -> &mut ControlOverride {
        if let Some(index) = self.overrides.iter().position(|override_| {
            override_.window_id == window_id && override_.control_id == control_id
        }) {
            return &mut self.overrides[index];
        }
        self.overrides.push(ControlOverride {
            window_id: window_id.to_owned(),
            control_id: control_id.to_owned(),
            properties: Vec::new(),
        });
        self.overrides
            .last_mut()
            .expect("a runtime override was just added")
    }

    fn source_window_id<'a>(&'a self, window_id: &'a str) -> &'a str {
        self.cloned_windows
            .iter()
            .find(|clone| clone.destination == window_id)
            .map_or(window_id, |clone| clone.source.as_str())
    }

    fn source_control_id<'a>(&'a self, window_id: &str, control_id: &'a str) -> &'a str {
        self.cloned_windows
            .iter()
            .find(|clone| clone.destination == window_id && control_id == clone.destination)
            .map_or(control_id, |clone| clone.source.as_str())
    }
}

/// An authoritative request from game code to one client's UI.
///
/// This enum is transport-neutral so the runtime can use it for local testing
/// before the client/server connection protocol is selected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiCommand {
    /// Apply `property=value` assignments to a control.
    WinSet {
        /// Qualified or unqualified control address.
        control: String,
        /// Semicolon-separated assignment string.
        parameters: String,
    },
    /// Read one effective control property.
    WinGet {
        /// Qualified or unqualified control address.
        control: String,
        /// Property name to read.
        property: String,
    },
    /// Change a control's visible state.
    WinShow {
        /// Qualified or unqualified control address.
        control: String,
        /// Requested visibility.
        visible: bool,
    },
    /// Determine whether a control exists in the loaded skin.
    WinExists {
        /// Qualified or unqualified control address.
        control: String,
    },
    /// Copy runtime overrides from one control to another.
    WinClone {
        /// Source control address.
        source: String,
        /// Destination control address.
        destination: String,
    },
}

/// The deterministic result of applying a [`UiCommand`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiCommandReply {
    /// A mutating command completed successfully.
    Applied,
    /// A requested property value.
    Property(String),
    /// Whether a requested control exists.
    Exists(bool),
}

/// The headless client-side state for one connected player.
///
/// A session owns no gameplay state. It accepts authoritative [`UiCommand`]s
/// from the server and queues local interaction events for the server to
/// consume at a deterministic scheduling boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClientSession {
    ui: UiState,
    events: Vec<UiEvent>,
}

impl ClientSession {
    /// Creates a player session using a parsed DMF control tree as its skin.
    #[must_use]
    pub fn new(tree: ControlTree) -> Self {
        Self {
            ui: UiState::new(tree),
            events: Vec::new(),
        }
    }

    /// Returns the current headless UI state.
    #[must_use]
    pub const fn ui(&self) -> &UiState {
        &self.ui
    }

    /// Applies an authoritative server command to this player's UI.
    ///
    /// # Errors
    ///
    /// Returns an error when the command cannot be applied to the loaded skin.
    pub fn apply_command(&mut self, command: UiCommand) -> Result<UiCommandReply, UiStateError> {
        self.ui.apply(command)
    }

    /// Queues a local UI event for delivery to the server.
    pub fn push_event(&mut self, event: UiEvent) {
        self.events.push(event);
    }

    /// Drains local UI events in the order they occurred.
    #[must_use]
    pub fn take_events(&mut self) -> Vec<UiEvent> {
        std::mem::take(&mut self.events)
    }

    /// Handles a capability-limited message from an embedded browser control.
    ///
    /// # Errors
    ///
    /// Returns an error when the sender origin is not allowed, the source is
    /// not a browser control, or the requested UI operation cannot be applied.
    pub fn handle_browser_message(
        &mut self,
        source_origin: &str,
        source_control: &str,
        policy: &BrowserPolicy,
        message: BrowserBridgeRequest,
    ) -> Result<BrowserBridgeReply, BrowserBridgeError> {
        if source_origin != policy.origin {
            return Err(BrowserBridgeError::OriginDenied {
                expected: policy.origin.clone(),
                actual: source_origin.to_owned(),
            });
        }
        let Some((window, control)) = source_control.split_once('.') else {
            return Err(BrowserBridgeError::NotBrowserControl(
                source_control.to_owned(),
            ));
        };
        if self
            .ui
            .tree()
            .control(window, control)
            .is_none_or(|node| node.control_type != ControlType::Browser)
        {
            return Err(BrowserBridgeError::NotBrowserControl(
                source_control.to_owned(),
            ));
        }
        match message {
            BrowserBridgeRequest::Ui(command) => self
                .apply_command(command)
                .map(BrowserBridgeReply::Ui)
                .map_err(BrowserBridgeError::Ui),
            BrowserBridgeRequest::Command { command } => {
                self.push_event(UiEvent::Command { command });
                Ok(BrowserBridgeReply::Accepted)
            }
            BrowserBridgeRequest::Topic { topic } => {
                self.push_event(UiEvent::BrowserTopic {
                    control: source_control.to_owned(),
                    topic,
                });
                Ok(BrowserBridgeReply::Accepted)
            }
        }
    }
}

/// An interaction emitted by one client UI session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiEvent {
    /// A configured macro or client command was invoked.
    Command {
        /// The command text to route through the server's client-command path.
        command: String,
    },
    /// A named control was activated, such as by a button click.
    ControlActivated {
        /// Fully-qualified control address.
        control: String,
    },
    /// Keyboard state changed while the client UI had focus.
    Key {
        /// Platform-neutral key identifier supplied by the shell.
        key: String,
        /// Whether the key was pressed (`true`) or released (`false`).
        pressed: bool,
    },
    /// A browser control sent a BYOND topic request.
    BrowserTopic {
        /// Fully-qualified browser control address.
        control: String,
        /// Topic payload or URL supplied by the browser bridge.
        topic: String,
    },
}

/// Per-browser capability policy supplied by the client shell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BrowserPolicy {
    /// The sole origin permitted to call the native Dream64 bridge.
    pub origin: String,
}

/// A message requested by JavaScript through the Dream64 `window.Byond` bridge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BrowserBridgeRequest {
    /// Invoke a client UI operation.
    Ui(UiCommand),
    /// Send a client command to the server.
    Command {
        /// Command text to dispatch through the client-command path.
        command: String,
    },
    /// Send a browser topic to the server.
    Topic {
        /// Topic payload supplied by the browser page.
        topic: String,
    },
}

/// The result of a [`BrowserBridgeRequest`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BrowserBridgeReply {
    /// The message was accepted and emitted as an event.
    Accepted,
    /// The result from a UI operation.
    Ui(UiCommandReply),
}

/// A browser page attempted to use a bridge capability it does not hold.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BrowserBridgeError {
    /// The sender did not match the browser control's configured origin.
    OriginDenied {
        /// Allowed origin.
        expected: String,
        /// Origin reported by the webview.
        actual: String,
    },
    /// The message did not originate from a declared browser control.
    NotBrowserControl(String),
    /// A valid browser message contained an invalid UI request.
    Ui(UiStateError),
}

/// JavaScript bootstrap injected only into browser controls by the future shell.
///
/// The host must serialize requests as JSON and route them through
/// [`ClientSession::handle_browser_message`]. The bridge intentionally exposes
/// no filesystem, process, or arbitrary native APIs.
pub const WEBVIEW2_BYOND_BRIDGE_BOOTSTRAP: &str = r#"(() => {
  const send = message => window.ipc.postMessage(JSON.stringify(message));
  window.cef_to_byond = url => send({ kind: "byond", url: String(url) });
  window.Byond = Object.freeze({
    command(command) { send({ kind: "command", command }); },
    topic(topic) { send({ kind: "topic", topic }); },
    ui(command) { send({ kind: "ui", command }); }
  });
  window.addEventListener("DOMContentLoaded", () => send({ kind: "ready" }), { once: true });
})();"#;

#[derive(Clone, Debug, Eq, PartialEq)]
struct ControlOverride {
    window_id: String,
    control_id: String,
    properties: Vec<RuntimeProperty>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ClonedWindow {
    source: String,
    destination: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RuntimeProperty {
    name: String,
    value: String,
}

/// A UI command could not be applied to the loaded skin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiStateError {
    /// The supplied control name resolves to no loaded control.
    UnknownControl(String),
    /// An unqualified control name exists in more than one window.
    AmbiguousControl(String),
    /// A `winset` parameter lacks a property name or `=` separator.
    MalformedAssignment(String),
    /// A quoted `winset` parameter did not terminate.
    UnterminatedAssignmentQuote,
}

fn parse_assignments(parameters: &str) -> Result<Vec<(String, String)>, UiStateError> {
    let mut assignments = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut escaped = false;
    for (offset, character) in parameters.char_indices() {
        if escaped {
            escaped = false;
        } else if matches!(character, '\\') && quote.is_some() {
            escaped = true;
        } else if quote == Some(character) {
            quote = None;
        } else if quote.is_none() && matches!(character, '\"' | '\'') {
            quote = Some(character);
        } else if quote.is_none() && character == ';' {
            assignments.push(parse_assignment(&parameters[start..offset])?);
            start = offset + character.len_utf8();
        }
    }
    if quote.is_some() {
        return Err(UiStateError::UnterminatedAssignmentQuote);
    }
    if !parameters[start..].trim().is_empty() {
        assignments.push(parse_assignment(&parameters[start..])?);
    }
    Ok(assignments)
}

fn parse_assignment(source: &str) -> Result<(String, String), UiStateError> {
    let Some((name, value)) = source.split_once('=') else {
        return Err(UiStateError::MalformedAssignment(source.trim().to_owned()));
    };
    let name = name.trim();
    if name.is_empty() {
        return Err(UiStateError::MalformedAssignment(source.trim().to_owned()));
    }
    let value = value.trim();
    let value = if let Some(quote) = value
        .chars()
        .next()
        .filter(|quote| matches!(quote, '\"' | '\''))
    {
        match quoted_content(value, quote) {
            QuotedResult::Complete { decoded, consumed } if value[consumed..].trim().is_empty() => {
                decoded
            }
            QuotedResult::Unterminated => return Err(UiStateError::UnterminatedAssignmentQuote),
            _ => return Err(UiStateError::MalformedAssignment(source.trim().to_owned())),
        }
    } else {
        value.to_owned()
    };
    Ok((name.to_owned(), value))
}

struct Parser<'source> {
    source: &'source str,
    document: Document,
    current_section: Option<usize>,
}

impl<'source> Parser<'source> {
    fn new(source: &'source str) -> Self {
        Self {
            source,
            document: Document {
                source_len: source.len(),
                ..Document::default()
            },
            current_section: None,
        }
    }

    fn run(mut self) -> Document {
        let mut start = 0usize;
        while start < self.source.len() {
            let newline = self.source[start..]
                .find('\n')
                .map_or(self.source.len(), |offset| start + offset);
            let content_end = newline
                .checked_sub(1)
                .filter(|index| self.source.as_bytes()[*index] == b'\r')
                .unwrap_or(newline);
            self.parse_line(start, content_end);
            start = if newline < self.source.len() {
                newline + 1
            } else {
                self.source.len()
            };
        }
        self.document
    }

    fn parse_line(&mut self, start: usize, end: usize) {
        let line = &self.source[start..end];
        let indentation = line.len() - line.trim_start_matches([' ', '\t']).len();
        let trimmed = &line[indentation..];
        if trimmed.is_empty() {
            return;
        }
        if is_full_line_comment(trimmed) {
            self.document.comments.push(Comment {
                raw: trimmed.to_owned(),
                span: SourceSpan::new(start + indentation, end),
            });
            return;
        }
        let inline_comment = inline_comment_start(trimmed);
        if let Some(comment_start) = inline_comment {
            self.document.comments.push(Comment {
                raw: trimmed[comment_start..].to_owned(),
                span: SourceSpan::new(absolute_offset(start, indentation, comment_start), end),
            });
        }
        let code_end = inline_comment.unwrap_or(trimmed.len());
        let code = trimmed[..code_end].trim_end();
        if code.is_empty() {
            return;
        }
        let absolute = start + indentation;

        let (keyword, tail) = leading_keyword(code);
        match keyword {
            "window" | "menu" | "macro" if !tail.trim_start().starts_with('=') => {
                self.parse_section(keyword, tail, absolute, end);
                return;
            }
            "elem" => {
                self.parse_element(tail, absolute, end);
                return;
            }
            _ => {}
        }
        if code.contains('=') {
            self.parse_property(code, absolute, end);
            return;
        }
        self.error(
            DiagnosticKind::UnknownStatement,
            "expected window, menu, macro, elem, or key = value",
            SourceSpan::new(absolute, end),
        );
    }

    fn parse_section(&mut self, keyword: &str, tail: &str, start: usize, end: usize) {
        let Some(name) = self.parse_quoted_name(tail, start + keyword.len()) else {
            return;
        };
        if self
            .document
            .sections
            .iter()
            .any(|section| section.name() == name)
        {
            self.warning(
                DiagnosticKind::DuplicateIdentifier,
                format!("duplicate top-level DMF identifier {name:?}"),
                SourceSpan::new(start, end),
            );
        }
        let span = SourceSpan::new(start, end);
        let section = match keyword {
            "window" => Section::Window(Window {
                name,
                span,
                controls: Vec::new(),
            }),
            "menu" => Section::Menu(Menu {
                name,
                span,
                entries: Vec::new(),
            }),
            "macro" => Section::MacroSet(MacroSet {
                name,
                span,
                macros: Vec::new(),
            }),
            _ => unreachable!("caller recognizes section keywords"),
        };
        self.document.sections.push(section);
        self.current_section = Some(self.document.sections.len() - 1);
    }

    fn parse_element(&mut self, tail: &str, start: usize, end: usize) {
        let Some(section_index) = self.current_section else {
            self.error(
                DiagnosticKind::ElementOutsideSection,
                "elem must follow a window, menu, or macro header",
                SourceSpan::new(start, end),
            );
            return;
        };
        let trimmed_tail = tail.trim();
        let id = if trimmed_tail.is_empty() {
            None
        } else {
            let Some(id) = self.parse_quoted_name(tail, start + "elem".len()) else {
                return;
            };
            Some(id)
        };
        if id
            .as_deref()
            .is_some_and(|id| section_has_element(&self.document.sections[section_index], id))
        {
            self.warning(
                DiagnosticKind::DuplicateIdentifier,
                format!(
                    "duplicate elem identifier {:?} in this section",
                    id.as_deref().expect("duplicate lookup requires an id")
                ),
                SourceSpan::new(start, end),
            );
        }
        let span = SourceSpan::new(start, end);
        match &mut self.document.sections[section_index] {
            Section::Window(section) => section.controls.push(Control {
                id,
                span,
                properties: Vec::new(),
            }),
            Section::Menu(section) => section.entries.push(MenuEntry {
                id,
                span,
                properties: Vec::new(),
            }),
            Section::MacroSet(section) => section.macros.push(Macro {
                id,
                span,
                properties: Vec::new(),
            }),
        }
        extend_section_span(&mut self.document.sections[section_index], end);
    }

    fn parse_property(&mut self, code: &str, start: usize, end: usize) {
        let Some(section_index) = self.current_section else {
            self.error(
                DiagnosticKind::PropertyOutsideElement,
                "property must belong to an elem inside a section",
                SourceSpan::new(start, end),
            );
            return;
        };
        if !section_has_elements(&self.document.sections[section_index]) {
            self.error(
                DiagnosticKind::PropertyOutsideElement,
                "property must follow an elem header",
                SourceSpan::new(start, end),
            );
            return;
        }
        let assignment = code.find('=').expect("caller found an assignment");
        let key_source = &code[..assignment];
        let key = key_source.trim();
        let key_offset = key_source.find(key).unwrap_or(0);
        if key.is_empty()
            || !key
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        {
            self.error(
                DiagnosticKind::MalformedProperty,
                "property name may contain only letters, digits, '_', '-', and '.'",
                SourceSpan::new(start, start + assignment),
            );
            return;
        }
        let value_source = &code[assignment + 1..];
        let value_trimmed = value_source.trim();
        let value_offset = assignment + 1 + value_source.find(value_trimmed).unwrap_or(0);
        let value_span = SourceSpan::new(
            start + value_offset,
            start + value_offset + value_trimmed.len(),
        );
        let value = self.parse_value(value_trimmed, value_span);
        let duplicate = last_properties(&self.document.sections[section_index])
            .is_some_and(|properties| properties.iter().any(|property| property.key == key));
        if duplicate {
            self.warning(
                DiagnosticKind::DuplicateProperty,
                format!("property {key:?} repeats within this elem; source order is preserved"),
                SourceSpan::new(start + key_offset, start + key_offset + key.len()),
            );
        }
        let property = Property {
            key: key.to_owned(),
            span: SourceSpan::new(start, end),
            key_span: SourceSpan::new(start + key_offset, start + key_offset + key.len()),
            value,
        };
        last_properties_mut(&mut self.document.sections[section_index])
            .expect("element existence was checked")
            .push(property);
        extend_last_element_span(&mut self.document.sections[section_index], end);
        extend_section_span(&mut self.document.sections[section_index], end);
    }

    fn parse_quoted_name(&mut self, tail: &str, base: usize) -> Option<String> {
        let leading = tail.len() - tail.trim_start_matches([' ', '\t']).len();
        let value = tail[leading..].trim_end();
        let span = SourceSpan::new(base + leading, base + leading + value.len());
        let parsed = quoted_content(value, '"');
        match parsed {
            QuotedResult::Complete { decoded, consumed } if value[consumed..].trim().is_empty() => {
                Some(decoded)
            }
            QuotedResult::Complete { consumed, .. } => {
                self.error(
                    DiagnosticKind::TrailingCharacters,
                    "unexpected source after quoted identifier",
                    SourceSpan::new(span.start + consumed, span.end),
                );
                None
            }
            QuotedResult::Unterminated => {
                self.error(
                    DiagnosticKind::UnterminatedQuote,
                    "unterminated quoted identifier",
                    span,
                );
                None
            }
            QuotedResult::NotQuoted => {
                self.error(
                    DiagnosticKind::MalformedHeader,
                    "DMF section and elem identifiers must be double quoted",
                    span,
                );
                None
            }
        }
    }

    fn parse_value(&mut self, raw: &str, span: SourceSpan) -> PropertyValue {
        let quote = raw
            .chars()
            .next()
            .filter(|character| matches!(character, '"' | '\''));
        let Some(quote) = quote else {
            return PropertyValue {
                raw: raw.to_owned(),
                decoded: raw.to_owned(),
                kind: ValueKind::Bare,
                span,
            };
        };
        let kind = if quote == '"' {
            ValueKind::Quoted
        } else {
            ValueKind::Resource
        };
        match quoted_content(raw, quote) {
            QuotedResult::Complete { decoded, consumed } => {
                if !raw[consumed..].trim().is_empty() {
                    self.error(
                        DiagnosticKind::TrailingCharacters,
                        "unexpected source after quoted property value",
                        SourceSpan::new(span.start + consumed, span.end),
                    );
                }
                PropertyValue {
                    raw: raw.to_owned(),
                    decoded,
                    kind,
                    span,
                }
            }
            QuotedResult::Unterminated => {
                self.error(
                    DiagnosticKind::UnterminatedQuote,
                    "unterminated quoted property value",
                    span,
                );
                PropertyValue {
                    raw: raw.to_owned(),
                    decoded: raw[quote.len_utf8()..].to_owned(),
                    kind,
                    span,
                }
            }
            QuotedResult::NotQuoted => unreachable!("the first character was checked"),
        }
    }

    fn error(&mut self, kind: DiagnosticKind, message: impl Into<String>, span: SourceSpan) {
        self.document.diagnostics.push(Diagnostic {
            kind,
            severity: DiagnosticSeverity::Error,
            message: message.into(),
            span,
        });
    }

    fn warning(&mut self, kind: DiagnosticKind, message: impl Into<String>, span: SourceSpan) {
        self.document.diagnostics.push(Diagnostic {
            kind,
            severity: DiagnosticSeverity::Warning,
            message: message.into(),
            span,
        });
    }
}

enum QuotedResult {
    Complete { decoded: String, consumed: usize },
    Unterminated,
    NotQuoted,
}

fn quoted_content(source: &str, quote: char) -> QuotedResult {
    if !source.starts_with(quote) {
        return QuotedResult::NotQuoted;
    }
    let mut decoded = String::new();
    let mut escaped = false;
    for (offset, character) in source[quote.len_utf8()..].char_indices() {
        if escaped {
            match character {
                'n' => decoded.push('\n'),
                'r' => decoded.push('\r'),
                't' => decoded.push('\t'),
                '\\' => decoded.push('\\'),
                other if other == quote => decoded.push(other),
                other => {
                    decoded.push('\\');
                    decoded.push(other);
                }
            }
            escaped = false;
            continue;
        }
        if character == '\\' {
            escaped = true;
            continue;
        }
        if character == quote {
            return QuotedResult::Complete {
                decoded,
                consumed: quote.len_utf8() + offset + character.len_utf8(),
            };
        }
        decoded.push(character);
    }
    QuotedResult::Unterminated
}

fn leading_keyword(source: &str) -> (&str, &str) {
    let end = source.find(char::is_whitespace).unwrap_or(source.len());
    (&source[..end], &source[end..])
}

fn is_full_line_comment(source: &str) -> bool {
    source.starts_with("//") || source.starts_with('#') || source.starts_with(';')
}

fn inline_comment_start(source: &str) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut quote = None;
    let mut escaped = false;
    let mut index = 0usize;
    while index + 1 < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
            index += 1;
            continue;
        }
        if quote.is_some() && byte == b'\\' {
            escaped = true;
            index += 1;
            continue;
        }
        if quote == Some(byte) {
            quote = None;
            index += 1;
            continue;
        }
        if quote.is_none() && matches!(byte, b'"' | b'\'') {
            quote = Some(byte);
            index += 1;
            continue;
        }
        if quote.is_none()
            && byte == b'/'
            && bytes[index + 1] == b'/'
            && index
                .checked_sub(1)
                .is_none_or(|previous| bytes[previous].is_ascii_whitespace())
        {
            return Some(index);
        }
        index += 1;
    }
    None
}

const fn absolute_offset(line_start: usize, indentation: usize, relative: usize) -> usize {
    line_start + indentation + relative
}

fn section_has_element(section: &Section, id: &str) -> bool {
    match section {
        Section::Window(section) => section
            .controls
            .iter()
            .any(|element| element.id.as_deref() == Some(id)),
        Section::Menu(section) => section
            .entries
            .iter()
            .any(|element| element.id.as_deref() == Some(id)),
        Section::MacroSet(section) => section
            .macros
            .iter()
            .any(|element| element.id.as_deref() == Some(id)),
    }
}

fn section_has_elements(section: &Section) -> bool {
    match section {
        Section::Window(section) => !section.controls.is_empty(),
        Section::Menu(section) => !section.entries.is_empty(),
        Section::MacroSet(section) => !section.macros.is_empty(),
    }
}

fn last_properties(section: &Section) -> Option<&[Property]> {
    match section {
        Section::Window(section) => section
            .controls
            .last()
            .map(|element| element.properties.as_slice()),
        Section::Menu(section) => section
            .entries
            .last()
            .map(|element| element.properties.as_slice()),
        Section::MacroSet(section) => section
            .macros
            .last()
            .map(|element| element.properties.as_slice()),
    }
}

fn last_properties_mut(section: &mut Section) -> Option<&mut Vec<Property>> {
    match section {
        Section::Window(section) => section
            .controls
            .last_mut()
            .map(|element| &mut element.properties),
        Section::Menu(section) => section
            .entries
            .last_mut()
            .map(|element| &mut element.properties),
        Section::MacroSet(section) => section
            .macros
            .last_mut()
            .map(|element| &mut element.properties),
    }
}

fn extend_section_span(section: &mut Section, end: usize) {
    match section {
        Section::Window(section) => section.span.end = end,
        Section::Menu(section) => section.span.end = end,
        Section::MacroSet(section) => section.span.end = end,
    }
}

fn extend_last_element_span(section: &mut Section, end: usize) {
    match section {
        Section::Window(section) => {
            section
                .controls
                .last_mut()
                .expect("caller checked element existence")
                .span
                .end = end;
        }
        Section::Menu(section) => {
            section
                .entries
                .last_mut()
                .expect("caller checked element existence")
                .span
                .end = end;
        }
        Section::MacroSet(section) => {
            section
                .macros
                .last_mut()
                .expect("caller checked element existence")
                .span
                .end = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ControlTree, ControlTreeDiagnosticKind, ControlType, DiagnosticKind, DiagnosticSeverity,
        Section, ValueKind, parse,
    };

    #[test]
    fn parses_synthetic_monk_compatibility_shapes() {
        let source = include_str!("../fixtures/compatibility.dmf");
        let document = parse(source);

        assert!(document.diagnostics.is_empty());
        assert_eq!(document.sections.len(), 3);
        assert_eq!(document.comments.len(), 1);
        let Section::MacroSet(macros) = &document.sections[0] else {
            panic!("first section should be a macro set");
        };
        let Section::Menu(menu) = &document.sections[1] else {
            panic!("second section should be a menu");
        };
        let Section::Window(window) = &document.sections[2] else {
            panic!("third section should be a window");
        };
        assert_eq!(macros.macros.len(), 1);
        assert_eq!(menu.entries.len(), 2);
        assert_eq!(menu.entries[0].id, None);
        assert_eq!(window.controls.len(), 2);
        let icon = window.controls[1]
            .properties
            .iter()
            .find(|property| property.key == "icon")
            .expect("resource property should exist");
        assert_eq!(icon.value.kind, ValueKind::Resource);
        assert_eq!(icon.value.decoded, "icons\\ui\\app.png");
        let status = window.controls[1]
            .properties
            .iter()
            .find(|property| property.key == "on-status")
            .expect("quoted command should exist");
        assert_eq!(status.value.kind, ValueKind::Quoted);
        assert!(status.value.decoded.contains("\"status.text="));
        assert_eq!(document.source_len, source.len());
    }

    #[test]
    fn retains_property_order_raw_values_and_byte_spans() {
        let source = "window \"w\"\r\n\telem \"map\"\r\n\t\ttype = MAP\r\n\t\tsize = 640x480\r\n";
        let document = parse(source);
        let Section::Window(window) = &document.sections[0] else {
            panic!("section should be a window");
        };
        let properties = &window.controls[0].properties;

        assert_eq!(properties[0].key, "type");
        assert_eq!(properties[0].value.raw, "MAP");
        assert_eq!(properties[1].key, "size");
        assert_eq!(properties[1].value.raw, "640x480");
        for property in properties {
            assert_eq!(
                &source[property.key_span.start..property.key_span.end],
                property.key
            );
            assert_eq!(
                &source[property.value.span.start..property.value.span.end],
                property.value.raw
            );
        }
    }

    #[test]
    fn retains_comments_without_treating_colors_as_comments() {
        let source = "# full-line comment\nwindow \"w\"\n\telem \"label\"\n\t\ttext-color = #ffffff // trailing comment\n";
        let document = parse(source);
        let Section::Window(window) = &document.sections[0] else {
            panic!("section should be a window");
        };

        assert_eq!(document.comments[0].raw, "# full-line comment");
        assert_eq!(document.comments[1].raw, "// trailing comment");
        assert_eq!(window.controls[0].properties[0].value.raw, "#ffffff");
        assert!(document.diagnostics.is_empty());
    }

    #[test]
    fn reports_and_recovers_from_actionable_errors() {
        let source = "elem \"orphan\"\nwindow missing-quotes\nwindow \"w\"\n\ttype = MAIN\n\telem \"c\"\n\t\tname = \"unterminated\n\t\ttype = MAP\n\t\ttype = BROWSER\nunknown syntax\n";
        let document = parse(source);
        let kinds: Vec<_> = document
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.kind)
            .collect();

        assert!(kinds.contains(&DiagnosticKind::ElementOutsideSection));
        assert!(kinds.contains(&DiagnosticKind::MalformedHeader));
        assert!(kinds.contains(&DiagnosticKind::PropertyOutsideElement));
        assert!(kinds.contains(&DiagnosticKind::UnterminatedQuote));
        assert!(kinds.contains(&DiagnosticKind::DuplicateProperty));
        assert!(kinds.contains(&DiagnosticKind::UnknownStatement));
        assert!(document.diagnostics.iter().all(|diagnostic| {
            diagnostic.span.end <= source.len() && !diagnostic.message.is_empty()
        }));
        assert!(document.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == DiagnosticKind::DuplicateProperty
                && diagnostic.severity == DiagnosticSeverity::Warning
        }));
    }

    #[test]
    fn lowers_addressable_controls_without_discarding_anonymous_elements() {
        let document =
            parse("window \"main\"\n\telem \"map\"\n\t\ttype = map\n\telem\n\t\ttype = LABEL\n");
        let tree = ControlTree::from_document(&document);

        assert!(tree.diagnostics.is_empty());
        assert_eq!(tree.windows.len(), 1);
        assert_eq!(tree.windows[0].controls.len(), 2);
        assert_eq!(tree.windows[0].controls[0].control_type, ControlType::Map);
        assert_eq!(tree.windows[0].controls[1].control_type, ControlType::Label);
        assert_eq!(
            tree.control("main", "map").map(|node| node.id.as_deref()),
            Some(Some("map"))
        );
        assert!(tree.control("main", "missing").is_none());
    }

    #[test]
    fn reports_duplicate_control_identifiers_deterministically() {
        let document = parse(
            "window \"main\"\n\telem \"map\"\n\t\ttype = MAP\n\telem \"map\"\n\t\ttype = BROWSER\n",
        );
        let tree = ControlTree::from_document(&document);

        assert_eq!(tree.diagnostics.len(), 1);
        assert_eq!(
            tree.diagnostics[0].kind,
            ControlTreeDiagnosticKind::DuplicateControlId
        );
        assert_eq!(
            tree.control("main", "map").map(|node| node.control_type),
            Some(ControlType::Map)
        );
    }

    #[test]
    fn ui_state_overrides_skin_properties_and_parses_quoted_assignments() {
        let document = parse(
            "window \"main\"\n\telem \"chat\"\n\t\ttype = OUTPUT\n\t\tis-visible = true\n\t\ttext = \"ready\"\n",
        );
        let mut state = super::UiState::new(ControlTree::from_document(&document));

        assert_eq!(state.winget("main.chat", "text"), Ok("ready".to_owned()));
        state
            .winset("main.chat", "text=\"hello; world\"; is-visible=false")
            .expect("the control exists and the assignments are valid");

        assert_eq!(state.winget("chat", "text"), Ok("hello; world".to_owned()));
        assert_eq!(state.winget("chat", "is-visible"), Ok("false".to_owned()));
        let all = state.winget_all("chat").unwrap();
        assert_eq!(all["text"], "hello; world");
        assert_eq!(all["is-visible"], "false");
        assert_eq!(all["type"], "OUTPUT");
        assert!(state.winexists("main.chat"));
        assert!(!state.winexists("main.missing"));
        assert_eq!(state.winexists_type("main.chat"), "OUTPUT");
        assert_eq!(state.winexists_type("main.missing"), "");
    }

    #[test]
    fn monkestation_right_click_macro_elements_are_winset_targets() {
        let document = parse(
            "macro \"default\"\n\telem \"Shift\"\n\t\tname = \"SHIFT\"\n\t\tcommand = \".winset :map.right-click=false\"\n\telem \"ShiftUp\"\n\t\tname = \"SHIFT+UP\"\n\t\tcommand = \".winset :map.right-click=true\"\nwindow \"mapwindow\"\n\telem \"map\"\n\t\ttype = MAP\n\t\tright-click = true\n",
        );
        let tree = ControlTree::from_document(&document);
        assert_eq!(tree.auxiliary.len(), 1);
        let mut state = super::UiState::new(tree);

        for (target, parameters) in [
            ("mapwindow.map", "right-click=false"),
            ("ShiftUp", "command=\".winset :map.right-click=false\""),
            ("Shift", "command=\".winset :map.right-click=true\""),
            ("ShiftUp", "command=\".winset :map.right-click=true\""),
            ("Shift", "command=\".winset :map.right-click=false\""),
        ] {
            state
                .winset(target, parameters)
                .unwrap_or_else(|error| panic!("{target} must resolve like BYOND: {error:?}"));
        }
        assert_eq!(
            state.winget("ShiftUp", "command"),
            Ok(".winset :map.right-click=true".to_owned())
        );
        assert_eq!(
            state.winget("Shift", "command"),
            Ok(".winset :map.right-click=false".to_owned())
        );
    }

    #[test]
    fn byond_type_selectors_resolve_default_controls_case_insensitively() {
        let document = parse(
            "window \"secondary\"\n\telem \"other_input\"\n\t\ttype = INPUT\nwindow \"main\"\n\telem \"input\"\n\t\ttype = INPUT\n\t\tis-default = true\n\telem \"map\"\n\t\ttype = MAP\n\t\tis-default = true\n",
        );
        let mut state = super::UiState::new(ControlTree::from_document(&document));

        state.winset(":map", "right-click=false").unwrap();
        assert_eq!(state.winget(":MAP", "right-click"), Ok("false".to_owned()));
        assert_eq!(state.winget(":Input", "type"), Ok("INPUT".to_owned()));
        assert_eq!(state.winget(":input", "is-default"), Ok("true".to_owned()));
    }

    #[test]
    fn winset_parent_creates_dynamic_macro_element() {
        let document = parse(
            "macro \"default\"\n\telem \"North\"\n\t\tname = \"W\"\n\t\tcommand = \".north\"\n",
        );
        let mut state = super::UiState::new(ControlTree::from_document(&document));

        state
            .winset("default-", "parent=default;name=T;command=.tgui-say say")
            .expect("parent=default creates a runtime macro element");

        assert!(state.winexists("default-"));
        assert_eq!(state.winget("default-", "parent"), Ok("default".to_owned()));
        assert_eq!(state.winget("default-", "name"), Ok("T".to_owned()));
        assert_eq!(
            state.winget("default.default-", "command"),
            Ok(".tgui-say say".to_owned())
        );
        assert_eq!(
            state.winget("default.*", "command"),
            Ok(concat!(
                "default.North.command=.north;",
                "default.default-.command=.tgui-say say"
            )
            .to_owned())
        );
    }

    #[test]
    fn ui_state_clones_runtime_overrides_without_changing_skin_defaults() {
        let document = parse(
            "window \"main\"\n\telem \"source\"\n\t\ttype = LABEL\n\t\ttext = \"skin\"\n\telem \"target\"\n\t\ttype = LABEL\n\t\ttext = \"target skin\"\n",
        );
        let mut state = super::UiState::new(ControlTree::from_document(&document));
        state
            .winset("main.source", "text=runtime")
            .expect("source control should exist");
        state
            .winclone("main.source", "main.target")
            .expect("both controls should exist");

        assert_eq!(
            state.winget("main.target", "text"),
            Ok("runtime".to_owned())
        );
        assert_eq!(state.winget("main.source", "type"), Ok("LABEL".to_owned()));
    }

    #[test]
    fn winclone_creates_a_runtime_window_and_accepts_tgui_children() {
        let document = parse(
            "window \"popupwindow\"\n\telem \"popupwindow\"\n\t\ttype = MAIN\n\t\tsize = 120x120\n",
        );
        let mut state = super::UiState::new(ControlTree::from_document(&document));

        state
            .winclone("popupwindow", "tgui-window-1")
            .expect("BYOND winclone creates a new window id");
        state
            .winset("tgui-window-1", "title=Character Setup;is-visible=true")
            .expect("the cloned main control is addressable by its new window id");
        state
            .winset(
                "tgui-window-1.browser",
                "parent=tgui-window-1;type=BROWSER;pos=0,0;size=640x480",
            )
            .expect("runtime children can be parented to the cloned window");

        assert!(state.winexists("tgui-window-1"));
        assert!(state.winexists("tgui-window-1.browser"));
        assert_eq!(
            state.winget("tgui-window-1", "size"),
            Ok("120x120".to_owned())
        );
        assert_eq!(
            state.winget("tgui-window-1", "title"),
            Ok("Character Setup".to_owned())
        );
        assert_eq!(
            state.winget("tgui-window-1.browser", "type"),
            Ok("BROWSER".to_owned())
        );
    }

    #[test]
    fn winget_semicolon_control_list_returns_byond_keyed_params() {
        let document = parse(
            "window \"main\"\n\telem \"split\"\n\t\ttype = CHILD\n\t\tsize = 1920x1030\nwindow \"mapwindow\"\n\telem \"mapwindow\"\n\t\ttype = MAIN\n\t\tsize = 1248x1030\n",
        );
        let state = super::UiState::new(ControlTree::from_document(&document));
        assert_eq!(
            state.winget("main.split;mapwindow", "size"),
            Ok("main.split.size=1920x1030;mapwindow.size=1248x1030".to_owned())
        );
    }

    #[test]
    fn ui_commands_have_transport_neutral_deterministic_replies() {
        let document = parse("window \"main\"\n\telem \"status\"\n\t\ttype = LABEL\n");
        let mut state = super::UiState::new(ControlTree::from_document(&document));

        assert_eq!(
            state.apply(super::UiCommand::WinSet {
                control: "main.status".to_owned(),
                parameters: "text=connected".to_owned(),
            }),
            Ok(super::UiCommandReply::Applied)
        );
        assert_eq!(
            state.apply(super::UiCommand::WinGet {
                control: "main.status".to_owned(),
                property: "text".to_owned(),
            }),
            Ok(super::UiCommandReply::Property("connected".to_owned()))
        );
        assert_eq!(
            state.apply(super::UiCommand::WinExists {
                control: "main.missing".to_owned(),
            }),
            Ok(super::UiCommandReply::Exists(false))
        );
    }

    #[test]
    fn client_sessions_isolate_ui_state_and_preserve_event_order() {
        let document = parse("window \"main\"\n\telem \"status\"\n\t\ttype = LABEL\n");
        let tree = ControlTree::from_document(&document);
        let mut first = super::ClientSession::new(tree.clone());
        let second = super::ClientSession::new(tree);

        first
            .apply_command(super::UiCommand::WinSet {
                control: "main.status".to_owned(),
                parameters: "text=first".to_owned(),
            })
            .expect("the control exists");
        first.push_event(super::UiEvent::Key {
            key: "W".to_owned(),
            pressed: true,
        });
        first.push_event(super::UiEvent::Command {
            command: ".say hello".to_owned(),
        });

        assert_eq!(
            first.ui().winget("main.status", "text"),
            Ok("first".to_owned())
        );
        assert_eq!(second.ui().winget("main.status", "text"), Ok(String::new()));
        assert_eq!(
            first.take_events(),
            vec![
                super::UiEvent::Key {
                    key: "W".to_owned(),
                    pressed: true,
                },
                super::UiEvent::Command {
                    command: ".say hello".to_owned(),
                },
            ]
        );
        assert!(first.take_events().is_empty());
    }

    #[test]
    fn browser_bridge_requires_the_configured_origin_and_a_browser_control() {
        let document = parse(
            "window \"main\"\n\telem \"browser\"\n\t\ttype = BROWSER\n\telem \"label\"\n\t\ttype = LABEL\n",
        );
        let mut session = super::ClientSession::new(ControlTree::from_document(&document));
        let policy = super::BrowserPolicy {
            origin: "https://tgui.example".to_owned(),
        };

        assert_eq!(
            session.handle_browser_message(
                "https://tgui.example",
                "main.browser",
                &policy,
                super::BrowserBridgeRequest::Topic {
                    topic: "?src=ui;action=refresh".to_owned(),
                },
            ),
            Ok(super::BrowserBridgeReply::Accepted)
        );
        assert_eq!(
            session.take_events(),
            vec![super::UiEvent::BrowserTopic {
                control: "main.browser".to_owned(),
                topic: "?src=ui;action=refresh".to_owned(),
            }]
        );
        assert!(matches!(
            session.handle_browser_message(
                "https://evil.example",
                "main.browser",
                &policy,
                super::BrowserBridgeRequest::Command {
                    command: ".quit".to_owned(),
                },
            ),
            Err(super::BrowserBridgeError::OriginDenied { .. })
        ));
        assert!(matches!(
            session.handle_browser_message(
                "https://tgui.example",
                "main.label",
                &policy,
                super::BrowserBridgeRequest::Command {
                    command: ".quit".to_owned(),
                },
            ),
            Err(super::BrowserBridgeError::NotBrowserControl(_))
        ));
    }
}

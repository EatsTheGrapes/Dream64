# User interface architecture

## Guiding decision

Match BYOND's observable interface behavior, not its visual age or internal
widget implementation. Existing projects should keep their `.dmf` files and
browser UI bundles. New projects can opt into a modern declarative format that
compiles to the same internal control tree.

## Player client

The client is a native shell with independently composited surfaces:

```text
DMF / modern UI document
          |
          v
typed control tree <-> UI command protocol <-> game connection
   |            |
   |            `-> Chromium webview + window.Byond bridge
   `-> GPU map, HUD, text, filters, particles, and animation
```

The first compatibility slice covers the controls used by Monkestation's
`interface/skin.dmf`:

- main windows and panes;
- child/split layouts and anchors;
- map surfaces;
- browser surfaces for TGUI, chat, stat panels, and popups;
- input, output, label, and button controls;
- menus and macro sets;
- focus, visibility, size, position, saved properties, and commands.

The compatibility API includes:

- `winset`, `winget`, `winclone`, `winshow`, and `winexists`;
- `browse`, `browse_rsc`, and `output` routing;
- default map/input/output/browser selection;
- `.winset`, `.output`, `.reconnect`, `.quit`, screenshot, and game commands;
- mouse, keyboard, focus, drag/drop, and map-coordinate event translation;
- browser storage and the JavaScript `Byond` object used by TGUI.

Web content runs with explicit origin, capability, and resource policies. Game
HTML does not receive arbitrary native or filesystem access through the bridge.

## Development application

The first-party development UI should feel familiar to Dream Maker users while
being built from reusable services:

- project and object/type trees;
- DM editor with semantic navigation, completion, rename, and diagnostics;
- DMM map editor with GPU previews and validation;
- DMI icon/state and animation editor;
- DMF visual editor backed by the typed control tree;
- build output, tests, debugger, profiler, replay timeline, and runtime inspector.

Compiler intelligence is exposed over LSP and debugging over DAP so the native
application, VS Code, and other editors share the same behavior. The map, icon,
and skin editors use stable engine protocols instead of reaching into compiler
or runtime memory.

## Server administration

Dream Daemon compatibility remains headless-first. An authenticated optional
dashboard consumes metrics, logs, profiler samples, replay controls, and safe
administration commands over a versioned API. Hosting a server never requires a
desktop session or GUI process.

## Delivery order

1. Parse DMF into a lossless syntax tree and typed control tree.
2. Implement UI protocol state transitions and headless conformance tests.
3. Build the minimal player shell: window, map surface, browser surface, input,
	and output.
4. Load Monkestation's skin and unmodified TGUI bundle.
5. Add remaining BYOND controls and interaction parity.
6. Build the development application on the stabilized compiler/debugger APIs.
7. Add the optional server dashboard and the modern opt-in UI document format.

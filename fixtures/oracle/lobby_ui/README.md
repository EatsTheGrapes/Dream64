# Lobby UI oracle

This BYOND 516.1680 probe covers the initial DreamSeeker lobby contract. The
DMF declares one default main window containing map, browser, and output
controls. A real DreamSeeker connection produced this ordered server log:

```text
client.New begin key=Guest-... mob=Guest-...
winset returned
output calls returned
browse_rsc returned
browse returned
client still connected=1 mob=Guest-...
```

The assigned mob exists before `/client/New()` begins. The map control needs no
explicit server command: DreamSeeker renders the client's normal eye/mob map
stream. UI mutations are then ordered `winset`, control-targeted `output`,
resource delivery, and `browse`; all return without waiting for a UI response.

Run manually with DreamDaemon on a local port, then connect DreamSeeker to its
`byond://127.0.0.1:<port>` URL. `lobby_ui.out` is written by DM and is ignored as
a per-run result.

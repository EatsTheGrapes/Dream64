# TGS integration contract

Dream64 integrates with tgstation-server as a native third engine. It must not
pretend to be BYOND or OpenDream: those engine types select different artifact
extensions, launch arguments, health transports, and shutdown behavior.

## Dream64-side process contract

The packaged engine exposes three independently replaceable programs:

- `dream64-compiler`: accepts a `.dme`, keeps incremental state in its private
  cache, writes one self-contained runtime image (`.d64`), and exits nonzero
  without installing an invalid deployment.
- `dream64-server`: consumes a compiler-produced deployment and never compiles.
  TGS supplies the game port, private topic port, watchdog token, world params,
  deployment identity, random seed, and log destination.
- `dream64-client`: connects to a Dream64 server and is not installed or
  launched by the TGS host watchdog.

The distinct `dream64-compiler` and `dream64-server` executables enforce the
compiler/runtime ownership rule. The server rejects the legacy `compile`
command instead of silently doing compiler work during startup. Production
launches pass the `.d64` directly, so neither the `.dme` source tree nor the
compiler's private incremental cache is part of the runtime deployment.

The compiler validates the DME include graph on every invocation. An unchanged
project reuses the committed `.d64` without rewriting it. On a change, the
parsed-syntax cache is keyed per source file, so unchanged files are restored
while changed files are reparsed. Semantic dependency and procedure-lowering
incrementality are the next cache tier; until that tier is complete, those
stages conservatively rebuild after any source change.

The server contract still needs stable command-line forms for these dynamic TGS
values:

```text
dream64-server run <project.d64> \
  --port <game-port> \
  --topic-port <private-port> \
  --watchdog-token <secret> \
  --world-params <encoded-params> \
  --deployment <git-or-testmerge-identity> \
  --random-seed <nonzero-u64>
```

Secrets must not be printed in process logs. The topic listener is loopback-only
and requires the watchdog token on every request.

## Required TGS patch

Against current TGS `dev`, the native patch has these bounded changes:

1. Add `Dream64 = 2` to
   `Tgstation.Server.Api/Models/EngineType.cs` and the corresponding API rights.
2. Add `.d64` to the engine switch in
   `Components/Deployment/DmbProviderBase.cs`.
3. Implement `Dream64Installer` and `Dream64Installation` behind the existing
   `IEngineInstaller` / `IEngineInstallation` interfaces.
4. Register the installer in the engine dictionary in
   `Tgstation.Server.Host/Core/Application.cs`.
5. Add install, compile-argument, launch-argument, shutdown, deployment-provider,
   persistence, and watchdog tests alongside the BYOND/OpenDream cases.

`Dream64Installation` supplies:

- `ServerExePath` and `CompilerExePath` from the uploaded Dream64 release;
- `FormatCompilerArguments` -> `<deployment-project.dme>`;
- `FormatServerArguments` -> the stable `run` command above;
- standard-output and file-logging capabilities;
- graceful shutdown through the authenticated loopback topic endpoint;
- `server.env` and `compiler.env` through TGS's existing environment loader.

TGS already lowers deployment process priority and can raise live-server
priority. Dream64 must not duplicate that policy when TGS owns the process.

## DMAPI and readiness

Monkestation already calls the TGS DMAPI hooks. Dream64 must provide host-backed
behavior for at least:

- `TgsNew`
- `TgsInitializationComplete`
- `TgsReboot`
- `TgsEndProcess`
- the authenticated `TGS_TOPIC` request path

`TgsInitializationComplete` is the authoritative transition from initializing
to ready. TGS must not advertise the deployment or allow player readiness before
that transition.

## Hot deployment extension

Ordinary TGS stages a compiled deployment and starts it on reboot. Dream64 adds
an optional blue/green stage:

1. Compile the changed Git/test-merge deployment at low priority.
2. Launch it on a private standby control address with an explicit deployment
   identity and random seed.
3. Run normal map selection and subsystem initialization while the old round
   remains authoritative.
4. On `TgsInitializationComplete`, mark the standby hot.
5. On `TgsReboot`, stop accepting new work on the old runtime, activate the
   matching standby, release the public port, and let the standby acquire it.
6. Fall back to an ordinary cold launch if no valid matching standby exists.

The active and standby deployments must never share mutable game directories.
Every handoff validates engine semantics, project fingerprint, deployment ID,
random seed, runtime artifact, and ready-world identity.

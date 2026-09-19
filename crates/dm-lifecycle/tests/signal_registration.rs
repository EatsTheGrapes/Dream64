//! Monkestation's signal registration (`code/datums/signals.dm`) run verbatim
//! through the full runtime.
//!
//! `RegisterSignal` must compile to exactly the production bytecode shape: the
//! VM recognises that shape and runs first registrations natively
//! (`try_run_register_signal_fast_path`). So the macros and helpers it touches
//! -- `QDELETED`, `stack_trace`, `log_signal`, `RegisterSignals` -- are
//! reproduced as the codebase defines them, and these tests exercise the native
//! path alongside the interpreted `UnregisterSignal` and `_SendSignal`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use dm_compiler::{Compilation, CompilerDatabase};
use dm_lifecycle::{LifecycleIndex, build_initialization_plan, execute_initialization_plan};
use dm_map::parse;
use dm_runtime::RuntimeImage;
use dm_semantics::ProcedureRegistry;
use dm_value::Value;
use dm_world::{allocate_world, build_plan};

static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

struct TestProject {
    root: PathBuf,
}

impl TestProject {
    fn compile(types: &str) -> (Self, Compilation) {
        let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "dream64-signal-registration-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("test project directory should be created");
        fs::write(root.join("world.dme"), "#include \"types.dm\"\n")
            .expect("environment should be written");
        fs::write(root.join("types.dm"), types).expect("types should be written");
        let compilation = CompilerDatabase::new()
            .compile(root.join("world.dme"))
            .expect("fixture should compile");
        (Self { root }, compilation)
    }
}

impl Drop for TestProject {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// `code/datums/signals.dm` verbatim, with the defines and helpers it depends
/// on reproduced as Monkestation declares them. Only the helpers' *bodies* are
/// stubbed: a trace is recorded into `global.result` instead of CRASHing.
const SIGNALS_DM: &str = r#"
#define NONE 0
#define QDELING(X) (X.gc_destroyed)
#define QDELETED(X) (isnull(X) || QDELING(X))
#define stack_trace(message) _stack_trace(message, __FILE__, __LINE__)
#define SEND_SIGNAL(target, sigtype, arguments...) ( !target._listen_lookup?[sigtype] ? NONE : target._SendSignal(sigtype, list(target, ##arguments)) )

var/global/result = ""

/proc/_stack_trace(message, file, line)
	global.result += "|trace:[message]"
/proc/log_signal(text, list/data)
	return

/datum
	var/gc_destroyed
	var/list/_signal_procs
	var/list/_listen_lookup

/datum/proc/RegisterSignal(datum/target, signal_type, proctype, override = FALSE)
	if(QDELETED(src) || QDELETED(target))
		return

	if (islist(signal_type))
		var/static/list/known_failures = list()
		var/list/signal_type_list = signal_type
		var/message = "([target.type]) is registering [signal_type_list.Join(", ")] as a list, the older method. Change it to RegisterSignals."

		if (!(message in known_failures))
			known_failures[message] = TRUE
			stack_trace("[target] [message]")

		RegisterSignals(target, signal_type, proctype, override)
		return

	var/list/procs = (_signal_procs ||= list())
	var/list/target_procs = (procs[target] ||= list())
	var/list/lookup = (target._listen_lookup ||= list())

	var/exists = target_procs[signal_type]
	target_procs[signal_type] = proctype

	if(exists)
		if(!override)
			var/override_message = "[signal_type] overridden. Use override = TRUE to suppress this warning.\nTarget: [target] ([target.type]) Proc: [proctype]"
			log_signal(override_message)
			stack_trace(override_message)
		return

	var/list/looked_up = lookup[signal_type]

	if(isnull(looked_up)) // Nothing has registered here yet
		lookup[signal_type] = src
	else if(!islist(looked_up)) // One other thing registered here
		lookup[signal_type] = list(looked_up, src)
	else // Many other things have registered here
		looked_up += src

/datum/proc/RegisterSignals(datum/target, list/signal_types, proctype, override = FALSE)
	for (var/signal_type in signal_types)
		RegisterSignal(target, signal_type, proctype, override)

/datum/proc/UnregisterSignal(datum/target, sig_type_or_types)
	var/list/lookup = target?._listen_lookup
	if(!_signal_procs || !_signal_procs[target] || !lookup)
		return
	if(!islist(sig_type_or_types))
		sig_type_or_types = list(sig_type_or_types)
	for(var/sig in sig_type_or_types)
		if(!_signal_procs[target][sig])
			if(!istext(sig))
				stack_trace("We're unregistering with something that isn't a valid signal \[[sig]\], you fucked up")
			continue
		switch(length(lookup[sig]))
			if(2)
				lookup[sig] = (lookup[sig]-src)[1]
			if(1)
				stack_trace("[target] ([target.type]) somehow has single length list inside _listen_lookup")
				if(src in lookup[sig])
					lookup -= sig
					if(!length(lookup))
						target._listen_lookup = null
						break
			if(0)
				if(lookup[sig] != src)
					continue
				lookup -= sig
				if(!length(lookup))
					target._listen_lookup = null
					break
			else
				lookup[sig] -= src

	_signal_procs[target] -= sig_type_or_types
	if(!_signal_procs[target].len)
		_signal_procs -= target

/datum/proc/_SendSignal(sigtype, list/arguments)
	var/target = _listen_lookup[sigtype]
	if(!length(target))
		var/datum/listening_datum = target
		return NONE | call(listening_datum, listening_datum._signal_procs[src][sigtype])(arglist(arguments))
	. = NONE
	var/list/queued_calls = list()
	for(var/i in 1 to length(target))
		var/datum/listening_datum = target[i]
		queued_calls.Add(listening_datum, listening_datum._signal_procs[src][sigtype])
	for(var/i in 1 to length(queued_calls) step 2)
		. |= call(queued_calls[i], queued_calls[i + 1])(arglist(arguments))

/turf/boot
/area/boot
"#;

const MAP: &str = "\"a\" = (/turf/boot, /area/boot)\n(1,1,1) = {\"\na\n\"}\n";

/// Runs `scenario` (which must define `/world/New()`) against the verbatim
/// signal procs and returns the final value of `global.result`, or the runtime
/// failure that escaped `/world/New()`.
fn run(scenario: &str) -> Result<Value, String> {
    let (_project, compilation) = TestProject::compile(&format!("{SIGNALS_DM}\n{scenario}"));
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let world = build_plan(&parse(MAP).expect("map should parse"), &compilation);
    let plan = build_initialization_plan(&runtime, &index, &world, "boot.dmm");
    let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
    execute_initialization_plan(
        &compilation,
        &procedures,
        &index,
        &plan,
        &allocation,
        &mut runtime,
    )
    .map_err(|error| error.to_string())?;
    Ok(runtime
        .variables()
        .iter()
        .find(|variable| variable.path.ends_with("/result"))
        .expect("result global should exist")
        .value
        .clone())
}

/// The `_SendSignal` desync. The same decal element attached twice to one turf
/// re-registers with `override = TRUE`, which DM treats as "update the callback,
/// leave `_listen_lookup` alone". The native first-registration path used to
/// fall through and append the element again (`list(E, E)`); `UnregisterSignal`
/// then removed one copy, leaving the turf naming a listener with no
/// `_signal_procs[turf]` entry, and the next SEND_SIGNAL called a null proc.
#[test]
fn the_same_decal_attached_twice_detaches_without_a_dangling_listener() {
    let result = run(r#"
/datum/decal_like/proc/apply_overlay(datum/source)
	return 4
/datum/turf_like

/world/New()
	var/datum/turf_like/T = new
	var/datum/decal_like/E = new
	E.RegisterSignal(T, "update_overlays", /datum/decal_like/proc/apply_overlay, TRUE)
	E.RegisterSignal(T, "update_overlays", /datum/decal_like/proc/apply_overlay, TRUE)
	global.result += "shape=[islist(T._listen_lookup["update_overlays"]) ? "list" : "scalar"]"
	E.UnregisterSignal(T, "update_overlays")
	global.result += " lookup=[isnull(T._listen_lookup) ? "null" : "[length(T._listen_lookup)]"]"
	global.result += " after=[SEND_SIGNAL(T, "update_overlays")]"
"#);
    assert_eq!(result, Ok(Value::text("shape=scalar lookup=null after=0")));
}

/// A non-override re-registration warns (once, via bytecode) and must likewise
/// leave the listener in the lookup exactly once.
#[test]
fn a_warned_re_registration_still_leaves_one_listener() {
    let result = run(r#"
/datum/listener_like/proc/first(datum/source)
	return 1
/datum/listener_like/proc/second(datum/source)
	return 2
/datum/turf_like

/world/New()
	var/datum/turf_like/T = new
	var/datum/listener_like/L = new
	L.RegisterSignal(T, "ping", /datum/listener_like/proc/first)
	L.RegisterSignal(T, "ping", /datum/listener_like/proc/second)
	global.result += "|shape=[islist(T._listen_lookup["ping"]) ? "list" : "scalar"] sent=[SEND_SIGNAL(T, "ping")]"
"#)
    .expect("a warned re-registration must not raise");
    let Value::Text(result) = result else {
        panic!("result should be text, got {result:?}");
    };
    assert!(
        result.contains("|trace:ping overridden."),
        "the override warning should fire once through bytecode: {result}"
    );
    assert!(
        result.ends_with("|shape=scalar sent=2"),
        "the callback is replaced and the listener kept once: {result}"
    );
}

#[test]
fn decal_attach_then_detach_leaves_both_halves_empty() {
    let result = run(r#"
/datum/decal_like/proc/apply_overlay(datum/source)
	return 4
/datum/decal_like/proc/other(datum/source)
	return 8
/datum/dcs_like/proc/rotate(datum/source)
	return 16
/datum/turf_like

/world/New()
	var/datum/turf_like/T = new
	var/datum/decal_like/E = new
	var/datum/dcs_like/S = new
	E.RegisterSignal(T, "update_overlays", /datum/decal_like/proc/apply_overlay, TRUE)
	E.RegisterSignal(T, "clean_act", /datum/decal_like/proc/other, TRUE)
	E.RegisterSignal(T, "examine", /datum/decal_like/proc/other, TRUE)
	S.RegisterSignal(T, "dir_change", /datum/dcs_like/proc/rotate, TRUE)
	E.RegisterSignal(T, "shuttle_move", /datum/decal_like/proc/other, TRUE)
	global.result += "before=[SEND_SIGNAL(T, "update_overlays")]"
	E.UnregisterSignal(T, list("dir_change", "clean_act", "examine", "update_overlays", "shuttle_move", "smoothed_icon", "decals_rotating"))
	global.result += " mid=[isnull(T._listen_lookup) ? "null" : "[length(T._listen_lookup)]:[T._listen_lookup["update_overlays"] ? "uo-present" : "uo-absent"]"]"
	S.UnregisterSignal(T, "dir_change")
	global.result += " end=[isnull(T._listen_lookup) ? "null" : "[length(T._listen_lookup)]"]"
	global.result += " procs=[isnull(E._signal_procs) ? "null" : (E._signal_procs[T] ? "has-T" : "no-T")]"
	global.result += " after=[SEND_SIGNAL(T, "update_overlays")]"
"#);
    assert_eq!(
        result,
        Ok(Value::text(
            "before=4 mid=1:uo-absent end=null procs=no-T after=0"
        ))
    );
}

/// `count` distinct decal elements on one turf, each detaching inside the
/// multi-listener QDELETING dispatch and re-sending UPDATE_OVERLAYS from there
/// -- the nested shape of the original crash trace.
fn nested_detach_scenario(count: usize) -> String {
    let mut attach = String::new();
    for index in 1..=count {
        attach.push_str(&format!(
            "\tvar/datum/decal_like/E{index} = new\n\
             \tE{index}.RegisterSignal(T, \"update_overlays\", /datum/decal_like/proc/apply_overlay, TRUE)\n\
             \tE{index}.RegisterSignal(T, \"examine\", /datum/decal_like/proc/other, TRUE)\n\
             \tE{index}.RegisterSignal(T, \"qdeleting\", /datum/decal_like/proc/OnTargetDelete, TRUE)\n"
        ));
    }
    format!(
        r#"
/datum/decal_like/proc/apply_overlay(datum/source)
	global.result += "|overlay"
	return 4
/datum/decal_like/proc/other(datum/source)
	return 8
/datum/decal_like/proc/OnTargetDelete(datum/source)
	Detach(source)
/datum/decal_like/proc/Detach(datum/source)
	UnregisterSignal(source, list("dir_change", "examine", "update_overlays", "smoothed_icon"))
	SEND_SIGNAL(source, "update_overlays")
	UnregisterSignal(source, "qdeleting")
/datum/turf_like

/world/New()
	var/datum/turf_like/T = new
{attach}	SEND_SIGNAL(T, "qdeleting")
	global.result += "|end=[{end}]"
"#,
        end = r#"isnull(T._listen_lookup) ? "null" : "[length(T._listen_lookup)]""#,
    )
}

#[test]
fn decals_detaching_inside_a_qdeleting_dispatch_stay_consistent() {
    // One, two and three listeners take the single, `if(2)` and `else`
    // branches of `UnregisterSignal` respectively.
    assert_eq!(
        run(&nested_detach_scenario(1)),
        Ok(Value::text("|end=null"))
    );
    assert_eq!(
        run(&nested_detach_scenario(2)),
        Ok(Value::text("|overlay|end=null"))
    );
    assert_eq!(
        run(&nested_detach_scenario(3)),
        Ok(Value::text("|overlay|overlay|overlay|end=null"))
    );
}

#[test]
fn a_shared_element_survives_detaching_from_targets_in_scattered_order() {
    // SSdcs shares one decal element across every turf carrying that decal, so
    // its `_signal_procs` is a large datum-keyed assoc list that loses entries
    // from the middle. After every removal, re-verify every remaining target.
    let result = run(r#"
/datum/decal_like/proc/apply_overlay(datum/source)
	return 4
/datum/decal_like/proc/other(datum/source)
	return 8
/datum/turf_like

/world/New()
	var/datum/decal_like/E = new
	var/list/attached = list()
	for(var/i in 1 to 120)
		var/datum/turf_like/T = new
		attached += T
		E.RegisterSignal(T, "update_overlays", /datum/decal_like/proc/apply_overlay, TRUE)
		E.RegisterSignal(T, "examine", /datum/decal_like/proc/other, TRUE)
	var/seed = 17
	var/removals = 0
	var/procs_missing = 0
	var/send_wrong = 0
	while(length(attached))
		seed = (seed * 1103 + 12345) % 65536
		var/index = (seed % length(attached)) + 1
		var/datum/turf_like/gone = attached[index]
		attached.Cut(index, index + 1)
		E.UnregisterSignal(gone, list("update_overlays", "examine"))
		removals++
		for(var/datum/turf_like/T as anything in attached)
			if(!E._signal_procs || !E._signal_procs[T] || !E._signal_procs[T]["update_overlays"])
				procs_missing++
			else if(SEND_SIGNAL(T, "update_overlays") != 4)
				send_wrong++
	global.result = "removals=[removals] procs_missing=[procs_missing] send_wrong=[send_wrong] leftover=[length(E._signal_procs)]"
"#);
    assert_eq!(
        result,
        Ok(Value::text(
            "removals=120 procs_missing=0 send_wrong=0 leftover=0"
        ))
    );
}

#[test]
fn list_primitives_the_signal_procs_depend_on_match_byond() {
    let result = run(r#"
/datum/holder
	var/list/held

/world/New()
	// for-in iterates a snapshot, so removing during iteration skips nothing.
	var/list/plain = list("a", "b", "c", "d", "e")
	var/seen_plain = ""
	for(var/x in plain)
		seen_plain += x
		plain -= x
	// The `_clear_signal_refs` shape: assoc list reached through a field.
	var/datum/holder/H = new
	H.held = list("a" = 1, "b" = 2, "c" = 3, "d" = 4, "e" = 5)
	var/seen_assoc = ""
	for(var/k in H.held)
		seen_assoc += k
		H.held -= k
	// `_signal_procs[target] -= _signal_procs[target]` via _clear_signal_refs.
	var/list/self = list("a" = 1, "b" = 2, "c" = 3, "d" = 4)
	self -= self
	// ChangeTurf's LAZYOR (|=) and .Copy() must carry associations.
	var/list/dest = list()
	dest |= list("x" = "X", "y" = "Y")
	var/list/copied = list("p" = "P", "q" = "Q").Copy()
	global.result = "plain=[seen_plain] assoc=[seen_assoc] self=[length(self)] or=[dest["x"]][dest["y"]] copy=[copied["p"]][copied["q"]]"
"#);
    assert_eq!(
        result,
        Ok(Value::text("plain=abcde assoc=abcde self=0 or=XY copy=PQ"))
    );
}

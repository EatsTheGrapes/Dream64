// BYOND behavioral oracle for shared turf overlay/underlay lists.
//
// Dream64 gives every station/space turf its own distinct heap `List` identity
// for `overlays`/`underlays`, even though ~1M turfs hold structurally identical
// content copied from a shared constant list (e.g. GLOB.fullbright_overlays).
// That identity explosion is what makes the interpreter's mark-sweep and
// `world.contents` iteration snapshot slow during the lighting phase.
//
// Before interning those identities we must pin exactly what BYOND 516.1680
// observes for the read-back, identity, mutation-isolation and reassignment
// behaviour of `atom.overlays`. These probes capture that reference.
//
// Compile with BYOND 516 Dream Maker, run the DMB with DreamDaemon:
//   dm.exe turf_overlays.dme
//   DreamDaemon.exe turf_overlays.dmb -trusted -close
//   type turf_overlays.out

var/const/ORACLE_OUTPUT = "turf_overlays.out"

// Stand-in for a shared "constant" overlay source list such as
// GLOB.fullbright_overlays[SSmapping.max_plane_offset].
var/list/SHARED_OVERLAYS = list()

/proc/oracle_emit(key, value)
	text2file("[key]=[value]\n", ORACLE_OUTPUT)

/world
	maxx = 5
	maxy = 5
	maxz = 1
	view = 3
	turf = /turf/floor

/turf/floor
/mob/probe
/obj/marker
	icon_state = "marker"

// Describes a list's observable content: length, then each element rendered
// as text in iteration order. `overlays` elements normalize to appearance
// state, so string rendering is the stable cross-engine projection.
/proc/describe(list/candidate)
	if(!islist(candidate))
		return "notlist"
	var/list/parts = list()
	for(var/entry in candidate)
		parts += "[entry]"
	return "len=[candidate.len]|[jointext(parts, ",")]"

/world/New()
	..()
	fdel(ORACLE_OUTPUT)

	SHARED_OVERLAYS += "alpha"
	SHARED_OVERLAYS += "beta"
	SHARED_OVERLAYS += "gamma"

	var/turf/floor/t1 = locate(1, 1, 1)
	var/turf/floor/t2 = locate(2, 1, 1)
	var/turf/floor/t3 = locate(3, 1, 1)

	// --- case 1: two turfs each append the same shared source ----------
	t1.overlays += SHARED_OVERLAYS
	t2.overlays += SHARED_OVERLAYS
	oracle_emit("t1_after_shared", describe(t1.overlays))
	oracle_emit("t2_after_shared", describe(t2.overlays))
	oracle_emit("islist_t1_overlays", islist(t1.overlays))

	// --- case 2: list `==` semantics (BYOND list == is identity) -------
	oracle_emit("t1_overlays_eq_self", t1.overlays == t1.overlays)
	oracle_emit("t1_overlays_eq_t2", t1.overlays == t2.overlays)
	oracle_emit("t1_overlays_eq_shared", t1.overlays == SHARED_OVERLAYS)

	// --- case 3: mutating one turf must not touch the sibling ----------
	t1.overlays += "delta"
	oracle_emit("t1_after_extra", describe(t1.overlays))
	oracle_emit("t2_after_t1_extra", describe(t2.overlays))
	oracle_emit("shared_after_t1_extra", describe(SHARED_OVERLAYS))

	// --- case 4: Cut() on one turf must not touch the sibling ---------
	t3.overlays += SHARED_OVERLAYS
	t3.overlays.Cut()
	oracle_emit("t3_after_cut", describe(t3.overlays))
	oracle_emit("t1_after_t3_cut", describe(t1.overlays))
	oracle_emit("shared_after_t3_cut", describe(SHARED_OVERLAYS))

	// --- case 5: re-appending the shared source stacks (dupes allowed) -
	t2.overlays += SHARED_OVERLAYS
	oracle_emit("t2_after_second_shared", describe(t2.overlays))
	oracle_emit("t1_after_t2_second_shared", describe(t1.overlays))

	// --- case 6: whole-field reassignment -------------------------------
	t1.overlays = list("solo")
	oracle_emit("t1_after_reassign", describe(t1.overlays))
	oracle_emit("t2_after_t1_reassign", describe(t2.overlays))

	// --- case 7: underlays behave the same -----------------------------
	var/turf/floor/u1 = locate(1, 2, 1)
	var/turf/floor/u2 = locate(2, 2, 1)
	u1.underlays += SHARED_OVERLAYS
	u2.underlays += SHARED_OVERLAYS
	u1.underlays += "u-extra"
	oracle_emit("u1_after_extra", describe(u1.underlays))
	oracle_emit("u2_after_u1_extra", describe(u2.underlays))

	// --- case 8: appended element identity (image objects) -------------
	var/list/shared_images = list()
	shared_images += new /obj/marker
	shared_images += new /obj/marker
	var/turf/floor/i1 = locate(4, 1, 1)
	var/turf/floor/i2 = locate(5, 1, 1)
	i1.overlays += shared_images
	i2.overlays += shared_images
	oracle_emit("i1_overlay_count", i1.overlays.len)
	oracle_emit("i2_overlay_count", i2.overlays.len)

	oracle_emit("oracle_complete", 1)
	shutdown()

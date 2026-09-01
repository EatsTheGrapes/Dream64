// BYOND behavioral oracle for step_to() / step_towards() / step().
//
// Dream64's step family performs a single greedy step toward the target and
// does not consult turf density. These probes pin the move sequences that the
// interpreter must keep reproducing after the destination-turf lookup was moved
// off the O(all datums) heap scan and onto the world geometry index.
//
// Scenarios:
//   open      - unobstructed diagonal approach across open floor
//   cardinal  - straight-line approach, target due east
//   min_range - step_to with Min=2 stops once inside the range ring
//   blocked   - a dense wall column sits between probe and target
//   pocket    - target sealed inside a one-tile wall pocket
//
// The "blocked"/"pocket" traces record what Dream64's greedy stepper does
// (walk into the obstruction's face and stall); BYOND's pathfinder would route
// around. They are captured so the divergence stays visible and any future
// pathfinding parity work has a reference.

var/const/ORACLE_OUTPUT = "step_to.out"

/proc/oracle_emit(key, value)
	text2file("[key]=[value]\n", ORACLE_OUTPUT)

/world
	maxx = 15
	maxy = 15
	maxz = 1
	view = 6
	turf = /turf/floor

/turf/floor
/turf/wall
	density = 1

/mob/probe
/obj/beacon

/proc/trace_to(mob/probe/m, atom/trg, count, minrange)
	var/list/seq = list()
	for(var/i in 1 to count)
		step_to(m, trg, minrange)
		seq += "[m.x],[m.y]"
	return jointext(seq, ";")

/proc/trace_towards(mob/probe/m, atom/trg, count)
	var/list/seq = list()
	for(var/i in 1 to count)
		step_towards(m, trg)
		seq += "[m.x],[m.y]"
	return jointext(seq, ";")

/proc/place(atom/movable/a, x, y)
	a.loc = locate(x, y, 1)

/world/New()
	..()
	fdel(ORACLE_OUTPUT)

	var/mob/probe/m = new
	var/obj/beacon/b = new

	// --- open: diagonal, no obstruction --------------------------------
	place(m, 2, 2)
	place(b, 8, 6)
	oracle_emit("open_trace", trace_to(m, b, 8, 0))

	// --- cardinal: due east -------------------------------------------
	place(m, 2, 10)
	place(b, 9, 10)
	oracle_emit("cardinal_trace", trace_to(m, b, 8, 0))

	// --- min_range: stop when within 2 tiles -------------------------
	place(m, 2, 13)
	place(b, 10, 13)
	oracle_emit("min_range_trace", trace_to(m, b, 10, 2))

	// --- step_towards mirrors step_to on open ground -----------------
	place(m, 2, 4)
	place(b, 8, 4)
	oracle_emit("towards_trace", trace_towards(m, b, 8))

	// --- blocked: dense wall column at x=6, y=1..15 ------------------
	for(var/wy in 1 to 15)
		new /turf/wall(locate(6, wy, 1))
	place(m, 3, 8)
	place(b, 10, 8)
	oracle_emit("blocked_trace", trace_to(m, b, 8, 0))

	// --- pocket: target sealed in walls -----------------------------
	new /turf/wall(locate(12, 2, 1))
	new /turf/wall(locate(14, 2, 1))
	new /turf/wall(locate(13, 1, 1))
	new /turf/wall(locate(13, 3, 1))
	place(b, 13, 2)
	place(m, 9, 2)
	oracle_emit("pocket_trace", trace_to(m, b, 8, 0))

	oracle_emit("oracle_complete", 1)
	shutdown()

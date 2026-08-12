/world
	maxx = 2
	maxy = 1
	maxz = 1
	turf = /turf/floor

/turf/floor

/mob/holder

/obj/item/probe
	var/constructor_x
	var/constructor_y
	var/constructor_z
	var/constructor_loc
	var/constructor_member

/obj/item/probe/New(where)
	constructor_x = x
	constructor_y = y
	constructor_z = z
	constructor_loc = (loc == where)
	constructor_member = (src in where.contents)

/world/New()
	. = ..()
	var/turf/first = locate(1, 1, 1)
	var/turf/second = locate(2, 1, 1)
	var/mob/holder/old_holder = new(first)
	var/mob/holder/new_holder = new(second)
	var/obj/item/probe/item = new(old_holder)
	var/list/result = list(
		item.constructor_x,
		item.constructor_y,
		item.constructor_z,
		item.constructor_loc,
		item.constructor_member,
		(item in old_holder.contents),
	)
	item.loc = new_holder
	result += list(
		item.x,
		item.y,
		item.z,
		(item in old_holder.contents),
		(item in new_holder.contents),
	)
	text2file(jointext(result, "|"), "_constructor_container_oracle.out")
	shutdown()

/world
	name = "DM64 compiler probe"

/datum/probe_base
	var/value = 1

/datum/probe_base/proc/compute(input = 2)
	return value + input

/datum/probe_base/child
	value = 3

/datum/probe_base/child/compute(input = 4)
	return ..(input) * 2

/proc/run_probe(input)
	var/list/values = list("first", "second", "key" = 5)
	var/datum/probe_base/child/instance = new
	return list(instance.compute(input), values[1], values["key"])

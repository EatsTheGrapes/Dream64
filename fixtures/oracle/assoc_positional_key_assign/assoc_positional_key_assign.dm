// Oracle: numeric-index assignment onto an associative entry's iteration slot.
//
// Ground truth for `L[n] = x` where position n currently holds `K = V`.

/proc/keys_of(list/L)
	var/s = "len=[length(L)] keys:"
	for(var/i = 1, i <= length(L), i++)
		var/k = L[i]
		s += " "
		if(isnull(k))
			s += "null"
		else
			s += "[k]"
	return s

/proc/val_for(list/L, k)
	var/v = L[k]
	if(isnull(v))
		return "null"
	return "[v]"

/world/New()
	..()
	var/out = ""

	var/list/L1 = list("a" = 1, "b" = 2, "c" = 3)
	L1[2] = "z"
	out += "case1_newkey: [keys_of(L1)] | z=[val_for(L1,"z")] b=[val_for(L1,"b")] a=[val_for(L1,"a")]\n"

	var/list/L2 = list("a" = 1, "b" = 2, "c" = 3)
	L2[2] = "a"
	out += "case2_collide: [keys_of(L2)] | a=[val_for(L2,"a")] b=[val_for(L2,"b")]\n"

	var/list/L3 = list("a" = 1, "b" = 2, "c" = 3)
	L3[2] = 5
	out += "case3_numkey: [keys_of(L3)] | k5=[val_for(L3,5)] b=[val_for(L3,"b")]\n"

	var/list/L4 = list("x", "a" = 1, "b" = 2)
	L4[2] = "z"
	out += "case4_mixed: [keys_of(L4)] | z=[val_for(L4,"z")] a=[val_for(L4,"a")] x=[val_for(L4,"x")]\n"

	var/list/L5 = list("a" = 1, "b" = 2, "c" = 3)
	L5[2] = "b"
	out += "case5_same: [keys_of(L5)] | b=[val_for(L5,"b")]\n"

	var/list/L6 = list("a" = 1, "b" = 2, "c" = 3)
	L6[2] = "z"
	L6["z"] = 99
	out += "case6_rewrite: [keys_of(L6)] | z=[val_for(L6,"z")]\n"

	// case 7: is the post-assign slot still associative? assign value via key.
	var/list/L7 = list("a" = 1, "b" = 2, "c" = 3)
	L7[2] = "z"
	out += "case7_assoc_after: cut? "
	L7["z"] += 10
	out += "[keys_of(L7)] | z=[val_for(L7,"z")]\n"

	text2file(out, "assoc_positional_key_assign.out")
	shutdown()

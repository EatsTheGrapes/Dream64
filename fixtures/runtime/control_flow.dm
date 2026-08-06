/proc/clamp_probe(input)
	var/result = input
	if(result < 0)
		result = 0
	else
		if(result > 10)
			result = 10
	return result

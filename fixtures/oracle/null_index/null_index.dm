/world/New()
	..()
	var/list/values = null
	var/result = values[1]
	text2file("result=[isnull(result)]", "null_index.out")
	shutdown()

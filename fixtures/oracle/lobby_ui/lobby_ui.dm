world
	name = "Dream64 lobby UI oracle"
	view = 5
	turf = /turf/oracle
	mob = /mob/oracle

/turf/oracle
	name = "oracle turf"

/mob/oracle
	name = "oracle mob"

/client/New()
	..()
	text2file("client.New begin key=[key] mob=[mob]", "lobby_ui.out")
	winset(src, "main", "title=Dream64 Lobby Oracle")
	text2file("winset returned", "lobby_ui.out")
	src << output("operator output routed", "output")
	text2file("output calls returned", "lobby_ui.out")
	src << browse_rsc('oracle.txt', "oracle.txt")
	text2file("browse_rsc returned", "lobby_ui.out")
	src << browse("<html><body><h1>Lobby</h1><img src='oracle.txt'></body></html>", "window=browser")
	text2file("browse returned", "lobby_ui.out")
	spawn(10)
		text2file("client still connected=[src != null] mob=[mob]", "lobby_ui.out")

/world/New()
	..()
	new /obj/marker(locate(2, 1, 1))

/obj/marker
	name = "oracle marker"

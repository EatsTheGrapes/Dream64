// These procedures are intentionally not invoked by world/New. Each is a
// differential entry point for checking that a JIT exits through the normal VM
// error machinery, retaining the DM proc and source location in the stack.

/proc/jit_error_divide_by_zero(divisor)
	return 1 / divisor

/proc/jit_error_bad_subtract()
	return "text" - 1

/proc/jit_error_null_call()
	var/callback
	return call(callback)()

/proc/jit_error_nested()
	return jit_error_bad_subtract()

/world/New()
	..()
	fdel("jit_errors.out")
	world.log = file("jit_errors.out")
	var/list/options = params2list(world.params)
	var/error_case = options["case"]
	world.log << "case=[error_case]"
	spawn(0)
		switch(error_case)
			if("divide_by_zero")
				jit_error_divide_by_zero(0)
			if("bad_subtract")
				jit_error_bad_subtract()
			if("null_call")
				jit_error_null_call()
			if("nested")
				jit_error_nested()
	sleep(2)
	world.log << "world_survived=1"
	shutdown()

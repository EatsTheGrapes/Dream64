var/const/ORACLE_OUTPUT = "jit_core.out"
var/list/oracle_trace = list()

/proc/oracle_emit(key, value)
	text2file("[key]=[value]\n", ORACLE_OUTPUT)

/proc/oracle_mark(label, value)
	oracle_trace += label
	return value

/proc/oracle_join_trace()
	return jointext(oracle_trace, ",")

/proc/oracle_default(value = 41)
	return value + 1

/proc/oracle_sum(left, right)
	return left * 10 + right

/proc/oracle_identity(value)
	return value

/world/New()
	..()
	fdel(ORACLE_OUTPUT)

	// Arithmetic is deliberately kept in binary32-exact ranges. A first JIT
	// must preserve DM's number coercions and modulo/division behavior.
	oracle_emit("arith_add", 7 + 5)
	oracle_emit("arith_sub", 7 - 5)
	oracle_emit("arith_mul", 7 * 5)
	oracle_emit("arith_div", 7 / 2)
	oracle_emit("arith_mod", 7 % 4)
	oracle_emit("arith_neg_mod", -7 % 4)
	oracle_emit("arith_precedence", 2 + 3 * 4)
	oracle_emit("arith_null_add", null + 5)

	// Comparisons cover numeric, text, null, mixed-type, and shallow
	// equivalence. Do not assume host-language comparison rules in native code.
	oracle_emit("cmp_num_eq", 1 == 1.0)
	oracle_emit("cmp_num_lt", 2 < 10)
	oracle_emit("cmp_text_eq", "a" == "a")
	oracle_emit("cmp_text_lt", "a" < "b")
	oracle_emit("cmp_null_eq", null == 0)
	oracle_emit("cmp_mixed_eq", "1" == 1)
	oracle_emit("cmp_equiv_num", 1 ~= 1.0)

	// Conditions, !, &&, and || are separate contracts. These values catch a
	// JIT that accidentally uses Rust/C truthiness or returns an operand.
	oracle_emit("truth_null", null ? 1 : 0)
	oracle_emit("truth_zero", 0 ? 1 : 0)
	oracle_emit("truth_number", -2 ? 1 : 0)
	oracle_emit("truth_empty_text", "" ? 1 : 0)
	oracle_emit("truth_text", "x" ? 1 : 0)
	oracle_emit("truth_not_null", !null)
	oracle_emit("truth_and", 2 && 3)
	oracle_emit("truth_or", 0 || 7)

	// Every expression below records evaluation order independently of its
	// result. Native calls must evaluate arguments left-to-right exactly once.
	oracle_trace.Cut()
	var/arithmetic_result = oracle_mark("left", 2) + oracle_mark("right", 3)
	oracle_emit("order_arith_result", arithmetic_result)
	oracle_emit("order_arith_trace", oracle_join_trace())

	oracle_trace.Cut()
	var/call_result = oracle_sum(oracle_mark("arg1", 2), oracle_mark("arg2", 3))
	oracle_emit("order_call_result", call_result)
	oracle_emit("order_call_trace", oracle_join_trace())

	oracle_trace.Cut()
	var/short_and = oracle_mark("and_left", 0) && oracle_mark("and_right", 1)
	oracle_emit("order_and_result", short_and)
	oracle_emit("order_and_trace", oracle_join_trace())

	oracle_trace.Cut()
	var/short_or = oracle_mark("or_left", 1) || oracle_mark("or_right", 0)
	oracle_emit("order_or_result", short_or)
	oracle_emit("order_or_trace", oracle_join_trace())

	// Calls cover missing/explicit-null defaults, excess arguments, and nested
	// calls. Calls the JIT cannot prove direct must fall back to the interpreter.
	oracle_emit("call_default_missing", oracle_default())
	oracle_emit("call_default_null", oracle_default(null))
	oracle_emit("call_extra_arg", oracle_identity(9, 10))
	oracle_emit("call_nested", oracle_identity(oracle_sum(2, 3)))

	// Suspending instructions are an explicit interpreter/deopt boundary for
	// the first JIT. This establishes spawn(0) and sleep(0) ordering.
	oracle_trace.Cut()
	oracle_trace += "before_spawn"
	spawn(0)
		oracle_trace += "spawned"
	oracle_trace += "after_spawn"
	sleep(0)
	oracle_trace += "after_sleep"
	oracle_emit("suspend_trace", oracle_join_trace())

	oracle_emit("oracle_complete", 1)
	shutdown()

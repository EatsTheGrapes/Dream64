// Faithful port of tgstation's timsort, exercised through the full Dream64
// frontend + semantics + lowering + VM pipeline (dm-lowering test harness).
//
// Mirrors monkestation:
//   code/__HELPERS/sorts/sort_instance.dm
//   code/__HELPERS/sorts/helpers.dm  (sortTim / CREATE_SORT_INSTANCE)
//   code/__HELPERS/_lists.dm          (sort_list, move_element, move_range, reverse_range)
//   code/__HELPERS/cmp.dm             (cmp_numeric_asc)

#define FALSE 0
#define TRUE 1
#define MIN_MERGE 32
#define MIN_GALLOP 7
#define GLOBAL_PROC_REF(X) (/proc/##X)

#define fetchElement(L, i) (associative) ? L[L[i]] : L[i]

/proc/cmp_numeric_asc(a, b)
	return a - b

/proc/move_element(list/inserted_list, from_index, to_index)
	if(from_index == to_index || from_index + 1 == to_index)
		return
	if(from_index > to_index)
		++from_index
	inserted_list.Insert(to_index, null)
	inserted_list.Swap(from_index, to_index)
	inserted_list.Cut(from_index, from_index + 1)

/proc/move_range(list/inserted_list, from_index, to_index, len = 1)
	var/distance = abs(to_index - from_index)
	if(len >= distance)
		if(from_index <= to_index)
			return
		from_index += len
		for(var/i in 1 to distance)
			inserted_list.Insert(from_index, null)
			inserted_list.Swap(from_index, to_index)
			inserted_list.Cut(to_index, to_index + 1)
	else
		if(from_index > to_index)
			from_index += len
		for(var/i in 1 to len)
			inserted_list.Insert(to_index, null)
			inserted_list.Swap(from_index, to_index)
			inserted_list.Cut(from_index, from_index + 1)

/proc/reverse_range(list/inserted_list, start = 1, end = 0)
	if(inserted_list.len)
		start = start % inserted_list.len
		end = end % (inserted_list.len + 1)
		if(start <= 0)
			start += inserted_list.len
		if(end <= 0)
			end += inserted_list.len + 1
		--end
		while(start < end)
			inserted_list.Swap(start++, end--)
	return inserted_list

/datum/sort_instance
	var/list/L
	var/cmp = GLOBAL_PROC_REF(cmp_numeric_asc)
	var/associative = 0
	var/minGallop = MIN_GALLOP
	var/list/runBases = list()
	var/list/runLens = list()

/datum/sort_instance/proc/timSort(start, end)
	runBases.Cut()
	runLens.Cut()
	var/remaining = end - start
	if(remaining < MIN_MERGE)
		var/initRunLen = countRunAndMakeAscending(start, end)
		binarySort(start, end, start+initRunLen)
		return
	var/minRun = minRunLength(remaining)
	do
		var/runLen = countRunAndMakeAscending(start, end)
		if(runLen < minRun)
			var/force = (remaining <= minRun) ? remaining : minRun
			binarySort(start, start+force, start+runLen)
			runLen = force
		runBases.Add(start)
		runLens.Add(runLen)
		mergeCollapse()
		start += runLen
		remaining -= runLen
	while(remaining > 0)
	mergeForceCollapse()
	minGallop = MIN_GALLOP
	return L

/datum/sort_instance/proc/binarySort(lo, hi, start)
	if(start <= lo)
		start = lo + 1
	var/list/L = src.L
	for(start in start to hi - 1)
		var/pivot = fetchElement(L,start)
		var/left = lo
		var/right = start
		while(left < right)
			var/mid = (left + right) >> 1
			if(call(cmp)(fetchElement(L,mid), pivot) > 0)
				right = mid
			else
				left = mid+1
		move_element(L, start, left)

/datum/sort_instance/proc/countRunAndMakeAscending(lo, hi)
	var/runHi = lo + 1
	if(runHi >= hi)
		return 1
	var/list/L = src.L
	var/last = fetchElement(L,lo)
	var/current = fetchElement(L,runHi++)
	if(call(cmp)(current, last) < 0)
		while(runHi < hi)
			last = current
			current = fetchElement(L,runHi)
			if(call(cmp)(current, last) >= 0)
				break
			++runHi
		reverse_range(L, lo, runHi)
	else
		while(runHi < hi)
			last = current
			current = fetchElement(L,runHi)
			if(call(cmp)(current, last) < 0)
				break
			++runHi
	return runHi - lo

/datum/sort_instance/proc/minRunLength(n)
	var/r = 0
	while(n >= MIN_MERGE)
		r |= (n & 1)
		n >>= 1
	return n + r

/datum/sort_instance/proc/mergeCollapse()
	while(runBases.len >= 2)
		var/n = runBases.len - 1
		if(n > 1 && runLens[n-1] <= runLens[n] + runLens[n+1])
			if(runLens[n-1] < runLens[n+1])
				--n
			mergeAt(n)
		else if(runLens[n] <= runLens[n+1])
			mergeAt(n)
		else
			break

/datum/sort_instance/proc/mergeForceCollapse()
	while(runBases.len >= 2)
		var/n = runBases.len - 1
		if(n > 1 && runLens[n-1] < runLens[n+1])
			--n
		mergeAt(n)

/datum/sort_instance/proc/mergeAt(i)
	var/base1 = runBases[i]
	var/base2 = runBases[i+1]
	var/len1 = runLens[i]
	var/len2 = runLens[i+1]
	runLens[i] += runLens[i+1]
	runLens.Cut(i+1, i+2)
	runBases.Cut(i+1, i+2)
	var/k = gallopRight(fetchElement(L,base2), base1, len1, 0)
	base1 += k
	len1 -= k
	if(len1 == 0)
		return
	len2 = gallopLeft(fetchElement(L,base1 + len1 - 1), base2, len2, len2-1)
	if(len2 == 0)
		return
	if(len1 <= len2)
		mergeLo(base1, len1, base2, len2)
	else
		mergeHi(base1, len1, base2, len2)

/datum/sort_instance/proc/gallopLeft(key, base, len, hint)
	var/list/L = src.L
	var/lastOffset = 0
	var/offset = 1
	if(call(cmp)(key, fetchElement(L,base+hint)) > 0)
		var/maxOffset = len - hint
		while(offset < maxOffset && call(cmp)(key, fetchElement(L,base+hint+offset)) > 0)
			lastOffset = offset
			offset = (offset << 1) + 1
		if(offset > maxOffset)
			offset = maxOffset
		lastOffset += hint
		offset += hint
	else
		var/maxOffset = hint + 1
		while(offset < maxOffset && call(cmp)(key, fetchElement(L,base+hint-offset)) <= 0)
			lastOffset = offset
			offset = (offset << 1) + 1
		if(offset > maxOffset)
			offset = maxOffset
		var/temp = lastOffset
		lastOffset = hint - offset
		offset = hint - temp
	++lastOffset
	while(lastOffset < offset)
		var/m = lastOffset + ((offset - lastOffset) >> 1)
		if(call(cmp)(key, fetchElement(L,base+m)) > 0)
			lastOffset = m + 1
		else
			offset = m
	return offset

/datum/sort_instance/proc/gallopRight(key, base, len, hint)
	var/list/L = src.L
	var/offset = 1
	var/lastOffset = 0
	if(call(cmp)(key, fetchElement(L,base+hint)) < 0)
		var/maxOffset = hint + 1
		while(offset < maxOffset && call(cmp)(key, fetchElement(L,base+hint-offset)) < 0)
			lastOffset = offset
			offset = (offset << 1) + 1
		if(offset > maxOffset)
			offset = maxOffset
		var/temp = lastOffset
		lastOffset = hint - offset
		offset = hint - temp
	else
		var/maxOffset = len - hint
		while(offset < maxOffset && call(cmp)(key, fetchElement(L,base+hint+offset)) >= 0)
			lastOffset = offset
			offset = (offset << 1) + 1
		if(offset > maxOffset)
			offset = maxOffset
		lastOffset += hint
		offset += hint
	++lastOffset
	while(lastOffset < offset)
		var/m = lastOffset + ((offset - lastOffset) >> 1)
		if(call(cmp)(key, fetchElement(L,base+m)) < 0)
			offset = m
		else
			lastOffset = m + 1
	return offset

/datum/sort_instance/proc/mergeLo(base1, len1, base2, len2)
	var/list/L = src.L
	var/cursor1 = base1
	var/cursor2 = base2
	if(len2 == 1)
		move_element(L, cursor2, cursor1)
		return
	if(len1 == 1)
		move_element(L, cursor1, cursor2+len2)
		return
	move_element(L, cursor2++, cursor1++)
	--len2
	outer:
		while(1)
			var/count1 = 0
			var/count2 = 0
			do
				if(call(cmp)(fetchElement(L,cursor2), fetchElement(L,cursor1)) < 0)
					move_element(L, cursor2++, cursor1++)
					--len2
					++count2
					count1 = 0
					if(len2 == 0)
						break outer
				else
					++cursor1
					++count1
					count2 = 0
					if(--len1 == 1)
						break outer
			while((count1 | count2) < minGallop)
			do
				count1 = gallopRight(fetchElement(L,cursor2), cursor1, len1, 0)
				if(count1)
					cursor1 += count1
					len1 -= count1
					if(len1 <= 1)
						break outer
				move_element(L, cursor2, cursor1)
				++cursor2
				++cursor1
				if(--len2 == 0)
					break outer
				count2 = gallopLeft(fetchElement(L,cursor1), cursor2, len2, 0)
				if(count2)
					move_range(L, cursor2, cursor1, count2)
					cursor2 += count2
					cursor1 += count2
					len2 -= count2
					if(len2 == 0)
						break outer
				++cursor1
				if(--len1 == 1)
					break outer
				--minGallop
			while((count1|count2) > MIN_GALLOP)
			if(minGallop < 0)
				minGallop = 0
			minGallop += 2
	if(len1 == 1)
		move_element(L, cursor1, cursor2+len2)

/datum/sort_instance/proc/mergeHi(base1, len1, base2, len2)
	var/list/L = src.L
	var/cursor1 = base1 + len1 - 1
	var/cursor2 = base2 + len2 - 1
	if(len2 == 1)
		move_element(L, base2, base1)
		return
	if(len1 == 1)
		move_element(L, base1, cursor2+1)
		return
	move_element(L, cursor1--, cursor2-- + 1)
	--len1
	outer:
		while(1)
			var/count1 = 0
			var/count2 = 0
			do
				if(call(cmp)(fetchElement(L,cursor2), fetchElement(L,cursor1)) < 0)
					move_element(L, cursor1--, cursor2-- + 1)
					--len1
					++count1
					count2 = 0
					if(len1 == 0)
						break outer
				else
					--cursor2
					--len2
					++count2
					count1 = 0
					if(len2 == 1)
						break outer
			while((count1 | count2) < minGallop)
			do
				count1 = len1 - gallopRight(fetchElement(L,cursor2), base1, len1, len1-1)
				if(count1)
					cursor1 -= count1
					move_range(L, cursor1+1, cursor2+1, count1)
					cursor2 -= count1
					len1 -= count1
					if(len1 == 0)
						break outer
				--cursor2
				if(--len2 == 1)
					break outer
				count2 = len2 - gallopLeft(fetchElement(L,cursor1), cursor1+1, len2, len2-1)
				if(count2)
					cursor2 -= count2
					len2 -= count2
					if(len2 <= 1)
						break outer
				move_element(L, cursor1--, cursor2-- + 1)
				--len1
				if(len1 == 0)
					break outer
				--minGallop
			while((count1|count2) > MIN_GALLOP)
			if(minGallop < 0)
				minGallop = 0
			minGallop += 2
	if(len2 == 1)
		cursor1 -= len1
		move_range(L, cursor1+1, cursor2+1, len1)

var/global/datum/sort_instance/sortInstance = new()

#define CREATE_SORT_INSTANCE(to_sort, cmp, associative, fromIndex, toIndex) \
	if(length(to_sort) < 2) { \
		return to_sort; \
	} \
	fromIndex = fromIndex % length(to_sort); \
	toIndex = toIndex % (length(to_sort) + 1); \
	if (fromIndex <= 0) { \
		fromIndex += length(to_sort); \
	} \
	if (toIndex <= 0) { \
		toIndex += length(to_sort) + 1; \
	} \
	var/datum/sort_instance/sorter = sortInstance; \
	if (isnull(sorter)) { \
		sorter = new; \
	} \
	sorter.L = to_sort; \
	sorter.cmp = cmp; \
	sorter.associative = associative;

/proc/sortTim(list/to_sort, cmp = GLOBAL_PROC_REF(cmp_numeric_asc), associative = FALSE, fromIndex = 1, toIndex = 0) as /list
	CREATE_SORT_INSTANCE(to_sort, cmp, associative, fromIndex, toIndex)
	sorter.timSort(fromIndex, toIndex)
	return to_sort

#undef CREATE_SORT_INSTANCE

/proc/sort_list(list/list_to_sort, cmp = GLOBAL_PROC_REF(cmp_numeric_asc))
	return sortTim(list_to_sort.Copy(), cmp)

/proc/cmp_assoc_val_asc(a, b)
	return a - b

/proc/build_list(count, mode)
	var/list/input = list()
	for(var/i in 1 to count)
		switch(mode)
			if(0)
				input += (count - i + 1)
			if(1)
				input += ((i * 7) % count) + 1
			else
				input += i
	return input

/proc/check_sorted(list/sorted, count)
	if(sorted.len != count)
		return "LENGTH CORRUPTED: got [sorted.len] expected [count]"
	for(var/i in 1 to count - 1)
		if(sorted[i] > sorted[i+1])
			return "NOT SORTED at [i]"
	return "OK"

/proc/run_repro(count, mode)
	return check_sorted(sort_list(build_list(count, mode)), count)

// Sort an associative list by its associated values, then immediately run a
// plain (non-associative) sort through the same reused GLOB.sortInstance.
/proc/run_repro_assoc_then_plain(count)
	var/list/assoc = list()
	for(var/i in 1 to count)
		assoc["k[i]"] = ((i * 13) % count) + 1
	sortTim(assoc, GLOBAL_PROC_REF(cmp_assoc_val_asc), TRUE)
	// now a plain sort — sortInstance.associative must be reset to FALSE
	return check_sorted(sort_list(build_list(count, 1)), count)

// Many sorts in a row through the shared instance, alternating orderings.
/proc/run_repro_repeated(count)
	for(var/pass in 1 to 6)
		var/result = check_sorted(sort_list(build_list(count, pass % 3)), count)
		if(result != "OK")
			return "pass [pass]: [result]"
	return "OK"

// Mirrors /obj/machinery/chem_dispenser: a merge-path-sized list literal declared
// as an instance-var initializer on a parent type, inherited unchanged by a child
// type, then sorted from the child's constructor. This is the shape that the
// pre-64f577d inherited-instance-initializer bug corrupted for linked-artifact
// map atoms, surfacing as "DM list position N exceeds length M" inside gallopRight.
// The literal is in a jumbled order so timSort takes the natural-run + merge path
// (countRunAndMakeAscending -> mergeCollapse -> mergeAt -> gallop*), not a single
// binarySort.
/obj/dispenser
	var/list/dispensable = list(19, 3, 44, 12, 27, 5, 38, 1, 22, 9, 33, 16, 41, 7, 25, 2, 30, 11, 36, 18, 43, 6, 21, 14, 28, 4, 39, 10, 24, 15, 34, 8, 42, 13, 26, 20, 37, 17, 31, 23, 40, 29, 45, 32, 35)

/obj/dispenser/fullupgrade

/obj/dispenser/fullupgrade/proc/sort_it()
	dispensable = sort_list(dispensable)
	var/n = dispensable.len
	for(var/i in 1 to n - 1)
		if(dispensable[i] > dispensable[i+1])
			return "NOT SORTED at [i]"
	return "OK"

/proc/run_repro_inherited()
	var/obj/dispenser/fullupgrade/d = new
	return d.sort_it()

// Minimal port of tgstation timsort (code/__HELPERS/sorts/sort_instance.dm)
// plus the list helper procs it depends on, for reproducing the Dream64
// "DM list position N exceeds length M" merge-path bug in isolation.
//
// The datum type-tree block is omitted (compile_module only accepts procedure
// definitions); sortTim() seeds every instance field explicitly instead, and
// all field access is written as src.<field>. fetchElement is hard-wired to the
// non-associative form because the fatal boot failure is a plain L[i] access.





/proc/cmp_numeric_asc(a, b)
	return a - b

/proc/cmp_datum_k_asc(datum/a, datum/b)
	return a.k - b.k

/datum/sort_instance/proc/docmp(a, b)
	return call(src.cmp)(a, b)

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

/datum/sort_instance/proc/timSort(start, end)
	src.runBases.Cut()
	src.runLens.Cut()
	var/remaining = end - start
	if(remaining < 32)
		var/initRunLen = src.countRunAndMakeAscending(start, end)
		src.binarySort(start, end, start+initRunLen)
		return
	var/minRun = src.minRunLength(remaining)
	do
		var/runLen = src.countRunAndMakeAscending(start, end)
		if(runLen < minRun)
			var/force = (remaining <= minRun) ? remaining : minRun
			src.binarySort(start, start+force, start+runLen)
			runLen = force
		src.runBases.Add(start)
		src.runLens.Add(runLen)
		src.mergeCollapse()
		start += runLen
		remaining -= runLen
	while(remaining > 0)
	src.mergeForceCollapse()
	src.minGallop = 7
	return src.L

/datum/sort_instance/proc/binarySort(lo, hi, start)
	if(start <= lo)
		start = lo + 1
	var/list/L = src.L
	for(start in start to hi - 1)
		var/pivot = L[start]
		var/left = lo
		var/right = start
		while(left < right)
			var/mid = (left + right) >> 1
			if(src.docmp(L[mid], pivot) > 0)
				right = mid
			else
				left = mid+1
		move_element(L, start, left)

/datum/sort_instance/proc/countRunAndMakeAscending(lo, hi)
	var/runHi = lo + 1
	if(runHi >= hi)
		return 1
	var/list/L = src.L
	var/last = L[lo]
	var/current = L[runHi++]
	if(src.docmp(current, last) < 0)
		while(runHi < hi)
			last = current
			current = L[runHi]
			if(src.docmp(current, last) >= 0)
				break
			++runHi
		reverse_range(L, lo, runHi)
	else
		while(runHi < hi)
			last = current
			current = L[runHi]
			if(src.docmp(current, last) < 0)
				break
			++runHi
	return runHi - lo

/datum/sort_instance/proc/minRunLength(n)
	var/r = 0
	while(n >= 32)
		r |= (n & 1)
		n >>= 1
	return n + r

/datum/sort_instance/proc/mergeCollapse()
	while(src.runBases.len >= 2)
		var/n = src.runBases.len - 1
		if(n > 1 && src.runLens[n-1] <= src.runLens[n] + src.runLens[n+1])
			if(src.runLens[n-1] < src.runLens[n+1])
				--n
			src.mergeAt(n)
		else if(src.runLens[n] <= src.runLens[n+1])
			src.mergeAt(n)
		else
			break

/datum/sort_instance/proc/mergeForceCollapse()
	while(src.runBases.len >= 2)
		var/n = src.runBases.len - 1
		if(n > 1 && src.runLens[n-1] < src.runLens[n+1])
			--n
		src.mergeAt(n)

/datum/sort_instance/proc/mergeAt(i)
	var/list/L = src.L
	var/base1 = src.runBases[i]
	var/base2 = src.runBases[i+1]
	var/len1 = src.runLens[i]
	var/len2 = src.runLens[i+1]
	src.runLens[i] += src.runLens[i+1]
	src.runLens.Cut(i+1, i+2)
	src.runBases.Cut(i+1, i+2)
	var/k = src.gallopRight(L[base2], base1, len1, 0)
	base1 += k
	len1 -= k
	if(len1 == 0)
		return
	len2 = src.gallopLeft(L[base1 + len1 - 1], base2, len2, len2-1)
	if(len2 == 0)
		return
	if(len1 <= len2)
		src.mergeLo(base1, len1, base2, len2)
	else
		src.mergeHi(base1, len1, base2, len2)

/datum/sort_instance/proc/gallopLeft(key, base, len, hint)
	var/list/L = src.L
	var/lastOffset = 0
	var/offset = 1
	if(src.docmp(key, L[base+hint]) > 0)
		var/maxOffset = len - hint
		while(offset < maxOffset && src.docmp(key, L[base+hint+offset]) > 0)
			lastOffset = offset
			offset = (offset << 1) + 1
		if(offset > maxOffset)
			offset = maxOffset
		lastOffset += hint
		offset += hint
	else
		var/maxOffset = hint + 1
		while(offset < maxOffset && src.docmp(key, L[base+hint-offset]) <= 0)
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
		if(src.docmp(key, L[base+m]) > 0)
			lastOffset = m + 1
		else
			offset = m
	return offset

/datum/sort_instance/proc/gallopRight(key, base, len, hint)
	var/list/L = src.L
	var/offset = 1
	var/lastOffset = 0
	if(src.docmp(key, L[base+hint]) < 0)
		var/maxOffset = hint + 1
		while(offset < maxOffset && src.docmp(key, L[base+hint-offset]) < 0)
			lastOffset = offset
			offset = (offset << 1) + 1
		if(offset > maxOffset)
			offset = maxOffset
		var/temp = lastOffset
		lastOffset = hint - offset
		offset = hint - temp
	else
		var/maxOffset = len - hint
		while(offset < maxOffset && src.docmp(key, L[base+hint+offset]) >= 0)
			lastOffset = offset
			offset = (offset << 1) + 1
		if(offset > maxOffset)
			offset = maxOffset
		lastOffset += hint
		offset += hint
	++lastOffset
	while(lastOffset < offset)
		var/m = lastOffset + ((offset - lastOffset) >> 1)
		if(src.docmp(key, L[base+m]) < 0)
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
				if(src.docmp(L[cursor2], L[cursor1]) < 0)
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
			while((count1 | count2) < src.minGallop)
			do
				count1 = src.gallopRight(L[cursor2], cursor1, len1, 0)
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
				count2 = src.gallopLeft(L[cursor1], cursor2, len2, 0)
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
				--src.minGallop
			while((count1|count2) > 7)
			if(src.minGallop < 0)
				src.minGallop = 0
			src.minGallop += 2
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
				if(src.docmp(L[cursor2], L[cursor1]) < 0)
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
			while((count1 | count2) < src.minGallop)
			do
				count1 = len1 - src.gallopRight(L[cursor2], base1, len1, len1-1)
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
				count2 = len2 - src.gallopLeft(L[cursor1], cursor1+1, len2, len2-1)
				if(count2)
					cursor2 -= count2
					len2 -= count2
					if(len2 <= 1)
						break outer
				move_element(L, cursor1--, cursor2-- + 1)
				--len1
				if(len1 == 0)
					break outer
				--src.minGallop
			while((count1|count2) > 7)
			if(src.minGallop < 0)
				src.minGallop = 0
			src.minGallop += 2
	if(len2 == 1)
		cursor1 -= len1
		move_range(L, cursor1+1, cursor2+1, len1)

/proc/sortTim(list/to_sort, cmp = /proc/cmp_numeric_asc, associative = 0, fromIndex = 1, toIndex = 0)
	if(length(to_sort) < 2)
		return to_sort
	fromIndex = fromIndex % length(to_sort)
	toIndex = toIndex % (length(to_sort) + 1)
	if (fromIndex <= 0)
		fromIndex += length(to_sort)
	if (toIndex <= 0)
		toIndex += length(to_sort) + 1
	var/datum/sort_instance/sorter = new
	sorter.L = to_sort
	sorter.cmp = cmp
	sorter.associative = associative
	sorter.minGallop = 7
	sorter.runBases = list()
	sorter.runLens = list()
	sorter.timSort(fromIndex, toIndex)
	return to_sort

/proc/sort_list(list/list_to_sort, cmp = /proc/cmp_numeric_asc)
	return sortTim(list_to_sort.Copy(), cmp)

/proc/run_repro(count, mode)
	var/list/input = list()
	for(var/i in 1 to count)
		switch(mode)
			if(0)
				input += (count - i + 1)
			if(1)
				input += ((i * 7) % count) + 1
			else
				input += i
	var/list/sorted = sort_list(input)
	if(sorted.len != count)
		return "LENGTH CORRUPTED: got [sorted.len] expected [count]"
	for(var/i in 1 to count - 1)
		if(sorted[i] > sorted[i+1])
			return "NOT SORTED at [i]: [sorted[i]] > [sorted[i+1]]"
	return "OK len=[sorted.len]"

/proc/run_repro_datum(count, mode)
	var/list/input = list()
	for(var/i in 1 to count)
		var/datum/d = new
		switch(mode)
			if(0)
				d.k = (count - i + 1)
			if(1)
				d.k = ((i * 7) % count) + 1
			else
				d.k = i
		input += d
	var/list/sorted = sort_list(input, /proc/cmp_datum_k_asc)
	if(sorted.len != count)
		return "LENGTH CORRUPTED: got [sorted.len] expected [count]"
	for(var/i in 1 to count)
		var/datum/d = sorted[i]
		if(!istype(d))
			return "NON-DATUM at [i]"
	for(var/i in 1 to count - 1)
		var/datum/x = sorted[i]
		var/datum/y = sorted[i+1]
		if(x.k > y.k)
			return "NOT SORTED at [i]: [x.k] > [y.k]"
	return "OK len=[sorted.len]"


// BYOND 516.1680 behavioural oracle for the /icon pixel pipeline.
//
// Loads a committed 32x32 greyscale template (states "box", "stripe") and
// exercises the raster ops Dream64's dm-icon crate must reproduce: Blend with a
// colour (ICON_MULTIPLY / ICON_OVERLAY / ICON_ADD), Scale, Crop, Flip, Turn,
// SwapColor, plus the Width/Height/icon_states metadata readers. Results are
// GetPixel colour strings and dimensions written to icon_ops.out.

var/const/ORACLE_OUTPUT = "icon_ops.out"

/proc/oracle_emit(key, value)
	text2file("[key]=[value]\n", ORACLE_OUTPUT)

/world/New()
	..()
	fdel(ORACLE_OUTPUT)

	var/icon/base = icon('template.dmi', "box")
	oracle_emit("base_dims", "[base.Width()]x[base.Height()]")
	oracle_emit("base_states", jointext(icon_states('template.dmi'), ","))
	oracle_emit("base_fill", base.GetPixel(4, 4))
	oracle_emit("base_hole", base.GetPixel(4, 28))

	// Blend a colour with ICON_MULTIPLY (the GAGS colourise step).
	var/icon/mult = new(base)
	mult.Blend("#4080c0", ICON_MULTIPLY)
	oracle_emit("multiply_fill", mult.GetPixel(4, 4))

	// Blend a colour with ICON_ADD.
	var/icon/added = new(base)
	added.Blend("#202020", ICON_ADD)
	oracle_emit("add_fill", added.GetPixel(4, 4))

	// Blend a half-alpha white with ICON_OVERLAY.
	var/icon/over = new(base)
	over.Blend("#ffffff80", ICON_OVERLAY)
	oracle_emit("overlay_fill", over.GetPixel(4, 4))

	// Scale up (nearest neighbour) then sample.
	var/icon/scaled = new(base)
	scaled.Scale(64, 64)
	oracle_emit("scaled_dims", "[scaled.Width()]x[scaled.Height()]")
	oracle_emit("scaled_fill", scaled.GetPixel(8, 8))
	oracle_emit("scaled_hole", scaled.GetPixel(8, 56))

	// Crop to the opaque quadrant; canvas shrinks to 16x16.
	var/icon/cropped = new(base)
	cropped.Crop(1, 1, 16, 16)
	oracle_emit("cropped_dims", "[cropped.Width()]x[cropped.Height()]")
	oracle_emit("cropped_fill", cropped.GetPixel(1, 1))
	oracle_emit("cropped_far", cropped.GetPixel(16, 16))

	// Flip vertically (NORTH): the bottom-left fill moves to the top-left.
	var/icon/flipped = new(base)
	flipped.Flip(NORTH)
	oracle_emit("flip_low", flipped.GetPixel(4, 4))
	oracle_emit("flip_high", flipped.GetPixel(4, 28))

	// Turn 90 degrees counter-clockwise.
	var/icon/turned = new(base)
	turned.Turn(90)
	oracle_emit("turn_dims", "[turned.Width()]x[turned.Height()]")

	// SwapColor the exact fill grey for magenta.
	var/icon/swapped = new(base)
	swapped.SwapColor("#808080", "#ff00ff")
	oracle_emit("swap_fill", swapped.GetPixel(4, 4))

	del(src)

package dev.xiaoxin.vpshield.utils

import android.content.Context
import android.graphics.Bitmap
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicBlur
import android.util.Log
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale

fun createBlurredCrop(
    context: Context,
    orig: Bitmap,
    left: Int,
    top: Int,
    right: Int,
    bottom: Int,
    radius: Float = 20f
): Bitmap {
    val l = left.coerceAtLeast(0)
    val t = top.coerceAtLeast(0)
    val r = right.coerceAtMost(orig.width)
    val b = bottom.coerceAtMost(orig.height)
    val w = (r - l).coerceAtLeast(1)
    val h = (b - t).coerceAtLeast(1)

    // Crop the region
    val crop = Bitmap.createBitmap(orig, l, t, w, h)

    // Downscale for extra-strong blur (acts similar to enlarging Gaussian kernel size)
    val scaleFactor = 0.15f // 15% of original size (adaptive kernel on server maxes with >=51 so this is robust)
    val smallW = (w * scaleFactor).coerceAtLeast(1f).toInt()
    val smallH = (h * scaleFactor).coerceAtLeast(1f).toInt()
    val downscaled = crop.scale(smallW, smallH)

    val rs = RenderScript.create(context)
    try {
        val input = Allocation.createFromBitmap(rs, downscaled)
        val output = Allocation.createTyped(rs, input.type)
        val script = ScriptIntrinsicBlur.create(rs, Element.U8_4(rs))
        // Honor provided radius (server logic uses k = max(51, w//3)|1; radius ~= (k-1)/2 capped by 25)
        script.setRadius(radius.coerceIn(0.1f, 25f))
        script.setInput(input)
        script.forEach(output)
        val blurredSmall = createBitmap(smallW, smallH)
        output.copyTo(blurredSmall)
        val blurred = blurredSmall.scale(w, h)
        // Cleanup
        crop.recycle(); downscaled.recycle(); blurredSmall.recycle()
        return blurred
    } catch (e: Exception) {
        Log.e("BlurHelper", "Blur failed, returning original crop", e)
        return crop // fallback (cropped original)
    } finally { rs.destroy() }
}
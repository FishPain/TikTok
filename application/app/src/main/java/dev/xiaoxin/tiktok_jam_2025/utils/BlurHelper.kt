import android.content.Context
import android.graphics.Bitmap
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicBlur
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

    // Downscale for extra-strong blur
    val scaleFactor = 0.15f // 15% of original size
    val smallW = (w * scaleFactor).coerceAtLeast(1f).toInt()
    val smallH = (h * scaleFactor).coerceAtLeast(1f).toInt()
    val downscaled = crop.scale(smallW, smallH)

    // Create RenderScript context
    val rs = RenderScript.create(context)
    val input = Allocation.createFromBitmap(rs, downscaled)
    val output = Allocation.createTyped(rs, input.type)

    // Create and configure blur script
    val script = ScriptIntrinsicBlur.create(rs, Element.U8_4(rs))
    script.setRadius(25f) // Maximum blur
    script.setInput(input)
    script.forEach(output)

    // Copy result back to bitmap
    val blurredSmall = createBitmap(smallW, smallH)
    output.copyTo(blurredSmall)

    // Upscale back to original crop size
    val blurred = blurredSmall.scale(w, h)

    // Clean up
    rs.destroy()
    crop.recycle()
    downscaled.recycle()
    blurredSmall.recycle()

    return blurred
}
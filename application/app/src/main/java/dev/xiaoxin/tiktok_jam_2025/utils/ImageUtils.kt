package dev.xiaoxin.tiktok_jam_2025.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.exifinterface.media.ExifInterface

object ImageUtils {

    fun loadBitmapFromUri(context: Context, uri: Uri, maxSide: Int = 1600): Bitmap {
        // Step 1: Decode bounds to compute inSampleSize
        val optsBounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri).use { stream ->
            BitmapFactory.decodeStream(stream, null, optsBounds)
        }
        val (w, h) = optsBounds.outWidth to optsBounds.outHeight
        val longest = maxOf(w, h)
        val inSample = if (longest > maxSide) Integer.highestOneBit(longest / maxSide).coerceAtLeast(1) else 1

        // Step 2: Decode actual bitmap with inSampleSize
        val optsBitmap = BitmapFactory.Options().apply {
            inSampleSize = inSample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }

        val bmp: Bitmap = context.contentResolver.openInputStream(uri).use { stream ->
            BitmapFactory.decodeStream(stream, null, optsBitmap)
                ?: throw IllegalArgumentException("Couldn't decode bitmap")
        }

        // Step 3: Fix rotation using EXIF if possible
        context.contentResolver.openInputStream(uri).use { exifStream ->
            if (exifStream != null) {
                try {
                    val exif = ExifInterface(exifStream)
                    val orientation = exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                    return when (orientation) {
                        ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(bmp, 90)
                        ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(bmp, 180)
                        ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(bmp, 270)
                        else -> bmp
                    }
                } catch (_: Exception) {
                    // If EXIF fails, return original bitmap
                }
            }
        }

        return bmp
    }

    private fun rotateBitmap(src: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return src
        val matrix = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
        src.recycle()
        return rotated
    }
}

package dev.xiaoxin.tiktok_jam_2025.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.core.graphics.scale

class AgeEstimator(
    private val context: Context,
    modelFilename: String = "age_estimator_model.tflite",
    private val errorMargin: Float = 2f
) {
    private val interpreter: Interpreter =
        Interpreter(FileUtil.loadMappedFile(context, modelFilename))
    private val faceDetector = FaceDetection(context)

    init {
        faceDetector.createDetector()
    }

    fun processImage(bitmap: Bitmap): List<Result> {
        val faces = faceDetector.detect(bitmap).detections()
        Log.d("FaceDetection", "Faces detected: ${faces.size}")
        val results = mutableListOf<Result>()

        for (face in faces) {
            val rect = Rect(
                face.boundingBox().left.toInt(),
                face.boundingBox().top.toInt(),
                face.boundingBox().right.toInt(),
                face.boundingBox().bottom.toInt()
            )

            val crop = cropFace(bitmap, rect)
            val inputTensor = preprocess(crop, 200, 200)

            val output = Array(1) { FloatArray(1) }
            interpreter.run(inputTensor, output)
            val ageEstimate = output[0][0] * 116f
            Log.d("Age estimate:", ageEstimate.toString())
            val maskRecommendation = if (ageEstimate + errorMargin < 18f) "Yes" else "No"
            Log.d("Mask recommendation:", maskRecommendation)
            results.add(Result(rect, ageEstimate, maskRecommendation))
        }
        return results
    }

    private fun cropFace(bitmap: Bitmap, rect: Rect): Bitmap {
        val x = rect.left.coerceAtLeast(0)
        val y = rect.top.coerceAtLeast(0)
        val w = rect.width().coerceAtMost(bitmap.width - x)
        val h = rect.height().coerceAtMost(bitmap.height - y)
        return Bitmap.createBitmap(bitmap, x, y, w, h)
    }

    private fun preprocess(bitmap: Bitmap, targetW: Int, targetH: Int): ByteBuffer {
        val resized = bitmap.scale(targetW, targetH)
        val buffer =
            ByteBuffer.allocateDirect(4 * targetW * targetH * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(targetW * targetH)
        resized.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
        buffer.rewind()
        return buffer
    }

    data class Result(val bbox: Rect, val age: Float, val mask: String)

    fun close() {
        faceDetector.close()
        interpreter.close()
    }
}

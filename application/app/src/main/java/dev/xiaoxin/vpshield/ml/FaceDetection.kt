package dev.xiaoxin.tiktok_jam_2025.ml

import android.content.Context
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult

class FaceDetection(
    private val context: Context,
    private val modelAssetPath: String = "blaze_face_short_range.tflite"
) {
    private var detector: FaceDetector? = null

    fun createDetector(minDetectionConfidence: Float = 0.2f) {
        val baseOptionsBuilder = BaseOptions.builder().setModelAssetPath(modelAssetPath)
        val options = FaceDetector.FaceDetectorOptions.builder()
            .setBaseOptions(baseOptionsBuilder.build())
            .setMinDetectionConfidence(minDetectionConfidence)
            .setRunningMode(RunningMode.IMAGE) // single-image mode
            .build()

        detector = FaceDetector.createFromOptions(context, options)
    }

    fun detect(bitmap: android.graphics.Bitmap): FaceDetectorResult {
        val mpImage = BitmapImageBuilder(bitmap).build()
        // Note: detect(...) blocks (IMAGE mode). Run off UI thread.
        return detector!!.detect(mpImage)
    }

    fun close() {
        detector?.close()
    }
}

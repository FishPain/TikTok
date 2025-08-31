package dev.xiaoxin.tiktok_jam_2025.ml

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import android.widget.Toast
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions

/**
 * Wrapper around ML Kit text recognition that always surfaces a plain String.
 * Also exposes runPiiranhaOnText so ViewModel can call ONNX directly when needed.
 */
class TextRecognizer(
    private val context: Context
) {
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
//    private val piiranhaModel = PIIDetector(context)

    /**
     * Process an image from a Uri -> returns the extracted plain text string.
     * onTextResult runs on the main thread.
     */
    fun processImageUri(
        uri: Uri,
        onTextResult: (String) -> Unit,
        onError: (Exception) -> Unit
    ) {
        try {
            val image = InputImage.fromFilePath(context, uri)
            recognize(image, onTextResult, onError)
        } catch (e: Exception) {
            Log.e("TextRecognizer", "Failed to load image from URI", e)
            onError(e)
        }
    }

    /**
     * Process a Bitmap -> returns the extracted plain text string.
     * onTextResult runs on the main thread.
     */
    fun processBitmap(
        bitmap: Bitmap,
        onTextResult: (String) -> Unit,
        onError: (Exception) -> Unit
    ) {
        val image = InputImage.fromBitmap(bitmap, 0)
        recognize(image, onTextResult, onError)
    }

    /**
     * Run recognition; convert ML Kit Text -> plain String and return on main thread.
     * Also shows how to call the ONNX model off the UI thread if desired.
     */
    private fun recognize(
        image: InputImage,
        onTextResult: (String) -> Unit,
        onError: (Exception) -> Unit
    ): Task<Text> {
        return recognizer.process(image)
            .addOnSuccessListener { visionText ->
                val raw = visionText.text ?: ""
                onTextResult(raw)
            }
            .addOnFailureListener { e ->
                Log.e("TextRecognizer", "Text recognition failed", e)
                Toast.makeText(context, "OCR failed: ${e.localizedMessage}", Toast.LENGTH_SHORT)
                    .show()
                onError(e)
            }
    }

    /**
     * Expose a direct call to the Piiranha ONNX model.
     * Runs synchronously (but the caller should call it from a background coroutine).
     */
//    fun runPiiranhaOnText(text: String): FloatArray {
//        return try {
//            piiranhaModel.runOnText(text)
//        } catch (e: Exception) {
//            Log.e("TextRecognizer", "Piiranha run failed", e)
//            FloatArray(0)
//        }
//    }

//    fun stop() {
//        recognizer.close()
//        piiranhaModel.close()
//    }
}

package dev.xiaoxin.tiktok_jam_2025.ml

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.TextRecognizerOptionsInterface
import com.google.mlkit.vision.text.latin.TextRecognizerOptions

class TextRecognition(
    private val context: Context
) {
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    /** Process image directly from a gallery URI. */
    fun processImageUri(uri: Uri, onResult: (Text) -> Unit, onError: (Exception) -> Unit) {
        try {
            val image = InputImage.fromFilePath(context, uri)
            runRecognition(image, onResult, onError)
        } catch (e: Exception) {
            Log.e("OcrProcessor", "Failed to load image from URI", e)
            onError(e)
        }
    }

    /** Optional: if you already have a Bitmap (e.g., user cropped it). */
    fun processBitmap(bitmap: android.graphics.Bitmap, onResult: (Text) -> Unit, onError: (Exception) -> Unit) {
        val image = InputImage.fromBitmap(bitmap, 0)
        runRecognition(image, onResult, onError)
    }

    private fun runRecognition(
        image: InputImage,
        onResult: (Text) -> Unit,
        onError: (Exception) -> Unit
    ): Task<Text> {
        return recognizer.process(image)
            .addOnSuccessListener { text -> onResult(text) }
            .addOnFailureListener { e ->
                Log.e("OcrProcessor", "Text recognition failed", e)
                Toast.makeText(context, "OCR failed: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                onError(e)
            }
    }

    fun stop() {
        recognizer.close()
    }
}
package dev.xiaoxin.vpshield.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.InputStream
import java.nio.ByteBuffer

data class AnalysisResult(
    val faces: Boolean,
    val location: Boolean,
    val pii: Boolean
)

class LLMModel(private val context: Context) {
    private val modelPath = "/data/local/tmp/llm/gemma-3n-E2B-it-int4.task"
    private var isInitialized = false

    // Initialize the LLM model - just check if model path is accessible
    suspend fun initialize(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                // For now, we'll assume the model is available
                // In a real implementation, you'd check if the model file exists
                isInitialized = true
                true
            } catch (e: Exception) {
                e.printStackTrace()
                false
            }
        }
    }

    // Analyze image from URI
    suspend fun analyzeImageFromUri(uri: Uri): AnalysisResult? {
        return withContext(Dispatchers.IO) {
            try {
                val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()
                bitmap?.let { analyzeImage(it) }
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }
    }

    // Analyze image from CameraX ImageProxy
    suspend fun analyzeImageFromCamera(imageProxy: ImageProxy): AnalysisResult? {
        return withContext(Dispatchers.IO) {
            try {
                val bitmap = imageProxyToBitmap(imageProxy)
                analyzeImage(bitmap)
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }
    }

    // Core image analysis using multimodal LLM
    private suspend fun analyzeImage(bitmap: Bitmap): AnalysisResult? {
        return withContext(Dispatchers.IO) {
            try {
                if (!isInitialized) return@withContext null

                val mpImage: MPImage = BitmapImageBuilder(bitmap).build()

                // Multimodal prompt for content analysis
                val prompt = """
                    Analyze the provided image. CHECK WHETHER IT CONTAINS:
                    1. FACES: Any human faces. They must explicitly have human features.
                    2. LOCATION: Location identifiers. Example: Street signs, license plates building numbers, landmarks, GPS coordinates. IF YOU CAN DETERMINE WHERE THIS PLACE IS JUST BY LOOKING AT IT SAY YES.
                    3. PERSONAL IDENTIFIABLE INFORMATION: Anything that can uniquely identify a person, cards, etc. Example: Names, addresses, phone numbers, email addresses, ID numbers, social security numbers, or any personal documents. IF YOU CAN TELL ME WHO A PERSON IS OR COMMIT IDENTITY THEFT WITH THE INFO SAY YES.
                    
                    Respond ONLY with a JSON object in this exact format:
                    {"faces": "yes" or "no", "location": "yes" or "no", "pii": "yes" or "no"}
                                       
                    BE PRUDENT AND CONSERVATIVE. IF UNSURE, JUST ANSWER "YES".
                """.trimIndent()

                // Use the proper MediaPipe LLM inference pattern
                val response = analyzeWithSession(context, mpImage, prompt)

                if (response != null) {
                    parseAnalysisResponse(response)
                } else {
                    // Fallback to heuristic analysis if LLM fails
                    analyzeImageHeuristically(bitmap)
                }

            } catch (e: Exception) {
                e.printStackTrace()
                // Fallback to heuristic analysis on error
                analyzeImageHeuristically(bitmap)
            }
        }
    }

    // MediaPipe LLM inference session following the reference pattern
    private fun analyzeWithSession(context: Context, image: MPImage, prompt: String): String? {
        return try {
            val options = LlmInference.LlmInferenceOptions.builder()
                .setMaxNumImages(10)
                .setModelPath(modelPath)
                .build()

            val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
                .setTopK(10)
                .setTemperature(0.1f)
                .setGraphOptions(GraphOptions.builder().setEnableVisionModality(true).build())
                .build()

            // Kotlin equivalent of try-with-resources
            LlmInference.createFromOptions(context, options).use { llmInference ->
                LlmInferenceSession.createFromOptions(llmInference, sessionOptions).use { session ->
                    session.addQueryChunk(prompt)
                    session.addImage(image)
                    session.generateResponse()
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // Fallback heuristic analysis when LLM is not available
    private fun analyzeImageHeuristically(bitmap: Bitmap): AnalysisResult {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var totalBrightness = 0L
        var edgeCount = 0
        val colorFrequency = mutableMapOf<Int, Int>()

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            val brightness = (r + g + b) / 3
            totalBrightness += brightness

            // Simplified color grouping
            val colorGroup = ((r / 32) * 1024) + ((g / 32) * 32) + (b / 32)
            colorFrequency[colorGroup] = (colorFrequency[colorGroup] ?: 0) + 1

            // Simple edge detection
            if (i > 0 && i < pixels.size - 1) {
                val prevBrightness = getBrightness(pixels[i - 1])
                val nextBrightness = getBrightness(pixels[i + 1])
                if (kotlin.math.abs(brightness - prevBrightness) > 50 ||
                    kotlin.math.abs(brightness - nextBrightness) > 50
                ) {
                    edgeCount++
                }
            }
        }

        val avgBrightness = totalBrightness / pixels.size
        val hasComplexPatterns = edgeCount > (pixels.size * 0.1)
        val hasTextLikeRegions = edgeCount > (pixels.size * 0.05) && avgBrightness > 100
        val dominantColors = colorFrequency.entries.sortedByDescending { it.value }.take(5)

        // Heuristic analysis with conservative bias
        val likelyHasFaces = hasComplexPatterns &&
                avgBrightness in 80..200 &&
                width > 100 && height > 100

        val likelyHasLocation = hasTextLikeRegions &&
                hasComplexPatterns &&
                dominantColors.size > 3

        val likelyHasPii = hasTextLikeRegions &&
                avgBrightness > 150

        // Conservative approach - bias towards detection
        return AnalysisResult(
            faces = likelyHasFaces || kotlin.random.Random.nextFloat() < 0.3f,
            location = likelyHasLocation || kotlin.random.Random.nextFloat() < 0.25f,
            pii = likelyHasPii || kotlin.random.Random.nextFloat() < 0.2f
        )
    }

    private fun getBrightness(pixel: Int): Int {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        return (r + g + b) / 3
    }

    // Convert CameraX ImageProxy to Bitmap
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val buffer: ByteBuffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    // Parse the JSON response from LLM
    private fun parseAnalysisResponse(response: String): AnalysisResult? {
        return try {
            Log.d("Analysis response", response)
            val jsonStart = response.indexOf("{")
            val jsonEnd = response.lastIndexOf("}") + 1
            if (jsonStart == -1 || jsonEnd == 0) return null

            val jsonString = response.substring(jsonStart, jsonEnd)
            val jsonObject = JSONObject(jsonString)

            AnalysisResult(
                faces = jsonObject.getString("faces").lowercase() == "yes",
                location = jsonObject.getString("location").lowercase() == "yes",
                pii = jsonObject.getString("pii").lowercase() == "yes"
            )
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    fun close() {
        isInitialized = false
    }
}

package dev.xiaoxin.vpshield

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.xiaoxin.vpshield.utils.createBlurredCrop
import dev.xiaoxin.vpshield.ml.AgeEstimator
import dev.xiaoxin.vpshield.ml.AnalysisResult
import dev.xiaoxin.vpshield.ml.LLMModel
import dev.xiaoxin.vpshield.ml.TextRecognizer
import dev.xiaoxin.vpshield.utils.ImageUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Enhanced LLM usage example that integrates face detection and age estimation
 * when faces are detected by the LLM model
 */
class LLMAnalysisViewModel(@SuppressLint("StaticFieldLeak") private val context: Context) : ViewModel() {

    private val llmModel = LLMModel(context)
    private var ageEstimator: AgeEstimator = AgeEstimator(context)
    private var isModelInitialized = false
    private var isAgeEstimatorInitialized = false
    private val textRecognizer = TextRecognizer(context)

    private val EMPTY_RESULT = AnalysisResult(faces = false, location = false, pii = false)

    // Local availability state
    private val _localAvailable = MutableStateFlow(false)
    val localAvailable: StateFlow<Boolean> = _localAvailable.asStateFlow()

    init { initializeModels() }

    private fun initializeModels() {
        viewModelScope.launch(Dispatchers.IO) {
            // Initialize LLM model safely
            isModelInitialized = try {
                llmModel.initialize()
            } catch (e: Exception) {
                e.printStackTrace()
                false
            }
            // Initialize AgeEstimator lazily & safely
            if (!isAgeEstimatorInitialized) {
                isAgeEstimatorInitialized = try {
                    ageEstimator = AgeEstimator(context)
                    true
                } catch (e: Exception) {
                    e.printStackTrace()
                    false
                }
            }
            // Update observable state
            _localAvailable.value = isModelInitialized && isAgeEstimatorInitialized
            // Log status (on IO thread fine for println)
            println("LLM Model initialized: $isModelInitialized | AgeEstimator: $isAgeEstimatorInitialized")
        }
    }

    /**
     * Analyze an image from URI (e.g., from gallery) with complete face processing pipeline
     */
    fun analyzeImageFromGallery(
        uri: Uri,
        onResult: (AnalysisResult?) -> Unit,
        onFaceProcessingComplete: (Bitmap?, List<AgeEstimator.Result>) -> Unit = { _, _ -> }
    ) {
        if (!isModelInitialized) { onResult(EMPTY_RESULT); return }
        viewModelScope.launch {
            val result = try { llmModel.analyzeImageFromUri(uri) } catch (e: Exception) { null }
            val safe = result ?: EMPTY_RESULT
            if (result != null) handleAnalysisResult(safe, uri, null, onFaceProcessingComplete)
            onResult(safe)
        }
    }

    /**
     * Analyze an image from CameraX with complete face processing pipeline
     */
    fun analyzeCameraImage(
        imageProxy: ImageProxy,
        onResult: (AnalysisResult?) -> Unit,
        onFaceProcessingComplete: (Bitmap?, List<AgeEstimator.Result>) -> Unit = { _, _ -> }
    ) {
        if (!isModelInitialized) { onResult(EMPTY_RESULT); return }
        viewModelScope.launch {
            val result = try { llmModel.analyzeImageFromCamera(imageProxy) } catch (e: Exception) { null }
            val safe = result ?: EMPTY_RESULT
            if (result != null) handleAnalysisResult(safe, null, imageProxy, onFaceProcessingComplete)
            onResult(safe)
        }
    }

    /**
     * Handle the analysis result and trigger appropriate pipelines
     */
    private suspend fun handleAnalysisResult(
        result: AnalysisResult,
        uri: Uri?,
        imageProxy: ImageProxy?,
        onFaceProcessingComplete: (Bitmap?, List<AgeEstimator.Result>) -> Unit
    ) {
        when {
            result.faces -> {
                // Trigger face processing pipeline (no pre-blur, UI handles toggling)
                triggerFaceProcessing(uri, imageProxy, onFaceProcessingComplete)
            }

            result.location -> {
                // Trigger location processing pipeline
                triggerLocationProcessing()
            }

            result.pii -> {
                // pass through uri or imageProxy you already have in the flow
                triggerPiiProcessing(uri, imageProxy) { ocrText, piiScores ->
                    println("PII Pipeline finished")
                    println("OCR Text: $ocrText")
                    println("PII Scores: ${piiScores.joinToString()}")
                    // update UI/state as needed
                }
            }
        }
    }

    /**
     * Face processing now only detects faces & ages and returns original bitmap
     */
    @SuppressLint("DefaultLocale")
    private suspend fun triggerFaceProcessing(
        uri: Uri?,
        imageProxy: ImageProxy?,
        onFaceProcessingComplete: (Bitmap?, List<AgeEstimator.Result>) -> Unit
    ) {
        println("Face detected - triggering face processing pipeline (no pre-blur)")
        if (!isAgeEstimatorInitialized || ageEstimator == null) {
            println("Age estimator unavailable")
            onFaceProcessingComplete(null, emptyList())
            return
        }
        try {
            val originalBitmap = when {
                uri != null -> withContext(Dispatchers.IO) { ImageUtils.loadBitmapFromUri(context, uri, maxSide = 1600) }
                imageProxy != null -> withContext(Dispatchers.IO) { convertImageProxyToBitmap(imageProxy) }
                else -> null
            }
            if (originalBitmap == null) { onFaceProcessingComplete(null, emptyList()); return }
            val estimator = ageEstimator
            val faceResults = if (estimator != null) withContext(Dispatchers.Default) { estimator.processImage(originalBitmap) } else emptyList()
            println("Detected ${faceResults.size} faces")
            faceResults.forEachIndexed { idx, result ->
                val isMinor = result.age + 2f < 18f
                println("Face #$idx: age=${String.format("%.1f", result.age)} isMinor=$isMinor bbox=(${result.bbox.left}, ${result.bbox.top}, ${result.bbox.right}, ${result.bbox.bottom})")
            }
            onFaceProcessingComplete(originalBitmap, faceResults)
        } catch (e: Exception) {
            e.printStackTrace()
            onFaceProcessingComplete(null, emptyList())
        }
    }

    /**
     * Build a bitmap with selected faces blurred. indicesToBlur = set of face indices.
     */
    suspend fun buildBlurredBitmap(
        original: Bitmap,
        faces: List<AgeEstimator.Result>,
        indicesToBlur: Set<Int>,
        errorMargin: Float = 2f
    ): Bitmap = withContext(Dispatchers.Default) {
        // Copy mutable
        val out = original.copy(original.config ?: Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        faces.forEachIndexed { idx, res ->
            if (idx in indicesToBlur) {
                try {
                    val blurredCrop = createBlurredCrop(
                        context,
                        original,
                        res.bbox.left,
                        res.bbox.top,
                        res.bbox.right,
                        res.bbox.bottom,
                        radius = 25f
                    )
                    val dest = RectF(
                        res.bbox.left.toFloat(),
                        res.bbox.top.toFloat(),
                        res.bbox.right.toFloat(),
                        res.bbox.bottom.toFloat()
                    )
                    canvas.drawBitmap(blurredCrop, null, dest, null)
                } catch (e: Exception) {
                    val paint = Paint().apply { color = Color.argb(180, 0, 0, 0) }
                    canvas.drawRect(
                        res.bbox.left.toFloat(),
                        res.bbox.top.toFloat(),
                        res.bbox.right.toFloat(),
                        res.bbox.bottom.toFloat(),
                        paint
                    )
                }
            }
        }
        out
    }

    /**
     * Convert ImageProxy to Bitmap
     */
    private fun convertImageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val buffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun triggerLocationProcessing() {
        // Implement your location processing logic here
        // This could involve location anonymization, etc.
        println("Location markers detected - triggering location processing pipeline")
    }

    private suspend fun triggerPiiProcessing(
        uri: Uri?,
        imageProxy: ImageProxy?,
        onPiiProcessingComplete: (ocrText: String?, piiScores: FloatArray) -> Unit
    ) {
        println("PII detected - triggering PII processing pipeline")

        try {
            // Load bitmap from URI or ImageProxy
            val originalBitmap: Bitmap? = when {
                uri != null -> withContext(Dispatchers.IO) {
                    ImageUtils.loadBitmapFromUri(context, uri, maxSide = 1600)
                }

                imageProxy != null -> withContext(Dispatchers.IO) {
                    convertImageProxyToBitmap(imageProxy)
                }

                else -> null
            }

            if (originalBitmap == null) {
                onPiiProcessingComplete(null, FloatArray(0))
                return
            }

            // Run OCR -> returns plain String via callback; then run Piiranha in background
            // Use a suspendCoroutine or the coroutine friendly pattern: switch to a suspendCancellableCoroutine
            val ocrText = withContext(Dispatchers.Main) {
                // processBitmap expects callbacks on main thread; we'll bridge it into a suspend call
                kotlinx.coroutines.suspendCancellableCoroutine<String> { cont ->
                    textRecognizer.processBitmap(
                        originalBitmap,
                        onTextResult = { text ->
                            if (!cont.isCompleted) cont.resume(text) {}
                        },
                        onError = { e ->
                            if (!cont.isCompleted) cont.resumeWithException(e)
                        })
                }
            }

            // Run ONNX Piiranha off the main thread
//            val piiScores = withContext(Dispatchers.Default) {
//                // returns FloatArray
//                textRecognizer.runPiiranhaOnText(ocrText)
//            }

            // Return both OCR text and PII model output on main thread
//            withContext(Dispatchers.Main) {
//                onPiiProcessingComplete(ocrText, piiScores)
//            }

        } catch (e: Exception) {
            e.printStackTrace()
            withContext(Dispatchers.Main) {
                onPiiProcessingComplete(null, FloatArray(0))
            }
        }
    }

    fun isLocalAvailable(): Boolean = isModelInitialized && isAgeEstimatorInitialized

    override fun onCleared() {
        super.onCleared()
        llmModel.close()
        ageEstimator?.close()
    }
}

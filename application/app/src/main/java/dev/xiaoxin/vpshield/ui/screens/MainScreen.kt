package dev.xiaoxin.vpshield.ui.screens

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Security
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.graphics.createBitmap
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.xiaoxin.vpshield.LLMAnalysisViewModel
import dev.xiaoxin.vpshield.ml.AgeEstimator
import dev.xiaoxin.vpshield.ml.AnalysisResult
import dev.xiaoxin.vpshield.network.analyzeImageFile
import dev.xiaoxin.vpshield.utils.ImageUtils
import dev.xiaoxin.vpshield.utils.createBlurredCrop
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

// Accent color
private val AccentColor = Color(0xFFF5C076)

// Dark theme colors
private val DarkColorScheme = darkColorScheme(
    primary = AccentColor,
    onPrimary = Color.Black,
    secondary = AccentColor,
    onSecondary = Color.Black,
    background = Color(0xFF121212),
    onBackground = Color.White,
    surface = Color(0xFF1E1E1E),
    onSurface = Color.White
)

private enum class ProcessingMode { LOCAL, CLOUD }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val viewModel: LLMAnalysisViewModel = viewModel { LLMAnalysisViewModel(context) }

    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var displayBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var faceResults by remember { mutableStateOf<List<AgeEstimator.Result>>(emptyList()) }
    var facesToBlur by remember { mutableStateOf<Set<Int>>(emptySet()) }
    var blurredFaceCrops by remember { mutableStateOf<Map<Int, Bitmap>>(emptyMap()) }
    var analysisResult by remember { mutableStateOf<AnalysisResult?>(null) }
    var cloudDetectionUris by remember { mutableStateOf<Map<String, List<String>>>(emptyMap()) }
    var processingMode by remember { mutableStateOf(ProcessingMode.LOCAL) }
    var lastUsedMode by remember { mutableStateOf<ProcessingMode?>(null) }
    var statusMessage by remember { mutableStateOf<String?>(null) }
    var isAnalyzing by remember { mutableStateOf(false) }
    var exporting by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        selectedImageUri = uri
        originalBitmap = null
        displayBitmap = null
        faceResults = emptyList()
        facesToBlur = emptySet()
        blurredFaceCrops = emptyMap()
        analysisResult = null
        cloudDetectionUris = emptyMap()
        statusMessage = null
        if (uri != null) {
            scope.launch(Dispatchers.IO) {
                try {
                    val bmp = ImageUtils.loadBitmapFromUri(context, uri, 1600)
                    withContext(Dispatchers.Main) { originalBitmap = bmp }
                } catch (e: Exception) {
                    if (e is CancellationException) throw e
                    withContext(Dispatchers.Main) { statusMessage = "Failed to load image" }
                }
            }
        }
    }

    fun hasNetwork(): Boolean {
        val connectivityManager =
            context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val network = connectivityManager.activeNetwork ?: return false
            val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
                    capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) ||
                    capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)
        } else {
            @Suppress("DEPRECATION")
            connectivityManager.activeNetworkInfo?.isConnected == true
        }
    }

    fun emptyBitmap(): Bitmap {
        return createBitmap(1, 1)
    }

    fun saveBitmap(bmp: Bitmap, suffix: String) {
        scope.launch(Dispatchers.IO) {
            try {
                exporting = true
                val name = "privacy_${System.currentTimeMillis()}_${suffix}.png"
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, name)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                }
                context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    ?.let { uri ->
                        context.contentResolver.openOutputStream(uri).use { out ->
                            out?.let { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
                        }
                    }
                withContext(Dispatchers.Main) {
                    statusMessage = "Saved as $name"
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusMessage = "Export error: ${e.message}"
                }
            } finally {
                exporting = false
            }
        }
    }

    // Convert (x,y,w,h) bbox to RectF
    fun bboxToRectF(nums: List<Float>, imgW: Int, imgH: Int): RectF {
        val x = nums[0].coerceAtLeast(0f)
        val y = nums[1].coerceAtLeast(0f)
        val w = nums[2].coerceAtLeast(0f)
        val h = nums[3].coerceAtLeast(0f)
        val left = x.coerceAtLeast(0f)
        val top = y.coerceAtLeast(0f)
        val right = (x + w).coerceAtMost(imgW.toFloat())
        val bottom = (y + h).coerceAtMost(imgH.toFloat())
        return RectF(left, top, right, bottom)
    }

    // Server-equivalent adaptive Gaussian radius: k = max(51, (w//3)|1); radius = (k-1)/2 capped at 25
    fun adaptiveRadius(rect: Rect): Float {
        val w = rect.width().coerceAtLeast(1)
        val kRaw = maxOf(51, (w / 3))
        val k = if (kRaw % 2 == 0) kRaw + 1 else kRaw
        val radius = (k - 1) / 2f
        return radius.coerceAtMost(25f)
    }

    suspend fun runRemote(uri: Uri) {
        isAnalyzing = true
        statusMessage = "Cloud analyzing..."
        try {
            val file = withContext(Dispatchers.IO) {
                val f = File.createTempFile("upl_", ".img", context.cacheDir)
                context.contentResolver.openInputStream(uri)
                    ?.use { inp -> FileOutputStream(f).use { inp.copyTo(it) } }
                f
            }

            val (resp, fetchedBmp) = analyzeImageFile(file)

            if (resp == null) {
                analysisResult = AnalysisResult(false, false, false)
                displayBitmap = originalBitmap ?: emptyBitmap()
                cloudDetectionUris = emptyMap()
                statusMessage = "Cloud analysis failed"
                return
            }

            analysisResult = AnalysisResult(
                faces = resp.face != null,
                location = resp.location != null,
                pii = resp.pii != null
            )

            // Store URIs for cloud detections instead of processing bounding boxes
            val uriMap = mutableMapOf<String, List<String>>()

//            resp.face?.mask?.let { faceMasks ->
//                val faceUris = faceMasks.mapIndexed { index, mask ->
//                    "Face ${index + 1}: ${mask.coordinate} (${mask.reason})"
//                }
//                if (faceUris.isNotEmpty()) {
//                    uriMap["faces"] = faceUris
//                }
//            }
//
//            resp.location?.mask?.let { locationMasks ->
//                val locationUris = locationMasks.mapIndexed { index, mask ->
//                    "Location ${index + 1}: ${mask.coordinate} (${mask.reason})"
//                }
//                if (locationUris.isNotEmpty()) {
//                    uriMap["location"] = locationUris
//                }
//            }
//
//            resp.pii?.mask?.let { piiMasks ->
//                val piiUris = piiMasks.mapIndexed { index, mask ->
//                    "PII ${index + 1}: ${mask.coordinate} (${mask.reason})"
//                }
//                if (piiUris.isNotEmpty()) {
//                    uriMap["pii"] = piiUris
//                }
//            }

            cloudDetectionUris = uriMap

            // Just display the original image for cloud mode
            displayBitmap = fetchedBmp ?: originalBitmap ?: emptyBitmap()

            statusMessage = "Cloud done"
            lastUsedMode = ProcessingMode.CLOUD
        } catch (e: Exception) {
            analysisResult = analysisResult ?: AnalysisResult(false, false, false)
            displayBitmap = originalBitmap ?: emptyBitmap()
            cloudDetectionUris = emptyMap()
            statusMessage = "Cloud error: ${e.message}".take(60)
            lastUsedMode = ProcessingMode.CLOUD
        } finally {
            isAnalyzing = false
        }
    }

    fun runLocal(uri: Uri) {
        isAnalyzing = true
        statusMessage = "Local analyzing faces..."
        try {
            viewModel.analyzeImageFromGallery(
                uri,
                onResult = { res ->
                    val safe = res ?: AnalysisResult(false, false, false)
                    analysisResult = safe.copy(location = false, pii = false)
                    if (res != null && (res.location || res.pii)) {
                        if (hasNetwork()) {
                            statusMessage = "Sensitive content -> cloud..."
                            scope.launch { runRemote(uri) }
                        } else {
                            statusMessage = "Sensitive content offline; faces only."
                        }
                    } else {
                        lastUsedMode = ProcessingMode.LOCAL
                        statusMessage = "Local done"
                    }
                    isAnalyzing = false
                },
                onFaceProcessingComplete = { bmp, faces ->
                    originalBitmap = bmp ?: originalBitmap ?: emptyBitmap()
                    faceResults = faces
                    // Update analysisResult faces flag so count reflects actual detected faces
                    analysisResult = (analysisResult ?: AnalysisResult(
                        false,
                        false,
                        false
                    )).copy(faces = faces.isNotEmpty())
                    if (faces.isNotEmpty() && bmp != null) {
                        scope.launch(Dispatchers.Default) {
                            val map = mutableMapOf<Int, Bitmap>()
                            faces.forEachIndexed { i, f ->
                                try {
                                    // ensure bbox is normalized before feeding createBlurredCrop
                                    val bbox = f.bbox
                                    map[i] = createBlurredCrop(
                                        context,
                                        bmp,
                                        bbox.left,
                                        bbox.top,
                                        bbox.right,
                                        bbox.bottom,
                                        25f
                                    )
                                } catch (_: Exception) { /* ignore per-face errors */
                                }
                            }
                            blurredFaceCrops = map
                        }
                        facesToBlur =
                            faces.mapIndexedNotNull { i, f -> if (f.age + 2f < 18f) i else null }
                                .toSet()
                    }
                }
            )
        } catch (e: Exception) {
            analysisResult = analysisResult ?: AnalysisResult(false, false, false)
            originalBitmap = originalBitmap ?: emptyBitmap()
            statusMessage = "Local error: ${e.message}".take(60)
            isAnalyzing = false
            lastUsedMode = ProcessingMode.LOCAL
        }
    }

    val localAvailable by viewModel.localAvailable.collectAsState(initial = false)

    MaterialTheme(colorScheme = DarkColorScheme) {
        Scaffold(topBar = {
            TopAppBar(
                title = { Text("Visual Private Shield", fontWeight = FontWeight.SemiBold) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkColorScheme.primary,
                    titleContentColor = DarkColorScheme.onPrimary
                )
            )
        }) { pad ->
            Column(
                Modifier
                    .fillMaxSize()
                    .padding(pad)
                    .padding(16.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Mode toggle
                val net = hasNetwork()
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text("Mode:", color = Color.White, fontWeight = FontWeight.SemiBold)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(
                            selected = processingMode == ProcessingMode.LOCAL,
                            onClick = { if (localAvailable) processingMode = ProcessingMode.LOCAL },
                            enabled = localAvailable
                        )
                        Text("Local", color = if (localAvailable) Color.White else Color.Gray)
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        val cloudEnabled = net
                        RadioButton(
                            selected = processingMode == ProcessingMode.CLOUD,
                            onClick = { if (cloudEnabled) processingMode = ProcessingMode.CLOUD },
                            enabled = cloudEnabled
                        )
                        Text("Cloud", color = if (cloudEnabled) Color.White else Color.Gray)
                    }
                }
                statusMessage?.let { Text(it, color = AccentColor, fontSize = 12.sp) }

                // Image area
                Box(
                    Modifier
                        .fillMaxWidth()
                        .height(300.dp)
                        .background(DarkColorScheme.surface, RoundedCornerShape(16.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    when {
                        lastUsedMode == ProcessingMode.CLOUD && displayBitmap != null -> Image(
                            bitmap = displayBitmap!!.asImageBitmap(),
                            contentDescription = "Cloud processed image",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )

                        originalBitmap != null -> {
                            val bmp = originalBitmap!!
                            if (lastUsedMode != ProcessingMode.LOCAL || faceResults.isEmpty()) {
                                Image(
                                    bitmap = bmp.asImageBitmap(),
                                    contentDescription = "Original image",
                                    modifier = Modifier.fillMaxSize(),
                                    contentScale = ContentScale.Fit
                                )
                            } else {
                                Canvas(Modifier.fillMaxSize()) {
                                    val cw = size.width
                                    val ch = size.height
                                    val iw = bmp.width.toFloat()
                                    val ih = bmp.height.toFloat()
                                    val s = minOf(cw / iw, ch / ih)
                                    val dx = (cw - iw * s) / 2f
                                    val dy = (ch - ih * s) / 2f
                                    drawIntoCanvas {
                                        it.nativeCanvas.drawBitmap(
                                            bmp,
                                            null,
                                            RectF(dx, dy, dx + iw * s, dy + ih * s),
                                            null
                                        )
                                    }
                                    faceResults.forEachIndexed { idx, f ->
                                        val l = dx + f.bbox.left * s
                                        val t = dy + f.bbox.top * s
                                        val r = dx + f.bbox.right * s
                                        val b = dy + f.bbox.bottom * s
                                        if (facesToBlur.contains(idx)) {
                                            blurredFaceCrops[idx]?.let { crop ->
                                                drawIntoCanvas { c ->
                                                    c.nativeCanvas.drawBitmap(
                                                        crop,
                                                        null,
                                                        RectF(l, t, r, b),
                                                        null
                                                    )
                                                }
                                            }
                                        }
                                        drawIntoCanvas { c ->
                                            val p = Paint().apply {
                                                style = Paint.Style.STROKE
                                                strokeWidth = 3f
                                                color =
                                                    if (facesToBlur.contains(idx)) 0xFFFFC107.toInt() else 0x80FFFFFF.toInt()
                                            }
                                            c.nativeCanvas.drawRect(l, t, r, b, p)
                                        }
                                    }
                                }
                            }
                        }

                        selectedImageUri != null -> Text("Loading preview...", color = Color.Gray)
                        else -> Text("No image selected", color = Color.Gray)
                    }
                    if (isAnalyzing || exporting) Box(
                        Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(.55f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(color = AccentColor)
                            Text(
                                if (exporting) "Exporting..." else "Working...",
                                color = Color.White,
                                modifier = Modifier.padding(top = 8.dp)
                            )
                        }
                    }
                }

                Button(
                    onClick = { launcher.launch("image/*") },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = DarkColorScheme.primary)
                ) {
                    Text("Upload Image", color = DarkColorScheme.onPrimary)
                }

                if (selectedImageUri != null && !isAnalyzing) {
                    Button(
                        onClick = {
                            displayBitmap = null
                            cloudDetectionUris = emptyMap()
                            if (processingMode == ProcessingMode.CLOUD) {
                                if (!hasNetwork()) {
                                    statusMessage = "Offline -> local"
                                    processingMode = ProcessingMode.LOCAL
                                } else {
                                    scope.launch { runRemote(selectedImageUri!!) }
                                    return@Button
                                }
                            }
                            runLocal(selectedImageUri!!)
                        },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(10.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = DarkColorScheme.secondary)
                    ) {
                        Text(
                            "Analyze (${
                                processingMode.name.lowercase().replaceFirstChar { it.uppercase() }
                            })",
                            color = DarkColorScheme.onSecondary
                        )
                    }
                }

                analysisResult?.let { res ->
                    Card(
                        Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = DarkColorScheme.surface),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Column(
                            Modifier.padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Text(
                                "Analysis Results",
                                color = AccentColor,
                                fontWeight = FontWeight.Bold
                            )
                            AnalysisResultItem(
                                Icons.Default.Face,
                                "Faces Detected",
                                res.faces,
                                if (res.faces) "${faceResults.size} face(s)" else null
                            )
                            AnalysisResultItem(
                                Icons.Default.LocationOn,
                                "Location Info",
                                res.location
                            )
                            AnalysisResultItem(
                                Icons.Default.Security,
                                "Personal Info (PII)",
                                res.pii
                            )
                        }
                    }
                }

                // Cloud detection URIs display
                if (lastUsedMode == ProcessingMode.CLOUD && cloudDetectionUris.isNotEmpty()) {
                    Card(
                        Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = DarkColorScheme.surface),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Column(
                            Modifier.padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Text(
                                "Cloud Detection Details",
                                color = AccentColor,
                                fontWeight = FontWeight.Bold
                            )

                            cloudDetectionUris.forEach { (category, uris) ->
                                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                    Text(
                                        category.replaceFirstChar { it.uppercase() },
                                        color = Color.White,
                                        fontWeight = FontWeight.SemiBold,
                                        fontSize = 14.sp
                                    )
                                    uris.forEach { uri ->
                                        Text(
                                            "• $uri",
                                            color = Color.LightGray,
                                            fontSize = 12.sp,
                                            modifier = Modifier.padding(start = 8.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                if (lastUsedMode == ProcessingMode.LOCAL && originalBitmap != null && faceResults.isNotEmpty()) {
                    Column(
                        Modifier
                            .fillMaxWidth()
                            .background(DarkColorScheme.surface, RoundedCornerShape(12.dp))
                            .padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            "Face Blur Controls",
                            color = AccentColor,
                            fontWeight = FontWeight.Bold
                        )
                        faceResults.forEachIndexed { idx, f ->
                            val sel = facesToBlur.contains(idx)
                            val ageText = String.format(Locale.US, "%.1f", f.age)
                            Row(
                                Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    "Face ${idx + 1}: $ageText ${if (f.age + 2f < 18f) " (Minor)" else ""}",
                                    color = Color.White,
                                    fontSize = 14.sp
                                )
                                Switch(
                                    checked = sel,
                                    onCheckedChange = {
                                        facesToBlur =
                                            if (sel) facesToBlur - idx else facesToBlur + idx
                                    },
                                    colors = SwitchDefaults.colors(
                                        checkedThumbColor = Color.DarkGray,
                                        checkedTrackColor = AccentColor
                                    )
                                )
                            }
                        }
                        if (blurredFaceCrops.isNotEmpty()) {
                            Row(
                                Modifier.horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                faceResults.forEachIndexed { idx, f ->
                                    val w = f.bbox.width().coerceAtLeast(1)
                                    val h = f.bbox.height().coerceAtLeast(1)
                                    val orig = try {
                                        Bitmap.createBitmap(
                                            originalBitmap!!,
                                            f.bbox.left.coerceAtLeast(0),
                                            f.bbox.top.coerceAtLeast(0),
                                            w.coerceAtMost(originalBitmap!!.width - f.bbox.left),
                                            h.coerceAtMost(originalBitmap!!.height - f.bbox.top)
                                        ).asImageBitmap()
                                    } catch (_: Exception) {
                                        null
                                    }
                                    val blur = blurredFaceCrops[idx]?.asImageBitmap()
                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        Text(
                                            "Face ${idx + 1}",
                                            color = Color.LightGray,
                                            fontSize = 12.sp
                                        )
                                        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                            orig?.let {
                                                Image(
                                                    it,
                                                    contentDescription = null,
                                                    modifier = Modifier.size(56.dp)
                                                )
                                            }
                                            blur?.let {
                                                Image(
                                                    it,
                                                    contentDescription = null,
                                                    modifier = Modifier.size(56.dp)
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if ((originalBitmap != null || displayBitmap != null) && analysisResult != null) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        originalBitmap?.let {
                            Button(
                                onClick = { saveBitmap(it, "original") },
                                modifier = Modifier.weight(1f),
                                enabled = !exporting
                            ) { Text("Export Original") }
                        }
                        if (lastUsedMode == ProcessingMode.LOCAL && faceResults.isNotEmpty() && facesToBlur.isNotEmpty() && originalBitmap != null) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        val b = viewModel.buildBlurredBitmap(
                                            originalBitmap!!,
                                            faceResults,
                                            facesToBlur
                                        )
                                        saveBitmap(b, "local_blurred")
                                    }
                                },
                                modifier = Modifier.weight(1f),
                                enabled = !exporting
                            ) { Text("Export Current") }
                        }
                        if (lastUsedMode == ProcessingMode.CLOUD && displayBitmap != null) {
                            Button(
                                onClick = { displayBitmap?.let { saveBitmap(it, "cloud") } },
                                modifier = Modifier.weight(1f),
                                enabled = !exporting
                            ) { Text("Export Cloud") }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun AnalysisResultItem(
    icon: ImageVector,
    label: String,
    detected: Boolean,
    additionalInfo: String? = null
) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                icon,
                contentDescription = null,
                tint = if (detected) Color.Red else Color.Green,
                modifier = Modifier.size(20.dp)
            )
            Column(Modifier.padding(start = 8.dp)) {
                Text(label, color = Color.White, fontSize = 14.sp)
                additionalInfo?.let { Text(it, color = Color.Gray, fontSize = 12.sp) }
            }
        }
        Text(
            if (detected) "DETECTED" else "Clear",
            color = if (detected) Color.Red else Color.Green,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
    }
}
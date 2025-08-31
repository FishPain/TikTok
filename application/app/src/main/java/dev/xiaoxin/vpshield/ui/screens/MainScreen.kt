package dev.xiaoxin.tiktok_jam_2025.ui.screens

import android.content.ContentValues
import android.graphics.Bitmap
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
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import dev.xiaoxin.tiktok_jam_2025.LLMAnalysisViewModel
import dev.xiaoxin.tiktok_jam_2025.data.CoordinateInfo
import dev.xiaoxin.tiktok_jam_2025.ml.AgeEstimator
import dev.xiaoxin.tiktok_jam_2025.ml.AnalysisResult
import dev.xiaoxin.tiktok_jam_2025.network.analyzeImageFile
import dev.xiaoxin.tiktok_jam_2025.utils.ImageUtils
import dev.xiaoxin.tiktok_jam_2025.utils.createBlurredCrop
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

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
fun MainScreen(navController: NavController) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val viewModel: LLMAnalysisViewModel = viewModel { LLMAnalysisViewModel(context) }

    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var displayBitmap by remember { mutableStateOf<Bitmap?>(null) } // remote processed or replacement
    var faceResults by remember { mutableStateOf<List<AgeEstimator.Result>>(emptyList()) }
    var facesToBlur by remember { mutableStateOf<Set<Int>>(emptySet()) }
    var blurredFaceCrops by remember { mutableStateOf<Map<Int, Bitmap>>(emptyMap()) }
    var analysisResult by remember { mutableStateOf<AnalysisResult?>(null) }
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
        val cm = context.getSystemService(ConnectivityManager::class.java) ?: return false
        val net = cm.activeNetwork ?: return false
        val caps = cm.getNetworkCapabilities(net) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    fun saveBitmap(bmp: Bitmap, suffix: String) {
        scope.launch(Dispatchers.IO) {
            try {
                exporting = true
                val name = "privacy_${System.currentTimeMillis()}_${suffix}.png"
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, name)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) put(
                        MediaStore.Images.Media.RELATIVE_PATH,
                        "Pictures/PrivacyAnalyzer"
                    )
                }
                val uri = context.contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    values
                )
                uri?.let {
                    context.contentResolver.openOutputStream(it)
                        ?.use { os -> bmp.compress(Bitmap.CompressFormat.PNG, 100, os) }
                }
            } finally {
                exporting = false
            }
        }
    }

    fun emptyBitmap(): Bitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)

    fun parseCoord(c: CoordinateInfo): Rect {
        val nums = c.coordinate.split(',', ' ', ';').mapNotNull { it.trim().toIntOrNull() }
        return if (nums.size >= 4) Rect(nums[0], nums[1], nums[2], nums[3]) else Rect(0, 0, 0, 0)
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
                displayBitmap = displayBitmap ?: emptyBitmap()
                statusMessage = "Cloud failed (defaulted)"
                lastUsedMode = ProcessingMode.CLOUD
            } else {
                analysisResult = AnalysisResult(
                    resp.face != null,
                    !resp.location.isNullOrEmpty(),
                    resp.pii != null
                )
                originalBitmap = originalBitmap ?: ImageUtils.loadBitmapFromUri(context, uri, 1600)
                displayBitmap = if (!resp.location.isNullOrEmpty() && fetchedBmp != null) {
                    fetchedBmp
                } else {
                    val base = originalBitmap ?: emptyBitmap()
                    val rects =
                        (resp.face?.mask.orEmpty() + resp.pii?.mask.orEmpty()).map { parseCoord(it) }
                    base.copy(base.config ?: Bitmap.Config.ARGB_8888, true).also { out ->
                        val canvas = android.graphics.Canvas(out)
                        rects.forEach { r ->
                            if (!r.isEmpty) try {
                                val crop = createBlurredCrop(
                                    context,
                                    base,
                                    r.left,
                                    r.top,
                                    r.right,
                                    r.bottom,
                                    25f
                                )
                                canvas.drawBitmap(crop, null, android.graphics.RectF(r), null)
                            } catch (_: Exception) {
                            }
                        }
                    }
                }
                statusMessage = "Cloud done"
                lastUsedMode = ProcessingMode.CLOUD
            }
        } catch (e: Exception) {
            analysisResult = analysisResult ?: AnalysisResult(false, false, false)
            displayBitmap = displayBitmap ?: emptyBitmap()
            statusMessage = "Cloud error (defaulted): ${e.message}".take(60)
            lastUsedMode = ProcessingMode.CLOUD
        } finally {
            isAnalyzing = false
        }
    }

    suspend fun runLocal(uri: Uri) {
        isAnalyzing = true
        statusMessage = "Local analyzing faces..."
        try {
            viewModel.analyzeImageFromGallery(
                uri,
                onResult = { res ->
                    val safe = res ?: AnalysisResult(faces = false, location = false, pii = false)
                    analysisResult = safe.copy(location = false, pii = false)
                    if (res != null && (res.location || res.pii)) {
                        if (hasNetwork()) {
                            statusMessage = "Sensitive content -> switching to cloud..."
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
                    if (!faces.isNullOrEmpty() && bmp != null) {
                        scope.launch(Dispatchers.Default) {
                            val map = mutableMapOf<Int, Bitmap>()
                            faces.forEachIndexed { i, f ->
                                try {
                                    map[i] = createBlurredCrop(
                                        context,
                                        bmp,
                                        f.bbox.left,
                                        f.bbox.top,
                                        f.bbox.right,
                                        f.bbox.bottom,
                                        25f
                                    )
                                } catch (_: Exception) {
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
            statusMessage = "Local error (defaulted): ${e.message}".take(60)
            isAnalyzing = false
            lastUsedMode = ProcessingMode.LOCAL
        }
    }

    // Observe localAvailable as Compose state
    val localAvailable by viewModel.localAvailable.collectAsState()

    MaterialTheme(colorScheme = DarkColorScheme) {
        Scaffold(topBar = {
            TopAppBar(
                title = { Text("Smart Privacy Analyzer") },
                colors = TopAppBarDefaults.topAppBarColors(
                    DarkColorScheme.primary,
                    DarkColorScheme.onPrimary
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
                // Mode Toggle
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
                        lastUsedMode == ProcessingMode.CLOUD && displayBitmap != null -> androidx.compose.foundation.Image(
                            displayBitmap!!.asImageBitmap(),
                            null,
                            Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )

                        originalBitmap != null -> {
                            val bmp = originalBitmap!!
                            if (lastUsedMode != ProcessingMode.LOCAL || faceResults.isEmpty()) {
                                androidx.compose.foundation.Image(
                                    bmp.asImageBitmap(),
                                    null,
                                    Modifier.fillMaxSize(),
                                    contentScale = ContentScale.Fit
                                )
                            } else {
                                Canvas(Modifier.fillMaxSize()) {
                                    val cw = size.width;
                                    val ch = size.height
                                    val iw = bmp.width.toFloat();
                                    val ih = bmp.height.toFloat()
                                    val s = minOf(cw / iw, ch / ih)
                                    val dx = (cw - iw * s) / 2f;
                                    val dy = (ch - ih * s) / 2f
                                    drawIntoCanvas {
                                        it.nativeCanvas.drawBitmap(
                                            bmp,
                                            null,
                                            android.graphics.RectF(
                                                dx,
                                                dy,
                                                dx + iw * s,
                                                dy + ih * s
                                            ),
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
                                            val p = android.graphics.Paint().apply {
                                                style =
                                                    android.graphics.Paint.Style.STROKE; strokeWidth =
                                                3f; color =
                                                if (facesToBlur.contains(idx)) 0xFFFFC107.toInt() else 0x80FFFFFF.toInt()
                                            }; c.nativeCanvas.drawRect(l, t, r, b, p)
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
                            CircularProgressIndicator(
                                color = AccentColor
                            ); Text(
                            if (exporting) "Exporting..." else "Working...",
                            color = Color.White,
                            modifier = Modifier.padding(top = 8.dp)
                        )
                        }
                    }
                }

                // Upload button
                Button(
                    onClick = { launcher.launch("image/*") },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.buttonColors(DarkColorScheme.primary)
                ) { Text("Upload Image", color = DarkColorScheme.onPrimary) }

                // Analyze button
                if (selectedImageUri != null && !isAnalyzing) {
                    Button(
                        onClick = {
                            displayBitmap = null
                            if (processingMode == ProcessingMode.CLOUD) {
                                if (!hasNetwork()) {
                                    statusMessage = "Offline -> local"; processingMode =
                                        ProcessingMode.LOCAL
                                } else {
                                    scope.launch { runRemote(selectedImageUri!!) }; return@Button
                                }
                            }
                            scope.launch { runLocal(selectedImageUri!!) }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(10.dp),
                        colors = ButtonDefaults.buttonColors(DarkColorScheme.secondary)
                    ) {
                        Text(
                            "Analyze (${
                                processingMode.name.lowercase().replaceFirstChar { it.uppercase() }
                            })",
                            color = DarkColorScheme.onSecondary
                        )
                    }
                }

                // Analysis summary (always show after analysis)
                analysisResult?.let { res ->
                    Card(
                        Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(DarkColorScheme.surface),
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
                                if (res.faces && faceResults.isNotEmpty()) "${faceResults.size} face(s)" else null
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

                // Local face blur controls (only show when in local mode and faces detected)
                if (lastUsedMode == ProcessingMode.LOCAL && originalBitmap != null && faceResults.isNotEmpty()) {
                    Column(
                        Modifier
                            .fillMaxWidth()
                            .background(DarkColorScheme.surface, RoundedCornerShape(12.dp))
                            .padding(12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            "Face Blur Controls",
                            color = AccentColor,
                            fontWeight = FontWeight.Bold
                        )
                        faceResults.forEachIndexed { idx, f ->
                            val sel = facesToBlur.contains(idx); Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                "Face ${idx + 1}: ${
                                    String.format(
                                        "%.1f",
                                        f.age
                                    )
                                }${if (f.age + 2f < 18f) " (Minor)" else ""}",
                                color = Color.White,
                                fontSize = 14.sp
                            ); Switch(
                            checked = sel,
                            onCheckedChange = {
                                facesToBlur = if (sel) facesToBlur - idx else facesToBlur + idx
                            },
                            colors = SwitchDefaults.colors(checkedThumbColor = Color.DarkGray, checkedTrackColor = AccentColor)
                        )
                        }
                        }
                        if (blurredFaceCrops.isNotEmpty()) {
                            Row(
                                Modifier.horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                faceResults.forEachIndexed { idx, f ->
                                    val w = f.bbox.width().coerceAtLeast(1);
                                    val h = f.bbox.height().coerceAtLeast(1);
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
                                    };
                                    val blur = blurredFaceCrops[idx]?.asImageBitmap(); Column(
                                    horizontalAlignment = Alignment.CenterHorizontally
                                ) {
                                    Text(
                                        "Face ${idx + 1}",
                                        color = Color.LightGray,
                                        fontSize = 12.sp
                                    ); Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                    orig?.let {
                                        androidx.compose.foundation.Image(
                                            it,
                                            null,
                                            Modifier.size(56.dp)
                                        )
                                    }; blur?.let {
                                    androidx.compose.foundation.Image(
                                        it,
                                        null,
                                        Modifier.size(56.dp)
                                    )
                                }
                                }
                                }
                                }
                            }
                        }
                    }
                }

                // Export buttons (always show after analysis)
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
                                        ); saveBitmap(b, "local_blurred")
                                    }
                                },
                                modifier = Modifier.weight(1f),
                                enabled = !exporting
                            ) { Text("Export Current") }
                        }
                        if (lastUsedMode == ProcessingMode.CLOUD && displayBitmap != null) {
                            Button(
                                onClick = {
                                    displayBitmap?.let {
                                        saveBitmap(
                                            it,
                                            "cloud"
                                        )
                                    }
                                },
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
                null,
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
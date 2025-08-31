// MainComposeFaceBlur.kt (single file simplified)
import android.graphics.Bitmap
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import dev.xiaoxin.tiktok_jam_2025.ml.AgeEstimator
import dev.xiaoxin.tiktok_jam_2025.utils.ImageUtils
import dev.xiaoxin.tiktok_jam_2025.utils.createBlurredCrop
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@OptIn(DelicateCoroutinesApi::class)
@Composable
fun FaceBlurScreen() {
    val ctx = LocalContext.current
    var originalBmp by remember { mutableStateOf<Bitmap?>(null) }
    var results by remember { mutableStateOf<List<AgeEstimator.Result>>(emptyList()) }
    var blurredCrops by remember { mutableStateOf<Map<Int, Bitmap>>(emptyMap()) }
    val estimator = remember { AgeEstimator(ctx) }

    val errorMargin = 2f // Should match AgeEstimator's errorMargin
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            GlobalScope.launch(Dispatchers.Main) {
                val bmp = withContext(Dispatchers.IO) {
                    ImageUtils.loadBitmapFromUri(
                        ctx,
                        uri,
                        maxSide = 1600
                    )
                }
                originalBmp = bmp
                val detResults = withContext(Dispatchers.Default) { estimator.processImage(bmp) }
                results = detResults
                if (Build.VERSION.SDK_INT >= 31) {
                    val map = mutableMapOf<Int, Bitmap>()
                    for ((i, res) in detResults.withIndex()) {
                        if (res.age + errorMargin < 18f) {
                            val b = createBlurredCrop(
                                ctx,
                                bmp,
                                res.bbox.left,
                                res.bbox.top,
                                res.bbox.right,
                                res.bbox.bottom,
                                radius = 25f
                            )
                            map[i] = b
                        }
                    }
                    blurredCrops = map
                } else {
                    blurredCrops = emptyMap()
                }
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(8.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Button(onClick = { launcher.launch("image/*") }) { Text("Pick image") }
            Spacer(Modifier.width(12.dp))
            Text("Faces: ${results.size}")
        }
        Spacer(Modifier.height(8.dp))

        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            originalBmp?.let { bmp ->
                val imageBitmap = bmp.asImageBitmap()
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val canvasW = size.width
                    val canvasH = size.height
                    val imgW = bmp.width.toFloat()
                    val imgH = bmp.height.toFloat()
                    val scale = minOf(canvasW / imgW, canvasH / imgH)
                    val leftOffset = (canvasW - imgW * scale) / 2f
                    val topOffset = (canvasH - imgH * scale) / 2f

                    drawIntoCanvas {
                        it.nativeCanvas.drawBitmap(
                            bmp,
                            null,
                            android.graphics.RectF(
                                leftOffset,
                                topOffset,
                                leftOffset + imgW * scale,
                                topOffset + imgH * scale
                            ),
                            null
                        )
                    }

                    for ((i, res) in results.withIndex()) {
                        val box = res.bbox
                        val l = leftOffset + box.left * scale
                        val t = topOffset + box.top * scale
                        val r = leftOffset + box.right * scale
                        val b = topOffset + box.bottom * scale
                        if (Build.VERSION.SDK_INT >= 31 && res.age + errorMargin < 18f) {
                            blurredCrops[i]?.let { cropBmp ->
                                drawIntoCanvas {
                                    it.nativeCanvas.drawBitmap(
                                        cropBmp,
                                        null,
                                        android.graphics.RectF(l, t, r, b),
                                        null
                                    )
                                }
                            }
                        } else if (Build.VERSION.SDK_INT < 31 && res.age + errorMargin < 18f) {
                            drawIntoCanvas {
                                val paint = android.graphics.Paint().apply {
                                    color = android.graphics.Color.argb(180, 0, 0, 0)
                                }
                                it.nativeCanvas.drawRect(l, t, r, b, paint)
                            }
                        }
                    }
                }
            } ?: Text("No image selected")
        }

        Spacer(Modifier.height(8.dp))
        Column {
            results.forEachIndexed { idx, res ->
                Text(
                    "Face #$idx: (${res.bbox.left}, ${res.bbox.top}) -> (${res.bbox.right}, ${res.bbox.bottom}) age=%.1f mask=%s".format(
                        res.age,
                        res.mask
                    )
                )
            }
        }
    }


    DisposableEffect(Unit) {
        onDispose { estimator.close() }
    }
}

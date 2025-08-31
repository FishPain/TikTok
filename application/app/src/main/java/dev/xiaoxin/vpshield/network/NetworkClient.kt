package dev.xiaoxin.vpshield.network

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import dev.xiaoxin.vpshield.data.ApiResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File

suspend fun analyzeImageFile(file: File): Pair<ApiResponse?, Bitmap?> {
    val requestFile = file.asRequestBody("image/*".toMediaTypeOrNull())
    val part = MultipartBody.Part.createFormData("file", file.name, requestFile)

    val response = ApiClient.apiService.analyzeImage(part)
    val apiResponse = if (response.isSuccessful) response.body() else null

    var fetchedBitmap: Bitmap? = null

    // If location URL exists, fetch the image from the S3 bucket
    if (!apiResponse?.location.isNullOrEmpty()) {
        fetchedBitmap = fetchBitmapFromUrl(apiResponse.location)
    } else if (!apiResponse?.face.isNullOrEmpty()) {
        fetchedBitmap = fetchBitmapFromUrl(apiResponse.face)
    } else if (!apiResponse?.pii.isNullOrEmpty()) {
        fetchedBitmap = fetchBitmapFromUrl(apiResponse.pii)
    }

    return Pair(apiResponse, fetchedBitmap)
}

suspend fun fetchBitmapFromUrl(url: String): Bitmap? = withContext(Dispatchers.IO) {
    return@withContext try {
        val client = OkHttpClient()
        val request = Request.Builder().url(url).build()
        val response = client.newCall(request).execute()
        if (response.isSuccessful) {
            response.body?.byteStream().use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        } else {
            null
        }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

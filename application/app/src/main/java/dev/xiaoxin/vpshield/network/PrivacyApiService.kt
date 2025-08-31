package dev.xiaoxin.tiktok_jam_2025.network

import dev.xiaoxin.tiktok_jam_2025.data.ApiResponse
import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface PrivacyApiService {
    @Multipart
    @POST("/analyze")
    suspend fun analyzeImage(
        @Part file: MultipartBody.Part
    ): Response<ApiResponse>
}

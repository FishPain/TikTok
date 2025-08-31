package dev.xiaoxin.vpshield.network

import dev.xiaoxin.vpshield.data.ApiResponse
import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface PrivacyApiService {
    @Multipart
    @POST("/api/mask/all")
    suspend fun analyzeImage(
        @Part file: MultipartBody.Part
    ): Response<ApiResponse>
}

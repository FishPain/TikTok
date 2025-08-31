package dev.xiaoxin.tiktok_jam_2025.network

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object ApiClient {
    private val logger = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }
    private val client = OkHttpClient.Builder()
        .addInterceptor(logger)
        .build()

    val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl("https://ec2-13-250-59-59.ap-southeast-1.compute.amazonaws.com:8000/")  // Replace with actual base URL
        .client(client)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val apiService: PrivacyApiService = retrofit.create(PrivacyApiService::class.java)
}

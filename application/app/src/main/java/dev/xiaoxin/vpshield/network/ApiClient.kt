package dev.xiaoxin.vpshield.network

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object ApiClient {
    private val logger = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }
    private val client = OkHttpClient.Builder()
        .connectTimeout(180, java.util.concurrent.TimeUnit.SECONDS) // connection timeout
        .readTimeout(180, java.util.concurrent.TimeUnit.SECONDS)    // read timeout
        .writeTimeout(180, java.util.concurrent.TimeUnit.SECONDS)   // write timeout
        .addInterceptor(logger)
        .addInterceptor { chain ->
            val original = chain.request()
            val requestWithApiKey = original.newBuilder()
                .header("x-api-key", "IAMASECRET")
                .build()
            chain.proceed(requestWithApiKey)
        }
        .build()

    val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl("http://ec2-18-136-120-44.ap-southeast-1.compute.amazonaws.com:8000/")  // Replace with actual base URL
        .client(client)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val apiService: PrivacyApiService = retrofit.create(PrivacyApiService::class.java)
}

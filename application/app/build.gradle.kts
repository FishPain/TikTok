plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "dev.xiaoxin.vpshield"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.xiaoxin.vpshield"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
        }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    packaging {
        resources {
            pickFirsts += listOf(
                "lib/armeabi-v7a/libmediapipe_tasks_vision_jni.so",
                "lib/arm64-v8a/libmediapipe_tasks_vision_jni.so",
                "lib/armeabi-v7a/libmediapipe_jni.so",
                "lib/arm64-v8a/libmediapipe_jni.so"
            )
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
        mlModelBinding = true
    }
}

dependencies {

    // MLKit Text Recognition v2 imports for PII function, only doing English for now
    implementation(libs.text.recognition)
//    implementation("com.google.mlkit:text-recognition-chinese:16.0.1")
//    implementation("com.google.mlkit:text-recognition-devanagari:16.0.1")
//    implementation("com.google.mlkit:text-recognition-japanese:16.0.1")
//    implementation("com.google.mlkit:text-recognition-korean:16.0.1")

    // MediaPipe Face Detection for face localization
    implementation(libs.tasks.vision)
    implementation(libs.androidx.exifinterface)

    // ONNX Runtime for Piiranha PII detector
    implementation(libs.onnxruntime.android)

    // MediaPipe GenAI library to use the Gemma LLM
    implementation("com.google.mediapipe:tasks-genai:0.10.27")

    // Retrofit for remote endpoints
    implementation(libs.retrofit)
    implementation(libs.converter.gson)
    //noinspection UseTomlInstead
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    //noinspection UseTomlInstead
    implementation("io.coil-kt.coil3:coil-compose:3.0.4")
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.material.icons.extended)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.camera.core)
    implementation(libs.play.services.mlkit.face.detection)
    implementation(libs.litert.support.api)
    implementation(libs.litert)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}
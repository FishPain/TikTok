package dev.xiaoxin.tiktok_jam_2025

import FaceBlurScreen
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dev.xiaoxin.tiktok_jam_2025.ui.theme.Tiktok_jam_2025Theme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Tiktok_jam_2025Theme {
                TikTokJamApplication()
            }
        }
    }
}

@Composable
fun TikTokJamApplication() {
    val navController = rememberNavController()
    NavHost(
        navController = navController,
        startDestination = "splash"
    ) {
        composable("splash") {
            FaceBlurScreen()
        }
//        composable("home") {
//            HomeScreen()
//        }
//        composable("settings") {
//            SettingScreen()
//        }
//        composable("") {  }
    }
}
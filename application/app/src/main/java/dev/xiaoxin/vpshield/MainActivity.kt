package dev.xiaoxin.vpshield

import FaceBlurScreen
import dev.xiaoxin.vpshield.ui.screens.MainScreen
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dev.xiaoxin.vpshield.ui.theme.VPShieldTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            VPShieldTheme {
                VPShieldApplication()
            }
        }
    }
}

@Composable
fun VPShieldApplication() {
    val navController = rememberNavController()
    NavHost(
        navController = navController,
        startDestination = "main"
    ) {
        composable("splash") {
            FaceBlurScreen()
        }
        composable("main") {
            MainScreen()
        }
    }
}
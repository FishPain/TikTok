package dev.xiaoxin.vpshield.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.LongBuffer
import java.nio.charset.StandardCharsets

class PIIDetector(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        try {
            val modelBytes = context.assets.open("piiranha.onnx").readBytes()
            session = env.createSession(modelBytes)
        } catch (e: Exception) {
            Log.e("PiiranhaModel", "Failed to load ONNX model", e)
            throw e
        }
    }

    fun runOnText(inputText: String): FloatArray {
        try {
            val tokenIds = encodeTextToLongArray(inputText)
            val tensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(tokenIds),
                longArrayOf(1, tokenIds.size.toLong())
            )
            val result = session.run(mapOf(session.inputNames.iterator().next() to tensor))
            val output = result[0].value
            if (output is Array<*>) {
                val floatOutput = output.map {
                    (it as? FloatArray) ?: FloatArray(0)
                }.flatMap { it.toList() }.toFloatArray()
                return floatOutput
            } else {
                // handle unexpected type
                return FloatArray(0)
            }
        } catch (e: Exception) {
            Log.e("PiiranhaModel", "ONNX inference failed", e)
            return FloatArray(0)
        }
    }

    private fun encodeTextToLongArray(text: String): LongArray {
        return text.toByteArray(StandardCharsets.UTF_8).map { it.toLong() }.toLongArray()
    }

    fun close() {
        session.close()
        env.close()
    }
}

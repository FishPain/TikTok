package dev.xiaoxin.tiktok_jam_2025.data

data class ApiResponse(
    val face: FaceInfo?,
    val location: String?,
    val pii: PiiInfo?
)

data class FaceInfo(
    val mask: List<CoordinateInfo>
)

data class PiiInfo(
    val mask: List<CoordinateInfo>
)

data class CoordinateInfo(
    val coordinate: String,
    val reason: String
)

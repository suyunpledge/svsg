"""SVSG V5.2.1 全局枚举定义。

所有跨层（L1 / L1.5 / L2 / L3）协议共享本模块中的枚举，
各层禁止自行定义重复语义的字符串字面量，保证契约唯一来源。
"""

from enum import StrEnum, unique


@unique
class ConfLevel(StrEnum):
    """实例置信度分档。

    映射阈值（classification_entropy / local_edge_sharpness 等）为
    V5.2.1 初始默认值，Phase 2 标定后调整，不影响本枚举定义。
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    # 超时降级（S5 静默降级）与人工复审（S6）强制使用的最低档位
    LOWEST = "lowest"


@unique
class SceneType(StrEnum):
    """图像场景几何类型（IR 顶层 scene_type 字段）。"""

    ORTHOGRAPHIC = "orthographic"  # 正视/俯视，无显著透视畸变
    PERSPECTIVE = "perspective"    # 透视场景：未提供相机内参时禁用距离类关系
    UNKNOWN = "unknown"            # 未知：按 PERSPECTIVE 保守处理


@unique
class RelationType(StrEnum):
    """几何关系类型（L1 几何关系计算器输出）。"""

    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    NEAREST_TO = "nearest_to"
    DISTANCE_TO = "distance_to"


#: 方向性关系：仅依赖 bbox 坐标序，透视场景下仍然可靠
DIRECTIONAL_RELATIONS: frozenset[RelationType] = frozenset(
    {
        RelationType.LEFT_OF,
        RelationType.RIGHT_OF,
        RelationType.ABOVE,
        RelationType.BELOW,
        RelationType.CONTAINS,
        RelationType.OVERLAPS,
    }
)

#: 距离类关系：像素到物理距离的映射依赖场景标定，透视未标定时禁止输出
DISTANCE_RELATIONS: frozenset[RelationType] = frozenset(
    {RelationType.NEAREST_TO, RelationType.DISTANCE_TO}
)


@unique
class MetricType(StrEnum):
    """关系度量单位。"""

    IMAGE_PLANE_PX = "image_plane_px"  # 图像平面像素距离（非物理距离）
    PHYSICAL_UNIT = "physical_unit"    # 经深度校正的物理距离（需相机内参）


@unique
class IntentType(StrEnum):
    """L2 意图路由器的 7 类意图。"""

    OBJECT_DETECTION = "OBJECT_DETECTION"
    COUNT = "COUNT"
    ATTRIBUTE_QUERY = "ATTRIBUTE_QUERY"
    SPATIAL_RELATION = "SPATIAL_RELATION"
    TEMPORAL_RELATION = "TEMPORAL_RELATION"
    EXISTENCE_CHECK = "EXISTENCE_CHECK"
    REGION_QUERY = "REGION_QUERY"


@unique
class StateCode(StrEnum):
    """L2 有限状态机状态码（S0~S6，含 S4a）。"""

    S0 = "S0"    # Received：收到 L1 IR，进入校验管道
    S1 = "S1"    # Valid：坐标合法、类型存在，送入 L3 推理
    S2 = "S2"    # Invalid Parameter：坐标越界、字段缺失（硬拒绝）
    S3 = "S3"    # Missing Dependency：引用未知 ID（软重试，最多 1 次）
    S4A = "S4a"  # Under Verification：调用 L1.5 视觉验证服务
    S4 = "S4"    # Semantic Conflict：Detector 与验证结果不一致
    S5 = "S5"    # Timeout：L1/L1.5/L3 响应超时
    S6 = "S6"    # Human Review Required：转人工队列


@unique
class VerificationStatus(StrEnum):
    """L1.5 视觉验证服务的执行状态。"""

    SUCCESS = "success"
    TIMEOUT = "timeout"      # L1.5 超时：静默降级，conf_level -> lowest，转 S6
    DEGRADED = "degraded"    # 服务降级运行（如验证率降至 10% 时）
    ERROR = "error"          # 服务内部错误，按超时同路径处理

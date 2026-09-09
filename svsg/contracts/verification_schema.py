"""L1.5 视觉验证服务与 L3 推理引擎的数据契约。

包含：
- VerificationReport / VerificationResult：L1.5 输出的结构化验证报告
  （structured_report），是 L2 证据锚定校验的比对基准。
- Claim / L3Answer：L3 的双通道输出。claims 为结构化断言，供锚定校验
  做确定性 JSON 字段比对（避免对自然语言做模糊解析）；final_answer
  仅面向人类阅读，不参与校验。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import ConfLevel, VerificationStatus

_STRICT = ConfigDict(populate_by_name=True, extra="forbid")

#: Claim.value 的合法类型（保持宽松：语义级比对由 L2 锚定校验器负责）
ClaimValue = str | int | float | bool | None


class VerificationResult(BaseModel):
    """单个实例的验证结果。"""

    model_config = _STRICT

    instance_id: int = Field(..., ge=1)
    # 细粒度分类结论；无法可靠判定时为 "ambiguous"——
    # 此时 L3 若断言具体类别，将被证据锚定校验无条件否决（触发 S4）
    class_label: str | None = Field(default=None, alias="class")
    # 属性描述，如 {"color": "silver", "head_type": "phillips"}
    attributes: dict[str, str] = Field(default_factory=dict)
    # 存在性验证结论；None 表示本次未验证存在性
    exists: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None


class VerificationReport(BaseModel):
    """L1.5 结构化验证报告（structured_report）。

    status=timeout/error 时 results 可能为空：此时 L2 走静默降级路径
    （conf_level -> lowest，直接转 S6，不经过 S4a 重试），本报告仅作
    审计留痕，不作为锚定校验的比对基准。
    """

    model_config = _STRICT

    report_id: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)
    status: VerificationStatus
    results: list[VerificationResult] = Field(default_factory=list)
    elapsed_ms: float | None = Field(default=None, ge=0.0)


class Claim(BaseModel):
    """L3 最终答案中的单条核心断言。"""

    model_config = _STRICT

    # 断言关联的实例；全局性断言（如总数）可为 None
    instance_id: int | None = Field(default=None, ge=1)
    # 断言字段：class / count / exists / attribute:<name> / relation:<type> ...
    field: str = Field(..., min_length=1)
    value: ClaimValue = None
    confidence: ConfLevel | None = None


class L3Answer(BaseModel):
    """L3 双通道输出。"""

    model_config = _STRICT

    claims: list[Claim] = Field(default_factory=list)
    final_answer: str = Field(..., min_length=1)

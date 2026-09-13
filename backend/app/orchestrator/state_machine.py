"""
Explicit state machine definition for the multi-agent pipeline.
"""
from enum import Enum
from typing import Dict, Set


class PipelineStage(str, Enum):
    UPLOADED = "UPLOADED"
    PROFILED = "PROFILED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    PLAN_READY = "PLAN_READY"
    PREPARING = "PREPARING"
    ANALYZING = "ANALYZING"
    VALIDATING = "VALIDATING"
    NEEDS_REPAIR = "NEEDS_REPAIR"
    VALIDATED = "VALIDATED"
    BUILDING_DASHBOARD = "BUILDING_DASHBOARD"
    READY = "READY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Valid deterministic transitions
ALLOWED_TRANSITIONS: Dict[PipelineStage, Set[PipelineStage]] = {
    PipelineStage.UPLOADED: {PipelineStage.PROFILED, PipelineStage.FAILED, PipelineStage.CANCELLED},
    PipelineStage.PROFILED: {PipelineStage.NEEDS_CLARIFICATION, PipelineStage.PLAN_READY, PipelineStage.FAILED, PipelineStage.CANCELLED},
    PipelineStage.NEEDS_CLARIFICATION: {PipelineStage.PLAN_READY, PipelineStage.CANCELLED, PipelineStage.FAILED},
    PipelineStage.PLAN_READY: {PipelineStage.PREPARING, PipelineStage.CANCELLED, PipelineStage.FAILED},
    PipelineStage.PREPARING: {PipelineStage.ANALYZING, PipelineStage.FAILED, PipelineStage.CANCELLED},
    PipelineStage.ANALYZING: {PipelineStage.VALIDATING, PipelineStage.FAILED, PipelineStage.CANCELLED},
    PipelineStage.VALIDATING: {PipelineStage.VALIDATED, PipelineStage.NEEDS_REPAIR, PipelineStage.FAILED, PipelineStage.CANCELLED},
    PipelineStage.NEEDS_REPAIR: {PipelineStage.PREPARING, PipelineStage.ANALYZING, PipelineStage.FAILED, PipelineStage.CANCELLED},
    PipelineStage.VALIDATED: {PipelineStage.BUILDING_DASHBOARD, PipelineStage.FAILED, PipelineStage.CANCELLED},
    PipelineStage.BUILDING_DASHBOARD: {PipelineStage.READY, PipelineStage.FAILED, PipelineStage.CANCELLED},
    PipelineStage.READY: {PipelineStage.PREPARING, PipelineStage.CANCELLED},  # Can rerun
    PipelineStage.FAILED: {PipelineStage.PLAN_READY, PipelineStage.PREPARING}, # Can retry
    PipelineStage.CANCELLED: set()
}


class StateMachineError(Exception):
    pass


def validate_transition(current: PipelineStage, target: PipelineStage) -> None:
    if target not in ALLOWED_TRANSITIONS.get(current, set()):
        raise StateMachineError(f"Invalid state transition: from {current.value} to {target.value}")

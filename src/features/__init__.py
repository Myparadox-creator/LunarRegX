from .base import BaseFeatureEngine
from .classical import ClassicalFeatureEngine
from .phase_structural import PhaseStructuralEngine

def __getattr__(name: str):
    if name == "LearnedFeatureEngine":
        from .learned_adapter import LearnedFeatureEngine
        return LearnedFeatureEngine
    elif name == "DeepCorrespondenceMatcher":
        from .deep_matcher import DeepCorrespondenceMatcher
        return DeepCorrespondenceMatcher
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "BaseFeatureEngine",
    "ClassicalFeatureEngine",
    "PhaseStructuralEngine",
    "LearnedFeatureEngine",
    "DeepCorrespondenceMatcher"
]

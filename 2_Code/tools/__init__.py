"""Tooling utilities for SPARROW.

Imports are intentionally lazy to avoid circular import issues with cli.runner.
"""

__all__ = [
    "EstimateWeightsResult",
    "SerCorrResult",
    "ranuni",
    "rannor",
    "estimate_with_weights",
    "revise_covbetaest_sercor",
]


def __getattr__(name: str):
    if name in {"ranuni", "rannor"}:
        from .rng import rannor, ranuni

        return {"ranuni": ranuni, "rannor": rannor}[name]
    if name in {"SerCorrResult", "revise_covbetaest_sercor"}:
        from .sercorr import SerCorrResult, revise_covbetaest_sercor

        return {
            "SerCorrResult": SerCorrResult,
            "revise_covbetaest_sercor": revise_covbetaest_sercor,
        }[name]
    if name in {"EstimateWeightsResult", "estimate_with_weights"}:
        from .weights import EstimateWeightsResult, estimate_with_weights

        return {
            "EstimateWeightsResult": EstimateWeightsResult,
            "estimate_with_weights": estimate_with_weights,
        }[name]
    raise AttributeError(name)

"""Sampling stack: deterministic RNG, GPU sampler orchestration, MTP verification."""

from cascade.sampling.rng import (
    CASE_NEXTN,
    CASE_SAMPLE,
    CASE_VERIFY,
    derive_uniform,
    derive_uniform_nextn,
    random_sampling_ref_pure,
    sample_cdf_inverse,
    splitmix64_to_uniform,
)

__all__ = [
    "CASE_SAMPLE",
    "CASE_NEXTN",
    "CASE_VERIFY",
    "splitmix64_to_uniform",
    "derive_uniform",
    "derive_uniform_nextn",
    "sample_cdf_inverse",
    "random_sampling_ref_pure",
]

"""Módulo de seguridad: Dead Man's Switch + Shamir Secret Sharing."""

from .deadman import (
    DeadManSwitch,
    HeartbeatSource,
    HeartbeatStatus,
    ShamirSecretSharing,
)

__all__ = [
    "DeadManSwitch",
    "HeartbeatSource",
    "HeartbeatStatus",
    "ShamirSecretSharing",
]

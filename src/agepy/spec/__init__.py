"""Submodule of the spectroscopy subgroup."""

from __future__ import annotations

from pint import UnitRegistry

__all__ = ["ureg", "Q_"]

ureg = UnitRegistry()
Q_ = ureg.Quantity

"""
Regression module.
"""

from .lasso import LassoRegression
from .linear import LinearRegression
from .polynomial import PolynomialRegression
from .ridge import RidgeRegression

__all__ = [
    "LinearRegression",
    "LassoRegression",
    "PolynomialRegression",
    "RidgeRegression"
]
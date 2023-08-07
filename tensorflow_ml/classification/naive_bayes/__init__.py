"""
Classification module
"""

from .bernoulli import BernoulliNaiveBayes
from .gaussian import GaussianNaiveBayes

__all__ = [

    "BernoulliNaiveBayes",
    "GaussianNaiveBayes"
]
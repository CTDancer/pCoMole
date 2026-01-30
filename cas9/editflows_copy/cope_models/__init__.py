"""
COPE Models Package

This package contains objective functions and constraints for COPE (Constrained Optimization for Peptide Engineering).
"""

# Import commonly used classes and functions for convenient access
from .objectives import (
    Solubility,
    Permeability,
    BindingAffinity,
    Halflife,
    Toxicity,
    Admetica,
    analyze_peptide_likeness,
)

from .constraints import (
    Peptidomimetic,
    Length,
)

# Also expose the helper function from is_peptidomimetic
from .is_peptidomimetic import is_peptidomimetic_not_natural

__all__ = [
    # Objectives
    'Solubility',
    'Permeability',
    'BindingAffinity',
    'Halflife',
    'Toxicity',
    'Admetica',
    'analyze_peptide_likeness',
    # Constraints
    'Peptidomimetic',
    'Length',
    # Helpers
    'is_peptidomimetic_not_natural',
]


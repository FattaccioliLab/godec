"""
GoDec - Low Rank Sparse Noise Decomposition
============================================

Python implementation of the Greedy Semi-Soft GoDec algorithm (GreBsmo)
by Tianyi Zhou and Dacheng Tao (AISTATS 2013).

Original MATLAB code: https://tianyizhou.wordpress.com/2014/05/23/aistats-2013-grebsmo-code-is-released/

References:
    Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Low-rank & Sparse Matrix
    Decomposition in Noisy Case", ICML 2011.

    Tianyi Zhou and Dacheng Tao, "Greedy Bilateral Sketch, Completion and
    Smoothing", AISTATS 2013.

Python implementation: J. Fattaccioli (Department of Chemistry, ENS)
Date: March 2020
"""

from .decompose import DecomposeGoDec

__all__ = ["DecomposeGoDec"]

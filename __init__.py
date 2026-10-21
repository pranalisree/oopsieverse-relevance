"""
Object relevance for damage-aware RL in OopsieVerse.

Extension package: nothing in ``damagesim``, ``oopsiebench`` or ``externals``
is modified. Each tracked object ``e`` gets a relevance ``rho_e`` in [0, 1] and
the RL reward becomes

    r(s_t) = R_task(s_t) - w * sum_e rho_e * d_e(t)

where ``d_e(t)`` is the health the object lost on this step. Binary scopes
(S1..S4 in the proposal) are the special case rho_e in {0, 1}.
"""

from relevance.weights import RelevanceConfig, load_relevance_config

__all__ = ["RelevanceConfig", "load_relevance_config"]

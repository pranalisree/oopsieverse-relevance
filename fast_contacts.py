"""
Faster drop-in for ``RSMechanicalDamageEvaluator._get_part_contacts``.

The base evaluator rescans every geom and every contact (through the Python
bindings) once per damageable part. With full-scene tracking (~500 parts) that
is ~95% of step time. Here the contact table is built once per step with numpy
and each part reads its slice. Returned contacts are identical (same dict
fields, same order), so damage values do not change.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from damagesim.robosuite.evaluators.mechanical import RSMechanicalDamageEvaluator


class ContactCache:
    """Contacts of the current sim step, grouped by body id."""

    def __init__(self):
        self._by_body: Dict[int, List[dict]] = {}

    def refresh(self, sim) -> None:
        self._by_body = {}
        model, data = sim.model._model, sim.data._data
        n = int(data.ncon)
        if n == 0:
            return
        con = data.contact
        geom1 = np.asarray(con.geom1[:n])
        geom2 = np.asarray(con.geom2[:n])
        efc = np.asarray(con.efc_address[:n])
        normals = np.asarray(con.frame[:n, :3])

        mags = np.zeros(n)
        valid = efc >= 0
        mags[valid] = np.abs(np.asarray(data.efc_force)[efc[valid]])

        geom_bodyid = np.asarray(model.geom_bodyid)
        body1, body2 = geom_bodyid[geom1], geom_bodyid[geom2]
        for i in np.nonzero(mags > 1e-6)[0]:
            for bid in {int(body1[i]), int(body2[i])}:
                self._by_body.setdefault(bid, []).append({
                    "body_id": bid,
                    "force": mags[i] * normals[i],
                    "magnitude": float(mags[i]),
                    "geom1": int(geom1[i]),
                    "geom2": int(geom2[i]),
                })

    def contacts_for(self, body_id: int) -> List[dict]:
        return self._by_body.get(body_id, [])


class FastRSMechanicalDamageEvaluator(RSMechanicalDamageEvaluator):
    """Reads contacts from a shared ``ContactCache`` set on ``_contact_cache``."""

    _contact_cache: ContactCache | None = None

    def _get_part_contacts(self, part_name: str) -> List[dict]:
        if self._contact_cache is None:
            return super()._get_part_contacts(part_name)
        bid = self._body_id(part_name)
        if self._sim() is None or bid is None:
            return []
        return self._contact_cache.contacts_for(bid)


def use_fast_contacts(objects, cache: ContactCache) -> None:
    """Swap every plain RS mechanical evaluator on *objects* to the cached version."""
    for obj in objects:
        for ev in obj.damage_evaluators:
            if type(ev) is RSMechanicalDamageEvaluator:
                ev.__class__ = FastRSMechanicalDamageEvaluator
            if isinstance(ev, FastRSMechanicalDamageEvaluator):
                ev._contact_cache = cache

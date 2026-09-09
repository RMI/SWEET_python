"""Waste arriving at a disposal site that the city did not send it.

A city-level ADST run shares the city's landfilled residual out across the
city's own sites: ``landfill_split_timeline`` is *fractions that must sum to
1.0* (``advanced_dst_city._validate_shares``), so every ton at every site came
from the city by construction. Real sites are not like that. A regional
landfill takes waste from neighbouring municipalities, from private haulers,
from a catchment the baseline never describes -- and the operator knows the
gate total, not this city's share of it.

So a site may state ``accepted_waste_mass``: everything crossing its weighbridge
in a year, from every source. Whatever that exceeds the city's own allocation is
modeled as a second stream at the same site.

Two things then have to be true, and this module is arranged so that both are
structural facts about the code rather than arithmetic that happens to come out
right.

**The city's emissions do not move.** A city is answerable for the waste it
generates, wherever that waste ends up; a neighbour's waste is the neighbour's
inventory. ``City.sum_landfill_emissions`` sums ``.emissions`` over every entry
of ``parameters.landfills`` -- so the surplus landfills built here are never put
in that list. Nothing about the city's total is recomputed, re-derived or
carefully cancelled: it is the same expression over the same objects, and the
test asserts frame equality rather than a tolerance.

**A site's total is its streams, summed exactly.** Deposited mass enters the
first-order-decay kernel once and multiplicatively (``model_v2``:
``ch4_produce = ks_values * L_0[waste] * waste_masses * exp_term * mcf_values``)
and every step after it -- capture, flare, oxidation, the unit conversion -- is
a mass-independent factor. ``E(0)`` is exactly ``0.0``. So emissions superpose,
measured at ~1e-16 relative, and "the city's share of this site's methane" is a
physical quantity rather than an allocation convention somebody had to invent
and defend.

That invariant needs every stream at a site to agree on every parameter but its
mass, which is why the surplus twin is built from the city stream's own kwargs
dict rather than a copy of it. Note what is *not* on that list: composition.
``k`` is computed once per variant from the city's generated mix
(``advanced_dst_city`` around the ``decomposition_rates`` call) and handed
identically to every landfill, so a stream's own mix moves only its
per-component deposited masses -- which the kernel is linear in. Two streams at
one site with entirely different mixes still superpose exactly.

So the surplus carrying the city's post-diversion residual composition is a
**modelling default, not a numerical requirement**. It is the default because it
is the defensible reading of a gate observation -- a landfill's intake is
residual in character whoever sent it, since somebody's diversion happened
upstream of it -- and because the user has no second composition to hand. A
later PR can let a site state its own mix for the surplus, and
``residual_composition`` is the one function it has to replace.

(The 27.2% figure quoted in PR #785 is about blending two mixes into a *single*
kernel run, where one ``k`` has to stand for both. That is not what happens
here: each stream gets its own ``Landfill`` and its own pass.)

The other way to get this wrong is to skip the second kernel run and attribute
by tonnage: ``city tons / site tons * site total``. That is wrong whenever the
city's share moves over time, because this year's emissions come from decades of
deposit cohorts each with their own split -- measured at up to 28% error for a
site with a growing neighbour, and in a direction that looks entirely plausible.
Each stream gets its own ``Landfill`` and its own pass through the kernel.

Why the functions here are free functions rather than ``Landfill`` methods: the
package already separates a landfill's construction from its definition --
``dst_common.build_landfill`` builds a :class:`~SWEET_python.landfill.Landfill`
from a different module and leaves the caller to assign ``waste_mass_df``. This
follows that convention rather than growing either ``landfill.py`` or the
9,000-line ``city_params.py``.
"""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from SWEET_python.city_params import City, CustomError
from SWEET_python.landfill import Landfill

__all__ = [
    "SiteEmissions",
    "residual_composition",
    "surplus_intake",
    "surplus_masses",
    "adopt_city_params",
    "emission_frames",
]


class SiteEmissions(BaseModel):
    """One site's methane, split by whose waste produced it.

    Each frame is years x the model's degradable components plus ``total``, in
    **tons of methane per year** -- the same units and the same conversion
    ``City.sum_landfill_emissions`` applies to the city's own figure, so the two
    outputs can be read side by side.

    ``city`` is the share the city's own headline total is built from; ``other``
    is waste that reached the gate from outside the baseline, and is an all-zero
    frame rather than ``None`` for the ordinary site, so no caller has to
    special-case it.
    """

    city: pd.DataFrame
    other: pd.DataFrame
    total: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)


def residual_composition(
    residual: pd.DataFrame, generated: pd.DataFrame
) -> pd.DataFrame:
    """Per-year component shares of what the city is left to bury; rows sum to 1.

    The mix the surplus is given. This is the single place that decision is
    made, and the one function a later "a site states its own mix" PR replaces
    -- nothing downstream of it assumes the two streams at a site agree, because
    ``k`` is city-wide and they superpose whether they agree or not.

    Not the *generated* mix. Composting and anaerobic digestion draw only on
    organics and recycling only on recyclables, so the residual differs from
    what the city generates in shape and not merely in scale; splitting a
    gate-observed tonnage by the generated mix overstates its degradable half
    and with it the methane. ``generated`` is the fallback for a year the city
    buries nothing at all, which would otherwise be 0/0 -- and in such a year
    the tonnage these shares scale is itself zero, so the choice is cosmetic.
    """
    totals = residual.sum(axis=1)
    shares = residual.div(totals, axis=0)

    empty = totals <= 0
    if empty.any():
        fallback_totals = generated.sum(axis=1)
        fallback = generated.div(fallback_totals.where(fallback_totals > 0), axis=0)
        shares.loc[empty, :] = fallback.loc[empty, :]

    return shares.fillna(0.0)


def surplus_intake(
    accepted: pd.Series, city_intake: pd.Series
) -> pd.Series:
    """Tons per year reaching a site's gate from outside this baseline.

    Compared at the *gate*, before combustion and before the open/close window,
    because that is where the stated total was measured: a site that burns its
    intake still accepted all of it.

    Floored at zero, which is the answer to a stated total *below* what the city
    sends. That is a real condition -- a gate figure and an allocation are
    measured by different people in different years -- and it is not a licence
    to shrink the city's stream. The city's waste has to go somewhere, this
    feature does not add anywhere else for it to go, and a city's emissions are
    not reduced by a neighbour's paperwork. So the site simply has no outside
    waste, and the caller is free to show the city's figure as the floor it is.
    """
    return (accepted - city_intake).clip(lower=0.0)


def surplus_masses(
    surplus: pd.Series, composition: pd.DataFrame
) -> pd.DataFrame:
    """The surplus tonnage split by component, at the city's residual mix."""
    return composition.mul(surplus, axis=0)


def adopt_city_params(source: Landfill, *surplus: Landfill) -> None:
    """Give the surplus landfills their site's own ``city_params_dict``.

    ``CityParameters.repopulate_attr_dicts`` pushes a freshly dumped parameter
    dict onto every landfill in ``parameters.landfills`` just before the engine
    runs. The surplus landfills are deliberately not in that list -- that list is
    what the city's total is summed over -- so they miss that pass, and would
    otherwise decay against whatever dict they were constructed with.

    Copying the object off the site's city-stream landfill, rather than
    rebuilding it, is what makes them exactly equal: two streams at one site
    must agree on every non-mass parameter or they do not superpose, and an
    equality that has to be maintained by hand is one that will eventually drift.

    Mirrors all three attributes ``repopulate_attr_dicts`` sets, not just the
    first. The other two only exist once a landfill has run, which a surplus
    stream has not yet -- but copying one of three is the kind of near-miss that
    survives a refactor and then stops being harmless.
    """
    for landfill in surplus:
        landfill.city_params_dict = source.city_params_dict
        if hasattr(landfill, "model"):
            landfill.model.city_params_dict = source.city_params_dict
            landfill.model.landfill_instance_attrs = landfill.model_dump()


def require_linear(landfill: Landfill) -> None:
    """Refuse to split a site whose emissions are not linear in deposited mass.

    ``Landfill.doing_fancy_ox`` is hardcoded ``False`` and its body derives an
    oxidation factor from one year's available methane and then clips it three
    times -- genuinely nonlinear in mass. Everything here assumes it stays off:
    with it on, two streams at a site stop superposing and "the city's share"
    silently becomes an allocation convention rather than a measurement, with no
    error and no visible symptom.

    So this raises rather than warns, and it is a runtime check rather than a
    test: the flag is an attribute anybody can set, and a test only fails for
    whoever runs the suite.
    """
    if getattr(landfill, "doing_fancy_ox", False):
        raise CustomError(
            "nonlinear_oxidation",
            "This site uses CALMIM oxidation, whose emissions are not linear in "
            "deposited mass, so its waste cannot be split by source.",
        )


def emission_frames(
    city_stream: Landfill, surplus_stream: Optional[Landfill]
) -> SiteEmissions:
    """One site's emissions, split by whose waste produced them."""
    require_linear(city_stream)
    city = _to_tons_ch4(city_stream.emissions)

    if surplus_stream is None or surplus_stream.emissions is None:
        other = city * 0.0
    else:
        # Both streams were built on the same open year, so the kernel gave them
        # the same index -- but reindex rather than trust it, because a silent
        # NaN here would read as a plausible number downstream.
        other = _to_tons_ch4(surplus_stream.emissions).reindex(
            index=city.index, columns=city.columns, fill_value=0.0
        )

    return SiteEmissions(city=city, other=other, total=city + other)


def _to_tons_ch4(emissions: pd.DataFrame) -> pd.DataFrame:
    """m^3 of methane to tons of it, the way the city's own total is converted.

    ``City.sum_landfill_emissions`` converts to tons of CO2e and then divides by
    28, which lands back on tons of methane. Reproduced rather than simplified
    so that a per-site figure and the city figure it has to reconcile with pass
    through identical arithmetic -- including the same rounding.
    """
    return emissions.map(City.convert_methane_m3_to_ton_co2e) / 28

"""Calculator method plans for FRUST workflows.

Workflow classes define stage ids such as ``"xtb_opt"`` or ``"dft_ts_opt"``.
``ScreeningPlan`` owns the inexpensive GFN-FF/g-xTB stages, while
``MethodPlan`` maps the complete executable stage graph to ``CalculatorSpec``
objects. The end-to-end workflow composes the two before dispatching stages to
``Stepper.xtb``, ``Stepper.gxtb``, or ``Stepper.orca``. Calculator plans change
methods and solvation; they do not change workflow targets or chemistry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any, Literal


_PRESETS: dict[str, "MethodPlan"] = {}
_BUILTINS_REGISTERED = False
_SCREENING_PRESETS: dict[str, "ScreeningPlan"] = {}
_SCREENING_BUILTINS_REGISTERED = False

CalculationLevel = Literal["low_cost", "dft_ranked", "full"]

_STAGE_ALIASES = {
    "dft_rank_sp": "dft_pre_sp",
    "dft_preopt": "dft_pre_opt",
    "dft_hessian": "hess",
    "dft_ts_opt": "optts",
    "dft_freq": "freq",
    "dft_solv_sp": "solv",
}
_LEGACY_STAGE_ALIASES = {legacy: canonical for canonical, legacy in _STAGE_ALIASES.items()}


@dataclass(frozen=True)
class ThermochemistrySpec:
    """Describe how analysis should assemble a molecular free energy.

    Parameters
    ----------
    mode : {"frequency_gibbs", "electronic_plus_thermal"}
        ``"frequency_gibbs"`` uses the Gibbs energy reported by the final
        frequency calculation. ``"electronic_plus_thermal"`` combines the
        analysis electronic energy with the frequency-stage thermal correction
        as ``E_analysis + (G_frequency - E_frequency)``.
    temperature_k : float, optional
        Temperature associated with the frequency thermochemistry in kelvin.
    energy_unit : {"hartree"}, optional
        Unit stored by the calculation result columns.
    """

    mode: str
    temperature_k: float = 298.15
    energy_unit: str = "hartree"

    def __post_init__(self) -> None:
        allowed = {"frequency_gibbs", "electronic_plus_thermal"}
        mode = str(self.mode).strip().lower()
        if mode not in allowed:
            raise ValueError(f"thermochemistry mode must be one of {sorted(allowed)}")
        if float(self.temperature_k) <= 0:
            raise ValueError("thermochemistry temperature_k must be positive")
        unit = str(self.energy_unit).strip().lower()
        if unit != "hartree":
            raise ValueError("thermochemistry energy_unit currently must be 'hartree'")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "temperature_k", float(self.temperature_k))
        object.__setattr__(self, "energy_unit", unit)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible thermochemistry description."""
        return {
            "schema_version": 1,
            "mode": self.mode,
            "temperature_k": self.temperature_k,
            "energy_unit": self.energy_unit,
        }


@dataclass(frozen=True)
class CalculatorSpec:
    """Engine-specific calculator configuration for one workflow stage.

    Parameters
    ----------
    engine : str
        Calculator engine name. Supported values are ``"xtb"`` for the xTB
        wrapper, ``"gxtb"`` for the OET g-xTB wrapper, and ``"orca"`` for ORCA
        calculations.
    options : dict
        Options forwarded to the corresponding :class:`frust.stepper.Stepper`
        method.
    detailed_inp_str : str, optional
        Extra xTB/g-xTB input cards.
    xtra_inp_str : str, optional
        Extra ORCA input block.
    solvent : str or None, optional
        Semantic solvent name used by the calculator. ORCA specifications
        created with :func:`orca` populate this field so workflow inspection
        can display solvent settings without parsing raw input text.
    method, basis : str or None, optional
        Semantic electronic-structure method and basis metadata. These values
        are retained separately from raw engine options so structure builders
        can select a matching reference geometry.
    solvation_model : str or None, optional
        Semantic solvation model. Built-in ORCA solvent calculations use
        ``"smd"``.
    kwargs : dict, optional
        Additional engine-specific keyword arguments forwarded to Stepper.

    Notes
    -----
    ``CalculatorSpec`` is intentionally small. Workflow stage ids and filtering
    behavior live in :class:`frust.workflows.core.StageDef`; this object only
    describes the calculator call used by that stage.
    """

    engine: str
    options: dict[str, Any] = field(default_factory=dict)
    detailed_inp_str: str = ""
    xtra_inp_str: str = ""
    solvent: str | None = None
    method: str | None = None
    basis: str | None = None
    solvation_model: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        engine = self.engine.lower()
        if engine not in {"xtb", "gxtb", "orca"}:
            raise ValueError(f"Unsupported calculator engine: {self.engine!r}")
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "options", dict(self.options or {}))
        solvent = None if self.solvent is None else str(self.solvent).strip()
        object.__setattr__(self, "solvent", solvent or None)
        method = None if self.method is None else str(self.method).strip()
        basis = None if self.basis is None else str(self.basis).strip()
        model = (
            None
            if self.solvation_model is None
            else str(self.solvation_model).strip().lower()
        )
        if solvent and not model:
            model = "smd"
        if model and not solvent:
            raise ValueError("solvation_model requires solvent metadata")
        object.__setattr__(self, "method", method or None)
        object.__setattr__(self, "basis", basis or None)
        object.__setattr__(self, "solvation_model", model or None)
        object.__setattr__(self, "kwargs", dict(self.kwargs or {}))

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible calculator description."""
        return {
            "engine": self.engine,
            "options": _json_compatible(self.options),
            "detailed_inp_str": self.detailed_inp_str,
            "xtra_inp_str": self.xtra_inp_str,
            "solvent": self.solvent,
            "method": self.method,
            "basis": self.basis,
            "solvation_model": self.solvation_model,
            "kwargs": _json_compatible(self.kwargs),
        }


@dataclass(frozen=True)
class ScreeningPlan:
    """Calculator choices for the inexpensive structure-screening stages.

    Parameters
    ----------
    name : str
        Human-readable screening-plan name.
    stages : mapping
        Calculator specifications for ``xtb_preopt``, ``xtb_sp``, and
        ``xtb_opt``. The built-in ``"gxtb-default"`` plan uses GFN-FF for
        preoptimization and direct g-xTB for ranking and optimization.

    Notes
    -----
    A screening plan deliberately contains no DFT settings. It can therefore
    be fingerprinted and reported independently from the optional downstream
    :class:`MethodPlan`.
    """

    name: str
    stages: Mapping[str, CalculatorSpec]

    def __post_init__(self) -> None:
        required = {"xtb_preopt", "xtb_sp", "xtb_opt"}
        normalized = {str(key): value for key, value in self.stages.items()}
        missing = sorted(required - set(normalized))
        extra = sorted(set(normalized) - required)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {missing}")
            if extra:
                details.append(f"unexpected {extra}")
            raise ValueError(
                "ScreeningPlan must define exactly the screening stages: "
                + "; ".join(details)
            )
        for stage_id, spec in normalized.items():
            if not isinstance(spec, CalculatorSpec):
                raise TypeError(f"Stage {stage_id!r} must be a CalculatorSpec")
        object.__setattr__(self, "stages", normalized)

    def to_dict(self, *, include_name: bool = True) -> dict[str, Any]:
        """Return a stable JSON-compatible screening-plan description."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "stages": {
                stage_id: self.stages[stage_id].to_dict()
                for stage_id in sorted(self.stages)
            },
        }
        if include_name:
            payload["name"] = self.name
        return payload

    def fingerprint(self) -> str:
        """Return a SHA-256 fingerprint of the screening calculator settings."""
        encoded = json.dumps(
            self.to_dict(include_name=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MethodPlan:
    """Calculator choices for a complete FRUST workflow graph.

    Parameters
    ----------
    name : str
        Human-readable method-plan name.
    stages : mapping
        Mapping from workflow stage ids to :class:`CalculatorSpec` objects.
    include_terminal_solv_sp : bool, optional
        Whether DFT workflow graphs should end with their separate
        ``dft_solv_sp`` calculation. Set this to ``False`` when the DFT
        ranking, optimization, and frequency stages already include solvent.
    thermochemistry : ThermochemistrySpec or None, optional
        Explicit rule used to assemble molecular free energies from the
        frequency and analysis stages.

    Notes
    -----
    Stage ids must match the ``StageDef.id`` values used by a workflow, unless a
    stage explicitly sets ``StageDef.method_stage``. For example, the screen TS
    workflow asks for keys such as ``"xtb_preopt"``, ``"dft_hessian"``,
    ``"dft_ts_opt"``, and ``"dft_freq"``. Plans with
    ``include_terminal_solv_sp=True`` also require ``"dft_solv_sp"``.

    A method plan can contain more stages than a specific workflow will run.
    For example, the built-in ``"r2scan-3c"`` preset contains TS-specific keys
    such as ``"dft_hessian"``, ``"dft_ts_opt"``, and ``"dft_freq"``, but
    ``ft.workflows.raw_mols(..., dft=True)`` uses the molecule stages
    ``prepare -> xtb_preopt -> xtb_sp -> xtb_opt -> dft_rank_sp -> dft_opt ->
    dft_freq -> dft_solv_sp``. Use ``wf.show_stages()`` on a workflow object to see the
    active stages and resource-group names before running or submitting.
    """

    name: str
    stages: Mapping[str, CalculatorSpec]
    include_terminal_solv_sp: bool = True
    thermochemistry: ThermochemistrySpec | None = None

    def __post_init__(self) -> None:
        normalized: dict[str, CalculatorSpec] = {}
        for stage_id, spec in self.stages.items():
            if not isinstance(spec, CalculatorSpec):
                raise TypeError(f"Stage {stage_id!r} must be a CalculatorSpec")
            normalized[str(stage_id)] = spec
        object.__setattr__(self, "stages", normalized)
        object.__setattr__(
            self,
            "include_terminal_solv_sp",
            bool(self.include_terminal_solv_sp),
        )
        if self.thermochemistry is not None and not isinstance(
            self.thermochemistry, ThermochemistrySpec
        ):
            raise TypeError("thermochemistry must be a ThermochemistrySpec or None")

    def to_dict(self, *, include_name: bool = True) -> dict[str, Any]:
        """Return a stable JSON-compatible method-plan description.

        Parameters
        ----------
        include_name : bool, optional
            Include the human-facing plan name. Fingerprints deliberately omit
            it so differently named but scientifically identical plans match.
        """
        payload: dict[str, Any] = {
            "schema_version": 1,
            "include_terminal_solv_sp": self.include_terminal_solv_sp,
            "thermochemistry": (
                None if self.thermochemistry is None else self.thermochemistry.to_dict()
            ),
            "stages": {
                stage_id: self.stages[stage_id].to_dict()
                for stage_id in sorted(self.stages)
            },
        }
        if include_name:
            payload["name"] = self.name
        return payload

    def fingerprint(self) -> str:
        """Return a SHA-256 fingerprint of scientific calculator settings."""
        encoded = json.dumps(
            self.to_dict(include_name=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def for_stage(self, stage_id: str) -> CalculatorSpec:
        """Return the calculator spec for one workflow stage.

        Parameters
        ----------
        stage_id : str
            Workflow stage id, such as ``"xtb_opt"`` or ``"dft_ts_opt"``.

        Returns
        -------
        CalculatorSpec
            Calculator settings for the stage.

        Raises
        ------
        KeyError
            If the method plan has no matching stage key.
        """
        try:
            return self.stages[stage_id]
        except KeyError as exc:
            alias = _STAGE_ALIASES.get(stage_id) or _LEGACY_STAGE_ALIASES.get(stage_id)
            if alias in self.stages:
                return self.stages[alias]
            available = ", ".join(sorted(self.stages))
            raise KeyError(
                f"Method plan {self.name!r} has no stage {stage_id!r}. "
                f"Available stages: {available}"
            ) from exc

    def replace(self, **stages: CalculatorSpec) -> "MethodPlan":
        """Return a copy with selected stage specs replaced.

        Parameters
        ----------
        **stages : CalculatorSpec
            Replacement specs keyed by workflow stage id.

        Returns
        -------
        MethodPlan
            New method plan with the selected stages changed. The original
            method plan is unchanged.

        Examples
        --------
        >>> method = preset("r2scan-3c").replace(
        ...     xtb_opt=xtb(gfn=2, opt=True),
        ... )
        """
        updated = dict(self.stages)
        for stage_id, spec in stages.items():
            if not isinstance(spec, CalculatorSpec):
                raise TypeError(f"Replacement for {stage_id!r} must be a CalculatorSpec")
            canonical = _LEGACY_STAGE_ALIASES.get(stage_id)
            legacy = _STAGE_ALIASES.get(stage_id)
            if canonical in updated:
                resolved_stage = canonical
            elif legacy in updated:
                resolved_stage = legacy
            else:
                resolved_stage = stage_id
            updated[resolved_stage] = spec
        return dataclass_replace(self, stages=updated)

    def with_stage(self, stage_id: str, spec: CalculatorSpec) -> "MethodPlan":
        """Return a copy with one stage spec replaced.

        Parameters
        ----------
        stage_id : str
            Workflow stage id to replace.
        spec : CalculatorSpec
            Replacement calculator settings.

        Returns
        -------
        MethodPlan
            New method plan with one stage replaced.
        """
        return self.replace(**{stage_id: spec})


def xtb(
    *,
    gfn: int | None = None,
    gfnff: bool = False,
    opt: bool = False,
    ohess: bool = False,
    options: Mapping[str, Any] | None = None,
    detailed_inp_str: str = "",
    **kwargs: Any,
) -> CalculatorSpec:
    """Build an xTB calculator specification.

    Parameters
    ----------
    gfn : int or None, optional
        xTB GFN level. For example, ``gfn=2`` produces ``{"gfn": 2}`` options.
    gfnff : bool, optional
        Use GFN-FF. Cannot be combined with ``gfn``.
    opt : bool, optional
        Add the xTB optimization keyword.
    ohess : bool, optional
        Add the xTB Hessian keyword.
    options : mapping or None, optional
        Additional Stepper xTB options. Automatically generated options are
        added only when the key is absent.
    detailed_inp_str : str, optional
        Extra xTB input cards passed to ``Stepper.xtb``.
    **kwargs
        Extra keyword arguments forwarded to ``Stepper.xtb``.

    Returns
    -------
    CalculatorSpec
        Spec with ``engine="xtb"``.

    Examples
    --------
    >>> xtb(gfnff=True, opt=True)
    >>> xtb(gfn=2)
    """
    opts = dict(options or {})
    if gfnff and gfn is not None:
        raise ValueError("xTB spec cannot combine `gfnff=True` with `gfn=`")
    if gfnff:
        opts.setdefault("gfnff", None)
    elif gfn is not None:
        opts.setdefault("gfn", int(gfn))
    if opt:
        opts.setdefault("opt", None)
    if ohess:
        opts.setdefault("ohess", None)
    return CalculatorSpec(
        engine="xtb",
        options=opts,
        detailed_inp_str=detailed_inp_str,
        kwargs=kwargs,
    )


def gxtb(
    *,
    job: str = "sp",
    options: Mapping[str, Any] | None = None,
    detailed_inp_str: str = "",
    **kwargs: Any,
) -> CalculatorSpec:
    """Build a g-xTB calculator specification.

    Parameters
    ----------
    job : {"sp", "opt", "ohess"}, optional
        g-xTB job type. This intentionally does not accept xTB-only options
        such as ``gfn=2``.
    options : mapping, optional
        Additional g-xTB options forwarded as-is.
    detailed_inp_str : str, optional
        Extra g-xTB input cards passed to ``Stepper.gxtb``.
    **kwargs
        Extra keyword arguments forwarded to ``Stepper.gxtb``.

    Returns
    -------
    CalculatorSpec
        Spec with ``engine="gxtb"``.

    Examples
    --------
    Built-in method presets use direct g-xTB for both the ``xtb_sp`` ranking
    stage and the constrained ``xtb_opt`` optimization stage:

    >>> gxtb(job="opt")
    """
    opts = dict(options or {})
    job_name = job.lower()
    if job_name in {"opt", "ohess"}:
        opts.setdefault(job_name, None)
    elif job_name != "sp":
        raise ValueError("g-xTB job must be one of 'sp', 'opt', or 'ohess'")
    return CalculatorSpec(
        engine="gxtb",
        options=opts,
        detailed_inp_str=detailed_inp_str,
        kwargs=kwargs,
    )


def screening_preset(name: str = "gxtb-default") -> ScreeningPlan:
    """Return a registered inexpensive screening-plan preset.

    Parameters
    ----------
    name : str, optional
        Preset name. The built-in ``"gxtb-default"`` plan runs GFN-FF
        preoptimization followed by direct g-xTB single-point ranking and
        optimization.

    Returns
    -------
    ScreeningPlan
        Reusable low-cost stage plan.
    """
    _ensure_screening_presets()
    key = _preset_key(name)
    try:
        return _SCREENING_PRESETS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_SCREENING_PRESETS))
        raise KeyError(
            f"Unknown screening preset {name!r}. Available: {available}"
        ) from exc


def register_screening_preset(name: str, plan: ScreeningPlan) -> ScreeningPlan:
    """Register a screening preset for the current Python session.

    Parameters
    ----------
    name : str
        Preset lookup name.
    plan : ScreeningPlan
        Screening plan to register.

    Returns
    -------
    ScreeningPlan
        The supplied plan.
    """
    if not isinstance(plan, ScreeningPlan):
        raise TypeError("plan must be a ScreeningPlan")
    _SCREENING_PRESETS[_preset_key(name)] = plan
    return plan


def apply_screening_plan(method: MethodPlan, screening: ScreeningPlan) -> MethodPlan:
    """Return a method plan using the selected inexpensive screening stages.

    Parameters
    ----------
    method : MethodPlan
        Downstream DFT plan.
    screening : ScreeningPlan
        Low-cost plan whose three stages replace the corresponding entries in
        ``method``.

    Returns
    -------
    MethodPlan
        Composed calculator map used internally by a workflow.
    """
    if not isinstance(method, MethodPlan):
        raise TypeError("method must be a MethodPlan")
    if not isinstance(screening, ScreeningPlan):
        raise TypeError("screening must be a ScreeningPlan")
    return method.replace(**dict(screening.stages))


def with_ranking_solvation(
    method: MethodPlan,
    ranking_solvation: str = "method",
) -> tuple[MethodPlan, dict[str, str | None]]:
    """Resolve solvation for DFT single points on g-xTB geometries.

    Parameters
    ----------
    method : MethodPlan
        DFT plan containing ``dft_rank_sp``.
    ranking_solvation : str, optional
        ``"method"`` inherits the solvent used for the plan's final analysis
        energy, ``"gas"`` requests no implicit solvent, and any other value is
        interpreted as an SMD solvent name such as ``"chloroform"`` or
        ``"toluene"``.

    Returns
    -------
    MethodPlan
        Plan with a resolved ``dft_rank_sp`` specification.
    dict
        JSON-compatible ``model`` and ``solvent`` metadata.
    """
    requested = str(ranking_solvation).strip()
    if not requested:
        raise ValueError("ranking_solvation cannot be empty")
    if requested.casefold() == "method":
        solvent = _analysis_solvent(method)
    elif requested.casefold() == "gas":
        solvent = None
    else:
        solvent = requested

    rank_spec = method.for_stage("dft_rank_sp")
    if rank_spec.engine != "orca":
        raise ValueError("DFT ranking solvation currently requires an ORCA dft_rank_sp")
    updated = _replace_orca_solvation(rank_spec, solvent)
    resolved = method.with_stage("dft_rank_sp", updated)
    requested_value = (
        requested.casefold()
        if requested.casefold() in {"method", "gas"}
        else requested
    )
    return resolved, {
        "requested": requested_value,
        "model": None if solvent is None else "smd",
        "solvent": solvent,
    }


def orca(
    *,
    method: str,
    basis: str | None = None,
    job: str = "sp",
    solvent: str | None = None,
    xtra_inp_str: str = "",
    **kwargs: Any,
) -> CalculatorSpec:
    """Build a conventional ORCA calculator specification.

    Parameters
    ----------
    method : str
        ORCA method keyword, for example ``"wB97X-D3"`` or ``"R2SCAN"``.
    basis : str or None, optional
        Basis keyword. Use ``None`` for composite methods that should not add a
        separate basis keyword.
    job : {"sp", "opt", "optts", "freq"}, optional
        ORCA job type. FRUST expands this into standard simple-input keywords
        such as ``SP``, ``Opt``, ``OptTS``, or ``Freq`` plus ``TightSCF`` and
        ``NoSym``.
    solvent : str or None, optional
        SMD solvent name. When supplied, a CPCM/SMD block is prepended to
        ``xtra_inp_str``.
    xtra_inp_str : str, optional
        Additional ORCA input block passed to ``Stepper.orca``.
    **kwargs
        Extra keyword arguments forwarded to ``Stepper.orca``.

    Returns
    -------
    CalculatorSpec
        Spec with ``engine="orca"``.

    Examples
    --------
    >>> orca(method="wB97X-D3", basis="6-31G**", job="opt")
    >>> orca(method="R2SCAN", basis="def2-SVPD", job="sp", solvent="chloroform")
    """
    options = _orca_options(method, basis, job)
    extra = _solvent_block(solvent) if solvent else ""
    if xtra_inp_str.strip():
        extra = (extra + "\n" + xtra_inp_str.strip()).strip()
    return CalculatorSpec(
        engine="orca",
        options=options,
        xtra_inp_str=extra,
        solvent=solvent,
        method=method,
        basis=basis,
        solvation_model="smd" if solvent else None,
        kwargs=kwargs,
    )


def orca_composite(
    method: str,
    *,
    job: str = "sp",
    solvent: str | None = None,
    xtra_inp_str: str = "",
    **kwargs: Any,
) -> CalculatorSpec:
    """Build an ORCA composite-method specification.

    Parameters
    ----------
    method : str
        ORCA composite method keyword, such as ``"r2SCAN-3c"``.
    job : {"sp", "opt", "optts", "freq"}, optional
        ORCA job type.
    solvent : str or None, optional
        SMD solvent name.
    xtra_inp_str : str, optional
        Additional ORCA input block.
    **kwargs
        Extra keyword arguments forwarded to ``Stepper.orca``.

    Returns
    -------
    CalculatorSpec
        ORCA spec with no separate basis keyword.

    Notes
    -----
    This is the right helper for methods such as ``r2SCAN-3c`` where ORCA's
    method keyword already includes the basis/model definition.
    """
    return orca(
        method=method,
        basis=None,
        job=job,
        solvent=solvent,
        xtra_inp_str=xtra_inp_str,
        **kwargs,
    )


def with_ts_mode_following(
    plan: MethodPlan,
    *,
    mode_roles: tuple[str, ...] | list[str],
    active_roles: tuple[str, ...] | list[str] | None = None,
    active_atoms_factor: float | None = None,
    recalc_hess: int | None = None,
    trust_radius: float | None = None,
    max_step: float | None = None,
    stage: str = "dft_ts_opt",
    tight_opt: bool = True,
) -> MethodPlan:
    """Return a method plan with role-based ORCA TS mode following.

    Parameters
    ----------
    plan : MethodPlan
        Method plan whose transition-state optimization stage should be
        customized.
    mode_roles : tuple or list of str
        Chemical roles defining the internal coordinate ORCA should follow.
        Two roles define a bond, three an angle, and four a dihedral.
    active_roles : tuple or list of str or None, optional
        Chemical roles passed to ORCA ``TS_Active_Atoms``.
    active_atoms_factor : float or None, optional
        ORCA ``TS_Active_Atoms_Factor``. Requires ``active_roles``.
    recalc_hess : int or None, optional
        Number of OptTS cycles between exact Hessian recalculations.
    trust_radius : float or None, optional
        ORCA trust radius. Positive values are adaptive and negative values
        keep the absolute radius fixed.
    max_step : float or None, optional
        Positive ORCA maximum component of the optimization step.
    stage : str, optional
        Method-plan stage to customize. Defaults to ``"dft_ts_opt"``.
    tight_opt : bool, optional
        Add the ORCA ``TightOpt`` simple-input keyword. Defaults to ``True``.

    Returns
    -------
    MethodPlan
        Copy of ``plan`` with only the selected ORCA stage changed.

    Raises
    ------
    TypeError
        If ``plan`` is not a :class:`MethodPlan`.
    ValueError
        If the selected stage does not use ORCA.

    Examples
    --------
    Configure TS3 using chemical roles rather than fixed atom indices:

    >>> method = preset("r2scan-3c-solv")
    >>> method = with_ts_mode_following(
    ...     method,
    ...     mode_roles=("pin_B", "substrate_C"),
    ...     active_roles=("cat_B", "transfer_H", "pin_B", "substrate_C"),
    ...     active_atoms_factor=1.5,
    ...     recalc_hess=3,
    ...     trust_radius=0.15,
    ... )
    """
    if not isinstance(plan, MethodPlan):
        raise TypeError("plan must be a MethodPlan")

    spec = plan.for_stage(stage)
    if spec.engine != "orca":
        raise ValueError(
            f"TS mode following requires an ORCA stage; {stage!r} uses {spec.engine!r}"
        )

    options = dict(spec.options)
    if tight_opt:
        options.setdefault("TightOpt", None)

    kwargs = dict(spec.kwargs)
    kwargs.update(
        {
            "ts_mode": tuple(mode_roles),
            "ts_active_atoms": (
                None if active_roles is None else tuple(active_roles)
            ),
            "ts_active_atoms_factor": active_atoms_factor,
            "recalc_hess": recalc_hess,
            "trust_radius": trust_radius,
            "max_step": max_step,
        }
    )
    updated = dataclass_replace(spec, options=options, kwargs=kwargs)
    return plan.with_stage(stage, updated)


def preset(name: str) -> MethodPlan:
    """Return a registered workflow method preset.

    Parameters
    ----------
    name : str
        Preset name. Matching is case-insensitive and treats underscores like
        hyphens. Built-in values are:

        - ``"r2scan-3c"``: use the ORCA ``r2SCAN-3c`` composite method for DFT
          stages.
        - ``"wb97xd3-631g"``: use ORCA ``wB97X-D3`` with ``6-31G**`` for most
          DFT stages and ``6-31+G**`` for the solvent single-point stage. This
          is the workflow default when ``method=None``.
        - ``"r2scan-3c-solv"``: use solvent-inclusive ORCA ``r2SCAN-3c`` DFT
          stages with SMD chloroform and no terminal solvent single point.
        - ``"wb97xd3-631g-solv"``: use solvent-inclusive ORCA
          ``wB97X-D3/6-31G**`` DFT stages with SMD chloroform and no terminal
          solvent single point.
        - ``"r2scan-def2svp"``: use ORCA ``R2SCAN`` with the ``def2-SVP`` basis
          for DFT stages.

    Returns
    -------
    MethodPlan
        Registered method plan.

        The returned plan is a reusable stage-to-calculator map. A workflow may
        use only some of its keys depending on the chemistry and ``dft`` value.
        For example, raw molecule DFT workflows use ``dft_opt``, ``dft_freq``,
        and ``dft_solv_sp`` but do not use TS-only ``dft_hessian`` or
        ``dft_ts_opt`` stages. Call
        ``wf.show_stages()`` after constructing a workflow to inspect the active
        subset.

    Raises
    ------
    KeyError
        If no preset with that name is registered.
    """
    _ensure_builtin_presets()
    key = _preset_key(name)
    try:
        return _PRESETS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_PRESETS))
        raise KeyError(f"Unknown workflow method preset {name!r}. Available: {available}") from exc


def register_preset(name: str, method: MethodPlan) -> MethodPlan:
    """Register a method preset for the current Python session.

    Parameters
    ----------
    name : str
        Preset name to register. Names are normalized with the same rules as
        :func:`preset`.
    method : MethodPlan
        Method plan to store.

    Returns
    -------
    MethodPlan
        The same method plan, so registration can be used inline.

    Examples
    --------
    >>> custom = preset("r2scan-3c").replace(xtb_opt=xtb(gfn=2, opt=True))
    >>> register_preset("my-r2scan-xtb-opt", custom)
    >>> preset("my-r2scan-xtb-opt") is custom
    True
    """
    if not isinstance(method, MethodPlan):
        raise TypeError("method must be a MethodPlan")
    _PRESETS[_preset_key(name)] = method
    return method


def _preset_key(name: str) -> str:
    """Normalize a method preset name.

    Parameters
    ----------
    name : str
        User-facing preset name.

    Returns
    -------
    str
        Lowercase hyphenated lookup key.
    """
    return str(name).strip().lower().replace("_", "-")


def _ensure_builtin_presets() -> None:
    """Register built-in method presets once."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    register_preset("r2scan-3c", _r2scan_3c())
    register_preset("wb97xd3-631g", _wb97xd3_631g())
    register_preset("r2scan-3c-solv", _r2scan_3c_solv())
    register_preset("wb97xd3-631g-solv", _wb97xd3_631g_solv())
    register_preset("r2scan-def2svp", _r2scan_def2svp())
    _BUILTINS_REGISTERED = True


def _ensure_screening_presets() -> None:
    """Register built-in inexpensive screening plans once."""
    global _SCREENING_BUILTINS_REGISTERED
    if _SCREENING_BUILTINS_REGISTERED:
        return
    register_screening_preset(
        "gxtb-default",
        ScreeningPlan(
            name="gxtb-default",
            stages={
                "xtb_preopt": xtb(gfnff=True, opt=True),
                "xtb_sp": gxtb(job="sp"),
                "xtb_opt": gxtb(job="opt"),
            },
        ),
    )
    _SCREENING_BUILTINS_REGISTERED = True


def _analysis_solvent(method: MethodPlan) -> str | None:
    """Return the solvent associated with a method plan's analysis energy."""
    stage_order = (
        "dft_solv_sp",
        "dft_freq",
        "dft_opt",
        "dft_ts_opt",
        "dft_rank_sp",
    )
    for stage_id in stage_order:
        if stage_id not in method.stages:
            continue
        solvent = method.for_stage(stage_id).solvent
        if solvent:
            return solvent
    return None


def _replace_orca_solvation(
    spec: CalculatorSpec,
    solvent: str | None,
) -> CalculatorSpec:
    """Return an ORCA specification with a replaced CPCM/SMD solvent block."""
    extra = spec.xtra_inp_str
    if spec.solvent:
        previous = _solvent_block(spec.solvent)
        extra = extra.replace(previous, "", 1).lstrip("\n")
    if solvent:
        block = _solvent_block(solvent)
        extra = block if not extra else f"{block}\n{extra}"
    return dataclass_replace(
        spec,
        xtra_inp_str=extra,
        solvent=solvent,
        solvation_model="smd" if solvent else None,
    )


def _base_stages(
    *,
    dft_rank_sp: CalculatorSpec,
    dft_preopt: CalculatorSpec,
    dft_opt: CalculatorSpec,
    dft_hessian: CalculatorSpec,
    dft_ts_opt: CalculatorSpec,
    dft_freq: CalculatorSpec,
    dft_solv_sp: CalculatorSpec | None = None,
) -> dict[str, CalculatorSpec]:
    """Return common stage specs for built-in workflow presets.

    Parameters
    ----------
    dft_rank_sp, dft_preopt, dft_opt, dft_hessian, dft_ts_opt, dft_freq,
    dft_solv_sp : CalculatorSpec or None, optional
        Terminal solvent single-point calculator. Use ``None`` for a method
        plan whose DFT stages are solvent-inclusive.

    Returns
    -------
    dict of str to CalculatorSpec
        Stage-id mapping shared by built-in workflow method plans.
    """
    canonical = {
        "xtb_preopt": xtb(gfnff=True, opt=True),
        "xtb_sp": gxtb(job="sp"),
        "xtb_opt": gxtb(job="opt"),
        "dft_rank_sp": dft_rank_sp,
        "dft_preopt": dft_preopt,
        "dft_opt": dft_opt,
        "dft_hessian": dft_hessian,
        "dft_ts_opt": dft_ts_opt,
        "dft_freq": dft_freq,
    }
    if dft_solv_sp is not None:
        canonical["dft_solv_sp"] = dft_solv_sp
    return canonical


def _r2scan_3c() -> MethodPlan:
    """Build the ORCA r2SCAN-3c composite method preset."""
    method = "r2SCAN-3c"
    return MethodPlan(
        name="r2scan-3c",
        thermochemistry=ThermochemistrySpec("electronic_plus_thermal"),
        stages=_base_stages(
            dft_rank_sp=orca_composite(method, job="sp"),
            dft_preopt=orca_composite(method, job="opt"),
            dft_opt=orca_composite(method, job="opt"),
            dft_hessian=orca_composite(method, job="freq"),
            dft_ts_opt=orca_composite(method, job="optts"),
            dft_freq=orca_composite(method, job="freq"),
            dft_solv_sp=orca_composite(method, job="sp", solvent="chloroform"),
        ),
    )


def _wb97xd3_631g() -> MethodPlan:
    """Build the ORCA wB97X-D3/6-31G workflow method preset."""
    method = "wB97X-D3"
    return MethodPlan(
        name="wb97xd3-631g",
        thermochemistry=ThermochemistrySpec("electronic_plus_thermal"),
        stages=_base_stages(
            dft_rank_sp=orca(method=method, basis="6-31G**", job="sp"),
            dft_preopt=orca(method=method, basis="6-31G**", job="opt"),
            dft_opt=orca(method=method, basis="6-31G**", job="opt"),
            dft_hessian=orca(method=method, basis="6-31G**", job="freq"),
            dft_ts_opt=orca(method=method, basis="6-31G**", job="optts"),
            dft_freq=orca(method=method, basis="6-31G**", job="freq"),
            dft_solv_sp=orca(
                method=method, basis="6-31+G**", job="sp", solvent="chloroform"
            ),
        ),
    )


def _r2scan_3c_solv() -> MethodPlan:
    """Build the solvent-inclusive ORCA r2SCAN-3c method preset."""
    method = "r2SCAN-3c"
    return MethodPlan(
        name="r2scan-3c-solv",
        include_terminal_solv_sp=False,
        thermochemistry=ThermochemistrySpec("frequency_gibbs"),
        stages=_base_stages(
            dft_rank_sp=orca_composite(method, job="sp", solvent="chloroform"),
            dft_preopt=orca_composite(method, job="opt", solvent="chloroform"),
            dft_opt=orca_composite(method, job="opt", solvent="chloroform"),
            dft_hessian=orca_composite(method, job="freq", solvent="chloroform"),
            dft_ts_opt=orca_composite(method, job="optts", solvent="chloroform"),
            dft_freq=orca_composite(method, job="freq", solvent="chloroform"),
        ),
    )


def _wb97xd3_631g_solv() -> MethodPlan:
    """Build the solvent-inclusive ORCA wB97X-D3/6-31G workflow preset."""
    method = "wB97X-D3"
    return MethodPlan(
        name="wb97xd3-631g-solv",
        include_terminal_solv_sp=False,
        thermochemistry=ThermochemistrySpec("frequency_gibbs"),
        stages=_base_stages(
            dft_rank_sp=orca(
                method=method, basis="6-31G**", job="sp", solvent="chloroform"
            ),
            dft_preopt=orca(
                method=method, basis="6-31G**", job="opt", solvent="chloroform"
            ),
            dft_opt=orca(
                method=method, basis="6-31G**", job="opt", solvent="chloroform"
            ),
            dft_hessian=orca(
                method=method, basis="6-31G**", job="freq", solvent="chloroform"
            ),
            dft_ts_opt=orca(
                method=method, basis="6-31G**", job="optts", solvent="chloroform"
            ),
            dft_freq=orca(
                method=method, basis="6-31G**", job="freq", solvent="chloroform"
            ),
        ),
    )


def _r2scan_def2svp() -> MethodPlan:
    """Build the ORCA R2SCAN/def2-SVP workflow method preset."""
    method = "R2SCAN"
    return MethodPlan(
        name="r2scan-def2svp",
        thermochemistry=ThermochemistrySpec("electronic_plus_thermal"),
        stages=_base_stages(
            dft_rank_sp=orca(method=method, basis="def2-SVP", job="sp"),
            dft_preopt=orca(method=method, basis="def2-SVP", job="opt"),
            dft_opt=orca(method=method, basis="def2-SVP", job="opt"),
            dft_hessian=orca(method=method, basis="def2-SVP", job="freq"),
            dft_ts_opt=orca(method=method, basis="def2-SVP", job="optts"),
            dft_freq=orca(method=method, basis="def2-SVP", job="freq"),
            dft_solv_sp=orca(
                method=method, basis="def2-SVPD", job="sp", solvent="chloroform"
            ),
        ),
    )


def _orca_options(method: str, basis: str | None, job: str) -> dict[str, None]:
    """Build ORCA simple-input options for one job type.

    Parameters
    ----------
    method : str
        ORCA method keyword.
    basis : str or None
        Optional ORCA basis keyword.
    job : str
        Job type: ``"sp"``, ``"opt"``, ``"optts"``, or ``"freq"``.

    Returns
    -------
    dict of str to None
        Ordered simple-input keyword mapping passed through Stepper.
    """
    job_name = job.lower()
    job_keywords = {
        "sp": ("TightSCF", "SP", "NoSym"),
        "opt": ("TightSCF", "SlowConv", "Opt", "NoSym"),
        "optts": ("TightSCF", "SlowConv", "OptTS", "NoSym"),
        "freq": ("TightSCF", "SlowConv", "Freq", "NoSym"),
    }
    try:
        keywords = job_keywords[job_name]
    except KeyError as exc:
        supported = ", ".join(sorted(job_keywords))
        raise ValueError(f"Unsupported ORCA job {job!r}. Supported jobs: {supported}") from exc

    options: dict[str, None] = {method: None}
    if basis:
        options[basis] = None
    options.update({keyword: None for keyword in keywords})
    return options


def _solvent_block(solvent: str) -> str:
    """Return the ORCA CPCM/SMD solvent block used by FRUST defaults.

    Parameters
    ----------
    solvent : str
        SMD solvent name.

    Returns
    -------
    str
        ORCA input block enabling SMD through CPCM.
    """
    return f'%CPCM\nSMD TRUE\nSMDSOLVENT "{solvent}"\nend'


def _json_compatible(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(
        f"Method-plan value {value!r} of type {type(value).__name__} is not JSON-compatible"
    )

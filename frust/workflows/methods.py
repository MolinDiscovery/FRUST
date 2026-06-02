"""Calculator method plans for FRUST workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any


_PRESETS: dict[str, "MethodPlan"] = {}
_BUILTINS_REGISTERED = False


@dataclass(frozen=True)
class CalculatorSpec:
    """Engine-specific calculator configuration for one workflow stage.

    Parameters
    ----------
    engine : str
        Calculator engine name. Supported values are ``"xtb"``, ``"gxtb"``,
        and ``"orca"``.
    options : dict
        Options forwarded to the corresponding :class:`frust.stepper.Stepper`
        method.
    detailed_inp_str : str, optional
        Extra xTB/g-xTB input cards.
    xtra_inp_str : str, optional
        Extra ORCA input block.
    kwargs : dict, optional
        Additional engine-specific keyword arguments forwarded to Stepper.
    """

    engine: str
    options: dict[str, Any] = field(default_factory=dict)
    detailed_inp_str: str = ""
    xtra_inp_str: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        engine = self.engine.lower()
        if engine not in {"xtb", "gxtb", "orca"}:
            raise ValueError(f"Unsupported calculator engine: {self.engine!r}")
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "options", dict(self.options or {}))
        object.__setattr__(self, "kwargs", dict(self.kwargs or {}))


@dataclass(frozen=True)
class MethodPlan:
    """Calculator choices for a complete FRUST workflow graph.

    Parameters
    ----------
    name : str
        Human-readable method-plan name.
    stages : mapping
        Mapping from workflow stage ids to :class:`CalculatorSpec` objects.
    """

    name: str
    stages: Mapping[str, CalculatorSpec]

    def __post_init__(self) -> None:
        normalized: dict[str, CalculatorSpec] = {}
        for stage_id, spec in self.stages.items():
            if not isinstance(spec, CalculatorSpec):
                raise TypeError(f"Stage {stage_id!r} must be a CalculatorSpec")
            normalized[str(stage_id)] = spec
        object.__setattr__(self, "stages", normalized)

    def for_stage(self, stage_id: str) -> CalculatorSpec:
        """Return the calculator spec for one stage id."""
        try:
            return self.stages[stage_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.stages))
            raise KeyError(
                f"Method plan {self.name!r} has no stage {stage_id!r}. "
                f"Available stages: {available}"
            ) from exc

    def replace(self, **stages: CalculatorSpec) -> "MethodPlan":
        """Return a copy with selected stage specs replaced.

        Examples
        --------
        >>> method = preset("r2scan-3c").replace(
        ...     xtb_sp=gxtb(job="sp"),
        ...     xtb_opt=gxtb(job="opt"),
        ... )
        """
        updated = dict(self.stages)
        for stage_id, spec in stages.items():
            if not isinstance(spec, CalculatorSpec):
                raise TypeError(f"Replacement for {stage_id!r} must be a CalculatorSpec")
            updated[stage_id] = spec
        return dataclass_replace(self, stages=updated)

    def with_stage(self, stage_id: str, spec: CalculatorSpec) -> "MethodPlan":
        """Return a copy with one stage spec replaced."""
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
    """Build an xTB calculator specification."""
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


def orca(
    *,
    method: str,
    basis: str | None = None,
    job: str = "sp",
    solvent: str | None = None,
    xtra_inp_str: str = "",
    **kwargs: Any,
) -> CalculatorSpec:
    """Build a conventional ORCA calculator specification."""
    options = _orca_options(method, basis, job)
    extra = _solvent_block(solvent) if solvent else ""
    if xtra_inp_str.strip():
        extra = (extra + "\n" + xtra_inp_str.strip()).strip()
    return CalculatorSpec(
        engine="orca",
        options=options,
        xtra_inp_str=extra,
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
    """Build an ORCA composite-method specification without a basis keyword."""
    return orca(
        method=method,
        basis=None,
        job=job,
        solvent=solvent,
        xtra_inp_str=xtra_inp_str,
        **kwargs,
    )


def preset(name: str) -> MethodPlan:
    """Return a registered method preset by name."""
    _ensure_builtin_presets()
    key = _preset_key(name)
    try:
        return _PRESETS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_PRESETS))
        raise KeyError(f"Unknown workflow method preset {name!r}. Available: {available}") from exc


def register_preset(name: str, method: MethodPlan) -> MethodPlan:
    """Register a user method preset for the current Python session."""
    if not isinstance(method, MethodPlan):
        raise TypeError("method must be a MethodPlan")
    _PRESETS[_preset_key(name)] = method
    return method


def _preset_key(name: str) -> str:
    """Normalize a method preset name."""
    return str(name).strip().lower().replace("_", "-")


def _ensure_builtin_presets() -> None:
    """Register built-in method presets once."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    register_preset("r2scan-3c", _r2scan_3c())
    register_preset("wb97xd3-631g", _wb97xd3_631g())
    register_preset("r2scan-def2svp", _r2scan_def2svp())
    _BUILTINS_REGISTERED = True


def _base_stages(
    *,
    dft_pre_sp: CalculatorSpec,
    dft_pre_opt: CalculatorSpec,
    dft_opt: CalculatorSpec,
    hess: CalculatorSpec,
    optts: CalculatorSpec,
    freq: CalculatorSpec,
    solv: CalculatorSpec,
) -> dict[str, CalculatorSpec]:
    """Return common stage specs for FRUST workflow presets."""
    return {
        "xtb_preopt": xtb(gfnff=True, opt=True),
        "xtb_sp": xtb(gfn=2),
        "xtb_opt": xtb(gfn=2, opt=True),
        "dft_pre_sp": dft_pre_sp,
        "dft_pre_opt": dft_pre_opt,
        "dft_opt": dft_opt,
        "hess": hess,
        "optts": optts,
        "freq": freq,
        "solv": solv,
    }


def _r2scan_3c() -> MethodPlan:
    method = "r2SCAN-3c"
    return MethodPlan(
        name="r2scan-3c",
        stages=_base_stages(
            dft_pre_sp=orca_composite(method, job="sp"),
            dft_pre_opt=orca_composite(method, job="opt"),
            dft_opt=orca_composite(method, job="opt"),
            hess=orca_composite(method, job="freq"),
            optts=orca_composite(method, job="optts"),
            freq=orca_composite(method, job="freq"),
            solv=orca_composite(method, job="sp", solvent="chloroform"),
        ),
    )


def _wb97xd3_631g() -> MethodPlan:
    method = "wB97X-D3"
    return MethodPlan(
        name="wb97xd3-631g",
        stages=_base_stages(
            dft_pre_sp=orca(method=method, basis="6-31G**", job="sp"),
            dft_pre_opt=orca(method=method, basis="6-31G**", job="opt"),
            dft_opt=orca(method=method, basis="6-31G**", job="opt"),
            hess=orca(method=method, basis="6-31G**", job="freq"),
            optts=orca(method=method, basis="6-31G**", job="optts"),
            freq=orca(method=method, basis="6-31G**", job="freq"),
            solv=orca(method=method, basis="6-31+G**", job="sp", solvent="chloroform"),
        ),
    )


def _r2scan_def2svp() -> MethodPlan:
    method = "R2SCAN"
    return MethodPlan(
        name="r2scan-def2svp",
        stages=_base_stages(
            dft_pre_sp=orca(method=method, basis="def2-SVP", job="sp"),
            dft_pre_opt=orca(method=method, basis="def2-SVP", job="opt"),
            dft_opt=orca(method=method, basis="def2-SVP", job="opt"),
            hess=orca(method=method, basis="def2-SVP", job="freq"),
            optts=orca(method=method, basis="def2-SVP", job="optts"),
            freq=orca(method=method, basis="def2-SVP", job="freq"),
            solv=orca(method=method, basis="def2-SVPD", job="sp", solvent="chloroform"),
        ),
    )


def _orca_options(method: str, basis: str | None, job: str) -> dict[str, None]:
    """Build ORCA simple-input options for one job type."""
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
    """Return the ORCA CPCM/SMD solvent block used by FRUST defaults."""
    return f'%CPCM\nSMD TRUE\nSMDSOLVENT "{solvent}"\nend'

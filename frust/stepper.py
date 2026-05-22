# frustactivation/stepper.py
import logging
import sys
import os
from pathlib import Path
import re
import textwrap
from typing import Any, Callable
import numpy as np
import pandas as pd
from pandas import Series
from inspect import signature
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem.rdchem import Mol
from frust.utils.dirs import make_step_dir, prepare_base_dir
from frust.utils.provenance import (
    calculator_provenance,
    env_executable,
    env_path,
    gxtb_executable,
    oet_executable,
)
from frust.utils.slurm import detect_job_id
from frust.schema import (
    energy_columns,
    infer_group_columns,
    metadata_from_mapping,
    normalize_dataframe,
    output_column,
    parse_structure_name,
)


def _metadata_text(value: str | None) -> str | None:
    """Normalize optional free-text input metadata."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_list(value: list | tuple | None) -> list[str] | None:
    """Normalize optional list metadata to strings."""
    if value is None:
        return None
    return [str(item) for item in value]


def _make_logger(name: str, debug: bool) -> logging.Logger:
    """Create an instance-local logger so debug settings don't leak globally."""
    inst_logger = logging.getLogger(name)
    inst_logger.propagate = False
    inst_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if not inst_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        inst_logger.addHandler(handler)

    return inst_logger


def _logger_name(step_type: str | None, job_id: int | None) -> str:
    """Build a readable, stable logger name for one Stepper instance."""
    step_label = (step_type or "generic").upper()
    job_label = f"job{job_id}" if job_id is not None else "local"
    return f"{__name__}.{step_label}.{job_label}"

class Stepper:
    """Chain xTB and ORCA calculations over dataframe-based conformer tables.

    `Stepper` operates on pandas DataFrames, adds calculation outputs as new
    columns, and optionally manages run directories and saved engine files.
    :meth:`build_initial_df` can also build the initial dataframe from simple
    chemistry inputs such as SMILES strings, XYZ geometries, bare RDKit
    molecules, or raw FRUST structure dictionaries.
    """

    def __init__(
        self,
        step_type: str | None = None,
        output_base: Path | str | None = None,
        job_id: int | None = None,
        debug: bool = False,
        dump_each_step: bool = False,
        n_cores: int = 8,
        memory_gb: float = 20.0,
        save_calc_dirs: bool = False,
        save_output_dir: bool = True,
        work_dir: str | None = None,        
        **kwargs
    ):
        """Configure a Stepper instance.

        Parameters
        ----------
        step_type : str or None, optional
            Workflow label used for built-in constrained TS/INT calculations.
            Supported constrained values are ``TS1``, ``TS2``, ``TS3``,
            ``TS4``, and ``INT3``. Use ``"auto"`` to infer one constrained
            type from the dataframe built by :meth:`build_initial_df`. If
            ``None``, unconstrained workflows still work, but
            ``constraint=True`` in :meth:`xtb` or :meth:`orca` will raise an
            error.
        output_base : Path or str or None, optional
            Base directory under which saved outputs are created. The final run
            directory is created lazily on first save request, not at
            construction time.
        job_id : int or None, optional
            Optional job identifier used in output-directory naming and logger
            naming. If omitted, :func:`frust.utils.slurm.detect_job_id` is used
            to infer one when possible.
        debug : bool, optional
            If ``True``, use mock xTB and ORCA backends instead of the real
            engines and enable debug logging.
        dump_each_step : bool, optional
            Reserved for dataframe dumping workflows. Stored on the instance
            but not currently consumed by :class:`Stepper` itself.
        n_cores : int, optional
            Default core count passed to engine calls unless overridden on an
            individual :meth:`xtb` or :meth:`orca` call.
        memory_gb : float, optional
            Default memory setting in gigabytes used for ORCA calls.
        save_calc_dirs : bool, optional
            If ``True``, preserve full calculation directories for each row
            when output saving is enabled.
        save_output_dir : bool, optional
            If ``True``, enable output-directory creation and file saving.
            When ``False``, calculations still run but no output tree is
            created unless a caller explicitly requests per-step saving.
        work_dir : str or None, optional
            Scratch or working directory passed to the engine wrappers. If
            omitted, ``$SCRATCH`` is used when available, otherwise the current
            directory.

        Raises
        ------
        TypeError
            If unexpected keyword arguments are supplied.
        """
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Stepper keyword arguments: {unknown}")

        self._auto_step_type = step_type is not None and step_type.lower() == "auto"
        self.step_type      = None if self._auto_step_type else (step_type.upper() if step_type is not None else None)
        self.debug          = debug
        job_id              = detect_job_id(job_id, True)
        self.job_id         = job_id
        logger_step_type    = "AUTO" if self._auto_step_type else self.step_type
        self.logger         = _make_logger(_logger_name(logger_step_type, self.job_id), debug)
        self.dump_each_step = dump_each_step
        self.n_cores        = n_cores
        self.memory_gb      = memory_gb
        self.save_calc_dirs = save_calc_dirs
        self.save_output    = save_output_dir
        self.output_base    = Path(output_base) if output_base is not None else None

        self.base_dir = self.output_base if self.output_base is not None else Path(".")

        if self.debug:
            from tooltoad.xtb import mock_xtb_calculate
            from tooltoad.orca import mock_orca_calculate
            from tooltoad.gxtb import mock_gxtb_calculate
            self.xtb_fn  = mock_xtb_calculate
            self.orca_fn = mock_orca_calculate
            self.gxtb_fn = mock_gxtb_calculate
        else:
            from tooltoad.xtb import xtb_calculate
            from tooltoad.orca import orca_calculate
            from tooltoad.gxtb import gxtb_calculate
            self.xtb_fn  = xtb_calculate
            self.orca_fn = orca_calculate
            self.gxtb_fn = gxtb_calculate

        if work_dir:
            self.work_dir = work_dir
            os.makedirs(self.work_dir, exist_ok=True)
            self.logger.info(f"Working dir: {self.work_dir}")
        else:
            try:
                self.work_dir = os.environ["SCRATCH"]
            except Exception:
                self.work_dir = "."
            os.makedirs(self.work_dir, exist_ok=True)
            self.logger.info(f"Working dir: {self.work_dir}")

    def _ensure_base_dir(self) -> Path:
        """Create the output directory lazily on first use."""
        if self.save_output and not getattr(self, "_base_dir_prepared", False):
            self.base_dir = prepare_base_dir(self.output_base, self.job_id)
            self._base_dir_prepared = True
        return self.base_dir

    @staticmethod
    def _coord_columns(df: pd.DataFrame) -> list[str]:
        preferred = []
        if "coords_embedded" in df.columns:
            preferred.append("coords_embedded")
        preferred.extend(
            [
                c
                for c in df.columns
                if "coords" in c and c not in preferred and not c.endswith("-opt_coords")
            ]
        )
        preferred.extend([c for c in df.columns if c.endswith("-opt_coords") or c.endswith("-oc")])
        return preferred

    @classmethod
    def _last_coord_col(cls, df: pd.DataFrame) -> str:
        """Return the most specific available coordinate column."""
        cols = cls._coord_columns(df)
        if not cols:
            raise ValueError(
                "DataFrame must contain coordinates in 'coords_embedded' or a '*-oc'/'*-opt_coords' column"
            )
        return cols[-1]

    def _step_type_upper(self) -> str:
        """Normalize step_type for callers that only need TS dispatch."""
        return (self.step_type or "").upper()

    def _effective_n_cores(self, n_cores: int | None = None) -> int:
        """Return the effective core count for one calculator call."""
        return int(n_cores) if n_cores is not None else int(self.n_cores)

    def _validate_constraint_request(self, df: pd.DataFrame) -> str:
        """Validate that constraint mode is fully specified."""
        step_type = self._step_type_upper()
        if not step_type:
            raise ValueError(
                "`constraint=True` requires `Stepper(step_type=...)` so the correct TS/INT constraints can be selected"
            )
        if step_type not in {"TS1", "TS2", "TS3", "TS4", "INT3"}:
            raise ValueError(
                f"`constraint=True` is only supported for TS1/TS2/TS3/TS4/INT3, got {self.step_type!r}"
            )
        if "constraint_atoms" not in df.columns:
            raise ValueError(
                "`constraint=True` requires a 'constraint_atoms' column in the input DataFrame"
            )
        return step_type

    @staticmethod
    def _validate_required_columns(
        df: pd.DataFrame,
        *,
        needs_grouping: bool = False,
        needs_hessian: bool = False,
    ) -> None:
        missing = [col for col in ("atoms",) if col not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {', '.join(missing)}"
            )

        Stepper._last_coord_col(df)

        if needs_grouping and "substrate_name" not in df.columns:
            raise ValueError(
                "`lowest=` requires a 'substrate_name' column so conformers can be grouped before filtering"
            )

        if needs_hessian and not any(col.endswith(".hess") for col in df.columns):
            raise ValueError(
                "`use_last_hess=True` requires a prior '*.hess' column in the input DataFrame"
            )

    @staticmethod
    def _row_name(row: Series, index: object) -> str:
        """Pick a stable human-readable row label from available metadata."""
        for key in ("custom_name", "substrate_name", "moltype"):
            value = row.get(key)
            if value is not None and not pd.isna(value):
                return str(value)
        return f"row_{index}"

    @staticmethod
    def _row_conf_id(row: Series, index: object) -> str:
        """Pick a stable conformer/run identifier when cid is unavailable."""
        value = row.get("cid")
        if value is not None and not pd.isna(value):
            return str(value)
        return str(index)

    @staticmethod
    def _constraint_atoms(row: Series, min_size: int = 6) -> list[int]:
        """Validate and return constraint atoms for TS/INT workflows."""
        atoms = row.get("constraint_atoms")
        if atoms is None:
            raise ValueError("Missing 'constraint_atoms' for a constrained row")
        if not isinstance(atoms, (list, tuple, np.ndarray)):
            if pd.isna(atoms):
                raise ValueError("Missing 'constraint_atoms' for a constrained row")
            raise ValueError("'constraint_atoms' must be a sequence of atom indices")
        atoms = list(atoms)
        if len(atoms) < min_size:
            raise ValueError(
                f"'constraint_atoms' must contain at least {min_size} entries for constrained workflows"
            )
        return atoms

    @staticmethod
    def _plain_molecule_metadata(label: str, smiles: str | None) -> dict[str, Any]:
        """Build schema metadata for a plain molecule input."""
        return {
            "structure_id": f"MOL:{label}:structure",
            "custom_name": label,
            "substrate_name": label,
            "structure_type": "MOL",
            "molecule_role": "structure",
            "rpos": pd.NA,
            "smiles": smiles,
            "input_smiles": smiles,
        }

    @staticmethod
    def _mol_smiles(mol: Mol) -> str | None:
        """Return a best-effort SMILES string for an RDKit molecule."""
        try:
            return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
        except Exception:
            return None

    @staticmethod
    def _is_path_like(value: Any) -> bool:
        """Return True for pathlib/os path inputs."""
        return isinstance(value, (Path, os.PathLike))

    @classmethod
    def _looks_like_xyz_path(cls, value: Any) -> bool:
        """Return True when an input should be treated as an XYZ path."""
        if cls._is_path_like(value):
            return True
        return isinstance(value, str) and Path(value).suffix.lower() == ".xyz"

    @staticmethod
    def _looks_like_xyz_block(value: Any) -> bool:
        """Return True for strings that look like an XYZ geometry block."""
        if not isinstance(value, str) or "\n" not in value:
            return False
        lines = value.strip().splitlines()
        if len(lines) < 3:
            return False
        try:
            n_atoms = int(lines[0].strip())
        except ValueError:
            return False
        return n_atoms >= 1 and len(lines) >= n_atoms + 2

    @classmethod
    def _is_xyz_source(cls, value: Any) -> bool:
        """Return True when a value should be parsed as XYZ input."""
        return cls._looks_like_xyz_path(value) or cls._looks_like_xyz_block(value)

    @staticmethod
    def _parse_xyz_block(text: str, *, label: str) -> tuple[list[str], list[tuple[float, float, float]]]:
        """Parse an XYZ block into atoms and coordinates without bond perception."""
        lines = text.strip().splitlines()
        try:
            n_atoms = int(lines[0].strip())
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid XYZ block for {label!r}: first line must be an atom count") from exc

        atom_lines = lines[2 : 2 + n_atoms]
        if len(atom_lines) != n_atoms:
            raise ValueError(
                f"Invalid XYZ block for {label!r}: expected {n_atoms} atom lines, got {len(atom_lines)}"
            )

        atoms: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for line_no, line in enumerate(atom_lines, start=3):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid XYZ block for {label!r}: line {line_no} must contain element, x, y, z"
                )
            atom = parts[0]
            try:
                xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid XYZ block for {label!r}: line {line_no} has non-numeric coordinates"
                ) from exc
            atoms.append(atom)
            coords.append(xyz)

        return atoms, coords

    @classmethod
    def _read_xyz_source(cls, source: str | Path | os.PathLike, *, label: str) -> tuple[list[str], list[tuple[float, float, float]]]:
        """Read atoms and coordinates from an XYZ path or block."""
        if cls._looks_like_xyz_path(source):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"XYZ file does not exist for {label!r}: {path}")
            return cls._parse_xyz_block(path.read_text(), label=label)
        if cls._looks_like_xyz_block(source):
            return cls._parse_xyz_block(str(source), label=label)
        raise ValueError(f"Input for {label!r} is not an XYZ path or XYZ block")

    @staticmethod
    def _xyz_to_mol(atoms: list[str], coords: list[tuple[float, float, float]]) -> Mol:
        """Create an RDKit molecule with atoms and one conformer, preserving XYZ geometry."""
        rw_mol = Chem.RWMol()
        for symbol in atoms:
            rw_mol.AddAtom(Chem.Atom(symbol))
        mol = rw_mol.GetMol()
        conf = Chem.Conformer(len(atoms))
        for idx, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
        return mol

    @staticmethod
    def _smiles_to_mol(smiles: str, label: str) -> Mol:
        """Parse a SMILES string into an RDKit molecule with a clear error."""
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES for {label!r}: {smiles!r}")
        return mol

    @classmethod
    def _smiles_records_to_raw_mols(
        cls,
        smiles_values: list[str],
        labels: list[str],
    ) -> dict[str, tuple[Mol, dict[str, Any]]]:
        """Convert named SMILES records into raw molecules plus metadata."""
        raw: dict[str, tuple[Mol, dict[str, Any]]] = {}
        for smiles, label in zip(smiles_values, labels):
            if pd.isna(smiles):
                raise ValueError(f"Missing SMILES for {label!r}")
            smiles_text = str(smiles)
            raw[label] = (
                cls._smiles_to_mol(smiles_text, label),
                cls._plain_molecule_metadata(label, smiles_text),
            )
        return raw

    @classmethod
    def _xyz_records_to_embedded(
        cls,
        xyz_values: list[str | Path | os.PathLike],
        labels: list[str],
    ) -> dict[str, tuple[Mol, list[int], dict[str, Any]]]:
        """Convert named XYZ sources into embedded records with preserved coordinates."""
        embedded: dict[str, tuple[Mol, list[int], dict[str, Any]]] = {}
        for source, label in zip(xyz_values, labels):
            atoms, coords = cls._read_xyz_source(source, label=label)
            mol = cls._xyz_to_mol(atoms, coords)
            embedded[label] = (
                mol,
                [0],
                cls._plain_molecule_metadata(label, None),
            )
        return embedded

    @classmethod
    def _rdkit_records_to_embedded(
        cls,
        mols: list[Mol],
        labels: list[str],
        *,
        n_confs: int | None,
        n_cores: int,
        optimization: str,
        max_iters: int,
    ) -> dict[str, tuple]:
        """Normalize bare RDKit molecules, preserving conformers where present."""
        embedded: dict[str, tuple] = {}
        raw_to_embed: dict[str, tuple[Mol, dict[str, Any]]] = {}

        for mol, label in zip(mols, labels):
            if not isinstance(mol, Mol):
                raise TypeError(f"Expected an RDKit Mol for {label!r}, got {type(mol).__name__}")
            smiles = cls._mol_smiles(mol)
            metadata = cls._plain_molecule_metadata(label, smiles)
            if mol.GetNumConformers() > 0:
                embedded[label] = (
                    mol,
                    [conf.GetId() for conf in mol.GetConformers()],
                    metadata,
                )
            else:
                raw_to_embed[label] = (mol, metadata)

        if raw_to_embed:
            from frust.embedder import embed_mols

            embedded.update(
                embed_mols(
                    raw_to_embed,
                    n_confs=n_confs,
                    n_cores=n_cores,
                    optimization=optimization,
                    max_iters=max_iters,
                )
            )

        return embedded

    @staticmethod
    def _sequence_is_cids(value: Any) -> bool:
        """Return True for conformer-id sequences in embedded tuple values."""
        if isinstance(value, (str, bytes, dict)):
            return False
        if not isinstance(value, (list, tuple, np.ndarray)):
            return False
        return all(isinstance(item, (int, np.integer)) for item in value)

    @classmethod
    def _dict_value_kind(cls, value: Any) -> str:
        """Classify one build_initial_df dictionary value."""
        if cls._is_xyz_source(value):
            return "xyz"
        if isinstance(value, str):
            return "smiles"
        if isinstance(value, Mol):
            return "rdkit_mol"
        if not isinstance(value, tuple):
            return "unknown"
        if len(value) == 2:
            first, second = value
            if isinstance(first, Mol) and isinstance(second, dict):
                return "raw_mol"
            if isinstance(first, Mol) and cls._sequence_is_cids(second):
                return "embedded_mol"
        if len(value) == 3:
            first, second, third = value
            if isinstance(first, Mol) and cls._sequence_is_cids(second) and isinstance(third, dict):
                return "embedded_mol"
            if isinstance(first, Mol) and cls._sequence_is_cids(second) and isinstance(third, str):
                return "raw_ts"
        if len(value) == 4:
            first, second, third, fourth = value
            if (
                isinstance(first, Mol)
                and cls._sequence_is_cids(second)
                and isinstance(third, (list, tuple, np.ndarray))
                and isinstance(fourth, str)
            ):
                return "embedded_ts"
        if len(value) == 5:
            first, second, third, fourth, fifth = value
            if (
                isinstance(first, Mol)
                and cls._sequence_is_cids(second)
                and isinstance(third, (list, tuple, np.ndarray))
                and isinstance(fourth, str)
                and isinstance(fifth, (list, tuple))
            ):
                return "embedded_ts"
        return "unknown"

    @classmethod
    def _dict_kind(cls, data: dict[str, Any]) -> str:
        """Classify a dictionary input by requiring one consistent value kind."""
        if not data:
            raise ValueError("build_initial_df received an empty dictionary")
        kinds = {cls._dict_value_kind(value) for value in data.values()}
        if "unknown" in kinds:
            raise ValueError(
                "Could not classify build_initial_df dictionary input; expected SMILES, XYZ, RDKit molecules, raw molecules, raw TS/INT structures, or embedded structures"
            )
        embedded = {"embedded_mol", "embedded_ts"}
        if kinds <= embedded:
            return "embedded"
        if len(kinds) > 1:
            raise ValueError(
                "Mixed build_initial_df dictionary input is not supported; pass one kind of input at a time"
            )
        return kinds.pop()

    @staticmethod
    def _labels_from_names(
        count: int,
        *,
        name: str | None,
        names: list[str] | tuple[str, ...] | None,
        allow_name: bool,
        allow_names: bool,
    ) -> list[str]:
        """Resolve single or batch names into stable molecule labels."""
        if name is not None:
            if not allow_name or count != 1:
                raise ValueError("`name=` is only valid for a single SMILES string input")
            return [str(name)]
        if names is not None:
            if not allow_names:
                raise ValueError("`names=` is only valid for batch SMILES list inputs")
            if len(names) != count:
                raise ValueError(
                    f"`names=` must contain {count} labels, got {len(names)}"
                )
            return [str(item) for item in names]
        return [f"mol_{i}" for i in range(count)]

    @staticmethod
    def _df_labels(df: pd.DataFrame) -> list[str]:
        """Choose labels for a SMILES dataframe input."""
        for col in ("substrate_name", "name", "custom_name"):
            if col in df.columns:
                return [str(value) for value in df[col].tolist()]
        return [f"mol_{i}" for i in range(len(df))]

    def _raw_smiles_input_to_mols(
        self,
        structures: Any,
        *,
        name: str | None,
        names: list[str] | tuple[str, ...] | None,
    ) -> dict[str, tuple[Mol, dict[str, Any]]]:
        """Normalize SMILES-like user inputs into raw molecule dictionaries."""
        if isinstance(structures, str):
            labels = self._labels_from_names(
                1,
                name=name,
                names=names,
                allow_name=True,
                allow_names=False,
            )
            return self._smiles_records_to_raw_mols([structures], labels)

        if isinstance(structures, pd.DataFrame):
            if name is not None or names is not None:
                raise ValueError(
                    "Use dataframe columns such as 'substrate_name' to label dataframe inputs; `name=` and `names=` are not accepted with dataframe input"
                )
            if "smiles" not in structures.columns:
                raise ValueError("DataFrame input to build_initial_df must contain a 'smiles' column")
            labels = self._df_labels(structures)
            return self._smiles_records_to_raw_mols(structures["smiles"].tolist(), labels)

        if isinstance(structures, (list, tuple)):
            if not all(isinstance(item, str) for item in structures):
                raise ValueError("List input to build_initial_df must contain only SMILES strings")
            labels = self._labels_from_names(
                len(structures),
                name=name,
                names=names,
                allow_name=False,
                allow_names=True,
            )
            return self._smiles_records_to_raw_mols(list(structures), labels)

        if isinstance(structures, dict) and self._dict_kind(structures) == "smiles":
            if name is not None or names is not None:
                raise ValueError("Named SMILES dictionaries already provide labels; do not pass `name=` or `names=`")
            return self._smiles_records_to_raw_mols(
                [str(value) for value in structures.values()],
                [str(key) for key in structures.keys()],
            )

        raise TypeError(f"Unsupported SMILES input type: {type(structures).__name__}")

    def _xyz_input_to_embedded(
        self,
        structures: Any,
        *,
        name: str | None,
        names: list[str] | tuple[str, ...] | None,
    ) -> tuple[dict[str, tuple[Mol, list[int], dict[str, Any]]], str]:
        """Normalize XYZ inputs into embedded dictionaries."""
        if self._is_xyz_source(structures):
            if self._looks_like_xyz_path(structures):
                default = Path(structures).stem
                labels = [str(name) if name is not None else default]
                if names is not None:
                    raise ValueError("`names=` is only valid for batch XYZ inputs")
                input_kind = "xyz_path"
            else:
                if name is None:
                    raise ValueError("`name=` is required for a single XYZ block input")
                if names is not None:
                    raise ValueError("`names=` is only valid for batch XYZ inputs")
                labels = [str(name)]
                input_kind = "xyz_block"
            return self._xyz_records_to_embedded([structures], labels), input_kind

        if isinstance(structures, pd.DataFrame) and "xyz" in structures.columns and "smiles" not in structures.columns:
            if name is not None or names is not None:
                raise ValueError(
                    "Use dataframe columns such as 'substrate_name' to label dataframe inputs; `name=` and `names=` are not accepted with dataframe input"
                )
            labels = self._df_labels(structures)
            return self._xyz_records_to_embedded(structures["xyz"].tolist(), labels), "xyz_dataframe"

        if isinstance(structures, (list, tuple)):
            if not structures:
                raise ValueError("build_initial_df received an empty XYZ list")
            if not all(self._is_xyz_source(item) for item in structures):
                raise ValueError("List input to build_initial_df cannot mix XYZ inputs with other input types")
            labels = self._labels_from_names(
                len(structures),
                name=name,
                names=names,
                allow_name=False,
                allow_names=True,
            )
            return self._xyz_records_to_embedded(list(structures), labels), "xyz_list"

        if isinstance(structures, dict) and self._dict_kind(structures) == "xyz":
            if name is not None or names is not None:
                raise ValueError("Named XYZ dictionaries already provide labels; do not pass `name=` or `names=`")
            return self._xyz_records_to_embedded(
                list(structures.values()),
                [str(key) for key in structures.keys()],
            ), "xyz_dict"

        raise TypeError(f"Unsupported XYZ input type: {type(structures).__name__}")

    def _rdkit_input_to_embedded(
        self,
        structures: Any,
        *,
        name: str | None,
        names: list[str] | tuple[str, ...] | None,
        n_confs: int | None,
        n_cores: int,
        optimization: str,
        max_iters: int,
    ) -> tuple[dict[str, tuple], str]:
        """Normalize bare RDKit Mol inputs into embedded dictionaries."""
        if isinstance(structures, Mol):
            labels = self._labels_from_names(
                1,
                name=name,
                names=names,
                allow_name=True,
                allow_names=False,
            )
            return self._rdkit_records_to_embedded(
                [structures],
                labels,
                n_confs=n_confs,
                n_cores=n_cores,
                optimization=optimization,
                max_iters=max_iters,
            ), "rdkit_mol"

        if isinstance(structures, (list, tuple)) and structures and all(isinstance(item, Mol) for item in structures):
            labels = self._labels_from_names(
                len(structures),
                name=name,
                names=names,
                allow_name=False,
                allow_names=True,
            )
            return self._rdkit_records_to_embedded(
                list(structures),
                labels,
                n_confs=n_confs,
                n_cores=n_cores,
                optimization=optimization,
                max_iters=max_iters,
            ), "rdkit_mol_list"

        if isinstance(structures, dict) and self._dict_kind(structures) == "rdkit_mol":
            if name is not None or names is not None:
                raise ValueError("Named RDKit Mol dictionaries already provide labels; do not pass `name=` or `names=`")
            return self._rdkit_records_to_embedded(
                list(structures.values()),
                [str(key) for key in structures.keys()],
                n_confs=n_confs,
                n_cores=n_cores,
                optimization=optimization,
                max_iters=max_iters,
            ), "rdkit_mol_dict"

        raise TypeError(f"Unsupported RDKit Mol input type: {type(structures).__name__}")

    @staticmethod
    def _unique_constrained_types(df: pd.DataFrame) -> set[str]:
        """Return constrained TS/INT structure types present in a dataframe."""
        if "structure_type" not in df.columns:
            return set()
        constrained = {"TS1", "TS2", "TS3", "TS4", "INT3"}
        return {
            str(value).upper()
            for value in df["structure_type"].dropna().unique()
            if str(value).upper() in constrained
        }

    def _resolve_auto_step_type(self, df: pd.DataFrame) -> None:
        """Resolve or validate the Stepper constraint type from dataframe metadata."""
        constrained_types = self._unique_constrained_types(df)
        if self._auto_step_type:
            if len(constrained_types) > 1:
                types = ", ".join(sorted(constrained_types))
                raise ValueError(f"Cannot infer one step_type from mixed constrained structure types: {types}")
            if constrained_types:
                self.step_type = next(iter(constrained_types))
            return

        if self.step_type in {"TS1", "TS2", "TS3", "TS4", "INT3"} and constrained_types:
            if constrained_types != {self.step_type}:
                types = ", ".join(sorted(constrained_types))
                raise ValueError(
                    f"Stepper(step_type={self.step_type!r}) does not match dataframe structure_type values: {types}"
                )

    def _infer_ts_type(self, raw_ts_dict: dict[str, Any], ts_type: str | None) -> str:
        """Choose the TS/INT embedding type for raw TS dictionary inputs."""
        constrained = {"TS1", "TS2", "TS3", "TS4", "INT3"}
        if ts_type is not None:
            inferred = ts_type.upper()
        elif self.step_type in constrained:
            inferred = self.step_type
        else:
            parsed_types = {
                parse_structure_name(name).structure_type.upper()
                for name in raw_ts_dict
            }
            parsed_types = parsed_types & constrained
            if len(parsed_types) != 1:
                types = ", ".join(sorted(parsed_types)) or "none"
                raise ValueError(
                    "Raw TS/INT dictionaries require one inferable TS type "
                    f"or an explicit `ts_type=`, got {types}"
                )
            inferred = next(iter(parsed_types))

        if inferred not in constrained:
            raise ValueError(f"`ts_type=` must be one of TS1/TS2/TS3/TS4/INT3, got {ts_type!r}")
        return inferred

    def _with_initial_df_attrs(
        self,
        df: pd.DataFrame,
        *,
        input_kind: str,
        workflow: str | None,
        n_confs: int | None,
        n_cores: int,
        optimization: str | None = None,
        max_iters: int | None = None,
        select_mols: str | list[str] | None = None,
        ts_type: str | None = None,
        ts_optimize: bool | None = None,
    ) -> pd.DataFrame:
        """Attach normalized initial-dataframe provenance to attrs."""
        configured_step_type = "auto" if self._auto_step_type else self.step_type
        df.attrs["frust_initial_df"] = {
            "input_kind": input_kind,
            "workflow": workflow,
            "n_confs": n_confs,
            "n_cores": n_cores,
            "optimization": optimization,
            "max_iters": max_iters,
            "select_mols": select_mols,
            "ts_type": ts_type,
            "ts_optimize": ts_optimize,
            "step_type": configured_step_type,
            "resolved_step_type": self.step_type,
        }
        return df

    def build_initial_df(
        self,
        structures: Any,
        *,
        name: str | None = None,
        names: list[str] | tuple[str, ...] | None = None,
        n_confs: int | None = 1,
        n_cores: int | None = None,
        optimization: str = "none",
        max_iters: int = 100,
        workflow: str | None = None,
        select_mols: str | list[str] = "all",
        ts_type: str | None = None,
        ts_optimize: bool | None = None,
        optimize: bool | None = None,
    ) -> pd.DataFrame:
        """Build a FRUST conformer dataframe from embedded or raw structures.

        Parameters
        ----------
        structures : dict, str, pathlib.Path, list, pandas.DataFrame, rdkit.Chem.Mol
            Input structures. Existing embedded dictionaries keep the previous
            behavior. Raw SMILES inputs, XYZ paths/blocks, bare RDKit
            molecules, raw molecule dictionaries, and raw TS/INT dictionaries
            are normalized before dataframe construction. XYZ geometry is
            preserved as supplied.
        name : str, optional
            Label for a single SMILES, XYZ block/path, or RDKit molecule
            input. Written to ``substrate_name``.
        names : list[str], optional
            Labels for batch SMILES, XYZ, or RDKit molecule lists.
        n_confs : int or None, optional
            Number of conformers generated for raw inputs. Defaults to ``1``
            for quick calculator-style setup. Use ``None`` for FRUST's
            rotatable-bond heuristic.
        n_cores : int or None, optional
            Core count for RDKit embedding. Defaults to ``self.n_cores``.
        optimization : str, optional
            Force-field optimization passed to :func:`frust.embedder.embed_mols`
            for plain molecule inputs.
        workflow : str or None, optional
            Explicit workflow expansion for SMILES inputs. Currently only
            ``"mols"`` is supported; bare SMILES always means a plain molecule.
        select_mols : str or list[str], optional
            Selection forwarded to ``create_mol_per_rpos`` when
            ``workflow="mols"``.
        ts_type : str or None, optional
            Explicit TS/INT embedding type for raw TS dictionaries when it
            cannot be inferred from structure names.
        ts_optimize : bool or None, optional
            Whether raw TS/INT embedding should run constrained UFF
            optimization. Defaults to ``False``.
        optimize : bool or None, optional
            Backward-friendly alias for ``ts_optimize``.

        Returns
        -------
        pandas.DataFrame
            One row per conformer with FRUST structure metadata, atoms, and
            ``coords_embedded``.
        """
        embed_cores = self.n_cores if n_cores is None else n_cores

        if workflow is not None:
            workflow_name = workflow.lower()
            if workflow_name != "mols":
                raise ValueError(f"Unsupported build_initial_df workflow: {workflow!r}")
            from frust.utils.mols import create_mol_per_rpos

            if isinstance(structures, pd.DataFrame):
                if name is not None or names is not None:
                    raise ValueError(
                        "Use dataframe columns such as 'substrate_name' to label dataframe inputs; `name=` and `names=` are not accepted with dataframe input"
                    )
                if "smiles" not in structures.columns:
                    raise ValueError("DataFrame input to build_initial_df must contain a 'smiles' column")
                ligand_df = structures.copy()
            else:
                smiles_mols = self._raw_smiles_input_to_mols(
                    structures,
                    name=name,
                    names=names,
                )
                ligand_df = pd.DataFrame(
                    {
                        "smiles": [
                            metadata["input_smiles"]
                            for _, metadata in smiles_mols.values()
                        ],
                    }
                )
            raw_mols = create_mol_per_rpos(
                ligand_df,
                return_format="dict",
                select_mols=select_mols,
            )
            structures = raw_mols
            name = None
            names = None

        xyz_like = (
            self._is_xyz_source(structures)
            or (
                isinstance(structures, pd.DataFrame)
                and "xyz" in structures.columns
                and "smiles" not in structures.columns
            )
            or (
                isinstance(structures, (list, tuple))
                and any(self._is_xyz_source(item) for item in structures)
            )
        )
        if xyz_like:
            embedded, input_kind = self._xyz_input_to_embedded(
                structures,
                name=name,
                names=names,
            )
            df = self._build_initial_df_from_embedded(embedded)
            self._resolve_auto_step_type(df)
            return self._with_initial_df_attrs(
                df,
                input_kind=input_kind,
                workflow=workflow,
                n_confs=None,
                n_cores=embed_cores,
            )

        rdkit_like = (
            isinstance(structures, Mol)
            or (
                isinstance(structures, (list, tuple))
                and bool(structures)
                and all(isinstance(item, Mol) for item in structures)
            )
        )
        if rdkit_like:
            embedded, input_kind = self._rdkit_input_to_embedded(
                structures,
                name=name,
                names=names,
                n_confs=n_confs,
                n_cores=embed_cores,
                optimization=optimization,
                max_iters=max_iters,
            )
            df = self._build_initial_df_from_embedded(embedded)
            self._resolve_auto_step_type(df)
            return self._with_initial_df_attrs(
                df,
                input_kind=input_kind,
                workflow=workflow,
                n_confs=n_confs,
                n_cores=embed_cores,
                optimization=optimization,
                max_iters=max_iters,
            )

        if isinstance(structures, dict):
            kind = self._dict_kind(structures)
            if kind == "xyz":
                embedded, input_kind = self._xyz_input_to_embedded(
                    structures,
                    name=name,
                    names=names,
                )
                df = self._build_initial_df_from_embedded(embedded)
                self._resolve_auto_step_type(df)
                return self._with_initial_df_attrs(
                    df,
                    input_kind=input_kind,
                    workflow=workflow,
                    n_confs=None,
                    n_cores=embed_cores,
                )
            if kind == "rdkit_mol":
                embedded, input_kind = self._rdkit_input_to_embedded(
                    structures,
                    name=name,
                    names=names,
                    n_confs=n_confs,
                    n_cores=embed_cores,
                    optimization=optimization,
                    max_iters=max_iters,
                )
                df = self._build_initial_df_from_embedded(embedded)
                self._resolve_auto_step_type(df)
                return self._with_initial_df_attrs(
                    df,
                    input_kind=input_kind,
                    workflow=workflow,
                    n_confs=n_confs,
                    n_cores=embed_cores,
                    optimization=optimization,
                    max_iters=max_iters,
                )
            if kind == "embedded":
                df = self._build_initial_df_from_embedded(structures)
                self._resolve_auto_step_type(df)
                return self._with_initial_df_attrs(
                    df,
                    input_kind="embedded_dict",
                    workflow=workflow,
                    n_confs=None,
                    n_cores=embed_cores,
                )
            if kind == "raw_mol":
                from frust.embedder import embed_mols

                embedded = embed_mols(
                    structures,
                    n_confs=n_confs,
                    n_cores=embed_cores,
                    optimization=optimization,
                    max_iters=max_iters,
                )
                df = self._build_initial_df_from_embedded(embedded)
                self._resolve_auto_step_type(df)
                input_kind = f"workflow_{workflow}" if workflow else "raw_mol_dict"
                return self._with_initial_df_attrs(
                    df,
                    input_kind=input_kind,
                    workflow=workflow,
                    n_confs=n_confs,
                    n_cores=embed_cores,
                    optimization=optimization,
                    max_iters=max_iters,
                    select_mols=select_mols if workflow else None,
                )
            if kind == "raw_ts":
                from frust.embedder import embed_ts

                if ts_optimize is None:
                    ts_optimize = bool(optimize) if optimize is not None else False
                inferred_ts_type = self._infer_ts_type(structures, ts_type)
                embedded = embed_ts(
                    structures,
                    ts_type=inferred_ts_type,
                    n_confs=n_confs,
                    n_cores=embed_cores,
                    optimize=ts_optimize,
                )
                df = self._build_initial_df_from_embedded(embedded)
                self._resolve_auto_step_type(df)
                return self._with_initial_df_attrs(
                    df,
                    input_kind="raw_ts_dict",
                    workflow=workflow,
                    n_confs=n_confs,
                    n_cores=embed_cores,
                    ts_type=inferred_ts_type,
                    ts_optimize=ts_optimize,
                )
            if kind == "smiles":
                raw_mols = self._raw_smiles_input_to_mols(
                    structures,
                    name=name,
                    names=names,
                )
                df = self.build_initial_df(
                    raw_mols,
                    n_confs=n_confs,
                    n_cores=embed_cores,
                    optimization=optimization,
                    max_iters=max_iters,
                )
                df.attrs["frust_initial_df"]["input_kind"] = "smiles_dict"
                return df

        raw_mols = self._raw_smiles_input_to_mols(
            structures,
            name=name,
            names=names,
        )
        if isinstance(structures, str):
            input_kind = "smiles"
        elif isinstance(structures, pd.DataFrame):
            input_kind = "smiles_dataframe"
        elif isinstance(structures, (list, tuple)):
            input_kind = "smiles_list"
        else:
            input_kind = "smiles"

        df = self.build_initial_df(
            raw_mols,
            n_confs=n_confs,
            n_cores=embed_cores,
            optimization=optimization,
            max_iters=max_iters,
        )
        df.attrs["frust_initial_df"]["input_kind"] = input_kind
        return df

    def _build_initial_df_from_embedded(self, embedded_dict: dict) -> pd.DataFrame:
        """
        Turn a dictionary of embedded‐conformer data into a tidy DataFrame.

        The dictionary keys can be either:
          1) TS names of the form 'TS(..._rpos(N))' (with a 4‐ or 5‐tuple value)
          2) plain molecule names (without '_rpos(...)') and a 2‐tuple value

        For TS entries:
            key = 'TS(molname_rpos(N))'
            value = (Mol_with_H, cids, keep_idxs, smiles[, energies])

        For plain‐mol entries:
            key = 'some_name'
            value = (Mol, cids)

        In legacy-input cases, structure metadata is parsed from the key as a
        fallback. New generated records should carry metadata directly.

        Returns
        -------
        pd.DataFrame
            One row per conformer with columns
              - structure_id        (stable structure identifier)
              - custom_name         (the original display/file key)
              - substrate_name      (parsed substrate identity)
              - structure_type      (MOL, TS1, TS2, TS3, TS4, INT3)
              - molecule_role       (ts, ligand, int2, mol2, ...)
              - rpos                (int or None)
              - constraint_atoms    (list[int] or NA)
              - cid                 (conformer ID)
              - smiles              (str or None)
              - atoms               (list of atomic symbols)
              - coords_embedded     (list of (x,y,z) tuples)
              - energy_uff          (float or None)
        """
        rows: list[dict] = []

        for name, val in embedded_dict.items():
            if len(val) == 2:
                mol, cids = val
                keep_idxs = None
                smi = None
                energies: list[tuple[float,int]] = []
                metadata = None
            elif len(val) == 3 and isinstance(val[2], dict):
                mol, cids, metadata = val
                keep_idxs = None
                smi = metadata.get("smiles") or metadata.get("input_smiles")
                energies = []
            elif len(val) == 4:
                mol, cids, keep_idxs, smi = val
                energies = []
                metadata = None
            elif len(val) == 5:
                mol, cids, keep_idxs, smi, energies = val
                metadata = None
            else:
                raise ValueError(f"Bad tuple length for {name}")

            meta = metadata_from_mapping(metadata, fallback_name=name, smiles=smi)

            e_map: dict[int,float] = {cid_val: e_val for (e_val, cid_val) in energies} if energies else {}

            atom_syms = [atom.GetSymbol() for atom in mol.GetAtoms()]

            for cid in cids:
                conf = mol.GetConformer(cid)
                coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

                rows.append({
                    "structure_id":     meta.structure_id,
                    "custom_name":      meta.custom_name,
                    "substrate_name":   meta.substrate_name,
                    "structure_type":   meta.structure_type,
                    "molecule_role":    meta.molecule_role,
                    "rpos":             meta.rpos,
                    "constraint_atoms": keep_idxs,
                    "cid":              cid,
                    "smiles":           meta.smiles or smi,
                    "input_smiles":     meta.input_smiles or smi,
                    "atoms":            atom_syms,
                    "coords_embedded":  coords,
                    "energy_uff":       e_map.get(cid, None)
                })

        return pd.DataFrame(rows)

    def _run_engine(
            self,
            df: pd.DataFrame,
            engine_fn: Callable[..., dict],
            prefix: str,
            build_inputs: Callable[[Series], dict],
            save_step: bool,
            lowest: int | None,
            save_files: list[str] | None = None,
            use_last_hess: bool = False,
        ) -> pd.DataFrame:
        """
        Generic runner for xTB or ORCA or any other engine.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing at least ``atoms`` and one coordinate
            column. Depending on the options used, additional columns may be
            required, such as ``substrate_name`` for ``lowest=`` filtering or a
            ``*.hess`` column for Hessian reuse.
        engine_fn : callable
            Backend calculation function such as the wrapped xTB or ORCA
            driver.
        prefix : str
            Prefix used to name output columns for this calculation stage.
        build_inputs : callable
            Row-wise callback that returns backend-specific keyword arguments.
            These inputs are merged with the generic engine inputs assembled by
            :class:`Stepper`.
        save_step : bool
            If ``True``, preserve the full saved directory for each processed
            row.
        lowest : int or None
            If set, keep only the lowest-energy rows per structure grouping before
            passing data to the engine.
        save_files : list of str or None, optional
            Specific files to save from the engine output when partial saving is
            requested.
        use_last_hess : bool, optional
            If ``True``, reuse the most recent ``*.hess`` column from the
            dataframe by passing it back into the engine as an input file.

        Returns
        -------
        pandas.DataFrame
            A copy of the input dataframe with new stage-prefixed output
            columns added.

        Notes
        -----
        - Rows with missing coordinates are skipped and receive
          ``*-NT=False`` and ``*-EE=NaN``.
        - Engine exceptions are caught, logged, and stored as a stage-specific
          ``*-error`` column instead of aborting the whole dataframe run.
        - Output directories are created lazily and only when saving is
          requested.
        """
        import pandas as _pd

        df_out    = normalize_dataframe(df)
        df_out.attrs.update(getattr(df, "attrs", {}))
        self._validate_required_columns(
            df_out,
            needs_grouping=lowest is not None,
            needs_hessian=use_last_hess,
        )
        coord_col = self._last_coord_col(df_out)
        all_row_data: list[dict[str, object]] = []

        if lowest is not None and lowest < 1:
            self.logger.warning(f"ignoring lowest={lowest!r}, must be ≥1")
            lowest = None

        if lowest:
            e_cols = energy_columns(df_out)
            if not e_cols:
                raise ValueError("cannot apply `lowest=` filter: no energy column found")
            last_energy = e_cols[-1]

            group_keys = infer_group_columns(df_out)
            if not group_keys:
                raise ValueError("cannot apply `lowest=` filter: no structure identity columns found")

            sort_keys = group_keys + [last_energy]
            df_out = (
                df_out
                .sort_values(sort_keys, na_position="last")
                .groupby(group_keys, dropna=False)
                .head(lowest)
            )

        for i, row in df_out.iterrows():
            coords = row[coord_col]
            # --- skip any row with no coords ---
            # if coords is None or (_pd.isna(coords) if not isinstance(coords, (list, tuple)) else False):
            if coords is None or (_pd.isna(coords) if not isinstance(coords, (list, tuple, np.ndarray)) else False):            
                all_row_data.append({
                    output_column(prefix, "normal_termination"): False,
                    output_column(prefix, "electronic_energy"):  np.nan
                })
                continue

            row_save_files = save_files
            save_full_calc = (self.save_calc_dirs and self.save_output) or save_step
            save_partial   = row_save_files is not None and not save_step

            if save_full_calc or save_partial:
                self._ensure_base_dir()
                row_name = self._row_name(row, i)
                row_conf_id = self._row_conf_id(row, i)
                dir_name = f"{row_name}_{row_conf_id}"
                engine_base = self.base_dir / prefix
                engine_base.mkdir(parents=True, exist_ok=True)
                created_dir = str(make_step_dir(engine_base, dir_name))
            else:
                created_dir = None

            if save_full_calc:
                calc_dir   = created_dir
                save_dir   = created_dir
                row_save_files = None
            elif save_partial:
                calc_dir   = None
                save_dir   = created_dir
            else:
                calc_dir   = None
                save_dir   = None
            
            if use_last_hess:
                last_hess_col = [col for col in df.columns if col.endswith(".hess")][-1]
                last_hess = {"private_input.hess": row[last_hess_col]}

            base_args = {
                "atoms":  row["atoms"],
                "coords": [list(c) for c in coords],
                "n_cores": self.n_cores,
                "scr": self.work_dir,
                "data2file": last_hess if use_last_hess else None
            }

            if calc_dir is not None:
                base_args["calc_dir"] = calc_dir
            if save_dir is not None:
                base_args["save_dir"] = save_dir
            if row_save_files is not None:
                base_args["save_files"] = row_save_files

            inputs = {**base_args, **build_inputs(row)}

            # Filter inputs to only those accepted by engine_fn (avoids unexpected kwargs)
            try:
                allowed = set(signature(engine_fn).parameters.keys())
                filtered = {k: v for k, v in inputs.items() if k in allowed}
                if self.debug:
                    dropped = sorted(set(inputs) - set(filtered))
                    if dropped:
                        self.logger.debug(f"[{prefix}] dropped unsupported kwargs: {dropped}")
                inputs = filtered
            except Exception:
                # If introspection fails for any reason, fall back to original inputs
                pass

            # step 3: run engine, catch exceptions
            row_name = self._row_name(row, i)
            self.logger.info(f"[{prefix}] row {i} ({row_name})…")
            try:
                out = engine_fn(**inputs) or {}
            except Exception as e:
                self.logger.exception(f"[{prefix}] row {i} ({row_name}) failed: {e}")
                out = {
                    "normal_termination": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            finally:
                if save_step and save_dir:
                    try:
                        import shutil
                        from pathlib import Path

                        row_conf_id = self._row_conf_id(row, i)
                        dir_name = f"{row_name}_{row_conf_id}"

                        candidates = list(Path(self.work_dir).rglob(dir_name))
                        src = max(candidates, key=lambda p: len(p.parts)) if candidates else None

                        if src is None:
                            self.logger.warning(f"No calc dir named '{dir_name}' found under {self.work_dir}")
                        else:
                            save_path = Path(save_dir)
                            if src.resolve() != save_path.resolve():
                                for p in src.iterdir():
                                    dst = save_path / p.name
                                    if p.is_dir():
                                        shutil.copytree(p, dst, dirs_exist_ok=True)
                                    else:
                                        shutil.copy2(p, dst)
                    except Exception as e:
                        self.logger.error(f"Failed to stage files from '{src}' to '{save_dir}': {e}")

            # step 4: enforce defaults
            out.setdefault("normal_termination", False)
            if out["normal_termination"]:
                out.setdefault("electronic_energy", np.nan)
            # step 5: pick only the keys we know how to handle
            allowed = {"normal_termination", "electronic_energy", "opt_coords", "vibs", "error"}
            def _filelike(k: str) -> bool:
                # Accept only filename-ish keys (no spaces), or our private_* stash
                if not isinstance(k, str) or " " in k:
                    return False
                if k.startswith("private_"):
                    return True
                return bool(re.search(
                    r"\.(hess|xyz|inp|out|log|gbw|molden|wfn|txt|json)$", k
                ))

            for k in out.keys():
                if _filelike(k):
                    allowed.add(k)

            if "vibs" in out and "gibbs_energy" in out:
                allowed.add("gibbs_energy")

            if self.debug:
                ignored = sorted([k for k in out.keys()
                                if '.' in k and k not in allowed])
                if ignored:
                    self.logger.debug(f"[{prefix}] ignoring non-file dot-keys: {ignored}")

            row_data: dict[str, object] = {}
            for key in allowed:
                if key in out:
                    col = output_column(prefix, key)
                    row_data[col] = out[key]

            all_row_data.append(row_data)

        # --- now assemble all_row_data back into df_out ---
        all_cols = sorted({c for rd in all_row_data for c in rd})
        column_arrays = {col: [] for col in all_cols}

        for rd in all_row_data:
            for col in all_cols:
                if col in rd:
                    column_arrays[col].append(rd[col])
                else:
                    # fill defaults for skipped rows
                    if col.endswith("-NT") or col.endswith("-normal_termination"):
                        column_arrays[col].append(False)
                    elif col.endswith("-EE") or col.endswith("-GE") or col.endswith("-electronic_energy") or col.endswith("-gibbs_energy"):
                        column_arrays[col].append(np.nan)
                    elif col.endswith("-error"):
                        column_arrays[col].append(None)
                    else:
                        column_arrays[col].append(None)

        for col, vals in column_arrays.items():
            df_out[col] = vals

        steps = dict(df_out.attrs.get("frust_steps", {}))
        steps[prefix] = {"engine": prefix.split("-", 1)[0], "columns": sorted(column_arrays)}
        df_out.attrs["frust_steps"] = steps

        return df_out

    def xtb(
        self,
        df: pd.DataFrame,
        name: str = "xtb",
        options: dict | None = None,
        detailed_inp_str: str = "",
        constraint: bool = False,
        save_step: bool = False,
        lowest: int | None = None,
        n_cores: int | None = None,
    ) -> pd.DataFrame:
        """Embed multiple conformers with xTB and optionally optimize and/or
        compute frequencies.

        Args:
            df (pd.DataFrame): A DataFrame containing conformers. Required
                columns:
                - ``atoms``: list of atomic symbols
                - one coordinate column, typically ``coords_embedded`` or a
                  prior ``*-oc`` column
                - ``constraint_atoms`` when ``constraint=True``
                - ``substrate_name`` when ``lowest`` is used
            name (str): Base name for the xTB step, used to prefix result
                columns.
            options (dict, optional): xTB driver options, e.g. ``{'gfn': 2,
                'opt': None}``. Defaults to ``{'gfn': 0}``.
            detailed_inp_str (str, optional): Additional xTB input block
                (cards) to include. Defaults to ``""``.
            constraint (bool, optional): If ``True``, applies predefined
                distance and angle constraints for supported ``step_type``
                values. Requires ``Stepper(step_type=...)`` with one of
                ``TS1``, ``TS2``, ``TS3``, ``TS4``, or ``INT3``. Defaults to
                ``False``.
            save_step (bool, optional): If ``True``, saves calculation
                directories for each conformer. Defaults to ``False``.
            lowest (int or None, optional): If set, retains only the
                lowest-energy ``N`` conformers per structure group. Defaults
                to ``None``.
            n_cores (int or None, optional): If set, overrides the Stepper’s
                default core count for this xTB call only. Defaults to
                ``None``.

        Returns:
            pd.DataFrame: The input DataFrame augmented with stage-prefixed xTB
            result columns, typically including:
                - ``{prefix}-NT``
                - ``{prefix}-EE``
                - ``{prefix}-oc`` for optimization jobs
                - ``{prefix}-vibs`` and ``{prefix}-GE`` for
                  frequency jobs
                - ``{prefix}-error`` when a row-level engine failure occurs
                - saved file-content columns when the backend returns them
        """
        opts = dict(options) if options else {"gfn": 0}
        effective_n_cores = self._effective_n_cores(n_cores)
        if constraint:
            self._validate_constraint_request(df)
        keys = list(opts)
        if name != "xtb":
            prefix = name
        else:
            level = keys[0]
            opt_flag = next((k for k in keys[1:] if k in ("opt", "ohess")), None)
            prefix = f"{name}-{level}" + (f"-{opt_flag}" if opt_flag else "")

        def build_xtb(row: pd.Series) -> dict:
            inp: dict[str, object] = {"options": opts}
            step_type = self._step_type_upper()

            # Per-call override: only affects xTB, not ORCA.
            if n_cores is not None:
                inp["n_cores"] = effective_n_cores

            # Only add the user‐provided card if it's non‐empty
            base_str = detailed_inp_str.strip()
            if base_str:
                inp["detailed_input_str"] = base_str

            block = None

            if step_type == "TS1" and constraint:
                B, N, H, C = 0, 1, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[B]}, {atom[H]}, 2.07696
                distance: {atom[N]}, {atom[H]}, 1.5127 
                distance: {atom[H]}, {atom[C]}, 1.29095
                distance: {atom[B]}, {atom[C]}, 1.68461
                distance: {atom[B]}, {atom[N]}, 3.06223
                angle: {atom[N]}, {atom[H]}, {atom[C]}, 170.1342
                angle: {atom[H]}, {atom[C]}, {atom[B]}, 87.4870
                $end
                """).strip()

            if step_type == "TS2" and constraint:
                BCat10, N17, H40, H41, C46 = 0, 1, 4, 3, 5  # noqa: F841
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat10]}, {atom[H41]}, 1.656
                distance: {atom[N17]}, {atom[H40]}, 1.961
                distance: {atom[BCat10]}, {atom[N17]}, 3.080
                angle: {atom[BCat10]}, {atom[H41]}, {atom[N17]}, 86.58
                $end
                """).strip()

            if step_type == "TS3" and constraint:
                BCat10, H11, BPin22, H21, C = 0, 2, 3, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[H21]}, {atom[BCat10]}, 1.376
                distance: {atom[H21]}, {atom[BPin22]}, 1.264
                distance: {atom[H21]}, {atom[C]}, 2.477
                distance: {atom[BCat10]}, {atom[C]}, 1.616
                distance: {atom[BPin22]}, {atom[C]}, 2.180
                distance: {atom[BPin22]}, {atom[BCat10]}, 2.007
                angle: {atom[BCat10]}, {atom[H21]}, {atom[BPin22]}, 98.89
                angle: {atom[BCat10]}, {atom[C]}, {atom[BPin22]}, 61.75
                $end
                """).strip()

            if step_type == "TS4" and constraint:
                BCat11, H12, H13, BPin37, C = 0, 2, 3, 4, 5 # noqa: F841
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat11]}, {atom[BPin37]}, 2.219
                distance: {atom[BPin37]}, {atom[H13]}, 1.868
                distance: {atom[C]}, {atom[H13]}, 2.489
                distance: {atom[BCat11]}, {atom[H13]}, 1.216
                distance: {atom[BCat11]}, {atom[C]}, 1.946
                distance: {atom[BPin37]}, {atom[C]}, 1.585
                angle: {atom[BCat11]}, {atom[H13]}, {atom[BPin37]}, 89.48
                angle: {atom[BCat11]}, {atom[C]}, {atom[BPin37]}, 77.13
                $end
                """).strip()

            if step_type == "INT3" and constraint:
                BCat10, BPin42, H11, C = 0, 3, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat10]}, {atom[H11]}, 1.279
                distance: {atom[BCat10]}, {atom[C]}, 1.688
                distance: {atom[BPin42]}, {atom[H11]}, 1.378
                distance: {atom[BPin42]}, {atom[C]}, 1.749
                angle: {atom[BCat10]}, {atom[H11]}, {atom[BPin42]}, 89.85
                angle: {atom[BCat10]}, {atom[C]}, {atom[BPin42]}, 66.22
                $end
                """).strip()

            if block:
                if "detailed_input_str" in inp:
                    inp["detailed_input_str"] += "\n\n" + block
                else:
                    inp["detailed_input_str"] = block

            return inp

        result = self._run_engine(
            df, self.xtb_fn, prefix, build_xtb, save_step, lowest
        )
        result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
            {
                "engine": "xtb",
                "options": opts,
                "input": {
                    "options": opts,
                    "detailed_inp_str": _metadata_text(detailed_inp_str),
                    "constraint": bool(constraint),
                    "save_step": bool(save_step),
                    "lowest": lowest,
                    "n_cores": effective_n_cores,
                },
                "calculator": calculator_provenance(
                    name="xtb",
                    mode="direct",
                    backend=self.xtb_fn,
                    resources={"n_cores": effective_n_cores},
                    executables={
                        "xtb": env_executable("XTB_EXE", fallback_command="xtb"),
                    },
                ),
            }
        )
        return result

    def gxtb(
        self,
        df: pd.DataFrame,
        name: str = "gxtb",
        options: dict | None = None,
        detailed_inp_str: str = "",
        constraint: bool = False,
        save_step: bool = False,
        lowest: int | None = None,
        n_cores: int | None = None,
    ) -> pd.DataFrame:
        """Run g-xTB v2 calculations through Tooltoad's g-xTB calculator."""
        opts = dict(options) if options else {}
        effective_n_cores = self._effective_n_cores(n_cores)
        if constraint:
            self._validate_constraint_request(df)
        keys = list(opts)
        if name != "gxtb":
            prefix = name
        else:
            opt_flag = next((k for k in keys if k in ("opt", "ohess")), None)
            prefix = f"{name}" + (f"-{opt_flag}" if opt_flag else "")

        def build_gxtb(row: pd.Series) -> dict:
            inp: dict[str, object] = {"options": opts}
            step_type = self._step_type_upper()

            if n_cores is not None:
                inp["n_cores"] = effective_n_cores

            base_str = detailed_inp_str.strip()
            if base_str:
                inp["detailed_input_str"] = base_str

            block = None

            if step_type == "TS1" and constraint:
                B, N, H, C = 0, 1, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[B]}, {atom[H]}, 2.07696
                distance: {atom[N]}, {atom[H]}, 1.5127
                distance: {atom[H]}, {atom[C]}, 1.29095
                distance: {atom[B]}, {atom[C]}, 1.68461
                distance: {atom[B]}, {atom[N]}, 3.06223
                angle: {atom[N]}, {atom[H]}, {atom[C]}, 170.1342
                angle: {atom[H]}, {atom[C]}, {atom[B]}, 87.4870
                $end
                """).strip()

            if step_type == "TS2" and constraint:
                BCat10, N17, H40, H41, C46 = 0, 1, 4, 3, 5  # noqa: F841
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat10]}, {atom[H41]}, 1.656
                distance: {atom[N17]}, {atom[H40]}, 1.961
                distance: {atom[BCat10]}, {atom[N17]}, 3.080
                angle: {atom[BCat10]}, {atom[H41]}, {atom[N17]}, 86.58
                $end
                """).strip()

            if step_type == "TS3" and constraint:
                BCat10, H11, BPin22, H21, C = 0, 2, 3, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[H21]}, {atom[BCat10]}, 1.376
                distance: {atom[H21]}, {atom[BPin22]}, 1.264
                distance: {atom[H21]}, {atom[C]}, 2.477
                distance: {atom[BCat10]}, {atom[C]}, 1.616
                distance: {atom[BPin22]}, {atom[C]}, 2.180
                distance: {atom[BPin22]}, {atom[BCat10]}, 2.007
                angle: {atom[BCat10]}, {atom[H21]}, {atom[BPin22]}, 98.89
                angle: {atom[BCat10]}, {atom[C]}, {atom[BPin22]}, 61.75
                $end
                """).strip()

            if step_type == "TS4" and constraint:
                BCat11, H12, H13, BPin37, C = 0, 2, 3, 4, 5  # noqa: F841
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat11]}, {atom[BPin37]}, 2.219
                distance: {atom[BPin37]}, {atom[H13]}, 1.868
                distance: {atom[C]}, {atom[H13]}, 2.489
                distance: {atom[BCat11]}, {atom[H13]}, 1.216
                distance: {atom[BCat11]}, {atom[C]}, 1.946
                distance: {atom[BPin37]}, {atom[C]}, 1.585
                angle: {atom[BCat11]}, {atom[H13]}, {atom[BPin37]}, 89.48
                angle: {atom[BCat11]}, {atom[C]}, {atom[BPin37]}, 77.13
                $end
                """).strip()

            if step_type == "INT3" and constraint:
                BCat10, BPin42, H11, C = 0, 3, 4, 5
                atom = [x + 1 for x in self._constraint_atoms(row)]
                block = textwrap.dedent(f"""
                $constrain
                force constant=50
                distance: {atom[BCat10]}, {atom[H11]}, 1.279
                distance: {atom[BCat10]}, {atom[C]}, 1.688
                distance: {atom[BPin42]}, {atom[H11]}, 1.378
                distance: {atom[BPin42]}, {atom[C]}, 1.749
                angle: {atom[BCat10]}, {atom[H11]}, {atom[BPin42]}, 89.85
                angle: {atom[BCat10]}, {atom[C]}, {atom[BPin42]}, 66.22
                $end
                """).strip()

            if block:
                if "detailed_input_str" in inp:
                    inp["detailed_input_str"] += "\n\n" + block
                else:
                    inp["detailed_input_str"] = block

            return inp

        result = self._run_engine(
            df, self.gxtb_fn, prefix, build_gxtb, save_step, lowest
        )
        result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
            {
                "engine": "gxtb",
                "options": opts,
                "input": {
                    "options": opts,
                    "detailed_inp_str": _metadata_text(detailed_inp_str),
                    "constraint": bool(constraint),
                    "save_step": bool(save_step),
                    "lowest": lowest,
                    "n_cores": effective_n_cores,
                },
                "calculator": calculator_provenance(
                    name="gxtb",
                    mode="direct_gxtb",
                    backend=self.gxtb_fn,
                    resources={"n_cores": effective_n_cores},
                    executables={
                        "gxtb": gxtb_executable(),
                    },
                ),
            }
        )
        return result


    def orca(
        self,
        df: pd.DataFrame,
        name: str = "orca",
        options: dict | None = None,
        xtra_inp_str: str = "",
        constraint: bool = False,
        save_step: bool = False,
        save_files: list[str] | None = None,
        lowest: int | None = None,
        uma: str | None = None,
        read_files: list | None = None,
        use_last_hess: bool = False,
        n_cores: int | None = None,
        uma_server: bool = True,
        uma_device: str = "cpu",
        uma_cache_dir: str | None = None,
        uma_offline: bool = False,
        uma_server_cores: int | None = None,
        uma_memory_per_thread_mib: int = 500,
        uma_keep_logs: bool | str = "on_failure",
        uma_log_dir: str | None = None,
        gxtb: bool = False,
        gxtb_exe: str | None = None,
        gxtb_ext_params: str | None = None,
        **kw        
    ) -> pd.DataFrame:
        """Run ORCA calculations and attach results to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame of conformers to compute. Must include:
                - ``atoms``: list of atomic symbols
                - one coordinate column, typically ``coords_embedded`` or a
                  prior ``*-oc`` column
                - ``constraint_atoms`` when ``constraint=True``
                - ``substrate_name`` when ``lowest`` is used
                - a prior ``*.hess`` column when ``use_last_hess=True``
            name (str): Base name for the ORCA step, used to prefix result
                columns.
            options (dict): ORCA input keywords, e.g.
                ``{'wB97X-D3': None, '6-31G**': None, 'OptTS': None,
                'Freq': None}``.
            xtra_inp_str (str, optional): Additional ORCA input block such as
                CPCM settings or custom geometry directives. Defaults to
                ``""``.
            constraint (bool, optional): If ``True``, applies predefined
                distance and angle constraints for supported ``step_type``
                values. Requires ``Stepper(step_type=...)`` with one of
                ``TS1``, ``TS2``, ``TS3``, ``TS4``, or ``INT3``. Defaults to
                ``False``.
            save_step (bool, optional): If ``True``, saves ORCA run
                directories for inspection. Defaults to ``False``.
            save_files (list[str] or None, optional): Specific ORCA output
                files to retain when partial saving is enabled. If omitted and
                instance-level output saving is enabled, defaults to
                ``["orca.out"]``.
            lowest (int or None, optional): If set, keeps only the
                lowest-energy conformer per structure group. Defaults to
                ``None``.
            uma (str or None, optional): Optional UMA task/profile identifier
                used to inject ORCA external optimization settings. Defaults to
                ``None``.
            read_files (list or None, optional): Deposit contents from files in
                the work directory directly into the dataframe, e.g.
                ``["input.hess"]``.
            use_last_hess (bool, optional): If ``True``, scan the dataframe for
                the most recent ``*.hess`` column and feed it back to ORCA as
                ``private_input.hess``. Defaults to ``False``.
            n_cores (int or None, optional): If set, overrides the Stepper's
                default core count for this ORCA call only. Defaults to
                ``None``.
            uma_server (bool, optional): If ``True``, runs UMA through OET's
                server/client mode. If ``False``, uses standalone ``oet_uma``.
                Defaults to ``True``.
            uma_device (str, optional): OET UMA device argument, typically
                ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
            uma_cache_dir (str or None, optional): Optional FairChem cache
                directory passed to OET UMA. Defaults to ``None``.
            uma_offline (bool, optional): If ``True``, asks OET UMA to use
                offline mode. Defaults to ``False``.
            uma_server_cores (int or None, optional): Total core budget passed
                to ``oet_server --nthreads``. Defaults to this ORCA call's
                core count.
            uma_memory_per_thread_mib (int, optional): Memory budget passed to
                ``oet_server --memory-per-thread``. Defaults to ``500``.
            uma_keep_logs (bool or str, optional): Server-log retention policy:
                ``"on_failure"`` preserves logs only when the UMA-backed ORCA
                step fails, ``True``/``"always"`` keeps logs, and
                ``False``/``"never"`` removes them. Defaults to
                ``"on_failure"``.
            uma_log_dir (str or None, optional): Directory for preserved UMA
                server logs. If omitted, transient logs are written to a temp
                directory and preserved failures are copied to ``UMA-logs``.
            gxtb (bool, optional): If ``True``, runs ORCA with OET g-xTB v2 as
                an external method provider. ORCA still owns ``Opt``,
                ``OptTS``, ``NEB-TS``, and related run types. Defaults to
                ``False``.
            gxtb_exe (str or None, optional): Optional path to the g-xTB v2
                ``xtb`` executable. If omitted, ``GXTB_EXE`` is used.
            gxtb_ext_params (str or None, optional): Extra parameters appended
                to OET ``oet_gxtb`` through ORCA ``Ext_Params``.

        Returns:
            pd.DataFrame: The input DataFrame extended with stage-prefixed ORCA
            output columns, typically including:
                - ``{prefix}-NT``
                - ``{prefix}-EE``
                - ``{prefix}-oc`` for optimization jobs
                - ``{prefix}-vibs`` and ``{prefix}-GE`` for
                  frequency jobs
                - ``{prefix}-error`` when a row-level engine failure occurs
                - saved file-content columns when ORCA returns them
        """
        opts = dict(options or {})
        if kw:
            unknown = ", ".join(sorted(kw))
            raise TypeError(f"Unexpected orca() keyword arguments: {unknown}")
        if uma is not None and gxtb:
            raise ValueError("orca() cannot use both UMA and g-xTB external methods")
        if gxtb and "Freq" in opts:
            raise ValueError(
                "ORCA Freq is not compatible with g-xTB ExtOpt. "
                "Use NumFreq for finite-difference frequencies with external g-xTB gradients."
            )
        if gxtb and "calc_hess" in (xtra_inp_str or "").lower():
            raise ValueError(
                "ORCA %geom Calc_Hess is not compatible with g-xTB ExtOpt. "
                "Use OptTS with the approximate Hessian, and add NumFreq only when you need "
                "a post-optimization numerical frequency check."
            )
        if gxtb and "ExtOpt" not in opts:
            opts = {"ExtOpt": None, **opts}
        if constraint:
            self._validate_constraint_request(df)
        if save_files is None and self.save_output:
            save_files = ["orca.out"]
        keys = list(opts)
        if len(keys) < 1:
            raise ValueError("`options` must include at least one ORCA method key")

        if name != "orca":
            prefix = name
        elif len(keys) == 1:
            prefix = f"{name}-{keys[0]}"
        else:
            func, basis = keys[0], keys[1]
            opt_flag = next((k for k in keys[2:] if k in ("OptTS", "Freq", "NumFreq", "NoSym")), None)
            prefix = f"{name}-{func}-{basis}" + (f"-{opt_flag}" if opt_flag else "")

        effective_n_cores = self._effective_n_cores(n_cores)
        generated_input_blocks = []
        if "Freq" in opts:
            generated_input_blocks.append("freq_calc_hess")
        if use_last_hess:
            generated_input_blocks.append("read_last_hess")
        if constraint:
            generated_input_blocks.append(f"{self._step_type_upper()}_constraints")

        def build_input_metadata(**extra: object) -> dict[str, object]:
            metadata: dict[str, object] = {
                "options": opts,
                "xtra_inp_str": _metadata_text(xtra_inp_str),
                "constraint": bool(constraint),
                "save_step": bool(save_step),
                "save_files": _metadata_list(save_files),
                "lowest": lowest,
                "read_files": _metadata_list(read_files),
                "use_last_hess": bool(use_last_hess),
                "n_cores": effective_n_cores,
                "memory_gb": self.memory_gb,
                "generated_input_blocks": generated_input_blocks or None,
            }
            metadata.update(extra)
            return metadata

        def build_orca_calculator(
            mode: str,
            *,
            executables: dict[str, object] | None = None,
            resources: dict[str, object] | None = None,
            uma_metadata: dict[str, object] | None = None,
        ) -> dict[str, object]:
            execs = {
                "orca": env_executable("ORCA_EXE"),
                "xtb": env_executable("XTB_EXE", fallback_command="xtb"),
            }
            execs.update(executables or {})

            resource_data: dict[str, object] = {
                "n_cores": effective_n_cores,
                "memory_gb": self.memory_gb,
            }
            resource_data.update(resources or {})

            metadata = {}
            if uma_metadata is not None:
                metadata["uma"] = uma_metadata

            return calculator_provenance(
                name="orca",
                mode=mode,
                backend=self.orca_fn,
                resources=resource_data,
                executables=execs,
                environment={
                    "OPEN_MPI_DIR": env_path("OPEN_MPI_DIR"),
                    "XTBPATH": env_path("XTBPATH"),
                },
                **metadata,
            )

        def build_orca(row: Series) -> dict:
            step_type = self._step_type_upper()
            inp = {
                "options": opts,
                "xtra_inp_str": xtra_inp_str.strip(),
                "memory": self.memory_gb,
                "n_cores": effective_n_cores,
                "read_files": read_files,
            }

            if "Freq" in opts:
                block = textwrap.dedent("""
                    %geom
                      Calc_Hess true
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if use_last_hess:
                block = textwrap.dedent(
                    """
                    %geom
                      inhess Read
                      InHessName "private_input.hess"
                    end
                    """
                ).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "TS1":
                atom = self._constraint_atoms(row)
                B, N, H, C = atom[0], atom[1], atom[4], atom[5]
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {B} {H} 2.07696 C}}
                      {{B {N} {H} 1.5127 C}}
                      {{B {H} {C} 1.29095 C}}
                      {{B {B} {C} 1.68461 C}}
                      {{B {B} {N} 3.06223 C}}
                      {{A {N} {H} {C} C}}
                      {{A {H} {C} {B} 87.4870 C}}
                    end
                    end
                """).strip() # {{A {N} {H} {C} 170.1342 C}}
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "TS2":
                atom = self._constraint_atoms(row)
                BCat, N17, H40, H41 = atom[0], atom[1], atom[4], atom[3]
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {BCat} {H41} 1.656 C}}
                      {{B {N17} {H40} 1.961 C}}
                      {{B {BCat} {N17} 3.080 C}}
                      {{A {BCat} {H41} {N17} 86.58 C}}
                    end
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "TS3":
                atom = self._constraint_atoms(row)
                BCat, H11, BPin, H21, C = atom[0], atom[2], atom[3], atom[4], atom[5]
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {H21} {BCat} 1.376 C}}
                      {{B {H21} {BPin} 1.264 C}}
                      {{B {H21} {C} 2.477 C}}
                      {{B {BCat} {C} 1.616 C}}
                      {{B {BPin} {C} 2.180 C}}
                      {{B {BPin} {BCat} 2.007 C}}
                      {{A {BCat} {H21} {BPin} 98.89 C}}
                      {{A {BCat} {C} {BPin} 61.75 C}}
                    end
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "TS4":
                atom = self._constraint_atoms(row)
                BCat, H12, H13, BPin, C = atom[0], atom[2], atom[3], atom[4], atom[5]  # noqa: F841
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {BCat} {BPin} 2.219 C}}
                      {{B {BPin} {H13} 1.868 C}}
                      {{B {C} {H13} 2.489 C}}
                      {{B {BCat} {H13} 1.216 C}}
                      {{B {BCat} {C} 1.946 C}}
                      {{B {BPin} {C} 1.585 C}}
                      {{A {BCat} {H13} {BPin} 89.48 C}}
                      {{A {BCat} {C} {BPin} 77.13 C}}
                    end
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            if constraint and step_type == "INT3":
                atom = self._constraint_atoms(row)
                BCat, BPin, H11, C = atom[0], atom[3], atom[4], atom[5]
                block = textwrap.dedent(f"""
                    %geom Constraints
                      {{B {BCat} {H11} 1.279 C}}
                      {{B {BCat} {C} 1.688 C}}
                      {{B {BPin} {H11} 1.378 C}}
                      {{B {BPin} {C} 1.749 C}}
                      {{A {BCat} {H11} {BPin} 89.85 C}}
                      {{A {BCat} {C} {BPin} 66.22 C}}
                    end
                    end
                """).strip()
                inp["xtra_inp_str"] += ("\n\n" + block) if inp["xtra_inp_str"] else block

            return inp

        if uma is None and not gxtb:
            result = self._run_engine(df, self.orca_fn, prefix, build_orca, save_step, lowest, save_files, use_last_hess)
            result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
                {
                    "engine": "orca",
                    "options": opts,
                    "input": build_input_metadata(),
                    "calculator": build_orca_calculator("direct"),
                }
            )
            return result

        if gxtb:
            from frust.utils.gxtb import gxtb_orca_block, resolve_gxtb_exe

            resolved_gxtb_exe, _gxtb_exe_source = resolve_gxtb_exe(gxtb_exe)
            client_block = gxtb_orca_block(gxtb_exe=str(resolved_gxtb_exe), ext_params=gxtb_ext_params)

            def build_orca_gxtb(row: Series) -> dict:
                inp = build_orca(row)
                xin = inp.get("xtra_inp_str", "")
                inp["xtra_inp_str"] = (xin + "\n\n" + client_block).strip() if xin else client_block
                return inp

            result = self._run_engine(
                df,
                self.orca_fn,
                prefix,
                build_orca_gxtb,
                save_step,
                lowest,
                save_files,
                use_last_hess,
            )
            result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
                {
                    "engine": "orca",
                    "options": opts,
                    "gxtb": True,
                    "input": build_input_metadata(
                        gxtb=True,
                        gxtb_exe=_metadata_text(gxtb_exe),
                        gxtb_ext_params=_metadata_text(gxtb_ext_params),
                    ),
                    "calculator": build_orca_calculator(
                        "orca_external_gxtb",
                        executables={
                            "oet_gxtb": oet_executable("oet_gxtb"),
                            "gxtb": gxtb_executable(
                                gxtb_exe,
                                resolved_path=resolved_gxtb_exe,
                            ),
                        },
                    ),
                }
            )
            calc_gxtb = result.attrs["frust_steps"][prefix]["calculator"]["executables"]["gxtb"]
            result.attrs["frust_steps"][prefix]["gxtb_exe"] = calc_gxtb["path"]
            result.attrs["frust_steps"][prefix]["gxtb_exe_source"] = calc_gxtb["source"]
            return result
        
        from frust.utils.uma import parse_uma_spec, uma_orca_block, uma_server as run_uma_server

        spec = parse_uma_spec(
            uma,
            device=uma_device,
            cache_dir=uma_cache_dir,
            offline=uma_offline,
        )

        def uma_calculator(
            *,
            server: bool,
            resources: dict[str, object] | None = None,
        ) -> dict[str, object]:
            executable_entries = (
                {
                    "oet_client": oet_executable("oet_client"),
                    "oet_server": oet_executable("oet_server"),
                }
                if server
                else {"oet_uma": oet_executable("oet_uma")}
            )
            return build_orca_calculator(
                "orca_external_uma",
                executables=executable_entries,
                resources=resources,
                uma_metadata={
                    "spec": uma,
                    "task": spec.task,
                    "model": spec.model,
                    "device": spec.device,
                    "cache_dir": spec.cache_dir,
                    "offline": spec.offline,
                    "server": server,
                },
            )

        def run_with_uma_block(
            client_block: str,
            calculator: dict[str, object],
            input_extra: dict[str, object] | None = None,
        ) -> pd.DataFrame:
            orig_build = build_orca

            def build_orca_uma(row: Series) -> dict:
                inp = orig_build(row)
                xin = inp.get("xtra_inp_str", "")
                inp["xtra_inp_str"] = (xin + "\n\n" + client_block).strip() if xin else client_block
                return inp

            result = self._run_engine(df, self.orca_fn, prefix, build_orca_uma, save_step, lowest, save_files)
            uma_input = {
                "uma": uma,
                "uma_task": spec.task,
                "uma_model": spec.model,
                "uma_server": uma_server,
                "uma_device": uma_device,
                "uma_cache_dir": _metadata_text(uma_cache_dir),
                "uma_offline": bool(uma_offline),
                "uma_server_cores": uma_server_cores,
                "uma_memory_per_thread_mib": int(uma_memory_per_thread_mib),
                "uma_keep_logs": uma_keep_logs,
                "uma_log_dir": _metadata_text(uma_log_dir),
            }
            uma_input.update(input_extra or {})
            result.attrs.setdefault("frust_steps", {}).setdefault(prefix, {}).update(
                {
                    "engine": "orca",
                    "options": opts,
                    "uma": uma,
                    "uma_task": spec.task,
                    "uma_model": spec.model,
                    "uma_server": uma_server,
                    "input": build_input_metadata(**uma_input),
                    "calculator": calculator,
                }
            )
            return result

        def uma_result_failed(result: pd.DataFrame) -> bool:
            nt_col = output_column(prefix, "normal_termination")
            err_col = output_column(prefix, "error")
            if nt_col in result and result[nt_col].eq(False).any():
                return True
            if err_col in result and result[err_col].notna().any():
                return True
            return False

        if not uma_server:
            client_block = uma_orca_block(spec, server=False)
            return run_with_uma_block(client_block, uma_calculator(server=False))

        server_cores = uma_server_cores if uma_server_cores is not None else effective_n_cores
        with run_uma_server(
            log_dir=uma_log_dir,
            keep_logs=uma_keep_logs,
            use_gpu=uma_device == "cuda",
            server_cores=server_cores,
            memory_per_thread_mib=uma_memory_per_thread_mib,
        ) as server_handle:
            client_block = uma_orca_block(spec, server=True, bind=server_handle.bind)
            result = run_with_uma_block(
                client_block,
                uma_calculator(
                    server=True,
                    resources={
                        "uma_server_cores": int(server_cores),
                        "uma_memory_per_thread_mib": int(uma_memory_per_thread_mib),
                    },
                ),
                input_extra={"uma_server_cores": int(server_cores)},
            )
            if uma_keep_logs == "on_failure" and uma_result_failed(result):
                server_handle.preserve()
            return result

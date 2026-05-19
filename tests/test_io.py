import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from frust.utils.io import write_xyz, write_xyz_structures


def _coords(offset: float = 0.0) -> np.ndarray:
    return np.array(
        [
            [0.0 + offset, 0.0, 0.0],
            [0.0 + offset, 0.0, 0.74],
        ]
    )


def _df(**extra_cols) -> pd.DataFrame:
    data = {
        "atoms": [["H", "H"], ["H", "H"]],
        "custom_name": ["first", "second"],
        "ligand_name": ["lig/a one", "lig/a one"],
        "substrate_name": ["substrate one", "substrate two"],
        "embedded_coords": [_coords(), _coords(1.0)],
        "orca-opt_coords": [_coords(2.0), _coords(3.0)],
    }
    data.update(extra_cols)
    return pd.DataFrame(data)


class WriteXYZTests(unittest.TestCase):
    def test_default_uses_latest_coordinate_column(self):
        with tempfile.TemporaryDirectory() as td:
            paths = write_xyz(_df(), td)

            self.assertEqual(
                [path.name for path in paths],
                ["first.xyz", "second.xyz"],
            )
            text = paths[0].read_text()
            self.assertTrue(text.startswith("2\n"))
            self.assertIn("H 2.00000000 0.00000000 0.00000000", text)

    def test_explicit_coordinate_column(self):
        with tempfile.TemporaryDirectory() as td:
            paths = write_xyz(_df(), td, coords_col="embedded_coords")

            text = paths[0].read_text()
            self.assertIn("H 0.00000000 0.00000000 0.00000000", text)

    def test_mapping_export_uses_subfolders_and_suffixes(self):
        with tempfile.TemporaryDirectory() as td:
            paths = write_xyz(
                _df(),
                td,
                coords_col={
                    "embedded": "embedded_coords",
                    "DFT": "orca-opt_coords",
                },
            )

            rel_paths = sorted(path.relative_to(td).as_posix() for path in paths)
            self.assertEqual(
                rel_paths,
                [
                    "DFT/first_DFT.xyz",
                    "DFT/second_DFT.xyz",
                    "embedded/first_embedded.xyz",
                    "embedded/second_embedded.xyz",
                ],
            )

    def test_name_column_fallbacks_and_index_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            df = _df().drop(columns=["custom_name"])
            paths = write_xyz(df, td)
            self.assertEqual(
                [path.name for path in paths],
                ["lig_a_one.xyz", "lig_a_one_2.xyz"],
            )

        with tempfile.TemporaryDirectory() as td:
            df = _df().drop(columns=["custom_name", "ligand_name"])
            paths = write_xyz(df, td)
            self.assertEqual(
                [path.name for path in paths],
                ["substrate_one.xyz", "substrate_two.xyz"],
            )

        with tempfile.TemporaryDirectory() as td:
            df = _df().drop(columns=["custom_name", "ligand_name", "substrate_name"])
            paths = write_xyz(df, td)
            self.assertEqual([path.name for path in paths], ["0.xyz", "1.xyz"])

    def test_filename_sanitization_and_duplicate_suffixes(self):
        with tempfile.TemporaryDirectory() as td:
            paths = write_xyz(_df(), td, name_col="ligand_name")

            self.assertEqual(
                [path.name for path in paths],
                ["lig_a_one.xyz", "lig_a_one_2.xyz"],
            )

    def test_missing_coordinate_column_raises_value_error(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                write_xyz(_df().drop(columns=["embedded_coords", "orca-opt_coords"]), td)

            self.assertIn("No coordinate columns found", str(ctx.exception))

    def test_missing_explicit_column_raises_key_error(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(KeyError) as ctx:
                write_xyz(_df(), td, coords_col="missing_coords")

            self.assertIn("missing_coords", str(ctx.exception))

    def test_missing_coordinates_raise_value_error_with_row_context(self):
        with tempfile.TemporaryDirectory() as td:
            df = _df()
            df.at[1, "orca-opt_coords"] = None

            with self.assertRaises(ValueError) as ctx:
                write_xyz(df, td)

            self.assertIn("orca-opt_coords", str(ctx.exception))
            self.assertIn("second", str(ctx.exception))

    def test_overwrite_false_raises_for_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            existing = Path(td) / "first.xyz"
            existing.write_text("existing")

            with self.assertRaises(FileExistsError):
                write_xyz(_df().iloc[[0]], td, overwrite=False)

    def test_deprecated_write_xyz_structures_preserves_old_layout(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertWarns(DeprecationWarning):
                result = write_xyz_structures(
                    _df().iloc[[0]],
                    td,
                    {"DFT": "orca-opt_coords"},
                )

            self.assertIsNone(result)
            self.assertTrue((Path(td) / "DFT" / "first_DFT.xyz").exists())


if __name__ == "__main__":
    unittest.main()

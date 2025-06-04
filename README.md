# FRUST - Frustrated Activation Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Early%20Development-orange)](https://github.com)

> **⚠️ Early Development Phase**: This project is in active development and may undergo significant changes.

A computational pipeline for automated frustrated Lewis pair (FLP) activation studies, designed to facilitate the discovery and characterization of FLP-mediated chemical transformations.

## Overview

FRUST (Frustrated Activation) is a research tool that automates the computational workflow for studying frustrated Lewis pair activation mechanisms. The pipeline integrates molecular transformation, conformational sampling, and quantum chemical calculations to systematically explore FLP activation pathways.

### Key Features

- **Automated molecular transformation** for FLP activation scenarios
- **Conformational sampling** using RDKit and UFF optimization
- **Transition state structure generation** and optimization
- **Integration with quantum chemistry packages** (xTB, ORCA)
- **Flexible pipeline configuration** for different computational levels
- **Command-line interface** for batch processing

## Installation

### Prerequisites

- Python 3.10 or higher
- RDKit (for molecular manipulation)
- Optional: xTB, ORCA (for quantum chemical calculations)

### Install from Source

```bash
git clone <repository-url>
cd FRUST
pip install -e .
```

### Dependencies

The core dependencies are automatically installed:
- `rdkit` - Molecular manipulation and conformer generation
- `numpy` - Numerical computations
- `pandas` - Data handling and analysis
- `matplotlib` - Visualization
- `tqdm` - Progress bars

## Quick Start

### Command Line Usage

Not implemented.

### Python API

...

## Project Structure

...

## Pipeline Workflow

1. **Molecular Transformation**: Convert input SMILES to transition state guess structures
2. **Conformer Generation**: Generate multiple conformations using RDKit
3. **UFF Optimization**: Pre-optimize structures with Universal Force Field
4. **Quantum Chemical Calculations**: Refine structures with xTB/ORCA
5. **Analysis & Output**: Process results and generate reports

## Development

### Development Environment

The project includes Jupyter notebooks in the `dev/` directory for development and testing:

- `dev0_pipe_init.ipynb` - Pipeline initialization
- `dev1_generic_lig_identi.ipynb` - Ligand identification
- `dev2_pipe_fix_dirs.ipynb` - Directory structure fixes
- `dev3_pipe_test_run.ipynb` - Test runs
- `dev4_pipe_test_constrains.ipynb` - Constraint testing

### Testing

Run the test suite:

```bash
python -m pytest tests/
```

### Example Data

The `datasets/` directory contains example data:
- `ir_borylation.csv` - Iridium-catalyzed borylation dataset with SMILES and active atom indices

## Configuration

The pipeline behavior can be customized through the `Settings` class:

```python
from frust.config import Settings

settings = Settings(
    debug=True,           # Enable debug mode
    live=False,           # Test vs live mode
    dump_each_step=True,  # Save intermediate results
    bonds_to_remove=[(10, 41), (10, 12)],  # Bond breaking patterns
    n_cores=8,            # Parallel processing
    memory_gb=20.0        # Memory allocation
)
```

## Contributing

This is a research project in early development. Contributions, suggestions, and collaborations are welcome!

### Development Setup

1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Run tests to ensure everything works
4. Create a new branch for your feature
5. Submit a pull request

## Research Context

This tool is being developed to support research in frustrated Lewis pair chemistry, particularly focusing on:

- Automated transition state searching for FLP activation
- High-throughput screening of ligand systems
- Computational catalyst design
- Mechanism elucidation for FLP-mediated reactions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Jacob Molin Nielsen**  
Email: jacob.molin@me.com

## Acknowledgments

- RDKit development team for molecular manipulation tools
- Quantum chemistry software developers (xTB, ORCA teams)
- The frustrated Lewis pair research community

---

*Note: This project is in active development. Features and API may change significantly during the early development phase.*
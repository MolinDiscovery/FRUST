import typer
from .config import Settings
from .pipeline import TSPipeline
import sys
from types import List

app = typer.Typer()

@app.command()
def ts(
    smiles: List[str] = typer.Argument(..., help="Ligand SMILES"),
    debug: bool = typer.Option(False),
    live: bool = typer.Option(False),
    dump_each: bool = typer.Option(False),
):
    """Run the TS‐search pipeline."""
    settings = Settings(
        debug=debug,
        live=live,
        dump_each_step=dump_each,
    )
    df = TSPipeline(settings).run(smiles)
    typer.echo(df.to_string())
    sys.exit(0)
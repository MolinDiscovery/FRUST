import importlib

mod = importlib.import_module("frust.pipes.run_ts_per_rpos")
steps = [
    obj for name, obj in mod.__dict__.items()
    if name.startswith("run_") and callable(obj)
]


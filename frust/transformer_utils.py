# frust/transformer_utils.py
from collections import deque

def rotated_maps(lig, old, reactive_old_idx, new_targets):
    """Return a list of rotated atom‑maps."""
    lig = list(lig)
    old = list(old)
    pos_old = old.index(reactive_old_idx)          # where 41 sits in old list

    maps = []
    for new_atom in new_targets:
        pos_new = lig.index(new_atom)              # where 4 / 5 / 6 sits now
        shift   = pos_old - pos_new                # how far to rotate right
        d = deque(lig)
        d.rotate(shift)                            # positive = rotate right
        maps.append(list(zip(d, old)))
    return maps
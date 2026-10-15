# Vibration-grid rendering regression

| Five-minute playback, 30 molecules | Original 2.4.2 | 2.5.5 with resource cleanup |
| --- | ---: | ---: |
| JavaScript heap, fresh → final | 16.6 → 325.0 MB | 16.5 → 16.9 MB |
| First model's picking shapes | 240 → 64,800 | 264 → 264 |
| Pending animation callbacks | 90 | 30 |

FRUST's `ft.plot_vibs` builds the dataframe scene once. Browser-side playback is
handled by tooltoad's renderer, py3Dmol-generated JavaScript, and 3Dmol.js.
The original engine accumulated picking shapes and did not explicitly release
replaced model geometry. Tooltoad now resets picking shapes and disposes replaced
child geometry/materials after the replacement renders. py3Dmol is pinned to 2.5.5.

## Already-open notebooks

**Restart the kernel, close/reopen the notebook view, then rerun setup/display
cells.** Restarting only Python does not clear the webview's JavaScript globals.
py3Dmol's one-time loader can therefore retain 3Dmol.js 2.4.2 after Python upgrades
to 2.5.5. Saved output also keeps its original JavaScript until regenerated.

Removing the old timer guard exposed that mixed-version case: 90 callbacks and
about 53 frames/sec per molecule instead of the requested 20. Tooltoad retains an
idempotent resume guard for cached engines; both cached 2.4.2 and fresh 2.5.5 now
run 30 callbacks at about 20 frames/sec. The user reported that reopening helped.

## Reproduce without changing the notebook

These tools read saved outputs and use separate temporary HTML/browser sessions:

```bash
npm install --prefix /tmp/frust-vib-profile playwright
export NODE_PATH=/tmp/frust-vib-profile/node_modules
node dev/vibration_render_profile/profile.cjs /path/to/review_modes.ipynb 300
```

`profile.cjs` accepts a saved notebook or HTML export. It records heap, frame and
callback counts, WebGL allocation/deletion counts, and listener retention. After
playback it triggers resize events and replaces the output three times.
`--fixed` can inject the installed UMA tooltoad guard into an original unguarded
output. Prefer a fresh export when checking the current installed renderer.

To rebuild the saved vibration inputs through the current `ft.plot_vibs` path:

```bash
conda run -n UMA python dev/vibration_render_profile/build_upgrade_grid.py \
  /path/to/review_modes.ipynb /tmp/vibration-grid.html
node dev/vibration_render_profile/check_cached_engine.cjs /tmp/vibration-grid.html
```

The cached-engine check runs the same output against a preloaded 2.4.2 engine and
a fresh 2.5.5 engine, asserting one pending callback per occupied cell and bounded
frame cadence. Set `CHROME_PATH` to override the default macOS Chrome executable.
Playwright/Chrome are development tools, not FRUST runtime dependencies.

## Evidence and limits

[measurements.json](measurements.json) retains the compact measurements:

- 518 FRUST fast tests passed; 13 slow tests were deselected.
- 109 focused visualization/scene/cleanup tests passed after installation and
  again after restoring cached-engine compatibility.
- Rotation, zoom, distance/angle measurements passed. Same-frame screenshots
  with and without resource cleanup were pixel-identical.
- On native ANGLE Metal / Apple M3 Max, a controlled same-cadence comparison
  measured approximately 9–10 ms GPU time per update of all 30 molecules for
  both versions. Canvas dimensions, draw counts, and uploaded bytes matched.

These measurements use separate Chrome 153 sessions. GPU timers are not iStat's
system-wide utilization percentage; geometry counters are not physical VRAM.
Old outputs can still retain viewers through upstream event listeners. This is a
remaining disposal limitation; reopening the notebook view clears those viewers.

[Upstream issue #860](https://github.com/3dmol/3Dmol.js/issues/860) contains the
standalone water reproduction in [upstream_repro.html](upstream_repro.html).
It demonstrates picking-shape growth without FRUST or private calculation data.
No user notebook, saved output, or calculation data was edited for this work.

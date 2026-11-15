# wave

## Wavefield Simulation Cache

Render the wavefield scene as usual:

```powershell
manim -pql wave_filed.py NumericalWavefield
```

The first render saves numerical snapshots under `media/wavefield_cache/`.
Later renders reuse them when only text, colors, layout, lattice spacing,
playback speed, or render quality changes. The numerical propagation does not
run again.

In `wave_filed.py`, `SIMULATION_SECONDS` controls physical simulation duration
(default 1.5 seconds), while `SIM_SECONDS` controls animation duration.
`WAVEFIELD_SAMPLE_STEPS` controls snapshot spacing in numerical time steps.
Playback is independent of render frame rate.

Changes to the velocity model, source, grid, boundary parameters, simulation
duration, snapshot spacing, or solver implementation automatically create a
new cache. Delete `media/wavefield_cache/` to force recomputation or reclaim
disk space. The default cache is about 48 MB and is already excluded from Git
by the `media` ignore rule.

## Rendered Video Cache

`NumericalWavefield` also caches a caption-free video under
`media/wavefield_video_cache/`. The first run renders the lattice and curves
once; subsequent runs decode that video and draw only the current text over
it. No physics calculation or lattice update is performed on a video-cache
hit. The final caption-compositing and video-encoding pass still runs, but is
much lighter than redrawing the lattice.

Edit captions in the `ANIMATION_TEXT` dictionary near the top of
`wave_filed.py`, then use the same render command. Title, receiver labels,
source label, axis captions, and trace captions do not invalidate the video.
Axis numbers are also drawn separately rather than baked into the video.

The terminal reports `Rendered video cache MISS` for the one-time build and
`Rendered video cache HIT` when reusing it. Manim's separate `Caching disabled`
message during caption compositing does not mean this video cache was missed.

Changing physics, lattice geometry or styling, scene layout, animation
duration, resolution, or frame rate rebuilds the rendered video as needed.
Deleting `media/wavefield_video_cache/` forces a fresh background render.
Transparent output, still-image output (`-s`), and dry runs use direct drawing
instead of the opaque video cache. PyAV, already required by Manim, handles
video playback; no additional package is needed.

## Ten Receiver Traces

Render the separate ten-trace scene with:

```powershell
manim -pql wave_filed.py ReceiverTraces
```

This scene extracts R1 through R10 from the same numerical wavefield cache.
All ten receivers are equally spaced at the same depth, to the right of the
source and inside the absorbing margins. R1 is nearest the source and R10
is farthest; R1/R2 move with the shared array. Spacing is a whole number of
grid cells (120 meters with the default model). The traces build over
the full simulation time as vertical wiggles, with R1 through R10 across the
top and time increasing downward on a common amplitude scale, preserving
relative amplitudes without clipping the recorded data.

`NumericalWavefield` displays all ten receiver locations using compact
geophone symbols with contact dots, coiled sensor housings, and staggered
labels. R1/R2 are gold and R3 through R10 are teal. The two existing live
trace panels still show only R1 and R2. Receiver locations are shared with
the ten-trace scene, so symbols and recorded traces stay consistent.

Changing symbol geometry rebuilds the background-video cache once, but does
not rerun numerical propagation. Receiver caption edits in `ANIMATION_TEXT`
reuse both caches.

## Presentation Sequence

`WavePresentation.py` adds six standalone scenes without changing either
of the existing simulation scenes:

| Order | Scene | Main visual |
| --- | --- | --- |
| 1 | `RecordedTraces` | The same ten cached recordings, with time downward |
| 2 | `AcousticAssumptions` | Traveling compression/expansion and modeling assumptions |
| 3 | `AcousticWaveEquation` | Pressure PDE, source wavelet, and candidate velocity model |
| 4 | `WaveEquationPhysics` | One spring, coupled oscillators, a stiff layer, and the continuum limit |
| 5 | `SourceEstimation` | Source measurements, estimation routes, and the Ricker model |
| 6 | `VelocityEstimation` | Illustrative moveout fit and waveform-matching objective |

`WaveEquationPhysics` directly runs the existing scenes from
`1_SpringOscillation.py`, `2_CoupledOscillation.py`,
`3_LargeCoupledOscillation.py`, `4_StiffLayerOscillation.py`, and
`6_Grid2DOscillation.py` in that order.

Render the separate videos:

```powershell
manim -ql WavePresentation.py RecordedTraces AcousticAssumptions AcousticWaveEquation WaveEquationPhysics SourceEstimation VelocityEstimation
```

Or render and preview one chapter:

```powershell
manim -pql WavePresentation.py AcousticWaveEquation
```

`PRESENTATION_PACE` in `WavePresentation.py` scales the animations and pauses
for these six chapters only. Its default is 1.0. No new acoustic-wave propagation
is needed when the existing physics cache is available.

### Speaker Notes

1. **RecordedTraces:** "These ten recordings are our measurements. We want
	an earth model that could explain them. Limited recordings do not identify
	a unique subsurface."
2. **AcousticAssumptions:** "To predict the signals, we first need a model of
	wave motion. Our simplest model keeps small pressure disturbances and P
	waves, assumes constant density, isotropic lossless behavior, and a 2D
	earth. It leaves out shear waves, attenuation, and anisotropy."
3. **AcousticWaveEquation:** "This pressure equation advances the wavefield
	in time. Its two key physical inputs are the source wavelet and wave-speed
	field. Source/receiver geometry, initial conditions, and boundary
	conditions must also be specified."
4. **WaveEquationPhysics:** "A single mass introduces inertia and a restoring
	force, but it does not make a wave. Coupling lets each mass pull on its
	neighbors, so disturbances propagate. Changing spring stiffness represents
	a material contrast that changes speed and splits energy into reflected and
	transmitted waves. As the spacing between many masses approaches zero, the
	discrete second difference becomes the spatial derivative in the continuum
	wave equation. This is a physical analogy for acoustic pressure, not a claim
	that one hanging spring directly derives the full acoustic PDE."
5. **SourceEstimation:** "We can measure the source with a source monitor or
	near-field sensor, use a known vibroseis pilot sweep and correlation, or
	estimate a wavelet from arrivals with a propagation model and suitable
	assumptions. Here we prescribe a Ricker wavelet, the negative second
	derivative of a Gaussian; we have not extracted it from the ten traces.
	The convolution relation assumes calibrated receiver response."
6. **VelocityEstimation:** "First-arrival tomography, reflection moveout,
	well logs, and checkshots can constrain an initial velocity model. Moveout
	yields RMS velocity under suitable assumptions, not interval velocity
	directly. We can then refine a model by matching predicted and observed
	waveforms, using an estimated source and prior constraints."

The moveout points in the final chapter are explicitly synthetic illustrations,
not arrival picks or an inversion of the cached recordings. The colored
velocity model in the PDE chapter is a candidate forward-model input, not a
claimed reconstruction. Source and velocity can trade off, and ten traces
alone do not uniquely recover an arbitrary 2D earth.

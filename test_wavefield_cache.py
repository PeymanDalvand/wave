"""Physics cache reuse and invalidation checks for the wavefield scene."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from manim import tempconfig

import wave_filed as scene
import WavePresentation as slides
from seismic_lib import AcousticWaveSolver2D, ricker_wavelet


class WavefieldCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        parameters = patch.multiple(
            scene, GRID_NZ=40, GRID_NX=40, SOURCE_ROW=10, SOURCE_COL=10,
            LAYER_2_START=20, LAYER_3_START=30, BOUNDARY_PAD=5, SIMULATION_SECONDS=0.012,
        )
        parameters.start()
        self.addCleanup(parameters.stop)

    def test_snapshots_match_solver_and_keep_final_partial_interval(self):
        frames, times = scene.load_wavefield_cache(self.directory)
        np.testing.assert_allclose(times, [0.0, 0.005, 0.01, 0.012])
        solver = AcousticWaveSolver2D(40, 40, 10.0, 0.001, scene.build_velocity_model(), boundary_pad=5)
        for index, target in enumerate(times):
            while solver.time_step < round(target / scene.TIME_STEP):
                pulse = ricker_wavelet(solver.time_step * scene.TIME_STEP, scene.SOURCE_FREQUENCY)
                solver.inject_source(10, 10, pulse * scene.SOURCE_AMPLITUDE)
                solver.advance()
            np.testing.assert_allclose(frames[index], solver.u_curr, rtol=1e-6, atol=1e-8)
        del frames

    def test_visual_changes_reuse_cache_without_simulating(self):
        frames, times = scene.load_wavefield_cache(self.directory)
        expected = frames.copy()
        del frames
        with patch.multiple(scene, BALL_STEP=7, SIM_SECONDS=2.0, SPEED_FACTOR=2.0):
            with patch.object(AcousticWaveSolver2D, "advance", side_effect=AssertionError("Unexpected simulation")):
                frames, cached_times = scene.load_wavefield_cache(self.directory)
                np.testing.assert_array_equal(frames, expected)
                np.testing.assert_array_equal(cached_times, times)
                del frames
        self.assertEqual(len(list(self.directory.glob("*.npy"))), 1)

    def test_physics_changes_invalidate_cache(self):
        frames, _ = scene.load_wavefield_cache(self.directory)
        del frames
        for changes in (
            {"SOURCE_FREQUENCY": 20.0},
            {"LAYER_2_START": 18},
            {"SIMULATION_SECONDS": 0.02},
        ):
            with patch.multiple(scene, **changes):
                frames, _ = scene.load_wavefield_cache(self.directory)
                del frames
        self.assertEqual(len(list(self.directory.glob("*.npy"))), 4)

    def test_interrupted_write_does_not_publish_partial_cache(self):
        with patch.object(AcousticWaveSolver2D, "advance", side_effect=RuntimeError("Interrupted")):
            with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                scene.load_wavefield_cache(self.directory)
        self.assertEqual(list(self.directory.iterdir()), [])
        with patch.object(Path, "replace", side_effect=OSError("Cannot publish")):
            with self.assertRaisesRegex(OSError, "Cannot publish"):
                scene.load_wavefield_cache(self.directory)
        self.assertEqual(list(self.directory.iterdir()), [])


class ReceiverTraceTests(unittest.TestCase):
    def test_ten_locations_are_uniform_ordered_and_shared(self):
        with tempconfig({"quality": "low_quality", "dry_run": True}):
            original = scene.NumericalWavefield()
            visuals, labels = original._build_layout()
            positions = scene.ReceiverTraces()._trace_receiver_positions()
            self.assertEqual(len(positions), 10)
            self.assertEqual(len(set(positions)), 10)
            self.assertEqual(positions, original._receiver_grids)
            self.assertEqual(len(original._receiver_markers), 10)
            self.assertEqual(len(original._trace_panels), 2)
            self.assertTrue(all(row == positions[0][0] for row, _ in positions))
            self.assertTrue(all(col > scene.SOURCE_COL for _, col in positions))
            columns = np.array([col for _, col in positions])
            spacing = np.diff(columns)
            self.assertTrue(np.all(spacing > 0))
            np.testing.assert_array_equal(spacing, np.full(9, spacing[0]))
            self.assertEqual(columns[0] - scene.SOURCE_COL, spacing[0])
            self.assertTrue(all(scene.BOUNDARY_PAD < col < scene.GRID_NX - scene.BOUNDARY_PAD
                                for _, col in positions))
            for marker, (row, col) in zip(original._receiver_markers, positions):
                np.testing.assert_allclose(marker[0].get_center(),
                                           original._grid_to_point(original._image, row, col))
            receiver_labels = labels[1]
            self.assertEqual(len(receiver_labels), 10)
            for label_index, label in enumerate(receiver_labels):
                for other in receiver_labels[label_index + 1:]:
                    separated = (
                        label.get_right()[0] <= other.get_left()[0]
                        or other.get_right()[0] <= label.get_left()[0]
                        or label.get_top()[1] <= other.get_bottom()[1]
                        or other.get_top()[1] <= label.get_bottom()[1]
                    )
                    self.assertTrue(separated, f"Receiver labels overlap: {label.text}, {other.text}")
            self.assertFalse(any(isinstance(member, scene.Text)
                                 for visual in visuals for member in visual.get_family()))

    def test_traces_use_exact_unclipped_cached_samples(self):
        frames = np.arange(4 * 200 * 200, dtype=np.float32).reshape(4, 200, 200)
        times = np.array([0.0, 0.01, 0.02, 0.03])
        with tempconfig({
            "quality": "low_quality", "frame_rate": 1,
            "dry_run": True, "disable_caching": True,
        }):
            with patch.object(scene, "load_wavefield_cache", return_value=(frames, times)):
                with patch.object(scene, "SIM_SECONDS", 0.1):
                    with patch.object(AcousticWaveSolver2D, "advance", side_effect=AssertionError("Physics rerun")):
                        animation = scene.ReceiverTraces()
                        animation.render()
            self.assertEqual(len(animation.receiver_labels), 10)
            self.assertEqual([label.text for label in animation.receiver_labels],
                             [f"R{index}" for index in range(1, 11)])
            self.assertEqual(animation.trace_values.shape, (4, 10))
            label_centers = np.array([label.get_center() for label in animation.receiver_labels])
            self.assertTrue(np.all(np.diff(label_centers[:, 0]) > 0))
            np.testing.assert_allclose(label_centers[:, 1], label_centers[0, 1])
            self.assertTrue(all(curve.get_start()[1] > curve.get_end()[1]
                                for curve in animation.trace_curves))
            for index, (row, col) in enumerate(animation.receiver_positions):
                np.testing.assert_array_equal(animation.trace_values[:, index], frames[:, row, col])


class PresentationSceneTests(unittest.TestCase):
    def test_opening_gather_uses_the_same_ten_recordings(self):
        frames = np.zeros((4, 200, 200), dtype=np.float32)
        times = np.array([0.0, 0.5, 1.0, 1.5])
        with tempconfig({"quality": "low_quality", "dry_run": True}):
            positions = scene.ReceiverTraces()._trace_receiver_positions()
            for index, (row, col) in enumerate(positions):
                frames[:, row, col] = np.arange(4) * (index + 1)
            with patch.object(scene, "load_wavefield_cache", return_value=(frames, times)) as load_cache:
                traces, labels, _ = slides._recorded_gather()
            load_cache.assert_called_once()
            self.assertEqual(len(traces), 10)
            self.assertEqual([label.text for label in labels], [f"R{index}" for index in range(1, 11)])
            for index, trace in enumerate(traces):
                baseline = -4.8 + index * (9.6 / 9)
                np.testing.assert_allclose(trace.get_start(), [baseline, 1.75, 0])
                np.testing.assert_allclose(trace.get_end(), [baseline + (index + 1) * 0.03, -1.95, 0],
                                           rtol=1e-6, atol=1e-7)

    def test_presentation_scenes_animate_without_solver_calls_or_clipping(self):
        frames = np.zeros((4, 200, 200), dtype=np.float32)
        times = np.array([0.0, 0.5, 1.0, 1.5])
        scene_types = (
            slides.RecordedTraces, slides.AcousticAssumptions, slides.AcousticWaveEquation,
            slides.SourceEstimation, slides.VelocityEstimation,
        )
        with tempconfig({"quality": "low_quality", "dry_run": True, "disable_caching": True}):
            with patch.object(slides, "PRESENTATION_PACE", 0.001):
                with patch.object(scene, "load_wavefield_cache", return_value=(frames, times)):
                    with patch.object(AcousticWaveSolver2D, "advance", side_effect=AssertionError("Physics rerun")):
                        for scene_type in scene_types:
                            with self.subTest(scene=scene_type.__name__):
                                animation = scene_type()
                                animation.render()
                                family_size = sum(len(mob.get_family()) for mob in animation.mobjects)
                                self.assertGreater(family_size, 5)
                                for mob in animation.mobjects:
                                    points = mob.get_all_points()
                                    self.assertTrue(np.isfinite(points).all())
                                    if len(points):
                                        self.assertLessEqual(float(np.abs(points[:, 0]).max()), 14.2222223 / 2 + 0.05)
                                        self.assertLessEqual(float(np.abs(points[:, 1]).max()), 4.05)

    def test_wave_equation_physics_reuses_five_scenes_in_order(self):
        calls = []
        scene_types = (
            (slides.SpringOscillationScene, "spring"),
            (slides.CoupledOscillationScene, "coupled"),
            (slides.LargeCoupledOscillationScene, "large"),
            (slides.StiffLayerOscillationScene, "stiff"),
            (slides.Grid2DOscillationScene, "grid"),
        )
        patches = [
            patch.object(scene_type, "construct", side_effect=lambda _scene, name=name: calls.append(name))
            for scene_type, name in scene_types
        ]
        animation = slides.WaveEquationPhysics()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with patch.object(animation, "clear", side_effect=lambda: calls.append("clear")):
                animation.construct()
        self.assertEqual(
            calls,
            ["spring", "clear", "coupled", "clear", "large", "clear", "stiff", "clear", "grid"],
        )


class RenderedWavefieldCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.video_directory = self.directory / "video"
        parameters = patch.multiple(
            scene, GRID_NZ=40, GRID_NX=40, SOURCE_ROW=10, SOURCE_COL=10,
            LAYER_2_START=20, LAYER_3_START=30, BOUNDARY_PAD=5,
            SIMULATION_SECONDS=0.012, SIM_SECONDS=0.3,
        )
        parameters.start()
        self.addCleanup(parameters.stop)
        settings = tempconfig({
            "pixel_width": 320, "pixel_height": 180, "frame_rate": 10,
            "preview": False, "show_in_file_browser": False, "disable_caching": True,
            "write_to_movie": True, "save_last_frame": False, "dry_run": False,
            "output_file": str(self.directory / "captions.mp4"),
            "partial_movie_dir": str(self.directory / "partials"),
        })
        settings.__enter__()
        self.addCleanup(settings.__exit__, None, None, None)
        self.frames, self.times = scene.load_wavefield_cache(self.directory / "physics")
        self.addCleanup(self.frames._mmap.close)

    def test_caption_changes_keep_video_key_but_visual_changes_invalidate(self):
        original = scene.rendered_wavefield_cache_path(self.frames, self.video_directory)
        with patch.dict(scene.ANIMATION_TEXT, {"title": "A different title", "source": "Shot"}):
            self.assertEqual(original, scene.rendered_wavefield_cache_path(self.frames, self.video_directory))
        for changes in ({"BALL_STEP": 7}, {"SIM_SECONDS": 0.6}, {"TRACE_WINDOW": 0.5}):
            with patch.multiple(scene, **changes):
                self.assertNotEqual(original, scene.rendered_wavefield_cache_path(self.frames, self.video_directory))
        with tempconfig({"frame_rate": 20}):
            self.assertNotEqual(original, scene.rendered_wavefield_cache_path(self.frames, self.video_directory))
        with tempconfig({"pixel_width": 640, "pixel_height": 360}):
            self.assertNotEqual(original, scene.rendered_wavefield_cache_path(self.frames, self.video_directory))

    def test_caption_edit_renders_without_lattice_or_solver_calls(self):
        with patch.object(scene, "load_wavefield_cache", return_value=(self.frames, self.times)):
            video_path = scene.load_rendered_wavefield_cache(self.frames, self.video_directory)
            self.assertTrue(scene._rendered_wavefield_is_valid(video_path))
            with patch.dict(scene.ANIMATION_TEXT, {"title": "Updated caption"}):
                with patch.object(scene.WavefieldAnimation, "render", side_effect=AssertionError("Background rerendered")):
                    reused = scene.load_rendered_wavefield_cache(self.frames, self.video_directory)
                    self.assertEqual(reused, video_path)
                with patch.object(scene, "load_rendered_wavefield_cache", return_value=video_path):
                    with patch.object(scene.LayerTexture, "update", side_effect=AssertionError("Lattice redrawn")):
                        with patch.object(AcousticWaveSolver2D, "advance", side_effect=AssertionError("Physics rerun")):
                            animation = scene.NumericalWavefield()
                            animation.render()
            self.assertTrue(Path(animation.renderer.file_writer.movie_file_path).exists())
            self.assertEqual(animation.mobjects[1].text.replace(" ", ""), "Updatedcaption")
            self.assertGreater(float(np.std(animation.renderer.get_frame()[..., :3])), 1.0)

    def test_interrupted_video_render_does_not_publish_cache(self):
        with patch.object(scene.WavefieldAnimation, "render", side_effect=RuntimeError("Interrupted")):
            with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                scene.load_rendered_wavefield_cache(self.frames, self.video_directory)
        self.assertEqual(list(self.video_directory.iterdir()), [])

    def test_invalid_video_is_rebuilt(self):
        cache_path = scene.rendered_wavefield_cache_path(self.frames, self.video_directory)
        self.video_directory.mkdir()
        cache_path.write_bytes(b"not a video")
        with patch.object(scene, "load_wavefield_cache", return_value=(self.frames, self.times)):
            rebuilt = scene.load_rendered_wavefield_cache(self.frames, self.video_directory)
        self.assertEqual(rebuilt, cache_path)
        self.assertTrue(scene._rendered_wavefield_is_valid(rebuilt))


if __name__ == "__main__":
    unittest.main()
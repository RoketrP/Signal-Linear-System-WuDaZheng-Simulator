"""Check rendered teaching examples against independent mathematical identities."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_allclose

import app


def render_example(renderer, **parameters):
    ui = MagicMock()
    ui.created_columns = []

    def columns(sizes):
        result = [MagicMock() for _ in range(sizes if isinstance(sizes, int) else len(sizes))]
        ui.created_columns.extend(result)
        return result

    ui.columns.side_effect = columns
    charts = []

    def slider(prefix, name, label, minimum, maximum, value, step):
        return parameters.get(name, value)

    def select(prefix, name, label, options, index=0):
        return parameters.get(name, options[index])

    with (
        patch.object(app, "st", ui),
        patch.object(app, "control_buttons"),
        patch.object(app, "note_block"),
        patch.object(app, "slider_param", side_effect=slider),
        patch.object(app, "select_param", side_effect=select),
        patch.object(app, "render_chart", side_effect=lambda *args, **kwargs: charts.append(args)),
    ):
        renderer()
    return ui, charts


class NumericalRegressions(unittest.TestCase):
    def test_convolution_curve_matches_current_integral(self):
        ui, _ = render_example(app.render_ch2_sec4, t0=1.0)
        figure = ui.plotly_chart.call_args.args[0]
        curve, marker = figure.data[3:5]
        self.assertAlmostEqual(np.interp(marker.x[0], curve.x, curve.y), marker.y[0], delta=0.007)
        expected = 1 - np.exp(-1.25)
        self.assertAlmostEqual(marker.y[0], expected, delta=0.007)

    def test_full_convolution_preserves_associativity(self):
        ui, _ = render_example(app.render_ch2_sec5)
        errors = {c.metric.call_args.args[0]: float(c.metric.call_args.args[1]) for c in ui.created_columns if c.metric.called}
        self.assertLess(errors["结合律误差"], 1e-10)
        self.assertLess(errors["微分性质误差"], 0.03)

    def test_continuous_fft_has_integral_normalization(self):
        t = np.linspace(-8, 8, 4096, endpoint=False)
        f, spectrum = app.fft_spectrum(np.exp(-t**2), t[1] - t[0])
        self.assertAlmostEqual(abs(spectrum[np.argmin(abs(f))]), np.sqrt(np.pi), places=7)

    def test_continuous_fft_uses_actual_time_origin(self):
        t = np.linspace(-8, 8, 4096, endpoint=False)
        f, spectrum = app.fft_spectrum(np.exp(-(t - 0.5)**2), t[1] - t[0], t_start=t[0])
        expected = np.sqrt(np.pi) * np.exp(-(np.pi * f)**2) * np.exp(-2j * np.pi * f * 0.5)
        assert_allclose(spectrum, expected, atol=1e-12)

    def test_triangle_wave_matches_its_series(self):
        _, charts = render_example(app.render_ch4_sec2, kind="三角波", harmonics=15)
        _, _, traces, *_ = charts[0]
        self.assertLess(np.max(abs(traces[0]["y"] - traces[1]["y"])), 0.015)
        assert_allclose(charts[1][1], np.arange(1, 30, 2))

    def test_square_wave_power_sums_towards_one(self):
        _, charts = render_example(app.render_ch4_sec5)
        _, harmonics, traces, *_ = charts[1]
        self.assertTrue(np.any(np.asarray(harmonics) < 0))
        total = np.sum(traces[0]["y"])
        self.assertGreater(total, 0.96)
        self.assertLess(total, 1.0)

    def test_repeated_laplace_pole(self):
        for a, b in [(1.0, 1.0), (0.2, 4.0), (3.0, 0.5), (1.0, 1.0000001)]:
            with self.subTest(a=a, b=b):
                _, charts = render_example(app.render_ch5_sec3, a=a, b=b)
                _, t, traces, *_ = charts[0]
                distance = abs(b - a)
                expected = t * np.exp(-a*t) if distance == 0 else np.exp(-min(a, b)*t) * (-np.expm1(-distance*t)) / distance
                assert_allclose(traces[0]["y"], expected, atol=1e-12)

    def test_repeated_z_pole(self):
        for a, b in [(0.6, 0.6), (0.95, 0.1), (0.2, 0.9)]:
            with self.subTest(a=a, b=b):
                _, charts = render_example(app.render_ch6_sec3, a=a, b=b)
                _, n, traces, *_ = charts[0]
                expected = (n + 1) * a**n if a == b else (a**(n+1) - b**(n+1)) / (a-b)
                assert_allclose(traces[0]["y"], expected, atol=1e-12)

    def test_unobservable_preset_is_actually_unobservable(self):
        ui, _ = render_example(app.render_ch8_sec3, preset="不可观测示例")
        self.assertIn("不可观测", ui.write.call_args.args[0])
        self.assertIn("系统可控", ui.write.call_args.args[0])

    def test_deconvolution_uses_the_observed_scaled_output(self):
        ui, _ = render_example(app.render_ch3_sec5, scale=1.5)
        self.assertIn("[1.5, 3.0, 1.5]", ui.write.call_args.args[0])

    def test_z_reversal_changes_indices(self):
        _, charts = render_example(app.render_ch6_sec2, property="反转性质")
        indices = charts[0][1]
        self.assertLess(min(indices), 0)
        traces = charts[0][2]
        reversed_signal = traces[1]["y"]
        assert_allclose(reversed_signal[np.asarray(indices) > 0], 0)
        self.assertAlmostEqual(reversed_signal[np.asarray(indices) == 0][0], 1)

    def test_z_final_value_is_the_limit(self):
        _, charts = render_example(app.render_ch6_sec2, property="终值定理", a=0.9)
        assert_allclose(charts[0][2][1]["y"], 0)

    def test_left_exponential_matches_its_reported_transform(self):
        _, charts = render_example(app.render_ch5_sec1, kind="左边指数信号", a=1.0)
        t, traces = charts[0][1:3]
        assert_allclose(traces[0]["y"][t < 0], -np.exp(-t[t < 0]))

    def test_discrete_transfer_function_includes_origin_poles(self):
        ui, _ = render_example(app.render_ch6_sec5, kind="移动平均")
        figure = ui.plotly_chart.call_args_list[0].args[0]
        poles = [trace for trace in figure.data if trace.name == "极点"]
        self.assertTrue(poles)
        assert_allclose(poles[0].x, [0, 0])

    def test_fft_filter_preserves_amplitude_and_delay(self):
        _, charts = render_example(app.render_ch4_sec7, fc=4.0, delay=0.0)
        t, traces = charts[0][1:3]
        self.assertLess(np.max(abs(traces[0]["y"] - traces[1]["y"])), 0.04)
        _, delayed_charts = render_example(app.render_ch4_sec7, fc=4.0, delay=0.4)
        interior = (t > -3) & (t < 3)
        delayed = delayed_charts[0][2][1]["y"]
        expected = np.interp(t[interior] - 0.4, t, traces[1]["y"])
        assert_allclose(delayed[interior], expected, atol=0.005)


if __name__ == "__main__":
    unittest.main()

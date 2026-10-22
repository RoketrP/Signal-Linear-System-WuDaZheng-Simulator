"""Exercise real Streamlit widget state across every textbook section."""

import base64
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import app


def render_section(section):
    import app
    app.RENDERERS[section]()


# PNG rendering is checked separately; widget smoke tests avoid repeatedly rendering the same labels.
SMALL_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


class AppFlows(unittest.TestCase):
    def check_errors(self, at):
        self.assertFalse(at.exception, [error.message for error in at.exception])

    def snapshot(self, at):
        return {widget.key: widget.value for kind in (at.slider, at.selectbox) for widget in kind}

    def test_all_sections_example_and_reset(self):
        with patch.object(app, "make_png_bytes", return_value=SMALL_PNG):
            for section in app.RENDERERS:
                with self.subTest(section=section):
                    at = AppTest.from_function(render_section, args=(section,)).run(timeout=30)
                    self.check_errors(at)
                    original = self.snapshot(at)
                    at.button[0].click().run(timeout=30)
                    self.check_errors(at)
                    at.button[1].click().run(timeout=30)
                    self.check_errors(at)
                    self.assertEqual(self.snapshot(at), original)

    def test_all_selectbox_branches(self):
        with patch.object(app, "make_png_bytes", return_value=SMALL_PNG):
            for section in app.RENDERERS:
                at = AppTest.from_function(render_section, args=(section,)).run(timeout=30)
                for index in range(len(at.selectbox)):
                    for option in list(at.selectbox[index].options):
                        with self.subTest(section=section, option=option):
                            at.selectbox[index].select(option).run(timeout=30)
                            self.check_errors(at)

    def test_nyquist_boundary_warns(self):
        section = app.BOOK_STRUCTURE["第四章 傅里叶变换和系统的频域分析"][7]
        with patch.object(app, "make_png_bytes", return_value=SMALL_PNG):
            at = AppTest.from_function(render_section, args=(section,)).run()
            at.slider[0].set_value(5.2).run()
            self.check_errors(at)
            self.assertTrue(any("无法保证" in warning.value for warning in at.warning))

    def test_png_export_is_readable(self):
        import io
        import numpy as np
        from PIL import Image

        png = app.make_png_bytes(np.arange(3), [{"name": "序列", "y": np.array([1, 2, 1])}], "测试", "n", "幅值", discrete=True)
        with Image.open(io.BytesIO(png)) as image:
            image.verify()


if __name__ == "__main__":
    unittest.main()

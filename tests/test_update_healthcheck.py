from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools.update_healthcheck import REQUIRED_ROUTES, validate_web_assets, validate_routes, validate_runtime, validate_syntax


class HealthcheckAssetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.web = self.root / "tools/gui_web"
        self.assets = self.web / "assets"
        self.assets.mkdir(parents=True)
        self.index = self.web / "index.html"
        self.index.write_text('<div id="root"></div><script type="module" src="/assets/main.js"></script><link rel="stylesheet" href="/assets/main.css">')
        (self.assets / "main.js").write_text('import "./lazy.js"; const deps=["assets/lazy.js"];')
        (self.assets / "lazy.js").write_text('export const value=1;')
        (self.assets / "main.css").write_text('@font-face{src:url(./font.woff2)} body{background:url(data:image/png;base64,aGVsbG8=)}')
        (self.assets / "font.woff2").write_bytes(b"font")

    def test_valid_graph_and_missing_lazy_chunk_or_font(self):
        self.assertEqual(validate_web_assets(self.root), 5)
        (self.assets / "main.js").write_text('import "./missing.js";')
        with self.assertRaisesRegex(RuntimeError, "asset is missing"):
            validate_web_assets(self.root)
        (self.assets / "main.js").write_text('export const value=1;')
        (self.assets / "main.css").write_text('body{background:url(./missing.woff2)}')
        with self.assertRaisesRegex(RuntimeError, "asset is missing"):
            validate_web_assets(self.root)

    def test_entry_is_required_and_remote_or_traversal_assets_rejected(self):
        for entry, pattern in [('<div id="root"></div>', "module script"),
                               ('<script type="module" src="/assets/main.js"></script>', "#root"),
                               ('<div id="root"></div><script type="module" src="https://example.com/x.js"></script>', "must be local"),
                               ('<div id="root"></div><script type="module" src="../../secret.js"></script>', "escapes")]:
            self.index.write_text(entry)
            with self.assertRaisesRegex(RuntimeError, pattern):
                validate_web_assets(self.root)

    def test_python_syntax_still_checked(self):
        script = self.root / "module.py"
        script.write_text("value = 1")
        with patch("tools.update_healthcheck.subprocess.run", return_value=SimpleNamespace(returncode=0)):
            self.assertEqual(validate_syntax(self.root, {"files": ["module.py"]}), 1)
            script.write_text("def invalid(:")
            with self.assertRaises(SyntaxError):
                validate_syntax(self.root, {"files": ["module.py"]})

    def test_manifest_assets_must_exist(self):
        with patch("tools.update_healthcheck.subprocess.run", return_value=SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(RuntimeError, "Managed SPA file is missing"):
                validate_syntax(self.root, {"files": ["tools/gui_web/missing.js"]})


class HealthcheckRuntimeTests(unittest.TestCase):
    def test_required_routes_and_live_slices(self):
        routes = [SimpleNamespace(path=path, methods={method}) for method, path in REQUIRED_ROUTES]
        app = SimpleNamespace(routes=routes, state=SimpleNamespace(gui=SimpleNamespace(providers={name: None for name in ("system", "training", "export")})), openapi=lambda: {})
        self.assertEqual(validate_routes(app), len(REQUIRED_ROUTES))
        app.routes.pop()
        with self.assertRaisesRegex(RuntimeError, "missing required routes"):
            validate_routes(app)
        app.routes = routes + [SimpleNamespace(path=path, methods={method}) for method, path in REQUIRED_ROUTES]
        app.state.gui.providers.pop("training")
        with self.assertRaisesRegex(RuntimeError, "snapshot providers"):
            validate_routes(app)

    def test_real_factory_is_offline_without_model_or_browser(self):
        root = Path(__file__).resolve().parents[1]
        with patch("tools.gui_server.services.engine.load_model", side_effect=AssertionError("loaded model")), \
             patch("webbrowser.open", side_effect=AssertionError("opened browser")):
            self.assertGreaterEqual(validate_runtime(root), len(REQUIRED_ROUTES))
        # Other suites legitimately exercise the thin entry point in this
        # interpreter. Prove the healthcheck itself does not import it in a
        # fresh process rather than depending on unittest discovery order.
        import subprocess
        import sys
        code = ("from pathlib import Path; import sys; "
                "from tools.update_healthcheck import validate_runtime; "
                f"validate_runtime(Path({str(root)!r})); "
                "assert 'easy_gui' not in sys.modules")
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

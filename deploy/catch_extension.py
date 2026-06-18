# deploy/catch_extension.py
# Patched to add G1 Catch V21 while keeping older examples if their files exist.

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate


def _try_import(label, import_fn):
    try:
        return import_fn()
    except Exception as exc:
        print(f"[CatchExtension] Skipping {label}: {exc}")
        return None


ExampleV5 = _try_import("G1 UROP V5", lambda: __import__("catch.example_v5", fromlist=["ExampleV5"]).ExampleV5)
G1DeployV8 = _try_import("G1 UROP V8/V9", lambda: __import__("catch.UROP_v8_deploy", fromlist=["G1DeployV8"]).G1DeployV8)
G1CatchV12Deploy = _try_import("G1 Catch V12", lambda: __import__("catch.g1_catch_v12", fromlist=["G1CatchV12Deploy"]).G1CatchV12Deploy)
G1CatchV21Deploy = _try_import("G1 Catch V21", lambda: __import__("catch.g1_catch_v21", fromlist=["G1CatchV21Deploy"]).G1CatchV21Deploy)


class CatchExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.category = "1.UROP_catching"
        self._registered = []

        self._register(ext_id, "G1 Catch V21", "Interactive UROP_v21 bulky-object catching deploy", G1CatchV21Deploy)
        self._register(ext_id, "G1 UROP V5", "Testing UROP V5 Policy", ExampleV5)
        self._register(ext_id, "G1 UROP V8", "G1 UROP box catching deployment V8", G1DeployV8)
        self._register(ext_id, "G1 UROP V9", "G1 UROP box catching deployment V9", G1DeployV8)
        self._register(ext_id, "G1 Catch V12", "G1 UROP box catching deployment V12", G1CatchV12Deploy)

    def _register(self, ext_id: str, name: str, overview: str, sample_cls):
        if sample_cls is None:
            return
        ui_handle = BaseSampleUITemplate(
            ext_id=ext_id,
            file_path=os.path.abspath(__file__),
            title=name,
            overview=overview,
            sample=sample_cls(),
        )
        get_browser_instance().register_example(
            name=name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )
        self._registered.append(name)

    def on_shutdown(self):
        for name in getattr(self, "_registered", []):
            try:
                get_browser_instance().deregister_example(name=name, category=self.category)
            except Exception:
                pass
        self._registered = []

# [/home/idim5080-2/mdj/myIsaacLabstudy/deploy/merged_extension.py]
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate

from merged.g1_loco_catch_merged_v0 import LocoCatchMergedDeployV0

class MergedExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.category = "0.UROP_main"

        ui_handle_1 = BaseSampleUITemplate(
            ext_id=ext_id, file_path=os.path.abspath(__file__),
            title="Deploy Merged v0", overview="UROP deployment locomotion and catching merged", sample=LocoCatchMergedDeployV0()
        )
        get_browser_instance().register_example(name="Deploy Merged v0", execute_entrypoint=ui_handle_1.build_window, ui_hook=ui_handle_1.build_ui, category=self.category)

        # Isaac Sim 켜질 때 'Robotics Examples' 창 자동 띄우기
        import omni.ui as ui
        ui.Workspace.show_window("Robotics Examples")

    def on_shutdown(self):
        get_browser_instance().deregister_example(name="Deploy Merged v0", category=self.category)

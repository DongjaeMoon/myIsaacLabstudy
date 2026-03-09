# [/home/idim5080-2/mdj/myIsaacLabstudy/deploy/catch_extension.py]
#[/home/dongjae/isaaclab/myIsaacLabstudy/deploy/catch_extension.py]
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate

from catch.example_v5 import ExampleV5
from catch.UROP_v8_deploy import G1DeployV8
from catch.g1_catch_v12 import G1CatchV12Deploy

class CatchExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.category = "1.UROP_catching"

        ui_handle_1 = BaseSampleUITemplate(
            ext_id=ext_id, file_path=os.path.abspath(__file__),
            title="G1 UROP V5", overview="Testing UROP V5 Policy", sample=ExampleV5()
        )
        get_browser_instance().register_example(name="G1 UROP V5", execute_entrypoint=ui_handle_1.build_window, ui_hook=ui_handle_1.build_ui, category=self.category)

        ui_handle_2 = BaseSampleUITemplate(
            ext_id=ext_id, file_path=os.path.abspath(__file__),
            title="G1 UROP V8", overview="G1 UROP box catching deployment V8", sample=G1DeployV8()
        )
        get_browser_instance().register_example(name="G1 UROP V8", execute_entrypoint=ui_handle_2.build_window, ui_hook=ui_handle_2.build_ui, category=self.category)

        ui_handle_3 = BaseSampleUITemplate(
            ext_id=ext_id, file_path=os.path.abspath(__file__),
            title="G1 UROP V9", overview="G1 UROP box catching deployment V9", sample=G1DeployV8()
        )
        get_browser_instance().register_example(name="G1 UROP V9", execute_entrypoint=ui_handle_3.build_window, ui_hook=ui_handle_3.build_ui, category=self.category)

        ui_handle_12 = BaseSampleUITemplate(
            ext_id=ext_id, file_path=os.path.abspath(__file__),
            title="G1 Catch V12", overview="G1 UROP box catching deployment V12", sample=G1CatchV12Deploy()
        )
        get_browser_instance().register_example(name="G1 Catch V12", execute_entrypoint=ui_handle_12.build_window, ui_hook=ui_handle_12.build_ui, category=self.category)

    def on_shutdown(self):
        get_browser_instance().deregister_example(name="G1 UROP V5", category=self.category)
        get_browser_instance().deregister_example(name="G1 UROP V8", category=self.category)
        get_browser_instance().deregister_example(name="G1 UROP V9", category=self.category)
        get_browser_instance().deregister_example(name="G1 Catch V12", category=self.category)

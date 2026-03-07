# [/home/idim5080-2/mdj/myIsaacLabstudy/deploy/practice_extension.py]
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate

from deploy_practice.example import Example
from deploy_practice.go2_example import Go2Example
from deploy_practice.example_v3 import ExampleV3
from deploy_practice.example_v4 import ExampleV4
from deploy_practice.example_h1 import H1Example

class PracticeExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.category = "My Study"

        ui_handle_1 = BaseSampleUITemplate(ext_id=ext_id, file_path=os.path.abspath(__file__), title="G1 Practice", overview="G1 Robot Loading", sample=Example())
        get_browser_instance().register_example(name="G1 Practice", execute_entrypoint=ui_handle_1.build_window, ui_hook=ui_handle_1.build_ui, category=self.category)

        ui_handle_2 = BaseSampleUITemplate(ext_id=ext_id, file_path=os.path.abspath(__file__), title="Go2 Running", overview="Go2 Robot Test", sample=Go2Example())
        get_browser_instance().register_example(name="Go2 Running", execute_entrypoint=ui_handle_2.build_window, ui_hook=ui_handle_2.build_ui, category=self.category)

        ui_handle_3 = BaseSampleUITemplate(ext_id=ext_id, file_path=os.path.abspath(__file__), title="G1 UROP V3", overview="Testing UROP V3 Policy", sample=ExampleV3())
        get_browser_instance().register_example(name="G1 UROP V3", execute_entrypoint=ui_handle_3.build_window, ui_hook=ui_handle_3.build_ui, category=self.category)

        ui_handle_4 = BaseSampleUITemplate(ext_id=ext_id, file_path=os.path.abspath(__file__), title="G1 UROP V4", overview="Testing UROP V4 Policy", sample=ExampleV4())
        get_browser_instance().register_example(name="G1 UROP V4", execute_entrypoint=ui_handle_4.build_window, ui_hook=ui_handle_4.build_ui, category=self.category)

        ui_handle_5 = BaseSampleUITemplate(ext_id=ext_id, file_path=os.path.abspath(__file__), title="H1 deploy practice", overview="Deployment practice", sample=H1Example())
        get_browser_instance().register_example(name="H1 deploy practice", execute_entrypoint=ui_handle_5.build_window, ui_hook=ui_handle_5.build_ui, category=self.category)

    def on_shutdown(self):
        get_browser_instance().deregister_example(name="G1 Practice", category=self.category)
        get_browser_instance().deregister_example(name="Go2 Running", category=self.category)
        get_browser_instance().deregister_example(name="G1 UROP V3", category=self.category)
        get_browser_instance().deregister_example(name="G1 UROP V4", category=self.category)
        get_browser_instance().deregister_example(name="H1 deploy practice", category=self.category)

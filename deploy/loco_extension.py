# [/home/idim5080-2/mdj/myIsaacLabstudy/deploy/loco_extension.py]
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate

from loco.g1_loco_v0 import G1LocoV0
from loco.g1_loco_v3 import G1LocoV3Deploy
from loco.g1_loco_v5 import G1LocoV5Deploy
from loco.g1_loco_flat_example import G1LocoFlatExample

class LocoExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.category = "2.UROP_loco"

        ui_handle_1 = BaseSampleUITemplate(
            ext_id=ext_id, file_path=os.path.abspath(__file__),
            title="G1 Loco v0", overview="G1 locomotion practice v0", sample=G1LocoV0()
        )
        get_browser_instance().register_example(name="G1 Loco v0", execute_entrypoint=ui_handle_1.build_window, ui_hook=ui_handle_1.build_ui, category=self.category)

        ui_handle_2 = BaseSampleUITemplate(
            ext_id=ext_id, file_path=os.path.abspath(__file__),
            title="G1 Loco v3", overview="G1 locomotion v3 deployment", sample=G1LocoV3Deploy()
        )
        get_browser_instance().register_example(name="G1 Loco v3", execute_entrypoint=ui_handle_2.build_window, ui_hook=ui_handle_2.build_ui, category=self.category)

        ui_handle_3 = BaseSampleUITemplate(
            ext_id=ext_id, file_path=os.path.abspath(__file__),
            title="G1 Loco v5", overview="G1 locomotion v5 deployment", sample=G1LocoV5Deploy()
        )
        get_browser_instance().register_example(name="G1 Loco v5", execute_entrypoint=ui_handle_3.build_window, ui_hook=ui_handle_3.build_ui, category=self.category)

        ui_handle_4 = BaseSampleUITemplate(
            ext_id=ext_id, file_path=os.path.abspath(__file__),
            title="G1 deploy practice", overview="Deployment practice", sample=G1LocoFlatExample()
        )
        get_browser_instance().register_example(name="G1 deploy practice", execute_entrypoint=ui_handle_4.build_window, ui_hook=ui_handle_4.build_ui, category=self.category)

    def on_shutdown(self):
        get_browser_instance().deregister_example(name="G1 Loco v0", category=self.category)
        get_browser_instance().deregister_example(name="G1 Loco v3", category=self.category)
        get_browser_instance().deregister_example(name="G1 Loco v5", category=self.category)
        get_browser_instance().deregister_example(name="G1 deploy practice", category=self.category)

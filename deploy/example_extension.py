#[/home/dongjae/isaaclab/myIsaacLabstudy/deploy/example_extension.py]
import os
import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate

# [중요] 새로 만든 파일들을 다 import 해옵니다.
from example import Example          # 기존 G1
from example_v3 import ExampleV3
from go2_example import Go2Example   # 새로 만든 Go2 (파일명과 클래스명 주의!)

class ExampleExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.category = "My Study"

        # ---------------------------------------------------------
        # 1. G1 로봇 메뉴 등록
        # ---------------------------------------------------------
        ui_handle_1 = BaseSampleUITemplate(
            ext_id=ext_id,
            file_path=os.path.abspath(__file__),
            title="G1 Practice",       # 메뉴에 뜰 이름
            overview="G1 Robot Loading",
            sample=Example()           # 실행할 클래스
        )
        get_browser_instance().register_example(
            name="G1 Practice",        # 메뉴 이름 (title과 같게)
            execute_entrypoint=ui_handle_1.build_window,
            ui_hook=ui_handle_1.build_ui,
            category=self.category
        )

        # ---------------------------------------------------------
        # 2. Go1 로봇 메뉴 등록 (복사해서 내용만 바꿈)
        # ---------------------------------------------------------
        ui_handle_2 = BaseSampleUITemplate(
            ext_id=ext_id,
            file_path=os.path.abspath(__file__),
            title="Go2 Running",       # 메뉴에 뜰 이름 (다르게 설정)
            overview="Go2 Robot Test",
            sample=Go2Example()        # 실행할 클래스 (Go1Example)
        )
        get_browser_instance().register_example(
            name="Go2 Running",        # 메뉴 이름
            execute_entrypoint=ui_handle_2.build_window,
            ui_hook=ui_handle_2.build_ui,
            category=self.category
        )
    
        ui_handle_3 = BaseSampleUITemplate(
            ext_id=ext_id,
            file_path=os.path.abspath(__file__),
            # [중요] title은 메뉴판에 보일 이름입니다. 겹치지 않게!
            title="G1 UROP V3",      
            overview="Testing UROP V3 Policy",
            sample=ExampleV3()       # [중요] 위에서 임포트한 V3 클래스 실행
        )
        get_browser_instance().register_example(
            name="G1 UROP V3",       # [중요] title과 똑같이 적어주세요
            execute_entrypoint=ui_handle_3.build_window,
            ui_hook=ui_handle_3.build_ui,
            category=self.category
        )

        return

    def on_shutdown(self):
        # 켜진 거 다 꺼줘야 에러가 안 납니다.
        get_browser_instance().deregister_example(name="G1 Practice", category=self.category)
        get_browser_instance().deregister_example(name="Go2 Running", category=self.category)
        get_browser_instance().deregister_example(name="G1 UROP V3", category=self.category)
        return
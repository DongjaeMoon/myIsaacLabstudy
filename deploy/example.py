# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim.examples.interactive.base_sample import BaseSample


from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction

# Note: checkout the required tutorials at https://docs.isaacsim.omniverse.nvidia.com/latest/index.html


class Example(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):









        world = self.get_world()
        world.scene.add_default_ground_plane()

        add_reference_to_stage(usd_path="/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v1/usd/G1_23DOF_UROP.usd", prim_path="/World/robot")

        robot = world.scene.add(Robot(prim_path="/World/robot", name="robot"))
        robot.set_world_pose(position=[0.0, 0.0, 1.0])
        return

    async def setup_post_load(self):

        self._world = self.get_world()


        self._robot = self._world.scene.get_object("robot")

        print(self._robot.get_world_pose())

        self._world.add_physics_callback(callback_name="physics_callback", callback_fn=self._robot_control)



        return



    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return


    def _robot_control(self, step_size):

        action = ArticulationAction(
            joint_positions=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        )

        self._robot.apply_action(action)
        return
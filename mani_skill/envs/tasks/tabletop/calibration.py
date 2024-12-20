from typing import Any, Dict, List, Union

import numpy as np
import sapien.physx as physx
import torch
import json

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

np.random.seed(41)

@register_env("Calibration-v1", max_episode_steps=200)
class CalibrationEnv(BaseEnv):
    
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[Panda]
    OBJECT_INFO_PATH = "/home/tennyyin/projects/active-perception/maniskill-envs/object_info.json"
    PARTNET_MOBILITY_DIR = "/home/tennyyin/.maniskill/data/partnet_mobility/dataset"

    # workspace dimensions
    TABLE_MAX_X = 0.4
    TABLE_MIN_X = -0.4
    TABLE_MAX_Y = 0.6
    TABLE_MIN_Y = -0.6
    OBJECT_MIN_DISTANCE = 0.1
    MAX_NUM_OBJECTS = 8
    MIN_NUM_OBJECTS = 1

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # load object info
        with open(self.OBJECT_INFO_PATH, "r") as f:
            self.all_object_info = json.load(f)
        self.all_object_ids = np.array(list(self.all_object_info.keys()))

        if reconfiguration_freq is None:
            # if not user set, we pick a number
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )


    @property
    def _default_sim_config(self):
        return SimConfig()


    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # use random number of objects and object types
        self.num_obj = np.random.randint(self.MIN_NUM_OBJECTS, self.MAX_NUM_OBJECTS)
        self.obj_ids = []
        self.objs = []
        for i in range(self.num_obj):
            oid = np.random.choice(self.all_object_ids)
            self.obj_ids.append(oid)

        # build articulations from partnet mobility
        for i, oid in enumerate(self.obj_ids):
            builder = articulations.get_articulation_builder(
                self.scene,
                f"partnet-mobility:{oid}",
                fix_root_link=False
            )
            builder.set_scene_idxs(scene_idxs=[0])
            obj = builder.build(name=f"{i}-{oid}")
            self.objs.append(obj)

    # TODO: move to utils
    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def generate_positions(self, num_obj:int):
        """
        Generate random positions for objects on the table
        """
        positions = []
        for _ in range(num_obj):
            while True:
                candidate_x = randomization.uniform(self.TABLE_MIN_X, self.TABLE_MAX_X, size=(1,))
                candidate_y = randomization.uniform(self.TABLE_MIN_Y, self.TABLE_MAX_Y, size=(1,))
                candidate_position = [candidate_x, candidate_y]
                if all(self.distance(candidate_position, p) > self.OBJECT_MIN_DISTANCE for p in positions):
                    positions.append(candidate_position)
                    break
        assert len(positions) == num_obj
        return positions

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)

            positions_list = self.generate_positions(self.num_obj)
            for i in range(self.num_obj):
                p = torch.zeros((b, 3))
                p[:, 0] = positions_list[i][0]
                p[:, 1] = positions_list[i][1]
                p[:, 2] = 0.05 #TODO define offset?

                # TODO: all objects standing upright
                q = randomization.random_quaternions(
                    n=b, lock_x=True, lock_y=True, bounds=(-torch.pi, torch.pi)
                )
                self.objs[i].set_pose(Pose.create_from_pq(p, q))

            # apply pose changes and update kinematics to get updated link poses.
            if physx.is_gpu_enabled():
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()


    def evaluate(self):
        return dict(success=torch.tensor([False]))
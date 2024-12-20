from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.agents.robots.xmate3.xmate3 import Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

WARNED_ONCE = False


@register_env("SingleBottle-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class SingleBottleEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    agent: Union[Panda, PandaWristCam, Fetch]
    goal_thresh = 0.005 # goal radius

    forward_qpos = np.array([
        0.5, # base rotation
        -1.03, 
        0, 
        -3.07, 
        0, 
        2.77, 
        np.pi/4, 
        0.04, 
        0.04])

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.00,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = "006_mustard_bottle"

        # reward - grounding dino
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")
        self.text = "mustard bottle. table. random. object. background."

        if reconfiguration_freq is None:
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
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # add object
        self._objs: List[Actor] = []
        for i in range(self.num_envs):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{self.model_id}",
            )
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{self.model_id}-{i}"))
        self.obj = Actor.merge(self._objs, name="ycb_object")

        # add goal site
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.goal_site)

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            # this value is used to set object pose so the bottom is at z=0
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            # xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]

            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            goal_xyz = torch.zeros((b, 3)) # goal position fixed to 0,0,0
            # goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            # goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Initialize robot arm to a higher position above the table than the default typically used for other table top tasks
            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
                self.agent.robot.set_qpos(self.forward_qpos)
            elif self.robot_uids == "xmate3_robotiq":
                qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.562, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

    def evaluate(self):
        obj_to_goal_pos = self.goal_site.pose.p - self.obj.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static(0.2)
        return dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=self.agent.is_grasping(self.obj),
            success=torch.tensor(False, dtype=torch.bool),
        )
        # return dict(
        #     is_grasped=is_grasped,
        #     obj_to_goal_pos=obj_to_goal_pos,
        #     is_obj_placed=is_obj_placed,
        #     is_robot_static=is_robot_static,
        #     is_grasping=self.agent.is_grasping(self.obj),
        #     success=torch.logical_and(is_obj_placed, is_robot_static),
        # )

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp.pose.p,
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.obj.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        reward += info["is_obj_placed"] * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * is_grasped

        reward[info["success"]] = 6
        return reward

    # def compute_normalized_dense_reward(
    #     self, obs: Any, action: torch.Tensor, info: Dict
    # ):
    #     return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
    
    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # get raw image
        raw_image_np = obs['sensor_data']['hand_camera']['rgb'].cpu().numpy().squeeze(axis=0)
        # load to PIL
        raw_image = Image.fromarray(raw_image_np)
        inputs = self.processor(images=raw_image, text=self.text, return_tensors="pt").to(self.device)
        raw_image.save("/home/tennyyin/projects/utilities/SimpleDreamer/test.png")
        
        # raw_human_np = obs['sensor_data']['base_camera']['rgb'].cpu().numpy().squeeze(axis=0)
        # raw_human = Image.fromarray(raw_human_np)
        # raw_human.save("/home/tennyyin/projects/utilities/SimpleDreamer/test-human.png")
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.2,
            target_sizes=[raw_image.size[::-1]]
        )

        bboxes = results[0]['boxes'] # top-left, bottom-right
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        labels = results[0]['labels']
        scores = results[0]['scores']
        area_thresh = 0.04 * (512*512)

        # find if there exist a 'mustard bottle' in the results with area > 0.1
        target_score = max( (score.item() for score, label, area in zip(scores, labels, areas) if label == 'mustard bottle' and area > area_thresh), default=0 )

        target_score = (np.exp(3.0*target_score)-1.82)/10
        return target_score
        # get results with text 'mustard bottle'
        results = [result for result in results if 'yellow mustard bottle' in result['labels']]
        if len(results) == 0:
            return 0.0
        # else return highest confidence score of 'mustard bottle'


        maxscores=results[0]['scores'].cpu().numpy()
        # if len(maxscores) == 0:
        #     return 0.0
        return np.max(maxscores)
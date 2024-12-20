"""

Collects observations at each end effector position from test_poses.py. Specify output directory on line 65, 66

camera angles are hard coded here.

"""
import torch
import numpy as np
import gymnasium as gym

# utilities
import tyro
import sapien
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R

# maniskill
import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

@dataclass
class Args:
    # Algorithm specific arguments
    env_id: str = "BlockObstacle-v1"# "SingleBottle-v1"
    num_states: int = 2000
    seed: int = 2
    # env specific arguments
    ws_xmin: float = -0.3
    ws_xmax: float = 0.3
    ws_ymin: float = -0.3
    ws_ymax: float = 0.3
    ws_zmin: float = 0.08
    ws_zmax: float = 0.6
    target_position_x: float = 0
    target_position_y: float = 0
    target_position_z: float = 0.1
    angular_threshold_deg: float = 40.0

    # save 
    odir: str = "./img"
    id: str = "test6"


def is_camera_looking_at_target(camera_position, camera_quaternion, target_position, angular_threshold_deg):
    # Compute target direction vector
    d_target = np.array(target_position) - np.array(camera_position)
    v_target = d_target / np.linalg.norm(d_target)

    # Convert quaternion to rotation matrix and extract forward vector
    r = R.from_quat(camera_quaternion)  # Quaternion as [x, y, z, w]
    v_forward = r.apply([0, 0, 1])  # Assuming forward is (0, 0, 1) in the camera frame

    # Compute alignment
    cos_theta = np.dot(v_forward, v_target)
    angular_threshold_rad = np.deg2rad(angular_threshold_deg)

    return cos_theta > np.cos(angular_threshold_rad)

def check_ef_limits(tcp_pose):
    """
    check if end effector pose is within workspace limits
    """
    xyz = tcp_pose[:3]
    quaternion = tcp_pose[3:]
    # check if xyz is within limits
    if xyz[0] < args.ws_xmin or xyz[0] > args.ws_xmax:
        return False
    if xyz[1] < args.ws_ymin or xyz[1] > args.ws_ymax:
        return False
    if xyz[2] < args.ws_zmin or xyz[2] > args.ws_zmax:
        return False

    # return True
    # check if camera is looking at target
    target_position = [args.target_position_x, args.target_position_y, args.target_position_z]
    
    return is_camera_looking_at_target(xyz, quaternion, target_position, args.angular_threshold_deg)
    
#  def check_collision(tcp_pose):
    # """
    # check if end effector pose is in collision
    # """
       
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    # env setup
    env_kwargs = dict(
      obs_mode="rgb", # TODO state or rgb?
      control_mode="pd_ee_delta_pose", 
      render_mode="rgb_array", 
      sim_backend="gpu"
      )
    envs = gym.make(
      args.env_id, 
      num_envs=1, 
      **env_kwargs
      )
    breakpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # states = torch.zeros(args.num_states).to(device)
    
    next_obs, _ = envs.reset(seed=args.seed)
    
    tcp_poses = np.zeros((args.num_states, 7))
    actions = np.zeros((args.num_states, 7))
    
    # start run
    for i in range(args.num_states):
        if i % 50 == 0:
            print(f"Collecting state {i}")
        
        # save info
        next_obs_rgb = next_obs['sensor_data']['hand_camera']['rgb'].cpu().numpy().squeeze(axis=0)
        img = Image.fromarray(next_obs_rgb.astype(np.uint8))
        img.save("./img/{}/{}.png".format(args.id, i))
            
        tcp_pose= next_obs['extra']['tcp_pose'].cpu().numpy().flatten()
        
        tcp_poses[i] = tcp_pose
        
        # sample next action
        acceptable_action = False
        state_dict = envs.unwrapped.get_state_dict()
        
        while not acceptable_action:
          # sample aciton
          action = envs.action_space.sample() # delta joint position
          
          # calculate expected tcp position after action
          next_obs, reward, terminations, truncations, infos = envs.step(action)
          new_tcp_pose = next_obs['extra']['tcp_pose'].cpu().numpy().flatten()
          
          acceptable_action = check_ef_limits(new_tcp_pose) # and check_collision(new_tcp_pose)
          
          envs.unwrapped.set_state_dict(state_dict)
        
        actions[i] = action
        next_obs, reward, terminations, truncations, infos = envs.step(action)

    # save tcp poses
    np.save("./tcp_poses_"+args.id+".npy", tcp_poses)
    np.save("./actions_"+args.id+".npy", actions)
      
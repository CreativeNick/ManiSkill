import gymnasium as gym
import torch

from mani_skill.envs.scenes.tasks import PickSubtaskTrainEnv
from mani_skill.envs.scenes.tasks.planner import plan_data_from_file
from mani_skill.utils import common
from mani_skill.utils.scene_builder.replicacad.rearrange import (
    ReplicaCADPrepareGroceriesTrainSceneBuilder,
    ReplicaCADPrepareGroceriesValSceneBuilder,
    ReplicaCADSetTableTrainSceneBuilder,
    ReplicaCADSetTableValSceneBuilder,
    ReplicaCADTidyHouseTrainSceneBuilder,
    ReplicaCADTidyHouseValSceneBuilder,
)
from mani_skill.utils.visualization.misc import images_to_video

SEED = 2024
TASK_PLAN_FP = "005_tomato_soup_can-bci=10.json"
SIM_STATE_FP = "grasp_issue_sim_states.pt"
NUM_ENVS = 1

plan_data = plan_data_from_file(TASK_PLAN_FP)
scene_builder_cls = {
    "ReplicaCADTidyHouseTrain": ReplicaCADTidyHouseTrainSceneBuilder,
    "ReplicaCADTidyHouseVal": ReplicaCADTidyHouseValSceneBuilder,
    "ReplicaCADPrepareGroceriesTrain": ReplicaCADPrepareGroceriesTrainSceneBuilder,
    "ReplicaCADPrepareGroceriesVal": ReplicaCADPrepareGroceriesValSceneBuilder,
    "ReplicaCADSetTableTrain": ReplicaCADSetTableTrainSceneBuilder,
    "ReplicaCADSetTableVal": ReplicaCADSetTableValSceneBuilder,
}[plan_data.dataset]

env: PickSubtaskTrainEnv = gym.make(
    # sapien_env kwargs
    "PickSubtaskTrain-v0",
    num_envs=NUM_ENVS,
    obs_mode="state",
    reward_mode="normalized_dense",
    control_mode="pd_joint_delta_pos",
    render_mode="cameras",
    shader_dir="default",
    robot_uids="fetch",
    sim_backend="cpu",
    # time limit
    max_episode_steps=100,
    # SequentialTaskEnv args
    task_plans=plan_data.plans,
    scene_builder_cls=scene_builder_cls,
    # SubtaskTrainEnv args
    robot_force_mult=0.001,
    robot_force_penalty_min=0.2,
    robot_cumulative_force_limit=5000,
    randomize_arm=True,
    randomize_base=True,
    randomize_loc=True,
    target_randomization=False,
)

sim_states = torch.load(SIM_STATE_FP)
actions = torch.load("7_actions.pt")
images = []

env.reset(seed=SEED)
env.set_state_dict(sim_states[0])
actors = env.scene.actors
env.render_human().paused = True
for action in actions[1:]:
    # env.set_state_dict(state)
    env.step(action.cpu().numpy())
    env.render_human()
    # print(env.scene.actors["env-0_005_tomato_soup_can-1"].pose.p)
    images.append(common.to_numpy(env.render().squeeze(0)))

images_to_video(images, ".", video_name="grasp_issue_vid", fps=20)

env.close()

from mani_skill.envs.scenes.tasks.planner import plan_data_from_file
from mani_skill.envs.scenes.tasks import (
    PickSubtaskTrainEnv,
)
from mani_skill.utils.scene_builder.replicacad.rearrange import (
    ReplicaCADTidyHouseTrainSceneBuilder,
    ReplicaCADPrepareGroceriesTrainSceneBuilder,
    ReplicaCADSetTableTrainSceneBuilder,
    ReplicaCADTidyHouseValSceneBuilder,
    ReplicaCADPrepareGroceriesValSceneBuilder,
    ReplicaCADSetTableValSceneBuilder,
)
from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils import common

import torch

import gymnasium as gym

SEED = 2024
TASK_PLAN_FP = "005_tomato_soup_can-bci=10.json"
# SIM_STATE_FP = "grasp_issue_sim_states.pt"
SIM_STATE_FP = "dexm3_exps/EVAL--PickSubtaskTrain-v0/rcad-vsac-bci10-bsize-dis/job_8_9181_tpgURG2d952633bd782759d7b8df6195d0c85b33b33eab_pd_joint_delta_pos_cont_rand_arm_rand_base_rand_loc_rforce_mult=0.001_rforce_pen_min=0.2_rforce_cum_lim=5000_mobile_torso_mobile_base_ac/latest/eval_videos/7.pt"
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
    obs_mode="rgbd",
    reward_mode="normalized_dense",
    control_mode="pd_joint_delta_pos",
    render_mode="cameras",
    shader_dir="default",
    robot_uids="fetch",
    sim_backend="gpu",
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
images = []

env.reset(seed=SEED)
actors = env.scene.actors
for state in sim_states:
    env.set_state_dict(state)
    env.step(action=torch.zeros(env.action_space.shape))
    print(
        env.subtask_objs[0].pose.p,
        env.agent.is_grasping(env.subtask_objs[0], max_angle=85),
    )
    print(
        env.scene.actors["env-0_005_tomato_soup_can-1"].pose.p,
        env.agent.is_grasping(
            env.scene.actors["env-0_005_tomato_soup_can-1"], max_angle=85
        ),
    )
    print("====")
    images.append(common.to_numpy(env.render().squeeze(0)))

images_to_video(images, ".", video_name="grasp_issue_vid", fps=20)

env.close()

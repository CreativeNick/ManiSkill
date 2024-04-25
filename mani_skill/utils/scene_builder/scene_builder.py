from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union, Tuple

import torch
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Actor, Articulation


class SceneBuilder:
    """Base class for defining scene builders that can be reused across tasks"""

    """Env which scenebuilder will build in."""
    env: BaseEnv

    """Whether this scene builder will add its own lighting when build is called. If False, ManiSkill will add some default lighting"""
    builds_lighting: bool = False

    """
    **Optional** list of scene configuration information that can be used to build/init scenes. Can be a dictionary, a path to a json file, or some other data.
    Some scenes will need to load config data, while others might not.
    """
    build_configs: List[Any] = None
    init_configs: List[Any] = None

    """
    Dictionaries mapping names to scene objects, movable objects, and articulations for easy reference.
    """
    scene_objects: Dict[str, Actor]
    movable_objects: Dict[str, Actor]
    articulations: Dict[str, Articulation]

    """
    Some scenes allow for mobile robots to move through these scene. In this case, a list of navigable positions per env_idx should be provided for easy initialization.
    Can be a discretized list, range, spaces.Box, etc
    """
    navigable_positions: list[Union[np.ndarray, spaces.Box, Any]]

    def __init__(self, env):
        self.env = env

    def build(self, build_config_idxs: List[int] = None):
        """
        Should create actor/articulation builders and only build objects into the scene without initializing pose, qpos, velocities etc.
        """
        raise NotImplementedError()

    def initialize(self, env_idx: torch.Tensor, init_config_idxs: List[int] = None):
        """
        Should initialize the scene, which can include e.g. setting the pose of all objects, changing the qpos/pose of articulations/robots etc.
        """
        raise NotImplementedError()

    def sample_build_config_idxs(self):
        """
        Sample idxs of build configs for easy scene randomization. Should be changed to fit shape of self.build_configs.
        """
        return torch.randint(
            low=0, high=len(self.build_configs), size=(self.env.num_envs,)
        ).tolist()

    def sample_init_config_idxs(self):
        """
        Sample idxs of init configs for easy scene randomization. Should be changed to fit shape of self.init_configs.
        """
        return torch.randint(
            low=0, high=len(self.init_configs), size=(self.env.num_envs,)
        ).tolist()

    @property
    def scene(self):
        return self.env._scene

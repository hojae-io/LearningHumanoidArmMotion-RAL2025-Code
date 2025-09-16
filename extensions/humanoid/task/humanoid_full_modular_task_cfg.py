# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp  # noqa: F401, F403
import extensions.humanoid.mdp as brl_mdp

##
# Pre-defined configs
##
from extensions.humanoid.assets import HUMANOID_FULL_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from extensions.humanoid.terrains import COBBLESTONE_ROAD_CFG


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane", # "plane" or "generator"
        terrain_generator=ROUGH_TERRAINS_CFG, # ROUGH_TERRAINS_CFG, COBBLESTONE_ROAD_CFG
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # static_friction=1.0,
            # dynamic_friction=1.0,
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = HUMANOID_FULL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    # # camera
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = brl_mdp.VelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        reference="body",
        ranges=brl_mdp.VelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-2.0, 2.0), ang_vel_z=(-1.5, 1.5),
        ),
    )

    # """ Stepping Commands """
    step_command = brl_mdp.StepCommandCfg(
        step_period=18, # 18
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    leg_joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=HUMANOID_FULL_CFG.actuators['legs'].joint_names_expr) #! The order of the joints are not correct.
    arm_joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr) #! The order of the joints are not correct.
    # arm_joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr, disable_action=True) #! The order of the joints are not correct.


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LegActorCfg(ObsGroup):
        """Observations for leg actor. (order preserved)"""

        base_height = ObsTerm(func=mdp.base_pos_z, noise=Gnoise(std=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(std=0.05))
        base_heading = ObsTerm(func=brl_mdp.base_heading, noise=Gnoise(std=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(std=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        phase_sin = ObsTerm(func=brl_mdp.phase_sin)
        phase_cos = ObsTerm(func=brl_mdp.phase_cos)
        
        joint_pos = ObsTerm(func=brl_mdp.joint_pos, noise=Gnoise(std=0.05), 
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['legs'].joint_names_expr)})
        joint_vel = ObsTerm(func=brl_mdp.joint_vel, noise=Gnoise(std=1.0),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['legs'].joint_names_expr)})
        last_leg_action = ObsTerm(func=mdp.last_action, params={"action_name": "leg_joint_pos"}, noise=Gnoise(std=0.1))
        CAM = ObsTerm(func=brl_mdp.centroidal_angular_momentum_mixed, noise=Gnoise(std=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # @configclass
    # class LegCriticCfg(LegActorCfg):
    #     """Observations for leg critic. (order preserved)"""

    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True

    @configclass
    class LegCriticCfg(ObsGroup):
        """Observations for leg critic. (order preserved)"""
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Gnoise(std=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(std=0.05))
        base_heading = ObsTerm(func=brl_mdp.base_heading, noise=Gnoise(std=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(std=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        phase_sin = ObsTerm(func=brl_mdp.phase_sin)
        phase_cos = ObsTerm(func=brl_mdp.phase_cos)
        
        joint_pos = ObsTerm(func=brl_mdp.joint_pos, noise=Gnoise(std=0.05))
        joint_vel = ObsTerm(func=brl_mdp.joint_vel, noise=Gnoise(std=1.0))
        last_leg_action = ObsTerm(func=mdp.last_action, params={"action_name": "leg_joint_pos"})
        last_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "arm_joint_pos"})
        CAM = ObsTerm(func=brl_mdp.centroidal_angular_momentum_mixed, noise=Gnoise(std=0.1))
        CAM_des = ObsTerm(func=brl_mdp.centroidal_angular_momentum_des_mixed)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ArmActorCfg(ObsGroup):
        """Observations for arm actor. (order preserved)"""
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Gnoise(std=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(std=0.05))
        base_heading = ObsTerm(func=brl_mdp.base_heading, noise=Gnoise(std=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(std=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.05))
        
        joint_pos = ObsTerm(func=brl_mdp.joint_pos, noise=Gnoise(std=0.05),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr)})
        joint_vel = ObsTerm(func=brl_mdp.joint_vel, noise=Gnoise(std=1.0),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr)})
        last_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "arm_joint_pos"}, noise=Gnoise(std=0.1))
        CAM = ObsTerm(func=brl_mdp.centroidal_angular_momentum_mixed, noise=Gnoise(std=0.1))
        CAM_des = ObsTerm(func=brl_mdp.centroidal_angular_momentum_des_mixed)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # @configclass
    # class ArmCriticCfg(ArmActorCfg):
    #     """Observations for arm critic. (order preserved)"""

    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True

    @configclass
    class ArmCriticCfg(ObsGroup):
        """Observations for arm critic. (order preserved)"""
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Gnoise(std=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(std=0.05))
        base_heading = ObsTerm(func=brl_mdp.base_heading, noise=Gnoise(std=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(std=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        phase_sin = ObsTerm(func=brl_mdp.phase_sin)
        phase_cos = ObsTerm(func=brl_mdp.phase_cos)
        
        joint_pos = ObsTerm(func=brl_mdp.joint_pos, noise=Gnoise(std=0.05))
        joint_vel = ObsTerm(func=brl_mdp.joint_vel, noise=Gnoise(std=1.0))
        last_leg_action = ObsTerm(func=mdp.last_action, params={"action_name": "leg_joint_pos"})
        last_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "arm_joint_pos"})
        CAM = ObsTerm(func=brl_mdp.centroidal_angular_momentum_mixed, noise=Gnoise(std=0.1))
        CAM_des = ObsTerm(func=brl_mdp.centroidal_angular_momentum_des_mixed)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    leg_actor: LegActorCfg = LegActorCfg()
    leg_critic: LegCriticCfg = LegCriticCfg()
    arm_actor: ArmActorCfg = ArmActorCfg()
    arm_critic: ArmCriticCfg = ArmCriticCfg()


@configclass
class EventsCfg:
    """Configuration for randomization by events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x":     (-0., 0.), 
                "y":     (-0., 0.), 
                "z":     (-0., 0.),
                "roll":  (-torch.pi/20, torch.pi/20), 
                "pitch": (-torch.pi/20, torch.pi/20), 
                "yaw":   (-torch.pi, torch.pi)
            },
            "velocity_range": {
                "x":     (-.1, .1),
                "y":     (-.1, .1),
                "z":     (-.1, .1),
                "roll":  (-.1, .1),
                "pitch": (-.1, .1),
                "yaw":   (-.1, .1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_range,
        mode="reset",
        params={
            "position_range": torch.tensor([[-0.1, 0.1], # right_hip_yaw
                                            [-0.1, 0.1], # left_hip_yaw
                                            [-0.1, 0.1], # right_hip_abad
                                            [-0.1, 0.1], # left_hip_abad
                                            [-0.2, 0.2], # right_hip_pitch
                                            [-0.2, 0.2], # left_hip_pitch
                                            [ 0.6, 0.7], # right_knee
                                            [ 0.6, 0.7], # left_knee
                                            [-0.3, 0.0], # right_ankle
                                            [-0.3, 0.0], # left_ankle
                                            [-0.1, 0.1], # right_shoulder_pitch
                                            [-0.1, 0.1], # left_shoulder_pitch
                                            [-0.1, 0.1], # right_shoulder_abad
                                            [-0.1, 0.1], # left_shoulder_abad
                                            [-0.1, 0.1], # right_shoulder_yaw
                                            [-0.1, 0.1], # left_shoulder_yaw
                                            [-0.1, 0.1], # right_elbow
                                            [-0.1, 0.1], # left_elbow
                                            ]),
            "velocity_range": torch.tensor([[-0.1, 0.1], # right_hip_yaw
                                            [-0.1, 0.1], # left_hip_yaw
                                            [-0.1, 0.1], # right_hip_abad
                                            [-0.1, 0.1], # left_hip_abad
                                            [-0.1, 0.1], # right_hip_pitch
                                            [-0.1, 0.1], # left_hip_pitch
                                            [-0.1, 0.1], # right_knee
                                            [-0.1, 0.1], # left_knee
                                            [-0.1, 0.1], # right_ankle
                                            [-0.1, 0.1], # left_ankle
                                            [-0.1, 0.1], # right_shoulder_pitch
                                            [-0.1, 0.1], # left_shoulder_pitch
                                            [-0.1, 0.1], # right_shoulder_abad
                                            [-0.1, 0.1], # left_shoulder_abad
                                            [-0.1, 0.1], # right_shoulder_yaw
                                            [-0.1, 0.1], # left_shoulder_yaw
                                            [-0.1, 0.1], # right_elbow
                                            [-0.1, 0.1], # left_elbow
                                            ]),
        },
    )

    # reset_ball = EventTerm(
    #     func=brl_mdp.throw_ball_uniform,
    #     mode="interval",
    #     interval_range_s=(3.0, 5.0),
    #     params={
    #         "pose_range": {
    #             "x":     (1., 1.), 
    #             "y":     (0., 0.), 
    #             "z":     (1., 1.),
    #             "roll":  (-0., 0.), 
    #             "pitch": (-0., 0.), 
    #             "yaw":   (-0., 0.)
    #         },
    #         "velocity_range": {
    #             "x":     (-2., -2.),
    #             "y":     (-0., 0.),
    #             "z":     (-.5, .5),
    #             "roll":  (-0., 0.),
    #             "pitch": (-0., 0.),
    #             "yaw":   (-0., 0.),
    #         },
    #     },
    # )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_xy_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class EventsDeployCfg:
    """Configuration for randomization by events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x":     (-0., 0.), 
                    "y":     (-0., 0.), 
                    "z":     (-0., 0.),
                    "roll":  (-0., 0.), 
                    "pitch": (-0., 0.), 
                    "yaw":   (-torch.pi, torch.pi)
                },
                "velocity_range": {
                    "x":     (-0., 0.),
                    "y":     (-0., 0.),
                    "z":     (-0., 0.),
                    "roll":  (-0., 0.),
                    "pitch": (-0., 0.),
                    "yaw":   (-0., 0.),
                },
            },
        )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # interval
    apply_external_force = EventTerm(
        func=brl_mdp.apply_external_force_torque_disturbance,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"force_range": (-15.0, 15.0), "torque_range": (-1.5, 1.5)},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    @configclass
    class LegRewardsCfg:
        """Reward terms for the leg actor-critic."""

        # * Regularization rewards * #
        action_smoothness1 = RewTerm(
            func=brl_mdp.action_smoothness1,
            weight=2e-3,
            params={"action_name": "leg_joint_pos"}
        )
        action_smoothness2 = RewTerm(
            func=brl_mdp.action_smoothness2,
            weight=2e-4,
            params={"action_name": "leg_joint_pos"}
        )
        joint_torque = RewTerm(
            func=brl_mdp.joint_torque_penalty,
            weight=1e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['legs'].joint_names_expr)}
        )
        joint_velocity = RewTerm(
            func=brl_mdp.joint_velocity_penalty,
            weight=2e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['legs'].joint_names_expr)}
        )
        base_lin_vel_z = RewTerm(
            func=brl_mdp.base_lin_vel_z_penalty,
            weight=1e-1,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_ang_vel_xy = RewTerm(
            func=brl_mdp.base_ang_vel_xy_penalty,
            weight=1e-2,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        joint_pos_limits = RewTerm(
            func=brl_mdp.joint_pos_limits_penalty,
            weight=10,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['legs'].joint_names_expr)}
        )
        joint_torque_limits = RewTerm(
            func=brl_mdp.joint_torque_limits_penalty,
            weight=1e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['legs'].joint_names_expr)}
        )
        joint_regularization = RewTerm(
            func=brl_mdp.joint_regularization_penalty,
            weight=1.,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # * Floating base rewards * #
        # base_height = RewTerm(
        #     func=brl_mdp.base_height_reward,
        #     weight=1.0,
        #     params={"asset_cfg": SceneEntityCfg("robot"), "base_height_target": 0.62}
        # )
        # base_heading = RewTerm(
        #     func=brl_mdp.base_heading_reward,
        #     weight=3.0,
        #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"}
        # )
        base_z_orientation = RewTerm(
            func=brl_mdp.base_z_orientation_reward,
            weight=0.7, # 1.0,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # tracking_lin_vel_world = RewTerm(
        #     func=brl_mdp.tracking_lin_vel_world_reward,
        #     weight=4.0,
        #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"}
        # )
        tracking_lin_vel_xy = RewTerm(
            func=brl_mdp.tracking_lin_vel_xy_reward,
            weight=4.6, # 4.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"}
        )
        tracking_yaw_vel = RewTerm(
            func=brl_mdp.tracking_yaw_vel_reward,
            weight=2.51, #1.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"}
        )
        # * Stepping rewards * #
        contact_schedule = RewTerm(
            func=brl_mdp.contact_schedule_reward,
            weight=3.0,
        )

        # * Termination rewards * #
        termination = RewTerm(
            func=brl_mdp.termination_penalty,
            weight=1.0,
            params={"group_name": "leg"}
        )

    @configclass
    class ArmRewardsCfg:
        """Reward terms for the arm actor-critic."""
        # * Regularization rewards * #
        action_smoothness1 = RewTerm(
            func=brl_mdp.action_smoothness1,
            weight=1e-3,
            params={"action_name": "arm_joint_pos"}
        )
        action_smoothness2 = RewTerm(
            func=brl_mdp.action_smoothness2,
            weight=1e-4,
            params={"action_name": "arm_joint_pos"}
        )
        joint_torque = RewTerm(
            func=brl_mdp.joint_torque_penalty,
            weight=5e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr)}
        )
        joint_velocity = RewTerm(
            func=brl_mdp.joint_velocity_penalty,
            weight=5e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr)}
        )
        joint_position = RewTerm(
            func=brl_mdp.joint_position_penalty,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr)}
        )
        joint_pos_limits = RewTerm(
            func=brl_mdp.joint_pos_limits_penalty,
            weight=10,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr)}
        )
        joint_torque_limits = RewTerm(
            func=brl_mdp.joint_torque_limits_penalty,
            weight=1e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr)}
        )
        dCAM_xy = RewTerm(
            func=brl_mdp.dCAM_xy_penalty,
            weight=5e-2,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        tracking_CAM_reward = RewTerm(
            func=brl_mdp.tracking_CAM_reward,
            weight=3.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"}
        )

        # * Termination rewards * #
        termination = RewTerm(
            func=brl_mdp.termination_penalty,
            weight=1.0,
            params={"group_name": "arm"}
        )

    leg: LegRewardsCfg = LegRewardsCfg()
    arm: ArmRewardsCfg = ArmRewardsCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    @configclass
    class LegTerminationCfg:
        # illegal_contact = DoneTerm(
        #     func=brl_mdp.illegal_contact,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", 
        #                                                                        "right_upper_leg",
        #                                                                        "right_lower_leg",
        #                                                                        "left_upper_leg",
        #                                                                        "left_lower_leg",]), "threshold": 1.0},
        # )
        illegal_contact = DoneTerm(
            func=brl_mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", 
                                                                               "right_upper_leg",
                                                                               "right_lower_leg",
                                                                               "left_upper_leg",
                                                                               "left_lower_leg",
                                                                               "right_upper_arm",
                                                                               "right_lower_arm",
                                                                               "right_hand",
                                                                               "left_upper_arm",
                                                                               "left_lower_arm",
                                                                               "left_hand",]), "threshold": 1.0},
        )
        base_termination = DoneTerm(
            func=brl_mdp.base_termination,
            params={
                "max_lin_vel": 15.0,
                "max_ang_vel": 10.0,
                "max_tilting": 0.8,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    @configclass
    class ArmTerminationCfg:
        # illegal_contact = DoneTerm(
        #     func=brl_mdp.illegal_contact,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base",
        #                                                                        "right_upper_arm",
        #                                                                        "right_lower_arm",
        #                                                                        "right_hand",
        #                                                                        "left_upper_arm",
        #                                                                        "left_lower_arm",
        #                                                                        "left_hand",]), "threshold": 1.0},
        # )
        illegal_contact = DoneTerm(
            func=brl_mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", 
                                                                               "right_upper_leg",
                                                                               "right_lower_leg",
                                                                               "left_upper_leg",
                                                                               "left_lower_leg",
                                                                               "right_upper_arm",
                                                                               "right_lower_arm",
                                                                               "right_hand",
                                                                               "left_upper_arm",
                                                                               "left_lower_arm",
                                                                               "left_hand",]), "threshold": 1.0},
        )

    @configclass
    class TimeOutTerminationCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    leg: LegTerminationCfg = LegTerminationCfg()
    arm: ArmTerminationCfg = ArmTerminationCfg()
    time_out: TimeOutTerminationCfg = TimeOutTerminationCfg()


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    pass


##
# Environment configuration
##


@configclass
class HumanoidFullModularEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    DEPLOYMENT_TRAINING = True # True, False
    ADD_BALL = False # True, False

    # Viewer
    # viewer = ViewerCfg(eye=(0., -3.0, 0.5), origin_type='asset_root', asset_name='robot')
    viewer = ViewerCfg(eye=(2.0, -2.0, 0.5), origin_type='asset_root', asset_name='robot')
    # viewer = ViewerCfg(eye=(0., -3., 1.), lookat=(0., 1., 0.5), origin_type='world')
    
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    if DEPLOYMENT_TRAINING:
        events: EventsDeployCfg = EventsDeployCfg()
        scene.robot.init_state.pos = (0., 0., 0.65)
        scene.robot.init_state.joint_pos = {'a01_right_hip_yaw':        0.,       'a06_left_hip_yaw':        0.,
                                            'a02_right_hip_abad':      -0.1,      'a07_left_hip_abad':       0.1,
                                            'a03_right_hip_pitch':     -0.724757, 'a08_left_hip_pitch':     -0.724757,
                                            'a04_right_knee':           1.412282, 'a09_left_knee':           1.412282,
                                            'a05_right_ankle':         -0.68752,  'a10_left_ankle':         -0.68752,
                                            'a11_right_shoulder_pitch': 0.,       'a15_left_shoulder_pitch': 0.,
                                            'a12_right_shoulder_abad': -0.1,      'a16_left_shoulder_abad':  0.1,
                                            'a13_right_shoulder_yaw':   0.,       'a17_left_shoulder_yaw':   0.,
                                            'a14_right_elbow':         -0.1,      'a18_left_elbow':         -0.1
                                            }
    else:
        events: EventsCfg = EventsCfg()

    if ADD_BALL:
        # Rigid Object
        scene.ball = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.SphereCfg(
                        radius=0.1,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.0),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="average",
                        restitution_combine_mode="average",
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=1.0
                    ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(10.0, 0.0, 0.),
                    )
                )

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 5
        # video recording settings
        self.video_length_s = 3
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if hasattr(self.scene, 'height_scanner'):
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if hasattr(self.scene, 'contact_forces'):
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class HumanoidFullModularEnvCfg_PLAY(HumanoidFullModularEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.sim.disable_contact_processing = False
        # make a smaller scene for play
        self.scene.num_envs = 3
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.scene.rf_GRF = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_foot", 
                                             debug_vis=False, # True, False
                                             filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
                                             visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/RightFootContactSensor"),
                                             max_contact_data_count_per_env=5,)
    
        self.scene.lf_GRF = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_foot",
                                             debug_vis=False, # True, False
                                             filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
                                             visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/LeftFootContactSensor"),
                                             max_contact_data_count_per_env=5,)

        # disable randomization for play
        self.observations.leg_actor.enable_corruption = False
        self.observations.arm_actor.enable_corruption = False
        # remove random pushing
        self.events.physics_material = None
        self.events.add_base_mass = None
        if hasattr(self.events, 'base_external_force_torque'):
            self.events.base_external_force_torque = None
        if hasattr(self.events, 'push_robot'):
            self.events.push_robot = None
        if hasattr(self.events, 'apply_external_force'):
            self.events.apply_external_force = None

        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x":     (-0., 0.), 
                    "y":     (-0., 0.), 
                    "z":     (-0., 0.),
                    "roll":  (-0., 0.), 
                    "pitch": (-0., 0.), 
                    "yaw":   (-0., 0.)
                },
                "velocity_range": {
                    "x":     (-0., 0.),
                    "y":     (-0., 0.),
                    "z":     (-0., 0.),
                    "roll":  (-0., 0.),
                    "pitch": (-0., 0.),
                    "yaw":   (-0., 0.),
                },
            },
        )
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0., 0.),
                "velocity_range": (-0., 0.),
            },
        )

        self.episode_length_s = int(1e7)
        self.commands.base_velocity.resampling_time_range = (int(1e7), int(1e7))
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

"""
TODO:

1) rotor_inertia (=joint armature) is not set yet (v)
2) In env.scene["robot"].data, body_names and joint_names are not in the correct order.
3) Camera View (v)
4) Velocity Arrow change / World-frame vel command (v)
5) Foot step arrow visualization (v)
6) Plot contact forces (v)
7) Termination reward (v)
8) Training speed too slow
9) URDF color update (v) 
10) Video recording speed to real-time (v)
11) Joint Jacobian transfer
12) Keyboard control of the velocity command (v)
13) Option to disable logging (v)
14) soft joint torque / vel limit is weird? (v)
15) Rest of the config setting match with IsaacGym (v)
16) Video recording / Screenshot / Animation for play script (v)
17) Make full joint urdf / arm only urdf
18) Find the difference between lab vanilla and IsaacGyn vanilla #! Because of normalization?
19) Rendering speed with --cpu is too slow
20) Save the code / Load the cfg files when running play script
21) Non-noisy critic observation (v)
22) With camera, I cannot start the training? (v)

"""


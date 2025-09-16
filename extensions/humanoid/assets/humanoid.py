
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, TorqueActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from extensions import ISAACLAB_BRL_ROOT_DIR
from extensions.humanoid.assets.humanoid_parameters import effort_limit, velocity_limit, stiffness, damping, \
                                                           armature, friction

LEG_PATTERNS = ["hip_yaw", "hip_abad", "hip_pitch", "knee", "ankle"]
ARM_PATTERNS = ["shoulder_pitch", "shoulder_abad", "shoulder_yaw", "elbow"]

##
# Configuration
##

HUMANOID_FIXED_ARMS_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        link_density=1.0e-3,
        asset_path=f"{ISAACLAB_BRL_ROOT_DIR}/resources/mit_humanoid/urdf/humanoid_fixed_arms_sf.urdf",
        # asset_path=f"{ISAACLAB_BRL_ROOT_DIR}/resources/mit_humanoid/urdf/humanoid_fixed_arms_sf_400g.urdf",
        # asset_path=f"{ISAACLAB_BRL_ROOT_DIR}/resources/mit_humanoid/urdf/humanoid_full_sf.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.1, #! Maybe it should be 0, check: https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_transfer_policy.html#joint-order
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0, #! Maybe it should be 64
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            '.*_hip_yaw': 0.,
            '.*_hip_abad': 0.,
            '.*_hip_pitch': -0.2,
            '.*_knee': 0.6,
            '.*_ankle': 0.,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    soft_joint_vel_limit_factor=0.9,
    soft_joint_torque_limit_factor=0.8,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[f".*_{p}" for p in LEG_PATTERNS],
            effort_limit={f".*_{p}": getattr(effort_limit, p) for p in LEG_PATTERNS},
            velocity_limit={f".*_{p}": getattr(velocity_limit, p) for p in LEG_PATTERNS},
            stiffness={f".*_{p}": getattr(stiffness, p) for p in LEG_PATTERNS},
            damping={f".*_{p}": getattr(damping, p) for p in LEG_PATTERNS},
            armature={f".*_{p}": getattr(armature, p) for p in LEG_PATTERNS},
            friction={f".*_{p}": getattr(friction, p) for p in LEG_PATTERNS},
            apply_humanoid_jacobian=True, # True, False
        ),
    },
)

HUMANOID_FULL_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        link_density=1.0e-3,
        asset_path=f"{ISAACLAB_BRL_ROOT_DIR}/resources/mit_humanoid/urdf/humanoid_full_sf.urdf",
        # asset_path=f"{ISAACLAB_BRL_ROOT_DIR}/resources/mit_humanoid/urdf/humanoid_full_sf_400g.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.1, #! Maybe it should be 0, check: https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_transfer_policy.html#joint-order
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0, #! Maybe it should be 64
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            '.*_hip_yaw': 0.,
            '.*_hip_abad': 0.,
            '.*_hip_pitch': -0.2,
            '.*_knee': 0.6,
            '.*_ankle': 0.,
            '.*_shoulder_pitch': 0.,
            '.*_shoulder_abad': 0.,
            '.*_shoulder_yaw': 0.,
            '.*_elbow': 0.,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    soft_joint_vel_limit_factor=0.9,
    soft_joint_torque_limit_factor=0.8,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[f".*_{p}" for p in LEG_PATTERNS],
            effort_limit={f".*_{p}": getattr(effort_limit, p) for p in LEG_PATTERNS},
            velocity_limit={f".*_{p}": getattr(velocity_limit, p) for p in LEG_PATTERNS},
            stiffness={f".*_{p}": getattr(stiffness, p) for p in LEG_PATTERNS},
            damping={f".*_{p}": getattr(damping, p) for p in LEG_PATTERNS},
            armature={f".*_{p}": getattr(armature, p) for p in LEG_PATTERNS},
            friction={f".*_{p}": getattr(friction, p) for p in LEG_PATTERNS},
            apply_humanoid_jacobian=True, # True, False
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[f".*_{p}" for p in ARM_PATTERNS],
            effort_limit={f".*_{p}": getattr(effort_limit, p) for p in ARM_PATTERNS},
            velocity_limit={f".*_{p}": getattr(velocity_limit, p) for p in ARM_PATTERNS},
            stiffness={f".*_{p}": getattr(stiffness, p) for p in ARM_PATTERNS},
            damping={f".*_{p}": getattr(damping, p) for p in ARM_PATTERNS},
            armature={f".*_{p}": getattr(armature, p) for p in ARM_PATTERNS},
            friction={f".*_{p}": getattr(friction, p) for p in ARM_PATTERNS},
        ),
    },
)
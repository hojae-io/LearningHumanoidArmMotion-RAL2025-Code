
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import TorqueActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from extensions import ISAACLAB_BRL_ROOT_DIR

##
# Configuration
##

PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        link_density=1.0e-3,
        asset_path=f"{ISAACLAB_BRL_ROOT_DIR}/resources/pendulum/urdf/pendulum.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0, 
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        joint_pos={
            'actuator': 0.,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    soft_joint_vel_limit_factor=1.0,
    soft_joint_torque_limit_factor=1.0,
    actuators={
        "legs": TorqueActuatorCfg(
            joint_names_expr=["actuator"],
            effort_limit={"actuator": 0.5},
            velocity_limit={"actuator": 20.0},
            stiffness={"actuator": 10.0},
            damping={"actuator": 0.5},
        ),
    },
)

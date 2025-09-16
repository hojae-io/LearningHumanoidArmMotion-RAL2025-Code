""" This file contains the humanoid parameters for the robot.
    - effort_limit: The maximum torque that can be applied to the joint.
    - velocity_limit: The maximum velocity that can be applied to the joint.
    - stiffness: The stiffness of the joint PD controller.
    - damping: The damping of the joint PD controller.
    - rotor_inertias: The inertia of the rotor.
    - gear_ratio: The gear ratio of the joint.
    - armature: The armature of the joint.
    - friction: The friction of the joint.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class effort_limit:
  hip_yaw         = 34.0
  hip_abad        = 34.0
  hip_pitch       = 72.0
  knee            = 144.0
  ankle           = 68.0
  shoulder_pitch  = 34.0
  shoulder_abad   = 34.0
  shoulder_yaw    = 34.0
  elbow           = 55.0

@dataclass(frozen=True)
class velocity_limit:
  hip_yaw         = 48.0
  hip_abad        = 48.0
  hip_pitch       = 40.0
  knee            = 20.0
  ankle           = 24.0
  shoulder_pitch  = 50.0
  shoulder_abad   = 50.0
  shoulder_yaw    = 50.0
  elbow           = 50.0

@dataclass(frozen=True)
class stiffness:
  hip_yaw         = 30.0
  hip_abad        = 30.0
  hip_pitch       = 30.0
  knee            = 30.0
  ankle           = 30.0
  shoulder_pitch  = 60.0
  shoulder_abad   = 60.0
  shoulder_yaw    = 60.0
  elbow           = 60.0

@dataclass(frozen=True)
class damping:
  hip_yaw         = 1.0
  hip_abad        = 1.0
  hip_pitch       = 1.0
  knee            = 1.0
  ankle           = 1.0
  shoulder_pitch  = 2.0
  shoulder_abad   = 2.0
  shoulder_yaw    = 2.0
  elbow           = 2.0

@dataclass(frozen=True)
class rotor_inertias:
  hip_yaw         = 1.6841e-4
  hip_abad        = 1.6841e-4
  hip_pitch       = 5.548e-4
  knee            = 5.548e-4
  ankle           = 1.6841e-4
  shoulder_pitch  = 1.6841e-4
  shoulder_abad   = 1.6841e-4
  shoulder_yaw    = 1.6841e-4
  elbow           = 1.6841e-4

@dataclass(frozen=True)
class gear_ratio:
  hip_yaw         = 6.0
  hip_abad        = 6.0
  hip_pitch       = 6.0
  knee            = 12.0
  ankle           = 12.0
  shoulder_pitch  = 6.0
  shoulder_abad   = 6.0
  shoulder_yaw    = 6.0
  elbow           = 9.0

@dataclass(frozen=True)
class armature:
  hip_yaw         = rotor_inertias.hip_yaw * gear_ratio.hip_yaw ** 2
  hip_abad        = rotor_inertias.hip_abad * gear_ratio.hip_abad ** 2
  hip_pitch       = rotor_inertias.hip_pitch * gear_ratio.hip_pitch ** 2
  knee            = rotor_inertias.knee * gear_ratio.knee ** 2
  ankle           = rotor_inertias.ankle * gear_ratio.ankle ** 2
  shoulder_pitch  = rotor_inertias.shoulder_pitch * gear_ratio.shoulder_pitch ** 2
  shoulder_abad   = rotor_inertias.shoulder_abad * gear_ratio.shoulder_abad ** 2
  shoulder_yaw    = rotor_inertias.shoulder_yaw * gear_ratio.shoulder_yaw ** 2
  elbow           = rotor_inertias.elbow * gear_ratio.elbow ** 2

# @dataclass(frozen=True)
# class armature:
#   hip_yaw         = 0.01188
#   hip_abad        = 0.01188
#   hip_pitch       = 0.01980
#   knee            = 0.07920
#   ankle           = 0.04752
#   shoulder_pitch  = 0.01188
#   shoulder_abad   = 0.01188
#   shoulder_yaw    = 0.01188
#   elbow           = 0.0304

@dataclass(frozen=True)
class friction:
  hip_yaw         = 0.0
  hip_abad        = 0.0
  hip_pitch       = 0.0
  knee            = 0.2
  ankle           = 0.1
  shoulder_pitch  = 0.0
  shoulder_abad   = 0.0
  shoulder_yaw    = 0.0
  elbow           = 0.1



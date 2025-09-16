from functools import singledispatch
import torch
import numpy as np

@singledispatch
def skew_symmetric(p):
    raise TypeError("Unsupported type! Only torch.Tensor and np.ndarray are supported.")

@skew_symmetric.register
def _(p: torch.Tensor) -> torch.Tensor:
    assert p.shape[-1] == 3, "Last dimension of input must be 3"

    # Ensure p has shape (num_envs, 3)
    if p.ndim == 1:
        p = p.unsqueeze(0)  # Convert (3,) -> (1, 3)

    zero = torch.zeros_like(p[..., 0])  # (num_envs,)
    px, py, pz = p[..., 0], p[..., 1], p[..., 2]  # Extract components

    # Construct batch of skew-symmetric matrices
    skew_matrices = torch.stack([
        torch.stack([zero, -pz, py], dim=-1),
        torch.stack([pz, zero, -px], dim=-1),
        torch.stack([-py, px, zero], dim=-1)
    ], dim=-2)  # Final shape (num_envs, 3, 3)

    return skew_matrices

@skew_symmetric.register
def _(p: np.ndarray) -> np.ndarray:
    assert p.shape[-1] == 3, "Last dimension of input must be 3"

    # Ensure p has shape (num_envs, 3)
    if p.ndim == 1:
        p = np.expand_dims(p, axis=0)  # Convert (3,) -> (1, 3)

    zero = np.zeros_like(p[..., 0])  # (num_envs,)
    px, py, pz = p[..., 0], p[..., 1], p[..., 2]  # Extract components

    # Construct batch of skew-symmetric matrices
    skew_matrices = np.stack([
        np.stack([zero, -pz, py], axis=-1),
        np.stack([pz, zero, -px], axis=-1),
        np.stack([-py, px, zero], axis=-1)
    ], axis=-2)  # Final shape (num_envs, 3, 3)

    return skew_matrices

@torch.jit.script
def convert_gen_coord_from_isaaclab_to_pin(gen_coord: torch.Tensor) -> torch.Tensor:
    """ Convert the generalized coordinates from IsaacLab to Pinocchio convention
        gen_coord: [N, 17] or [N, 25]
        if gen_coord has 17 elements, it is fixed-arm humanoid
            IsaacLab:  [x, y, z, qw, qx, qy, qz, a01, a06, a02, a07, a03, a08, a04, a09, a05, a10]
            Pinocchio: [x, y, z, qx, qy, qz, qw, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10]
        if gen_coord has 25 elements, it is full humanoid
            IsaacLab:  [x, y, z, qw, qx, qy, qz, a01, a06, a11, a15, a02, a07, a12, a16, a03, a08, a13, a17, a04, a09, a14, a18, a05, a10]
            Pinocchio: [x, y, z, qx, qy, qz, qw, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18]
            
    """
    if gen_coord.shape[-1] == 17:
        isaaclab_to_pin = [0, 1, 2, 4, 5, 6, 3, 7, 9, 11, 13, 15, 8, 10, 12, 14, 16]
    elif gen_coord.shape[-1] == 25:
        isaaclab_to_pin = [0, 1, 2, 4, 5, 6, 3, 7, 11, 15, 19, 23, 8, 12, 16, 20, 24, 9, 13, 17, 21, 10, 14, 18, 22]
    else:
        raise ValueError("Input tensor must have 17 or 25 elements")
    return gen_coord[:, isaaclab_to_pin]
    
@torch.jit.script
def convert_gen_coord_from_pin_to_isaaclab(gen_coord: torch.Tensor) -> torch.Tensor:
    """ Convert the generalized coordinates from Pinocchio to IsaacLab convention
        gen_coord: [N, 17] or [N, 25]
        if gen_coord has 17 elements, it is fixed-arm humanoid
            IsaacLab:  [x, y, z, qw, qx, qy, qz, a01, a06, a02, a07, a03, a08, a04, a09, a05, a10]
            Pinocchio: [x, y, z, qx, qy, qz, qw, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10]
        if gen_coord has 25 elements, it is full humanoid
            IsaacLab:  [x, y, z, qw, qx, qy, qz, a01, a06, a11, a15, a02, a07, a12, a16, a03, a08, a13, a17, a04, a09, a14, a18, a05, a10]
            Pinocchio: [x, y, z, qx, qy, qz, qw, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18]
    """
    if gen_coord.shape[-1] == 17:
        pin_to_isaaclab = [0, 1, 2, 6, 3, 4, 5, 7, 12, 8, 13, 9, 14, 10, 15, 11, 16]
    elif gen_coord.shape[-1] == 25:
        pin_to_isaaclab = [0, 1, 2, 6, 3, 4, 5, 7, 12, 17, 21, 8, 13, 18, 22, 9, 14, 19, 23, 10, 15, 20, 24, 11, 16]
    else:
        raise ValueError("Input tensor must have 17 or 25 elements")
    return gen_coord[:, pin_to_isaaclab]
    
@torch.jit.script
def convert_gen_vel_from_isaaclab_to_pin(gen_vel: torch.Tensor) -> torch.Tensor:
    """ Convert the generalized velocities from IsaacLab to Pinocchio convention
        gen_vel: [N, 16] or [N, 24]
        if gen_vel has 16 elements, it is fixed-arm humanoid
            IsaacLab:  [vx, vy, vz, wx, wy, wz, a01, a06, a02, a07, a03, a08, a04, a09, a05, a10]
            Pinocchio: [vx, vy, vz, wx, wy, wz, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10]
        if gen_vel has 24 elements, it is full humanoid
            IsaacLab:  [vx, vy, vz, wx, wy, wz, a01, a06, a11, a15, a02, a07, a12, a16, a03, a08, a13, a17, a04, a09, a14, a18, a05, a10]
            Pinocchio: [vx, vy, vz, wx, wy, wz, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18]
    """
    if gen_vel.shape[-1] == 16:
        isaaclab_to_pin = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 7, 9, 11, 13, 15]
    elif gen_vel.shape[-1] == 24:
        isaaclab_to_pin = [0, 1, 2, 3, 4, 5, 6, 10, 14, 18, 22, 7, 11, 15, 19, 23, 8, 12, 16, 20, 9, 13, 17, 21]
    else:
        raise ValueError("Input tensor must have 16 or 24 elements")
    return gen_vel[:, isaaclab_to_pin]

@torch.jit.script
def convert_gen_force_from_pin_to_isaaclab(gen_force: torch.Tensor) -> torch.Tensor:
    """ Convert the generalized forces from Pinocchio to IsaacLab convention
        gen_force: [N, 16] or [N, 24]
        if gen_force has 16 elements, it is fixed-arm humanoid
            Pinocchio: [fx, fy, fz, tau_x, tau_y, tau_z, tau_a01, tau_a02, tau_a03, tau_a04, tau_a05, tau_a06, tau_a07, tau_a08, tau_a09, tau_a10]
            Isaaclab:  [fx, fy, fz, tau_x, tau_y, tau_z, tau_a01, tau_a06, tau_a02, tau_a07, tau_a03, tau_a08, tau_a04, tau_a09, tau_a05, tau_a10]
        if gen_force has 24 elements, it is full humanoid
            Pinocchio: [fx, fy, fz, tau_x, tau_y, tau_z, tau_a01, tau_a02, tau_a03, tau_a04, tau_a05, tau_a06, tau_a07, tau_a08, tau_a09, tau_a10, tau_a11, tau_a12, tau_a13, tau_a14, tau_a15, tau_a16, tau_a17, tau_a18]
            Isaaclab:  [fx, fy, fz, tau_x, tau_y, tau_z, tau_a01, tau_a06, tau_a11, tau_a15, tau_a02, tau_a07, tau_a12, tau_a16, tau_a03, tau_a08, tau_a13, tau_a17, tau_a04, tau_a09, tau_a14, tau_a18, tau_a05, tau_a10]
    """
    if gen_force.shape[-1] == 16:
        pin_to_isaaclab = [0, 1, 2, 3, 4, 5, 6, 11, 7, 12, 8, 13, 9, 14, 10, 15]
    elif gen_force.shape[-1] == 24:
        pin_to_isaaclab = [0, 1, 2, 3, 4, 5, 6, 11, 16, 20, 7, 12, 17, 21, 8, 13, 18, 22, 9, 14, 19, 23, 10, 15]
    else:
        raise ValueError("Input tensor must have 16 or 24 elements") 
    return gen_force[:, pin_to_isaaclab]

@torch.jit.script
def convert_jacobian_from_pin_to_isaaclab(jacobian: torch.Tensor) -> torch.Tensor:
    """ Convert the Jacobian from Pinocchio to IsaacLab convention
        Jacobian: [N, 6, 16] or [N, 6, 24]
        if Jacobian has 16 elements, it is fixed-arm humanoid
            Pinocchio q_dot: [dx, dy, dz, wx, wy, wz, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10]
            Isaaclab q_dot:  [dx, dy, dz, wx, wy, wz, a01, a06, a02, a07, a03, a08, a04, a09, a05, a10]
        if Jacobian has 24 elements, it is full humanoid
            Pinocchio q_dot: [dx, dy, dz, wx, wy, wz, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18]
            Isaaclab q_dot:  [dx, dy, dz, wx, wy, wz, a01, a06, a11, a15, a02, a07, a12, a16, a03, a08, a13, a17, a04, a09, a14, a18, a05, a10]
    """
    if jacobian.shape[-1] == 16:
        pin_to_isaaclab = [0, 1, 2, 3, 4, 5, 6, 11, 7, 12, 8, 13, 9, 14, 10, 15]
    elif jacobian.shape[-1] == 24:
        pin_to_isaaclab = [0, 1, 2, 3, 4, 5, 6, 11, 16, 20, 7, 12, 17, 21, 8, 13, 18, 22, 9, 14, 19, 23, 10, 15]
    else:
        raise ValueError("Input tensor must have 16 or 24 elements")
    return jacobian[:, pin_to_isaaclab]

@torch.jit.script
def convert_mass_matrix_from_pin_to_isaaclab(mass_matrix: torch.Tensor) -> torch.Tensor:
    """ Convert the mass matrix from Pinocchio to IsaacLab convention
        mass_matrix: [N, 16, 16] or [N, 24, 24]
        if mass_matrix has 16 elements, it is fixed-arm humanoid
            Pinocchio: [dx, dy, dz, wx, wy, wz, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10]
            Isaaclab:  [dx, dy, dz, wx, wy, wz, a01, a06, a02, a07, a03, a08, a04, a09, a05, a10]
        if mass_matrix has 24 elements, it is full humanoid
            Pinocchio: [dx, dy, dz, wx, wy, wz, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18]
            Isaaclab:  [dx, dy, dz, wx, wy, wz, a01, a06, a11, a15, a02, a07, a12, a16, a03, a08, a13, a17, a04, a09, a14, a18, a05, a10]
    """
    if mass_matrix.shape[-1] == 16:
        pin_to_isaaclab = [0, 1, 2, 3, 4, 5, 6, 11, 7, 12, 8, 13, 9, 14, 10, 15]
    elif mass_matrix.shape[-1] == 24:
        pin_to_isaaclab = [0, 1, 2, 3, 4, 5, 6, 11, 16, 20, 7, 12, 17, 21, 8, 13, 18, 22, 9, 14, 19, 23, 10, 15]
    else:
        raise ValueError("Input tensor must have 16 or 24 elements")
    return mass_matrix[pin_to_isaaclab][:, pin_to_isaaclab]

@torch.jit.script
def convert_joint_order_from_isaaclab_to_pin(joint_order: torch.Tensor) -> torch.Tensor:
    """ Convert the joint order from IsaacLab to Pinocchio convention
        joint_order: [N, 10] or [N, 18]
        if joint_order has 10 elements, it is fixed-arm humanoid
            IsaacLab:  [a01, a06, a02, a07, a03, a08, a04, a09, a05, a10]
            Pinocchio: [a01, a02, a03, a04, a05, a06, a07, a08, a09, a10]
        if joint_order has 18 elements, it is full humanoid
            IsaacLab:  [a01, a06, a11, a15, a02, a07, a12, a16, a03, a08, a13, a17, a04, a09, a14, a18, a05, a10]
            Pinocchio: [a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18]
    """
    if joint_order.shape[-1] == 10:
        isaaclab_to_pin = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    elif joint_order.shape[-1] == 18:
        isaaclab_to_pin = [0, 4, 8, 12, 16, 1, 5, 9, 13, 17, 2, 6, 10, 14, 3, 7, 11, 15]
    else:
        raise ValueError("Input tensor must have 10 or 18 elements")
    return joint_order[:, isaaclab_to_pin]

def adjoint_matrix_twist(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """ Compute the adjoint matrix of a rotation matrix R and a position vector p
        For both Isaaclab and Pinocchio, the twist is [v, w] (instead of [w, v])
        Adjoint matrix: [R, skew(p) * R; 0, R] 
        R: [N, 3, 3]
        p: [N, 3]
        return: [N, 6, 6]
    """
    assert R.shape[-2:] == (3, 3), "Rotation matrix must have shape (3, 3)"
    assert p.shape[-1] == 3, "Position vector must have shape (3,)"

    # Ensure R has shape (num_envs, 3, 3)
    if R.ndim == 2:
        R = R.unsqueeze(0)
    # Ensure p has shape (num_envs, 3)
    if p.ndim == 1:
        p = p.unsqueeze(0)  # Convert (3,) -> (1, 3)

    # Compute skew-symmetric matrix of p
    skew_p = skew_symmetric(p)  # (N, 3, 3)

    # Construct adjoint matrix
    adjoint = torch.cat([torch.cat([R, torch.matmul(skew_p, R)], dim=-1),
                         torch.cat([torch.zeros_like(R), R], dim=-1)], dim=-2)  # (N, 6, 6)

    return adjoint

def adjoint_matrix_wrench(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """ Compute the adjoint matrix of a rotation matrix R and a position vector p
        For both Isaaclab and Pinocchio, the wrench is [F, tau] (instead of [tau, F])
        Adjoint matrix: [R, 0; skew(p) * R, R]
        R: [N, 3, 3]
        p: [N, 3]
        return: [N, 6, 6]
    """
    assert R.shape[-2:] == (3, 3), "Rotation matrix must have shape (3, 3)"
    assert p.shape[-1] == 3, "Position vector must have shape (3,)"

    # Ensure R has shape (num_envs, 3, 3)
    if R.ndim == 2:
        R = R.unsqueeze(0)
    # Ensure p has shape (num_envs, 3)
    if p.ndim == 1:
        p = p.unsqueeze(0)  # Convert (3,) -> (1, 3)

    # Compute skew-symmetric matrix of p
    skew_p = skew_symmetric(p)  # (N, 3, 3)

    # Construct adjoint matrix
    adjoint = torch.cat([torch.cat([R, torch.zeros_like(R)], dim=-1),
                         torch.cat([torch.matmul(skew_p, R), R], dim=-1)], dim=-2)  # (N, 6, 6)

    return adjoint


if __name__ == "__main__":
    p_torch_single = torch.tensor([1.0, 2.0, 3.0])  # Shape (3,)
    p_torch_batch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2, 3)

    p_numpy_single = np.array([1.0, 2.0, 3.0])  # Shape (3,)
    p_numpy_batch = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2, 3)

    print(skew_symmetric(p_torch_single))  # Output: (1, 3, 3)
    print(skew_symmetric(p_torch_batch))   # Output: (2, 3, 3)

    print(skew_symmetric(p_numpy_single))  # Output: (1, 3, 3)
    print(skew_symmetric(p_numpy_batch))   # Output: (2, 3, 3)

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatingArray: TypeAlias = npt.NDArray[np.float32 | np.float64]


def mrp_to_quaternion(mrp: FloatingArray) -> FloatingArray:
    """
    Convert Modified Rodrigues Parameters (MRP) to unit quaternions [w, x, y, z].

    MRP is defined as a 3-dimensional vector [p1, p2, p3], and the conversion
    formula follows the standard MRP to quaternion transformation.

    Args:
        mrp: Input array with shape [n, 3] (batch) or [3,] (single sample),
             where 3 corresponds to p1, p2, p3 of MRP.
             Supported dtypes: np.float32, np.float64.

    Returns:
        FloatingArray: Unit quaternion array with shape [n, 4] (batch) or [4,],
                       ordered as [w, x, y, z] (w: real part, x/y/z: imaginary parts).
                       dtype is the same as input mrp.

    Raises:
        ValueError: If input array's last dimension is not 3 (invalid MRP dimension).
        RuntimeError: If the norm of quaternion is zero (extreme floating point error).

    Examples:
        >>> mrp = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        >>> quat = mrp_to_quaternion(mrp)
        >>> print(quat.shape)
        (4,)
        >>> print(np.linalg.norm(quat))  # Unit norm
        1.0

        >>> mrp_batch = np.array([[0.1,0.2,0.3], [0.4,0.5,0.6]], dtype=np.float32)
        >>> quat_batch = mrp_to_quaternion(mrp_batch)
        >>> print(quat_batch.shape)
        (2, 4)
    """
    if mrp.shape[-1] != 3:
        raise ValueError(
            f"Invalid MRP shape: last dimension must be 3, got {mrp.shape[-1]}. "
            f"Input shape: {mrp.shape}"
        )

    # Calculate squared norm of each MRP vector ||p||² (keep last dimension for broadcasting)
    p_sq_norm = np.sum(mrp**2, axis=-1, keepdims=True)
    # Calculate denominator 1 + ||p||² (avoid division by zero, theoretically safe for MRP)
    denominator = 1.0 + p_sq_norm

    # Compute quaternion components (broadcasting compatible with 1D/2D input)
    q_w = (1.0 - p_sq_norm) / denominator  # Real part
    q_xyz = 2 * mrp / denominator  # Imaginary parts [x, y, z]

    # Fix: Use np.concatenate (NumPy official API) instead of np.concat
    quaternion = np.concatenate([q_w, q_xyz], axis=-1)

    # Calculate quaternion norm (axis=-1 for 1D/2D compatibility)
    quat_norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    # Check for zero norm to avoid division by zero
    if np.any(quat_norm < 1e-15):
        raise RuntimeError(
            "Quaternion norm is zero (extreme floating point error). "
            "Please check the input MRP array."
        )
    # Normalize to unit quaternion (compensate floating point errors)
    quaternion = quaternion / quat_norm

    return quaternion

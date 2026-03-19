import math
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch
from numpy import linalg as la

UNIT_VECTOR_Z = [0, 0, 1.]


def attitude_init(position_BP_N: torch.Tensor) -> torch.Tensor:
    position_PB_N = -position_BP_N
    position_PB_N_unit = torch.nn.functional.normalize(position_PB_N, dim=-1)
    implied_sensor_direction = torch.tensor([0, 0, 1.])

    cos_angle = torch.einsum(
        '...i, i -> ...',
        position_PB_N_unit,
        implied_sensor_direction,
    ).unsqueeze(-1)
    angle = torch.acos(cos_angle)

    axis = implied_sensor_direction.unsqueeze(0).cross(
        position_PB_N_unit, dim=-1
    )
    axis = torch.nn.functional.normalize(axis, dim=-1)

    attitude_BN = axis * torch.tan(angle / 4.)
    return attitude_BN


def is_utc_datetime_str(input_str: str, formats: list[str] = None) -> bool:
    default_formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S+00:00",
        "%Y-%m-%d %H:%M:%S+00:00",
    ]
    target_formats = formats if formats else default_formats

    for fmt in target_formats:
        try:
            dt = datetime.strptime(input_str, fmt)
            return True
        except ValueError:

            continue
    return False


def convert_to_utc(datetime_str: str) -> str:
    naive_dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
    utc_dt = naive_dt

    utc_str_with_z = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    utc_str_with_offset = utc_dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    return utc_str_with_z


class ClassicElements(TypedDict):
    a: float  # Semi-major axis (m)
    e: float  # Eccentricity
    i: float  # Inclination (rad)
    Omega: float  # Right Ascension of Ascending Node (rad)
    omega: float  # Argument of Periapsis (rad)
    f: float  # True Anomaly (rad)


def rv2elem(mu: float, rVec: np.ndarray, vVec: np.ndarray) -> ClassicElements:
    """
    Converts inertial Cartesian position and velocity vectors into classical orbital elements.
    Handles circular and equatorial signularities consistent with standard astrodynamics conventions.

    :param mu: Gravitational parameter (m^3/s^2)
    :param rVec: Position vector (m)
    :param vVec: Velocity vector (m/s)
    :return: Dictionary containing classical orbital elements
    """
    # Angular momentum
    hVec = np.cross(rVec, vVec)
    h = la.norm(hVec)

    # Node vector
    kn = np.array([0.0, 0.0, 1.0])
    nVec = np.cross(kn, hVec)
    n = la.norm(nVec)

    # Position and velocity magnitudes
    r = la.norm(rVec)
    v = la.norm(vVec)

    # Eccentricity vector
    # e = (v^2/mu - 1/r)r - (r.v/mu)v
    eVec = (1.0 / mu) * ((v**2 - mu / r) * rVec - np.dot(rVec, vVec) * vVec)
    e = la.norm(eVec)

    # Specific mechanical energy
    energy = v**2 / 2 - mu / r

    # Semi-major axis
    # For parabolic orbits (energy ~ 0), a is theoretically infinity.
    # orbitalMotion.py returns -rp for parabolic, but we stick to 'a' here or inf.
    if abs(energy) > 1e-10:
        a = -mu / (2 * energy)
    else:
        a = float('inf')

    # Inclination (0 <= i <= pi)
    i = math.acos(np.clip(hVec[2] / h, -1.0, 1.0))

    # Define thresholds for singularities
    EPS_E = 1e-11
    EPS_I = 1e-11

    # Right Ascension of Ascending Node (Omega)
    # If equatorial (i ~ 0), Omega is undefined (conventionally 0)
    if n > EPS_I:
        Omega = math.atan2(nVec[1], nVec[0])
    else:
        Omega = 0.0

    # Determine reference vector for Argument of Periapsis (omega)
    # If inclined, measure from Node (nVec). If equatorial, measure from inertial X-axis.
    if n > EPS_I:
        node_ref = nVec
    else:
        node_ref = np.array([1.0, 0.0, 0.0])

    # Argument of Periapsis (omega) and True Anomaly (f)
    if e > EPS_E:
        # Eccentric case:
        # omega is angle from node_ref to eVec
        # f is angle from eVec to rVec

        # Calculate omega
        # dot(cross(node_ref, eVec), hVec) gives signed magnitude projected on h
        sin_omega = np.dot(np.cross(node_ref, eVec), hVec) / h
        cos_omega = np.dot(node_ref, eVec)
        omega = math.atan2(sin_omega, cos_omega)

        # Calculate f
        sin_f = np.dot(np.cross(eVec, rVec), hVec) / h
        cos_f = np.dot(eVec, rVec)
        f = math.atan2(sin_f, cos_f)

    else:
        # Circular case (e ~ 0):
        # perigee is undefined, so omega = 0.
        # f becomes Argument of Latitude (if inclined) or True Longitude (if equatorial).
        # f is angle from node_ref to rVec.
        omega = 0.0

        sin_f = np.dot(np.cross(node_ref, rVec), hVec) / h
        cos_f = np.dot(node_ref, rVec)
        f = math.atan2(sin_f, cos_f)

    # Normalize angles to [0, 2pi)
    Omega = (Omega + 2 * np.pi) % (2 * np.pi)
    omega = (omega + 2 * np.pi) % (2 * np.pi)
    f = (f + 2 * np.pi) % (2 * np.pi)

    return {
        "a": float(a),
        "e": float(e),
        "i": float(i),
        "Omega": float(Omega),
        "omega": (omega),
        "f": f,
    }

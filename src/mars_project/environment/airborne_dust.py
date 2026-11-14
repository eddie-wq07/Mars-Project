"""Mass-carrying dust parcels with slip-corrected Stokes drag.

A parcel represents many 1–10 um grains, not a centimetre-sized grain. The
prescribed gas is dilute, isothermal and stationary (or a supplied uniform wind).
Fine-fraction release from mechanically detached MPM material is a closure, not
an experimentally calibrated entrainment law. No random lift or timed deletion.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Atmosphere:
    pressure: float = 610.0
    temperature: float = 210.0
    viscosity: float = 1.06e-5
    molar_mass: float = 0.04401
    grain_density: float = 2500.0
    gravity: float = -3.71
    wind: tuple = (0.0, 0.0, 0.0)

    @property
    def density(self):
        return self.pressure * self.molar_mass / (8.314462618 * self.temperature)

    @property
    def mean_free_path(self):
        return self.viscosity / self.pressure * np.sqrt(np.pi * 8.314462618 * self.temperature / (2 * self.molar_mass))

    def relaxation_time(self, radius):
        radius = np.asarray(radius)
        if np.any(radius <= 0):
            raise ValueError("Dust grain radii must be positive")
        kn = self.mean_free_path / radius
        slip = 1 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))
        return 2 * self.grain_density * radius**2 * slip / (9 * self.viscosity)


def advance_dust(position, velocity, radius, dt, atmosphere):
    """Exact constant-wind solution of dv/dt=g+(wind-v)/tau, including buoyancy.

    Valid for low particle Reynolds number. Reject use beyond that regime rather
    than silently treating this aerosol law as a high-speed sand/supersonic model.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Dust timestep must be finite and positive")
    wind = np.asarray(atmosphere.wind)
    re = 2 * np.asarray(radius) * atmosphere.density * np.linalg.norm(velocity - wind, axis=-1) / atmosphere.viscosity
    if np.any(re > 1):
        raise ValueError("Dust parcel Reynolds number exceeds the Stokes model range")
    tau = atmosphere.relaxation_time(radius)[..., None]
    g = np.array([0.0, 0.0, atmosphere.gravity * (1 - atmosphere.density / atmosphere.grain_density)])
    terminal = wind + tau * g
    decay = np.exp(-dt / tau)
    travel = -tau * np.expm1(-dt / tau)
    return position + terminal * dt + (velocity - terminal) * travel, terminal + (velocity - terminal) * decay


class DustParcels:
    """Conserved parcel mass, with persistent deposited mass and momentum ledgers."""

    def __init__(self, atmosphere=None, capacity=16384):
        self.atmosphere, self.capacity = atmosphere or Atmosphere(), capacity
        self.reset()

    def reset(self):
        self.position = np.empty((0, 3))
        self.velocity = np.empty((0, 3))
        self.radius = np.empty(0)
        self.mass = np.empty(0)
        self.source = np.empty(0, dtype=np.int64)
        self.emitted = self.deposited = 0.0
        self.drag_impulse = np.zeros(3)
        self.gravity_impulse = np.zeros(3)
        self.deposit_impulse = np.zeros(3)

    def emit(self, position, velocity, mass, source):
        # Four logarithmic size bins. Mass fractions are explicit assumptions;
        # particle number follows m/(rho*4*pi*r^3/3), never the rendered footprint.
        radii = np.array([1.0, 2.0, 5.0, 10.0]) * 1e-6
        weights = np.array([0.1, 0.4, 0.35, 0.15])
        n = min(len(position), (self.capacity - len(self.mass)) // len(radii))
        if not n:
            return np.empty(0, dtype=np.int64)
        self.position = np.concatenate((self.position, np.repeat(position[:n], 4, axis=0)))
        self.velocity = np.concatenate((self.velocity, np.repeat(velocity[:n], 4, axis=0)))
        self.mass = np.concatenate((self.mass, (mass[:n, None] * weights).ravel()))
        self.radius = np.concatenate((self.radius, np.tile(radii, n)))
        self.source = np.concatenate((self.source, np.repeat(source[:n], 4)))
        self.emitted += float(mass[:n].sum())
        return np.asarray(source[:n])

    def step(self, dt, surface_height):
        if not len(self.mass):
            return np.empty((0, 3)), np.empty(0), np.empty((0, 3))
        old_v = self.velocity.copy()
        self.position, self.velocity = advance_dust(self.position, self.velocity, self.radius, dt, self.atmosphere)
        gravity_impulse = np.array(
            [
                0.0,
                0.0,
                self.mass.sum()
                * self.atmosphere.gravity
                * (1 - self.atmosphere.density / self.atmosphere.grain_density)
                * dt,
            ]
        )
        self.gravity_impulse += gravity_impulse
        self.drag_impulse += np.sum(self.mass[:, None] * (self.velocity - old_v), axis=0) - gravity_impulse
        landed = self.position[:, 2] <= surface_height(self.position[:, :2]) + self.radius
        positions, masses = self.position[landed].copy(), self.mass[landed].copy()
        momentum = masses[:, None] * self.velocity[landed]
        self.deposited += float(masses.sum())
        self.deposit_impulse += momentum.sum(0)
        for name in ("position", "velocity", "radius", "mass", "source"):
            setattr(self, name, getattr(self, name)[~landed])
        return positions, masses, momentum

    def extinction_area(self):
        # Geometric-optics Q_ext=2 approximation; no arbitrary opacity multiplier.
        # Full wavelength-dependent Mie scattering remains outside this closure.
        return 3 * self.mass / (2 * self.atmosphere.grain_density * self.radius)

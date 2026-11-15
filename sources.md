# Environment physics: sources and assumptions

Based on `b6ecf1a` (`mars-4`); shallow-regolith and robot atmospheric-drag updates 2026-11-14.

- Assumed dry, cohesionless sand; retain `mars-4` appearance with slight, persistent deformation and no bouncing.
- Values are **current settings** unless marked **planned**; this is not a calibrated universal Mars soil.

## Gravity

| Environment | Applied gravity `(x, y, z)`, m/s² | Basis |
| --- | --- | --- |
| Earth | `(0, 0, -9.81)` | Rounded standard gravity, 9.80665 m/s². [Standard reference](https://pmc.ncbi.nlm.nih.gov/articles/PMC5287000/) |
| Mars | `(0, 0, -3.71)` | Rounded ESA value, 3.711 m/s²; about 37.8% of the chosen Earth gravity. [ESA](https://www.esa.int/articles/meet-mars) |

- Uniform gravity, set in [scene_runner.py](src/mars_project/scene_runner.py).

## Regolith: explicit sand assumptions

[NASA's geotechnical review, Table 1, page 3](https://ntrs.nasa.gov/api/citations/20200003046/downloads/20200003046.pdf)
lists Pathfinder/Sojourner **1290–1520 kg/m³ density, 30–40° friction angle, and 0–420 Pa cohesion**: engineering estimates, not direct measurements.

| Property | Current sand-case value | Basis and meaning |
| --- | --- | --- |
| Bulk density | **1400 kg/m³** | Assumed within the NASA range; includes pore space. |
| Internal friction | **`friction = 0.68`** | Newton example's internal-shear parameter; conversion to NASA's 30–40° friction angle remains unverified. |
| Compression stiffness | **`E = 10 MPa`; `ν = 0.3`** | Newton assumptions, not NASA measurements. Elastic bulk modulus `K = E/[3(1−2ν)] ≈ 8.33 MPa`. |
| Compression yield pressure | **`yield_pressure = 10¹² Pa`** | Numerical cap far above foot pressures; not measured bearing strength. |
| Cohesion | **Assumed zero** | NASA range's noncohesive endpoint; `yield_stress = 0 Pa`, `tensile_yield_ratio = 0`. |

- Elastic stiffness does not prescribe footprint depth or permanent compaction.
- Explicitly pinned defaults: hardening/dilatancy `0`, damping `0 s`, viscosity `0 Pa·s`, hardening/softening rates `1`.
- Provenance: [Newton sand example](https://github.com/newton-physics/newton/blob/main/newton/examples/assets/mpm_sand.usda), [local configuration](src/mars_project/environment/mpm_soil.py), installed `MPMParticleMaterialCfg` defaults.

### What the current simulation actually does

- Newton MPM solves **3D** motion with two-way MuJoCo robot coupling; rendered mesh moves **only vertically**.
- Surface height follows **material volume / column area**, assuming constant bulk density; horizontal redistribution leaves ruts/piles.
- Vertical jitter cannot raise this surface; bulk volume fades at particle heights **3–7 cm above the prior surface**, without velocity-sign switching.
- This depth-averaged approximation omits elastic volume changes and overhangs; it does not replace physical MPM contact forces.
- Domain **8.5 × 4 × 0.04 m**; **4 cm grid**, **2 cm point spacing**, **170,000 samples**, approximately **1904 kg**.
- **4 cm loose sand over firm ground** is our shallow-layer assumption, not a NASA measurement or target footprint depth.
- One solver cell through the layer is a coarse contact approximation, not resolved depth-dependent compaction.
- Evaluation corridor extension (2026-11-14): +X end moved from 5 m to 7 m for the 5 m goal and scanner margin; material law, layer depth and grid spacing are unchanged.
- Activity: **0.55 m** around feet plus disturbances; sleep threshold **0.01 m/s for 0.4 s**, **3-cell confining halo**.
- Solver: **100 iterations**, tolerance **10⁻⁵**, `P0` strain basis; `air_drag = 1` is numerical regularization, not atmospheric drag.
- Implementation: [soil_columns.py](src/mars_project/environment/soil_columns.py), [soil_runtime.py](src/mars_project/environment/soil_runtime.py), [soil_surface.py](src/mars_project/environment/soil_surface.py).

## Atmospheric resistance

| Quantity | Earth reference | Mars reference |
| --- | --- | --- |
| Pressure | **101325 Pa** | **610 Pa** |
| Temperature | **288.15 K** | **210 K** |
| Gas molar mass | **0.02897 kg/mol**, air | **0.04401 kg/mol**, pure-CO₂ approximation |
| Calculated density, `ρ = pM/(RT)` | **≈1.225 kg/m³** | **≈0.0154 kg/m³** |
| Dynamic viscosity used for dust | **1.81 × 10⁻⁵ Pa·s** | **1.06 × 10⁻⁵ Pa·s** |
| Wind | **0 m/s** | **0 m/s** |

- Scenario references: [NASA](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) gives Mars **400–870 Pa**, **0.016 ± 0.006 kg/m³**, average **≈214 K**.
- **Implemented robot drag:** `F = −½ρCdA|v_rel|v_rel`, where `v_rel = v_link − wind`. [NASA equation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/)
- Each of H1's **20 links** uses its visual-geometry bounding box, projected along airflow: `A = |nx|dy·dz + |ny|dx·dz + |nz|dx·dy`, with unit flow direction `n` in the link frame.
- Forces act at each link's centre of mass through the simulator's additive external-force API. The unchanged PPO still outputs **19 joint actions**.
- Assume **`Cd = 1`**, identical across environments; this is not a measured H1 coefficient.
- Equal speed/area/coefficient gives **≈98.75% less Mars drag** with these densities.
- Simple approximation: uniform still air; no lift, wake shielding, spin torque or aerodynamic buoyancy. Bounding boxes overestimate some silhouettes; link overlap is not subtracted.
- Forces refresh every **5 ms** with the current H1 settings and remain applied through solver substeps (**20 ms** if a backend owns four-step decimation; logged at startup). No CFD is solved.
- `--air-drag auto` follows gravity; `earth`, `mars`, or `off` override **robot drag only** for controlled checks. Existing dust atmosphere continues to follow gravity.
- **Motor damping unchanged:** `--joint-resistance-scale 0`; the disabled heuristic uses Mars `0.020 kg/m³`, adding **0** damping.
- Dust assumptions: radii **1/2/5/10 µm**, mass weights **10/40/35/15%**, grain density **2500 kg/m³**, release fraction **10⁻⁶**.
- Dust uses slip-corrected Stokes drag; release, viscosity, grain distribution and appearance are assumptions.
- Implementation: [aerodynamics.py](src/mars_project/environment/aerodynamics.py), [air_drag.py](src/mars_project/environment/air_drag.py), [airborne_dust.py](src/mars_project/environment/airborne_dust.py). Startup logs record density, coefficient, wind and every link's dimensions.

## Experiment scope

- Fixed PPO checkpoint and motor gains; **100 trials per condition**, **5 m goal**, **0.5 m tolerance**, **30 s simulation timeout**.
- Report success without falling, failures and arrival time; compare using the same rigid-body backend.
- Target fidelity: plausible support, slip and shallow residual deformation; no full granular or atmospheric simulation required.

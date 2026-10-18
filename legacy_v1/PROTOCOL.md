# Simulation protocol v0.1 — adaptive water and energy facade

## Status and claim boundary

This is a reproducible starting specification, not a validated high-fidelity simulation. No field experiment by this team is required or assumed. Published independent experimental datasets will be used where available. Numerical convergence establishes discretization reliability; it does not establish accuracy of material properties, atmospheric forcing or the physical model. No annual performance or superiority claim is made in phase 1.

The delivered Python primitives are verification aids. They do not constitute a coupled thermal solver, droplet CFD, a rain/drainage model, or an annual PV simulation. They must not be substituted for those models when claiming high fidelity.

## Reference geometry and states

SI units; x is across the facade, y is upward, z is outward. The panel centre is the geometric origin. The facade plane is z=-0.70 m. Each face has a 1.00 m x 1.00 m active rectangle. The conceptual panel thickness is 0.06 m, not a mass or thermal resistance specification.

The PV and dew faces share a mechanically switched frame. The sky-facing active normal is n=(0, cos(beta), sin(beta)), beta measured from horizontal. Flipping selects the exposed face; it is not a change in the underlying semiconductor. Both faces occupy one installation footprint over time. Their manufacturing area and cost remain separate. Frame occlusion, hinges, drainage, cable routing and wind loads remain mechanical design work. OBJ files are visualization meshes, not watertight CFD meshes or certified fabrication drawings.

Stage-1 visual settings (45° PV, 25° dew, 65° rain) are demonstration angles, not optimization results. A full building, actuator kinematics and structural loading have not been solved.

## Minimum forcing contract

Each weather interval needs a timestamp with timezone and interval semantics. Required quantities: dry-bulb air temperature, RH or dew point, station pressure, local wind speed/direction and measurement height, direct and diffuse irradiance, downwelling longwave irradiance or a documented estimate, precipitation amount/rate, and cloud cover. Wall and ground temperatures or their thermal models are required. Record source, station/grid elevation, geographical coordinates, quality flags and missingness.

Separate site elevation above sea level from mounting height above ground. A standard-atmosphere pressure estimate is a fallback, not a substitute for measured pressure or a local wind/temperature profile. No universal temperature lapse rate is imposed across building floors. Local flow, roughness, sky obstruction and atmospheric stability matter.

Hourly forcing does not become observed 1-second weather by interpolation. Use internal substeps to integrate state dynamics while preserving the original temporal resolution and uncertainty of the forcing. Keep rain intermittency and missing longwave forcing as explicit uncertainty sources.

## Coupled thermal and water model to implement in phase 2

Use at least two thermal nodes to retain the heat stored in the PV face and dew face when changing modes. Node temperatures must not reset on switching. Refine to through-thickness and in-plane conduction where the Biot number, assembly details or local gradients invalidate a lumped approximation.

For each surface i, with positive flux into the surface:

`C_i dT_i/dt = alpha_i G_i - P_PV_i/A_i + q_conv_i + q_LW_i + q_back_i + L_v j_i - q_cool_i`.

All terms are W/m² except C_i (J/m²/K) and dT/dt. Condensation j_i>0 releases latent heat. Evaporation j_i<0 consumes heat and must be limited by the available film. Include rain sensible heat and, when needed, droplet evaporation. The auxiliary electricity used for cooling is q_cool/COP, with temperature-dependent COP and fan/pump loads; it is not equal to q_cool. Active cooling is disabled in the initial reference design.

Longwave exchange must include sky, ground and facade, with consistent view factors summing to one. A gray diffuse approximation is an initial reference. Spectral/angular treatment is required when material claims depend on selective 8–13 micrometre emission or coatings. A passive radiator can reach below ambient only through a net heat balance. Cloud cover alone is not a condensation controller.

Mass-transfer driving force is the vapor-pressure or concentration difference between local ambient air and the saturated interface. A positive dew-point margin is necessary for condensation but does not determine its yield. Transfer coefficients need resolved flow or a correlation justified for orientation, length, Reynolds number and mixed/natural convection. Track film mass, drainage to the reservoir and re-evaporation separately. Exclude freezing/frost initially and flag out-of-domain conditions; do not silently count ice as liquid delivery.

For the rain reference, use local liquid-water volume fraction times the inward droplet velocity dotted with surface normal. In undisturbed monodisperse flow, `Q_L_h = R_mm_h A max(0,-v dot n)/abs(v_y)`. This only verifies geometry and units. Near a facade, replace local droplet velocities and interception efficiency with a justified size distribution and Lagrangian/Eulerian rain model. Include splashing, obstruction, overlapping catchment footprints and runoff/drainage efficiency. Do not multiply total facade area by horizontal rainfall to claim capture.

PV requires irradiance on the active surface, incidence-angle loss, shading, cell temperature and a datasheet-derived electrical model. Phase-1 PVWatts-style arithmetic is only a dimensional/limiting-case aid. Use pvlib with documented module parameters and versioned dependencies, and avoid double-counting an externally estimated temperature if solving a coupled thermal model.

Battery energy balance: `delta E_bat = eta_charge E_charge - E_discharge/eta_discharge`. Enforce capacity/SOC, charge/discharge powers, reserve and self-discharge. Report motor, sensor, controller, fan, pump and storage losses, curtailed power, exports, imports and final SOC. Keep water delivered, electricity generation and electricity consumption separate before defining an economic objective.

## High-fidelity work package

1. Verify psychrometrics and heat/mass-flux sign conventions.
2. Establish transient two-node reference solver; test dry thermal response against analytic limits.
3. Resolve flow and transfer around the actual geometry. Choose an OpenFOAM solver/model combination deliberately; a generic heat-transfer solver does not automatically implement humid-air condensation or rain collection.
4. Compute view factors and shading from the reference CAD; test reciprocity and closure. Refine the optical/radiative model against the actual material spectrum.
5. Couple transfer coefficients, droplets and drainage to the thermal/mass model. Track coupling residuals and conservation.
6. Build a parameter sweep and a validated surrogate for annual/control studies. Running a detailed CFD solution for every annual control step is neither necessary nor automatically more accurate.

## Verification and validation gates (proposed targets, not achieved)

- Geometry: active area, normal orientation, units, facade clearance and catchment overlap are checked. Manufacturing feasibility is a separate gate.
- Conservation: interval-integrated mass and energy residuals below 1% of the dominant throughput over the interval; use absolute tolerances when throughput is near zero. Include storage changes.
- Spatial convergence: at least three systematically refined meshes; report an observed-order or Grid Convergence Index analysis and inspect local quantities, not only totals.
- Temporal convergence: at least three time steps; target changes below 2% in cumulative water/energy and below 0.2 K in surface temperature. Refine or report sensitivity when these targets fail.
- Validation: reproduce an independent published setup, its forcing, orientation, material, collection mechanism and measurement uncertainty. Report temperature MAE/RMSE and water/PV bias and RMSE. Proposed goals of temperature MAE <=1 K and water bias <=20% are provisional and may be stricter or weaker than the available reference uncertainty. Do not manufacture reference data from an aggregate annual figure.
- Calibration and validation datasets must be separate. Track digitized figure uncertainty if raw data are unavailable. A literature comparison without matching boundary conditions is a plausibility check, not validation.
- Uncertainty: propagate measured/assumed ranges for emissivity, solar absorptance, heat capacity, thermal resistance, mass transfer, sky longwave input and runoff efficiency. Report quantiles, scenarios and sensitivity. Sample size is chosen from convergence of uncertainty estimates, not a fixed decorative number.

## Fair comparisons and control

Use common installation footprint, weather, horizon, initial/final storage conditions and accounting rules. Compare a practical fixed PV panel with rain gutter; a fixed dual-face or dual-function design; a simple rule controller; and a forecast-based controller. A best-in-hindsight angle/mode schedule is an oracle upper bound only. If separate devices use more area, state the extra area and cost.

Use chronologically separated training/validation/test periods and, where possible, hold out a climate/site. Reanalysis is useful as realized forcing, not as a perfect future forecast for a deployable controller. Model forecast errors explicitly. Define water demand, energy reserve and economic weights before selecting a policy. Include switching energy, cooldown time, minimum dwell time, wear limits and uncertainty-aware decision thresholds.

## Primary references and tool documentation

Accessed 2026-10-16. References guide the model; their reported performance is not the performance of this design.

1. ASHRAE Handbook, Fundamentals, Psychrometrics, SI (2021): https://handbook.ashrae.org/Handbooks/F21/SI/f21_ch01/F21_Ch01_si.aspx
2. PsychroLib documentation and source (ASHRAE-based formulations): https://psychrometrics.github.io/psychrolib/api_docs.html and https://github.com/psychrometrics/psychrolib
3. EnergyPlus, Outside Surface Heat Balance: https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/3.5-outside-surface-heat-balance.html
4. pvlib, ModelChain: https://pvlib-python.readthedocs.io/en/stable/reference/modelchain.html
5. OpenFOAM, standard solvers: https://www.openfoam.com/documentation/user-guide/a-reference/a.1-standard-solvers
6. Haechler et al. (2021), Exploiting radiative cooling for uninterrupted 24-hour water harvesting from the atmosphere, Science Advances: https://pmc.ncbi.nlm.nih.gov/articles/PMC8221617/
7. Li et al. (2021), Nighttime Radiative Cooling for Water Harvesting from Solar Panels, ACS Photonics: https://web.stanford.edu/~llfan/projects/acsphotonics_lingling.pdf
8. Samzadeh et al. (2021), Feasibility of Vertical Rainwater Harvesting via In-situ Measurement of Wind-driven Rain Loads on Building Facades in a Tropical Climate: https://ijbes.utm.my/index.php/ijbes/article/view/736
9. Ritter et al. (2019), Dew frequency across the US from a network of in situ radiometers: https://hess.copernicus.org/articles/23/1179/2019/hess-23-1179-2019.html

## Current unresolved inputs

Reference location/climate; real or typical year; facade orientation and surrounding geometry; actual PV module; available dew coating and spectral data; thermal assembly and heat capacities; rain/runoff characterization; local weather and longwave data; intended water demand; acceptable cost and energy constraints. None has been silently assigned a field-validated value.

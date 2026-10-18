"""Verified primitives for phase 1. Not a coupled or field-validated simulator.

Units are SI unless stated in a function name. Saturation-pressure equations
are the ASHRAE (2017) Hyland-Wexler coefficient forms used by PsychroLib.
Equations/coefficient provenance: PROTOCOL.md references 1 and 2.
No downloaded experimental or weather data are embedded in this module.
"""
from __future__ import annotations
import math

SIGMA = 5.670374419e-8
R_VAPOR = 461.52


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("All scalar inputs must be finite")


def saturation_pressure_pa(temp_C: float) -> float:
    """Saturation over ice below the triple point, over water above it.

    Valid equation domain -100 to 200 C; frost is not liquid-water delivery.
    Enhancement factor/nonideal moist-air corrections are not included.
    """
    _finite(temp_C)
    if not -100 <= temp_C <= 200:
        raise ValueError("Temperature outside -100..200 C equation domain")
    t = temp_C + 273.15
    if temp_C <= 0.01:
        log_p = (-5674.5359/t + 6.3925247 - 0.009677843*t
                 + 6.2215701e-7*t*t + 2.0747825e-9*t**3
                 - 9.484024e-13*t**4 + 4.1635019*math.log(t))
    else:
        log_p = (-5800.2206/t + 1.3914993 - 0.048640239*t
                 + 4.1764768e-5*t*t - 1.4452093e-8*t**3
                 + 6.5459673*math.log(t))
    return math.exp(log_p)


def dew_point_C(temp_air_C: float, rh_fraction: float) -> float:
    """Invert saturation pressure; below the triple point returns frost point."""
    _finite(temp_air_C, rh_fraction)
    if not 0 < rh_fraction <= 1:
        raise ValueError("RH must satisfy 0 < RH <= 1; supply a fraction")
    target = rh_fraction * saturation_pressure_pa(temp_air_C)
    lo, hi = -100.0, temp_air_C
    if target < saturation_pressure_pa(lo):
        raise ValueError("Dew/frost point below supported domain")
    for _ in range(100):
        mid = (lo+hi)/2
        if saturation_pressure_pa(mid) > target:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2


def standard_pressure_pa(site_elevation_m: float) -> float:
    """Standard-atmosphere pressure estimate, not measured local pressure."""
    _finite(site_elevation_m)
    if not -500 <= site_elevation_m <= 10000:
        raise ValueError("Supported reference elevation is -500..10000 m")
    return 101325.0 * (1 - 2.25577e-5*site_elevation_m)**5.25588


def active_normal(tilt_deg: float) -> tuple[float, float, float]:
    _finite(tilt_deg)
    if not 0 <= tilt_deg <= 90:
        raise ValueError("Tilt is 0..90 degrees from horizontal")
    b = math.radians(tilt_deg)
    return 0.0, math.cos(b), math.sin(b)


def ideal_rain_capture_L_h(area_m2: float, horizontal_rain_mm_h: float,
                           drop_velocity_m_s: tuple[float,float,float],
                           normal: tuple[float,float,float]) -> float:
    """Undisturbed monodisperse flux reference, excluding CFD and losses."""
    _finite(area_m2, horizontal_rain_mm_h, *drop_velocity_m_s, *normal)
    if area_m2 <= 0 or horizontal_rain_mm_h < 0:
        raise ValueError("Area must be positive and rainfall nonnegative")
    if abs(sum(v*v for v in normal)-1) > 1e-8:
        raise ValueError("Normal must be a unit vector")
    fall = -drop_velocity_m_s[1]
    if fall <= 0:
        raise ValueError("Droplet y velocity must be downward")
    inward = max(0.0, -sum(a*b for a,b in zip(drop_velocity_m_s, normal)))
    return horizontal_rain_mm_h * area_m2 * inward/fall


def net_longwave_loss_W_m2(surface_C: float, emissivity: float,
                           longwave_down_W_m2: float,
                           ground_C: float, wall_C: float,
                           view_sky: float, view_ground: float,
                           view_wall: float) -> float:
    """Gray/diffuse reference; positive means energy leaves the surface."""
    _finite(surface_C, emissivity, longwave_down_W_m2, ground_C, wall_C,
            view_sky, view_ground, view_wall)
    fs = (view_sky, view_ground, view_wall)
    if not 0 <= emissivity <= 1 or min(fs) < 0 or abs(sum(fs)-1) > 1e-8:
        raise ValueError("Invalid emissivity or view-factor closure")
    if min(surface_C,ground_C,wall_C) <= -273.15 or longwave_down_W_m2 < 0:
        raise ValueError("Invalid thermal boundary condition")
    incoming = (view_sky*longwave_down_W_m2
                + view_ground*SIGMA*(ground_C+273.15)**4
                + view_wall*SIGMA*(wall_C+273.15)**4)
    return emissivity*(SIGMA*(surface_C+273.15)**4-incoming)


def condensation_reference_kg_m2_s(air_C: float, rh_fraction: float,
                                    surface_C: float, hm_m_s: float) -> float:
    """Dilute-vapor, film-temperature reference, positive condensation only.

    No rain film, evaporation, drainage or latent-heat solver is implemented.
    hm is an external input, not a calibrated facade transfer coefficient.
    """
    _finite(air_C,rh_fraction,surface_C,hm_m_s)
    if not 0 <= rh_fraction <= 1 or hm_m_s < 0:
        raise ValueError("Invalid RH or mass transfer coefficient")
    if surface_C < 0:
        raise ValueError("Freezing excluded from liquid-condensation reference")
    delta_p = rh_fraction*saturation_pressure_pa(air_C)-saturation_pressure_pa(surface_C)
    film_K = 273.15+(air_C+surface_C)/2
    return hm_m_s*max(0.0,delta_p)/(R_VAPOR*film_K)


def pv_reference_W(area_m2: float, poa_W_m2: float, eta_25: float,
                   gamma_per_K: float, cell_C: float) -> float:
    """PVWatts-style arithmetic aid; module parameters must be supplied."""
    _finite(area_m2,poa_W_m2,eta_25,gamma_per_K,cell_C)
    if area_m2 <= 0 or poa_W_m2 < 0 or not 0 <= eta_25 <= 1:
        raise ValueError("Invalid PV area, irradiance or reference efficiency")
    efficiency = min(1.0,max(0.0,eta_25*(1+gamma_per_K*(cell_C-25))))
    return area_m2*poa_W_m2*efficiency

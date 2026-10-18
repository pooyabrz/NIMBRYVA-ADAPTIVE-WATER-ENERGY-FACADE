"""Local, source-backed weather and explicitly assumed facade orientation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from geometry import solar_vector, ray_category, panel_frame, view_factors

ROOT = Path(__file__).resolve().parents[1]


def model_config():
    c=json.loads((ROOT/'config/model.json').read_text(encoding='utf-8'))
    from assembly import validate_config
    validate_config(c)
    return c


def cities():
    return json.loads((ROOT/'config/cities.json').read_text(encoding='utf-8'))


def load_weather(city_id='rasht', year=2024, config=None, longwave_shift_hours=0):
    import pvlib
    c = next(x for x in cities() if x['id']==city_id)
    cfg = config or model_config()
    p = ROOT/'data/normalized'/f'{city_id}-{year}.csv.gz'
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    metadata = json.loads(p.with_name(f'{city_id}-{year}.metadata.json').read_text())
    if metadata['status']!='passed':
        raise ValueError('Weather QC failed; do not silently use missing forcing')
    # Required +/-1 h sensitivity; edges become unsupported, never zero-filled.
    if longwave_shift_hours:
        df['lw_down_W_m2'] = df.lw_down_W_m2.shift(longwave_shift_hours)
    if cfg.get('facade_orientation')=='equator_facing_reference':
        facade_az = 180. if c['lat']>=0 else 0.
    else:
        facade_az = cfg['facade_azimuth_north_cw_deg']
    midpoints = df.index-pd.Timedelta(minutes=30)
    sun = pvlib.solarposition.get_solarposition(midpoints, c['lat'], c['lon'], altitude=metadata['era5_elevation_m'])
    vec = solar_vector(sun.apparent_zenith.to_numpy(), sun.azimuth.to_numpy(), facade_az)
    df['sun_x'],df['sun_y'],df['sun_z'] = vec.T
    # Apply only mounting-height corrections, preserving site pressure already
    # in ERA5. No sea-level pressure reset or city-elevation temperature shift.
    if cfg.get('model_version')=='cassette-v2.0':
        from core import saturation_pressure_pa,dew_point_C
        from flow import mixing_ratio
        height=cfg['mounting_height_m']
        df['reference_air_C']=df.air_C;df['reference_pressure_Pa']=df.pressure_Pa
        df['reference_wind10_m_s']=df.wind_m_s
        e=np.array([saturation_pressure_pa(t)*r/100 for t,r in zip(df.air_C,df.rh_pct)])
        w=.621945*e/(df.pressure_Pa.to_numpy()-e)
        df['air_C']=df.air_C+cfg['local_lapse_K_m']*(height-2)
        df['pressure_Pa']=df.pressure_Pa*np.exp(-9.80665*height/(287.05*(df.air_C+273.15)))
        e=w*df.pressure_Pa.to_numpy()/(.621945+w)
        sat=np.array([saturation_pressure_pa(t) for t in df.air_C])
        df['rh_pct']=100*e/sat
        df['mount_supersaturation_unsupported']=(df.rh_pct>100).astype(int)
        df['dew_C']=[dew_point_C(t,min(1,max(1e-9,r/100))) for t,r in zip(df.air_C,df.rh_pct)]
        df['wind_m_s']=df.wind_m_s*(height/10)**cfg['terrain_alpha']
    delta = np.radians(df.wind_from_deg.to_numpy()-facade_az)
    df['wind_x'] = -df.wind_m_s*np.sin(delta)
    df['wind_z'] = -df.wind_m_s*np.cos(delta)
    for i,tilt in enumerate(cfg['tilts_deg']):
        normal = panel_frame(tilt)[2]
        visible = ray_category(vec,cfg['mounting_height_m'],cfg['wall_standoff_m'],cfg['building_height_m'],cfg['building_width_m'])==0
        cosine = np.maximum(0,vec@normal)
        fs,fg,_ = view_factors(tilt,cfg['mounting_height_m'],cfg['wall_standoff_m'],cfg['building_height_m'],cfg['building_width_m'])
        direct = df.dni_W_m2.to_numpy()*cosine*visible*(vec[:,1]>0)
        # Isotropic diffuse sky and ground-reflection reference, no spectral optics.
        df[f'poa_{i}'] = direct + df.dhi_W_m2*fs + cfg['ground_shortwave_albedo']*df.ghi_W_m2*fg
    df.attrs = {'city':c,'year':year,'site_elevation_m':metadata['era5_elevation_m'],
                'facade_azimuth_deg':facade_az,'physics_status':cfg['status']}
    return df

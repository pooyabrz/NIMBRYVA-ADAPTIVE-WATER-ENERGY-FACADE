"""Local, source-backed weather and explicitly assumed facade orientation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from geometry import solar_vector, ray_category, panel_frame, view_factors

ROOT = Path(__file__).resolve().parents[1]


def model_config():
    return json.loads((ROOT/'config/model.json').read_text(encoding='utf-8'))


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
    # 10 m reanalysis wind is used as-is. Building-height flow is NOT resolved.
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

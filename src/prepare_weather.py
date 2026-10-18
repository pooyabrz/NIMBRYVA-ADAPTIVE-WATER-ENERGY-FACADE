"""Normalize source units, verify coverage, and summarize the reference year."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def prepare(city: dict, year: int) -> tuple[dict,dict]:
    raw=ROOT/'data/raw'
    era=json.loads((raw/f"{city['id']}-era5-{year}.json").read_text())
    nasa=json.loads((raw/f"{city['id']}-power-longwave-{year}.json").read_text())
    h=era['hourly'];idx=pd.to_datetime(h['time'],utc=True)
    df=pd.DataFrame({k:v for k,v in h.items() if k!='time'},index=idx)
    mapping={'temperature_2m':'air_C','relative_humidity_2m':'rh_pct',
      'dew_point_2m':'dew_C','surface_pressure':'pressure_hPa','wind_speed_10m':'wind_m_s',
      'wind_direction_10m':'wind_from_deg','precipitation':'precip_mm_h','rain':'rain_mm_h',
      'snowfall':'snowfall_cm_h','cloud_cover':'cloud_pct','shortwave_radiation':'ghi_W_m2',
      'direct_normal_irradiance':'dni_W_m2','diffuse_radiation':'dhi_W_m2','is_day':'is_day'}
    df=df.rename(columns=mapping)
    df['pressure_Pa']=df.pop('pressure_hPa')*100
    unit=nasa['parameters']['ALLSKY_SFC_LW_DWN']['units']
    if unit not in ['Wh/m^2','W/m^2']: raise ValueError('Unexpected NASA longwave units '+unit)
    lw=nasa['properties']['parameter']['ALLSKY_SFC_LW_DWN']
    series=pd.Series(lw,dtype=float)
    series.index=pd.to_datetime(series.index,format='%Y%m%d%H',utc=True)
    # Hourly Wh/m² divided by a one-hour interval is numerically W/m².
    df['lw_down_W_m2']=series.mask(series<0).reindex(df.index)
    expected=pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq='h',inclusive='left',tz='UTC')
    errors=[]
    if not df.index.equals(expected): errors.append('Timestamp coverage/spacing differs from complete UTC year')
    missing=df.isna().sum().to_dict()
    if any(missing.values()):errors.append('Missing forcing; no automatic zero filling')
    domains={'rh_pct':(0,100),'wind_m_s':(0,100),'cloud_pct':(0,100),
      'rain_mm_h':(0,500),'precip_mm_h':(0,500),'pressure_Pa':(30000,110000),
      'ghi_W_m2':(0,1600),'dni_W_m2':(0,1700),'dhi_W_m2':(0,1600),'lw_down_W_m2':(0,750)}
    for k,(lo,hi) in domains.items():
        if ((df[k]<lo)|(df[k]>hi)).any():errors.append('Out-of-domain '+k)
    if (df.dew_C>df.air_C+.5).any():errors.append('Dew point exceeds air temperature by >0.5 K')
    if (df.rain_mm_h>df.precip_mm_h+.05).any():errors.append('Rain exceeds total precipitation')
    qc={'city':city['id'],'year':year,'status':'passed' if not errors else 'failed',
      'hours':len(df),'missing_by_column':missing,'errors':errors,
      'source':'ERA5 via Open-Meteo, hourly; NASA POWER hourly longwave',
      'era5_grid_lat_lon':[era['latitude'],era['longitude']],
      'era5_elevation_m':era['elevation'],'power_geometry':nasa.get('geometry'),
      'longwave_unit_original':unit,
      'alignment_warning':'ERA5 shortwave is preceding-hour average; meteorology is instantaneous. NASA UTC hour labels are associated directly as a working assumption; +/-1 h sensitivity required.',
      'resolution_warning':'Neither source resolves a facade microclimate. NASA longwave footprint differs from ERA5.',
      'climate_warning':f'{year} is one realized year, not a 30-year climate normal or a future forecast.'}
    night=df.is_day==0
    gap=(df.air_C-df.dew_C).clip(lower=0)
    summary={**city,'year':year,'hours':len(df),'elevation_source_m':era['elevation'],
      'air_mean_C':float(df.air_C.mean()),'air_p05_C':float(df.air_C.quantile(.05)),
      'air_p95_C':float(df.air_C.quantile(.95)),'rh_mean_pct':float(df.rh_pct.mean()),
      'rain_total_mm':float(df.rain_mm_h.sum()),'precip_total_mm':float(df.precip_mm_h.sum()),
      'ghi_annual_kWh_m2':float(df.ghi_W_m2.sum()/1000),
      'wind_mean_m_s':float(df.wind_m_s.mean()),'wind_p95_m_s':float(df.wind_m_s.quantile(.95)),
      'night_hours':int(night.sum()),'night_dew_gap_median_K':float(gap[night].median()),
      'night_gap_le_2K_pct':float(100*(gap[night]<=2).mean()),
      'subzero_air_hours':int((df.air_C<0).sum()),'qc_status':qc['status']}
    out=ROOT/'data/normalized';out.mkdir(exist_ok=True)
    target=out/f"{city['id']}-{year}.csv.gz"
    temporary=target.with_name(target.name+'.tmp')
    df.to_csv(temporary,index_label='timestamp_UTC',compression='gzip')
    # Re-read before an atomic replacement; interrupted writes never become a dataset.
    reread=pd.read_csv(temporary,compression='gzip')
    if len(reread)!=len(df):raise IOError('Incomplete normalized weather write')
    temporary.replace(target)
    (out/f"{city['id']}-{year}.metadata.json").write_text(json.dumps(qc,ensure_ascii=False,indent=2))
    return summary,qc


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--years',nargs='+',type=int,default=[2024]);a=p.parse_args()
    cities=json.loads((ROOT/'config/cities.json').read_text());summaries=[];quality=[]
    for c in cities:
        for y in a.years:
            s,q=prepare(c,y);summaries.append(s);quality.append(q)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    pd.DataFrame(summaries).to_csv(out/'city-climate-summary.csv',index=False)
    # Keep checks for previously prepared years when adding a new year.
    existing=out/'weather-quality.json'
    previous=json.loads(existing.read_text()) if existing.exists() else []
    replaced={(q['city'],q['year']) for q in quality}
    combined=[q for q in previous if (q['city'],q['year']) not in replaced]+quality
    existing.write_text(json.dumps(combined,ensure_ascii=False,indent=2))
    print(pd.DataFrame(summaries)[['name','hours','air_mean_C','rain_total_mm','ghi_annual_kWh_m2','elevation_source_m','qc_status']].round(2).to_string(index=False))
    raise SystemExit(1 if any(q['status']!='passed' for q in quality) else 0)

"""Download and preserve provenance for uniform city-year weather inputs.

ERA5 via Open-Meteo supplies meteorology and shortwave radiation. NASA POWER
supplies hourly all-sky downwelling longwave (CERES product). These have
different footprints. The default timestamp association is a documented
working alignment, not a demonstrated sub-hourly match. Sensitivity to +/-1 h
longwave shifts is required before a performance claim.
"""
from __future__ import annotations
import argparse, concurrent.futures, datetime as dt, hashlib, json, time
import urllib.request, urllib.parse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VARIABLES=['temperature_2m','relative_humidity_2m','dew_point_2m','surface_pressure',
 'wind_speed_10m','wind_direction_10m','precipitation','rain','snowfall',
 'cloud_cover','shortwave_radiation','direct_normal_irradiance','diffuse_radiation','is_day']


def get_json(url: str, path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'WaterFacadeResearchPrototype/0.2'})
            with urllib.request.urlopen(req,timeout=90) as response:
                raw=response.read()
            obj=json.loads(raw)
            if obj.get('error') or 'properties' not in obj and 'hourly' not in obj:
                raise RuntimeError(str(obj)[:300])
            path.write_bytes(raw)
            metadata={'url':url,'retrieved_at_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),
                      'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            path.with_suffix('.provenance.json').write_text(json.dumps(metadata,indent=2))
            return obj
        except Exception:
            if attempt==2: raise
            time.sleep(2*(attempt+1))


def download(city: dict, year: int) -> dict:
    out=ROOT/'data'/'raw';out.mkdir(parents=True,exist_ok=True)
    query={'latitude':city['lat'],'longitude':city['lon'],
           'start_date':f'{year}-01-01','end_date':f'{year}-12-31',
           'hourly':','.join(VARIABLES),'models':'era5','timezone':'UTC','wind_speed_unit':'ms'}
    era_path=out/f"{city['id']}-era5-{year}.json"
    era=get_json('https://archive-api.open-meteo.com/v1/archive?'+urllib.parse.urlencode(query),era_path)
    query={'parameters':'ALLSKY_SFC_LW_DWN','community':'RE','latitude':city['lat'],
           'longitude':city['lon'],'start':f'{year}0101','end':f'{year}1231',
           'format':'JSON','time-standard':'UTC'}
    nasa_path=out/f"{city['id']}-power-longwave-{year}.json"
    nasa=get_json('https://power.larc.nasa.gov/api/temporal/hourly/point?'+urllib.parse.urlencode(query),nasa_path)
    return {'city':city['id'],'year':year,'hours_era5':len(era['hourly']['time']),
            'hours_longwave':len(nasa['properties']['parameter']['ALLSKY_SFC_LW_DWN']),
            'era5_elevation_m':era['elevation'],'status':'downloaded'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--years',nargs='+',type=int,default=[2024])
    parser.add_argument('--cities',nargs='*')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    cities=json.loads((ROOT/'config/cities.json').read_text())
    if args.cities: cities=[c for c in cities if c['id'] in args.cities]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(download,c,y):(c['id'],y) for c in cities for y in args.years}
        for job in concurrent.futures.as_completed(jobs):
            try: row=job.result()
            except Exception as exc:
                cid,yr=jobs[job];row={'city':cid,'year':yr,'status':'failed','error':str(exc)}
            results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    results.sort(key=lambda r:(r['city'],r['year']))
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/download-log.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    raise SystemExit(1 if any(r['status']=='failed' for r in results) else 0)

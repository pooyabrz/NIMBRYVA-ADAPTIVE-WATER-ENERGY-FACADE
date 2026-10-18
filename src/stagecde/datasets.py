"""Protocol-locked multi-year inputs; no synthetic year relabelling."""
import argparse,concurrent.futures,hashlib,json
from assembly import ROOT
from weather import cities
from fetch_weather import download
from prepare_weather import prepare

def jobs():
    s=json.loads((ROOT/'config/research/protocol.json').read_text())['weather_split']
    return [(c,y,role) for role,years,cs in [('train',s['train_years'],s['train_cities']),('selection',s['selection_years'],s['train_cities']),
       ('test',s['test_years'],s['train_cities']+s['test_only_cities'])] for c in cs for y in years]

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);a=p.parse_args();out=ROOT/'results/research/E';out.mkdir(parents=True,exist_ok=True)
    citymap={c['id']:c for c in cities()};records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(download,citymap[c],y):(c,y,role) for c,y,role in jobs()}
        for f in concurrent.futures.as_completed(fs):
            c,y,role=fs[f]
            try:
                d=f.result();summary,q=prepare(citymap[c],y);path=ROOT/f'data/normalized/{c}-{y}.csv.gz'
                row={'city':c,'year':y,'role':role,'status':q['status'],'hours':q['hours'],'path':str(path.relative_to(ROOT)),
                   'normalized_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_source_warning':q['alignment_warning']}
            except Exception as e:row={'city':c,'year':y,'role':role,'status':'failed','error':str(e)}
            records.append(row);print(c,y,role,row['status'],flush=True)
    result={'protocol':json.loads((ROOT/'config/research/protocol.json').read_text())['weather_split'],'datasets':sorted(records,key=lambda r:(r['role'],r['city'],r['year'])),
      'test_year_used_for_tuning':False,'issued_forecasts_available':False,'independent_observational_weather':False,
      'weather_is_reanalysis_not_experimental_validation':True}
    (out/'weather-split.json').write_text(json.dumps(result,indent=2))
    if any(r['status']!='passed' for r in records):raise SystemExit(1)
if __name__=='__main__':main()

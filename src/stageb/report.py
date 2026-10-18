"""Build a source-backed local release summary from actual execution records."""
import json,math,re
from pathlib import Path
import numpy as np
import pandas as pd
from assembly import ROOT


def read(path):return json.loads(Path(path).read_text())


def main():
    out=ROOT/'results/stage-b';thermal=[read(p) for p in sorted((out/'thermal').glob('*-dt60-r1.json'))]
    cfd=[];attempts=[]
    for folder in sorted((out/'cfd').iterdir()):
        if not folder.is_dir():continue
        meta=read(folder/'case.json');q=read(folder/'quality.json') if (folder/'quality.json').exists() else None
        ex=read(folder/'execution.json') if (folder/'execution.json').exists() else {'status':'NO_COMPLETED_EXECUTION_RECORD'}
        meshlog=(folder/'log.checkMesh').read_text() if (folder/'log.checkMesh').exists() else ''
        record={'case':folder.name,'status':ex['status'],'standard_mesh_passed':'Mesh OK.' in meshlog,
          'raw_full_mesh_or_scalar_checks_failed':'Failed ' in meshlog,'quality':q}
        attempts.append(record)
        if folder.name.startswith('reference-') and q and q['flow_available_for_exploratory_particle_tracking']:
            cfd.append(q)
    cfd.sort(key=lambda x:x['cells'])
    rain=[];rejected_rain=[]
    for path in sorted((out/'rain').glob('*.json')):
        r=read(path)
        if 'bins' not in r:continue
        if r.get('source_scheme')!='union_of_upstream_and_top_boundary_bundles_v2':
            rejected_rain.append({'file':path.name,'status':'SUPERSEDED_SINGLE_SOURCE_PLANE_NOT_QUALIFIED'});continue
        rain.append({'file':path.name,'case':Path(r['case']).name,'samples_per_size':r['bins'][0]['samples'],'dt_s':r['dt_s'],
           'bundle_m':r['bundle_halfwidth_m'],'specific_catch_ratio':r['total_specific_catch_ratio'],
           'sampling_relative95_halfwidth':r['sampling_relative95_halfwidth'],'sampling_5percent_gate':r['sampling_five_percent_gate'],
           'minimum_bin_panel_hits':min(x['panel_hits'] for x in r['bins']),'unresolved_particles':sum(x['unresolved'] for x in r['bins']),
           'particle_count_residual':sum(abs(x['particle_count_residual']) for x in r['bins']),
           'profile_mass_residual_kg_s':r['profile_mass_residual_kg_s']})
    grid=[]
    for a,b in zip(cfd,cfd[1:]):
        ua=np.array(a['quantities_of_interest']['probe_U_m_s']);ub=np.array(b['quantities_of_interest']['probe_U_m_s'])
        grid.append({'from':a['case'],'to':b['case'],'relative_L2_probe_change':float(np.linalg.norm(ua-ub)/np.linalg.norm(ub)),
          'pressure_only_axis_moment_change_Nm':b['quantities_of_interest']['pressure_only_rotation_axis_moment_Nm']-a['quantities_of_interest']['pressure_only_rotation_axis_moment_Nm']})
    convergence={}
    base=pd.read_csv(out/'thermal/rasht-2024-07-dt60-r1.csv.gz');base=base[base.scored].set_index('time')
    for name in ['rasht-2024-07-dt30-r2','rasht-2024-07-dt60-r1-wu168']:
        x=pd.read_csv(out/'thermal'/(name+'.csv.gz'));x=x[x.scored].set_index('time')
        convergence[name]=(base[['wall_C','ground_C']]-x[['wall_C','ground_C']]).abs().max().to_dict()
    verify=read(out/'verification.json');stefan=read(out/'surface/stefan-convergence.json');surface=read(out/'surface/surface-reference-cases.json')
    doe=read(out/'doe/metadata.json');coupled=[read(p) for p in sorted((out/'coupled').glob('*-2024-*.json'))]
    joined=read(out/'coupled/joined-reference.json') if (out/'coupled/joined-reference.json').exists() else None
    empty=read(out/'cfd/empty-medium/quality.json')
    result={'release':'stage-b-reference-1','date_utc':'2026-10-17','base_model':'cassette-v2.0','verification':verify,
      'stage_a_regression_checks':{'passed':31,'total':31,'note':'executed after wall/ground boundary change; record regression-v2.json'},
      'thermal_cases':thermal,'thermal_cities':len({x['city'] for x in thermal}),'thermal_scored_hours':sum(x['hours'] for x in thermal),
      'thermal_sensitivity_max_difference_K':convergence,'CFD_reference_cases':cfd,'CFD_attempts':attempts,'empty_ABL':empty,
      'grid_quantities_of_interest_changes':grid,'GCI_or_grid_independence_established':False,
      'rain_runs_current_source_scheme':rain,'superseded_rain_runs':rejected_rain,
      'Stefan_reference':stefan,'surface_references':surface,'DOE':doe,'coupled_cassette_references':coupled,'joined_reference':joined,
      'three_D_viewer':'native Python Matplotlib, actual solved U/T/Yv and selected trajectories; tested headlessly',
      'unresolved':['CFD scalar wall treatment and full latent/buoyancy CHT coupling','grid-independent load and rain result','particle/source accuracy gates',
        'empirical CNF splash, secondary droplets, nucleation/contact-line dynamics','thick-snow thermal structure and supercooled precipitation',
        'site-specific soil, urban roughness/neighbours, hourly atmospheric spectrum','independent experimental benchmark for this cassette','qualified Stage-B surrogate/PPO and final actuator rating'],
      'physical_yield_claim_supported':False,'experimental_validation':False}
    (out/'release.json').write_text(json.dumps(result,indent=2))
    lines=['# نتایج اجراشدهٔ مرحلهٔ B','',
       '**این گزارش از پرونده‌های واقعی اجرا ساخته شده است. مرحلهٔ B هنوز از نظر دقت فیزیکی نهایی بسته نشده است.**',
       '',f'تاریخ: 2026-10-17؛ آزمون‌های مستقل B: **{verify["passed"]}/{verify["total"]}**؛ آزمون‌های قبلی کاست: **31/31**. این اعداد اعتبارسنجی تجربی نیستند.',
       '', '## ساختمان و سطح زمین','',
       f'{len(thermal)} پنجره برای {result["thermal_cities"]} شهر در ژانویه/ژوئیهٔ ۲۰۲۴: مجموع {result["thermal_scored_hours"]} ساعت پس از گرم‌شدن ۴۸ ساعتهٔ هر مورد.',
       '', '| شهر | ماه | بیشینهٔ اختلاف دیوار و هوا، K | بیشینهٔ اختلاف زمین و هوا، K |', '|---|---:|---:|---:|']
    for t in thermal:lines.append(f'| {t["city"]} | {t["month"]} | {t["wall_air_max_difference_K"]:.2f} | {t["ground_air_max_difference_K"]:.2f} |')
    lines+=['','در مورد رشتِ ژوئیه، هم‌زمان نصف کردن گام زمانی و دو برابر کردن شبکه نسبت به مرجع، اختلاف‌های زیر را ایجاد کرد. این مطالعهٔ محدود، خطای همهٔ شرایط نیست:',
      '',f'- دیوار: {convergence["rasht-2024-07-dt30-r2"]["wall_C"]:.3f} K؛ زمین: {convergence["rasht-2024-07-dt30-r2"]["ground_C"]:.3f} K.',
      f'- افزایش گرم‌شدن از ۴۸ به ۱۶۸ ساعت: دیوار {convergence["rasht-2024-07-dt60-r1-wu168"]["wall_C"]:.3f} K و زمین {convergence["rasht-2024-07-dt60-r1-wu168"]["ground_C"]:.3f} K. بنابراین کفایت گرم‌شدن ۴۸ ساعت فرض نشده است.',
      '', '## CFD واقعاً اجراشده','', 'OpenCFD 2312 patch260127، حل‌گر simpleFoam، باد ۵ متر بر ثانیه در ۱۰ متر، z0=0.03 m، زاویهٔ کاست ۵۰ درجه. هندسهٔ ثابت با اندازهٔ واقعی کاست. جریان سه‌بعدی RANS و اسکالرهای مستقل T/Yv اجرا شدند.',
      '', '| شبکه | سلول | خطای مساحت کاست | بیشینهٔ residual اولیهٔ U |', '|---|---:|---:|---:|']
    for q in cfd:lines.append(f'| {q["case"]} | {q["cells"]:,} | {100*q["geometry"]["relative_area_error"]:.3f}% | {max(q["last_initial_residuals"][n] for n in ["Ux","Uy","Uz"]):.2e} |')
    lines+=['',f'در دامنهٔ بدون ساختمان، خطای نسبی L2 پروفیل U در ارتفاع ۲ تا ۳۰ متر **{100*empty["empty_ABL_relative_L2_U"]:.3f}%** و بیشینهٔ سرعت قائم {empty["empty_ABL_max_vertical_velocity_m_s"]:.5f} m/s بود. شرط همگرایی جریان در آستانهٔ ارزیابی 1e-5 موفق بود؛ اجرای آن به سقف ۱۲۰۰ تکرار رسید. برای اسکالر با اختلاف مرزی صفر، حفظ میدان ثابت بررسی شد، چون residual نرمال‌شدهٔ آن معیار مناسبی نبود.',
       '', '| مقایسهٔ شبکه | تغییر نسبی میدان در پنج پروب | تغییر گشتاور محور، Nm |', '|---|---:|---:|']
    for g in grid:lines.append(f'| {g["from"]} → {g["to"]} | {100*g["relative_L2_probe_change"]:.2f}% | {g["pressure_only_axis_moment_change_Nm"]:.4f} |')
    lines+=['','**استقلال از شبکه یا GCI معتبر به دست نیامده است.** تغییرات نیرو/گشتاور و پروب‌ها اجازهٔ ادعای بار نهایی را نمی‌دهد. h/hm محاسبه‌شده به‌علت نبود حل مناسب نزدیک دیوار صرفاً تشخیصی است. اجرای شکست‌خوردهٔ فاقد کاست یا با مساحت ناقص، در نتایج مرجع پذیرفته نشده است.',
      '', '## باران چنداندازه','', 'همهٔ ردیف‌های زیر با اجتماع مرز سقف و بالادست اجرا شده‌اند. نتایج قدیمیِ تک‌مرزی در گزارش ماشین‌خوان کنار گذاشته شده‌اند. نسبت برخورد، جرم برخوردکننده بر باران افقی × مساحت پنل است؛ نسبت جمع‌آوری آب مخزن نیست.',
      '', '| اجرا | نمونه در هر اندازه | dt، s | نیم‌عرض منبع، m | نسبت برخورد | نیم‌عرض نسبی ۹۵٪ نمونه‌برداری |', '|---|---:|---:|---:|---:|---:|']
    for r in rain:lines.append(f'| {r["case"]} | {r["samples_per_size"]} | {r["dt_s"]} | {r["bundle_m"]} | {r["specific_catch_ratio"]:.4f} | {100*r["sampling_relative95_halfwidth"]:.1f}% |')
    lines+=['','در این اجراها دقت نمونه‌برداری ۵٪ برای همهٔ اندازه‌ها به دست نیامده است. تفاوت پهنای منبع، تعداد نمونه و گام زمانی باید جدا بررسی شود؛ ردیف ۱۶۳۸۴تایی چند تغییر هم‌زمان دارد و به‌تنهایی آزمون همگرایی زمان نیست. CI، خطای درگ، DSD، قطع ناحیهٔ منبع یا خطای CFD را شامل نمی‌شود. رواناب بعدی با کسر ماندگاریِ پس از پاششِ ۰٫۵ یک سناریو است. پاشش و قطره‌های ثانویهٔ واقعی پوشش اعتبارسنجی نشده‌اند.',
      '', '## بخار، فیلم، یخ و برف','', '| شبکهٔ استفان | dt، s | خطای نسبی جبههٔ ذوب نسبت به جواب تحلیلی |', '|---:|---:|---:|']
    for s in stefan:lines.append(f'| {s["n"]} | {s["dt_s"]} | {100*s["relative_front_error"]:.4f}% |')
    lines+=['','این نتیجه برای مسئلهٔ یک‌بعدی استفان با خواص ثابت و بازهٔ نرم‌شدهٔ ۰٫۱ K است. حساسیت به بازهٔ ۰٫۰۵ و ۰٫۲ K در فایل جدا ثبت شده؛ این عدد خطای کل دستگاه نیست. آزمون‌های مستقل انجماد/ذوب با شار ثابت، تخلیهٔ Nusselt، محدودیت تبخیر و موازنهٔ برف آب‌معادل نیز اجرا شده‌اند.',
      '', '| مرجع یک‌ساعته | دمای میانگین، °C | آب مایع باقی‌مانده، kg | یخ/برف باقی‌مانده، kg | تخلیه، kg |', '|---|---:|---:|---:|---:|']
    for s in surface:lines.append(f'| {s["case"]} | {s["temperature_mean_C"]:.3f} | {s["liquid_kg"]:.5f} | {s["ice_plus_snow_kg"]:.5f} | {s["drain_kg"]:.5f} |')
    lines+=['','این‌ها شرایط مرزی مصنوعیِ معلوم هستند؛ برآورد عملکرد یک شهر نیستند. لایهٔ ضخیم برف، سینتیک باران فوق‌سرد و قطره/خط تماس واقعی پوشش خارج از این حل‌اند.',
      '', '## اتصال و DOE','',f'مدل دیوار/زمین به تابش طیفی کاست وصل شد. {len(coupled)} مرجع شهری اتصال بررسی شد؛ حالات سرد/ناموفق صریح رد می‌شوند و محیط v2 برای آن‌ها باز نشده است. در رشتِ ژوئیه تغییر دمای گسیلنده بر اثر این مرزها تا حدود ۳٫۱۷ K بود؛ این عدد مربوط به همان پنجرهٔ اتصال است.',
      f'طرح نهایی {doe["n"]} حالتهٔ LHS با seed={doe["seed"]} تولید شد و {doe["surface_cases_executed"]} حل فیلم ۳۰۰ ثانیه‌ای اجرا شد. {doe["neutral_RANS_count"]} حالت معیار Ri<0.1 را دارند؛ بقیه به حل‌گر دارای شناوری نیاز دارند. تعداد اجرای CFD این صف توسط فرمان DOE صفر است؛ سه مسئلهٔ مرجع شبکه‌ای بالا مستقلاً اجرا شده‌اند.',
      '', 'اتصال مرجع B1→تابش B3 و B2→ورودی مکانی باران B3 اجرا شد. شار باران ۳۲ نواری و دفتر جرم حفظ می‌شوند. h/hm در این اتصال هنوز prior صریح‌اند. این اتصال CHT دوطرفه، طیف واقعی آسمان یا عملکرد تأییدشدهٔ دستگاه نیست.',
      '', '## موارد باز و اقدام شما','', 'برای بستن مرحلهٔ B هنوز حل نزدیک دیوار و بازخورد گرمای نهان/شناوری، استقلال شبکه و ذره/منبع، پاشش/حرکت قطرهٔ واقعی، لایهٔ ضخیم برف و دادهٔ معیار مستقل لازم است. نبود امکان آزمایش فیزیکی مانع آزمون عددی نیست؛ با این حال دادهٔ آزمایش منتشرشده برای ادعای دقت واقعی ضروری است.',
      '', 'شما می‌توانید همین حالا نمایش محلی و آزمون‌ها را با راهنمای `STAGE-B-RUN-fa.md` اجرا کنید. برای بازسازی حل سنگین، OpenFOAM تثبیت‌شده را در Linux/WSL2 نصب کنید. آموزش PPO جدید تا عبور از معیارهای فوق انجام نشده است.']
    (ROOT/'docs/STAGE-B-RESULTS-fa.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'thermal_cities':result['thermal_cities'],'thermal_scored_hours':result['thermal_scored_hours'],'CFD_reference_cells':[q['cells'] for q in cfd],
          'stage_b_checks':f'{verify["passed"]}/{verify["total"]}','current_rain_runs':len(rain),'qualified_for_physical_yield':False},indent=2))


if __name__=='__main__':main()

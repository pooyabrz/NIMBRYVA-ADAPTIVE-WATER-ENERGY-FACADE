"""Build the evidence report without turning partial gates into product claims."""
import hashlib,importlib.metadata,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
from assembly import ROOT,compile_assembly
from weather import model_config
from actuator import load_envelope,transition
from drainage import depinning_mass
from stagecde.convergence import richardson_three

OUT=ROOT/'results/research'
def read(path):return json.loads((OUT/path).read_text())

def main():
    numerical=[];gci=[]
    for name in ['numerics-coupled.json','numerics-coupled-pv.json','numerics-film.json']:
        d=read('C/'+name);pairs=d.get('coupled_comparisons',sum(d.get('film_comparisons',{}).values(),[]))
        numerical.append({'source':name,'comparisons':len(pairs),'passed_initial_targets':sum(p['all_targets_pass'] for p in pairs),
          'max_mass_residual_kg':max(r['mass_residual_kg_max'] for r in d['coupled']) if 'coupled' in d else max(r['mass_residual_kg'] for rows in d['films'].values() for r in rows)})
        if 'coupled' in d:
            rows=sorted([r for r in d['coupled'] if r['dt']==30],key=lambda r:r['grid'])
            for key in ['water','energy']:
                gci.append({'source':name,'quantity':key,'refinement':'wall/ground, fixed physical dt30s',**richardson_three(*[r[key] for r in rows],absolute_floor=1e-10)})
    c=model_config();assembly=compile_assembly(c);pin=[];loads=[]
    # Contact-angle uncertainties are source-reported scatter, not an ageing law.
    for name,adv,rec in [('nominal',161.2,159.4),('low_hysteresis',160.4,160.3),('high_hysteresis',162.,158.5)]:
        for angle in [20,50,80]:pin.append({'condition':name,'adv_deg':adv,'rec_deg':rec,'angle_deg':angle,
          'depinning_mass_kg':depinning_mass(c['dew_area_m2'],angle,20,adv,rec,c['droplet_footprints_m2']),
          'footprints_m2':c['droplet_footprints_m2'],'model':'Furmidge monodisperse population diagnostic, not droplet dynamics validation'})
    for bearing in [.1,.3,.6]:
        cc={**c,'bearing_friction_Nm':bearing}
        for wind in [2,8,12]:loads.append({'bearing_friction_Nm':bearing,'wind_m_s':wind,**load_envelope(cc,assembly,wind,film_mass=.1)})
    wetting={'contact_angle_sensitivity':pin,'mechanical_loss_sensitivity':loads,
      'electrical_motion_energy_note':'catalogue running-power envelope used; load-dependent motor/gear efficiency curve unavailable, not inferred from torque',
      'depinning_population_note':'C full-solver footprint stress cases 1e3 and 1e5 per m2 are both below nominal 1e6; they are not a symmetric confidence interval'}
    (OUT/'C/wetting-mechanics.json').write_text(json.dumps(wetting,indent=2))
    d=read('D/experiments.json');s=read('E/surrogate/test-metrics.json');assessment=read('E/assessment.json');t=read('E/control/test.json');replay=read('E/control/teacher-replay.json')
    sensitivity=read('C/sensitivity.json');tracking=read('C/tracking-coarse-medium-fine.json');turbulence=read('C/turbulence.json')
    df=pd.DataFrame(replay['episodes']);totals=df.groupby('controller')[['drain_L','pv_gross_Wh','motor_Wh','unsupported_hours','reward']].sum()
    model_ranking=[]
    sf=pd.DataFrame(t['episodes']);selected=set(replay['groups']);em=sf[sf.group.isin(selected)].groupby('controller')[['drain_L','pv_gross_Wh','motor_Wh']].sum()
    for kind in totals.index:model_ranking.append({'controller':kind,'teacher_conditional_drain_L':float(totals.loc[kind,'drain_L']),
      'emulator_conditional_drain_L':float(em.loc[kind,'drain_L']),'teacher_PV_Wh':float(totals.loc[kind,'pv_gross_Wh']),
      'teacher_motor_Wh':float(totals.loc[kind,'motor_Wh']),'teacher_unsupported_hours':int(totals.loc[kind,'unsupported_hours'])})
    cases=sensitivity['scenarios'];envelopes=[]
    for city in sorted({r['city'] for r in cases}):
        z=[r for r in cases if r['city']==city and r['status']=='COMPLETED']
        envelopes.append({'city':city,'conditional_hours':z[0]['hours'],'water_L_range':[min(r['water_L'] for r in z),max(r['water_L'] for r in z)],
          'PV_Wh_range':[min(r['PV_Wh'] for r in z),max(r['PV_Wh'] for r in z)],'type':'one-at-a-time scenario envelope, not combined or probabilistic uncertainty'})
    runtime={'python':platform.python_version(),'platform':platform.platform()}
    for package in ['numpy','scipy','pandas','matplotlib','pvlib','torch','stable-baselines3','gymnasium','openpyxl','xlrd']:
        try:runtime[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:runtime[package]='not installed in report runtime'
    (ROOT/'config/research/runtime-lock.json').write_text(json.dumps(runtime,indent=2))
    result={'version':'CDE-1','numerical_comparisons':numerical,'Richardson_diagnostics':gci,
      'numerical_error_is_model_error':False,'whole_device_grid_time_independence':False,
      'measured_validation':{'PV_linear_closure':{'MAPE_range_percent':[min(x['heldout_MAPE_percent'] for x in d['PV']['modules']),max(x['heldout_MAPE_percent'] for x in d['PV']['modules'])]},
        'wall_flux_closure_holdout_MAPE_percent':d['rain']['heldout_mean_abs_relative_percent'],
        'dew_rate_integrated_error_percent':[x['integrated_relative_error_percent'] for x in d['dew']['heldout']],
        'full_coupled_radiative_dew_validation':False,'3D_CFD_experimentally_validated':False,'unknown_facade_PV_BOM_validated':False},
      'surrogate_test':s['overall'],'after_projection':assessment['after_mass_projection'],'scenario_envelopes':envelopes,
      'controller_replay_diagnostics':model_ranking,'tracking_dt_comparisons':tracking['time_comparisons'],
      'SST_comparison_cases':len(turbulence['comparisons']),'SST_building_residual_gate':turbulence.get('relaxed_retry',turbulence['comparisons'][1]['SST_quality'])['flow_residual_gate_passed'],
      'qualification':{'C_whole_device':False,'D_whole_product':False,'E_high_fidelity_surrogate':False,'RL_superiority':False},
      'no_physical_prototype_or_new_experiment_performed':True,'all_product_yield_remains_conditional':True,
      'measurement_uncertainty':'rain ETOT/adhesion loss supplied by experiment; no invented matrix/dew metrology uncertainty',
      'climate_inference':'20 additional city-years available, policies evaluated on selected warm episodes, not continuous multi-year yield',
      'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/stagecde').glob('*.py'))},'runtime':runtime}
    result['software_checks']={'cassette':json.loads((ROOT/'results/cassette-v2/verification.json').read_text()),'stage_b':json.loads((ROOT/'results/stage-b/verification.json').read_text())['passed'],'research':read('checks.json')['passed']}
    result['warmup_reconstruction_audit']=read('E/warmup-audit.json')
    (OUT/'E/training-data-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'E/transitions').glob('*.npz'))},indent=2))
    (OUT/'release.json').write_text(json.dumps(result,indent=2))
    pv=result['measured_validation']['PV_linear_closure']['MAPE_range_percent'];mae=s['overall']['temperature_MAE_K_by_node'];train=read('E/transitions-train.json')
    lines=['# نتیجهٔ اجراهای C، D و E — نسخهٔ CDE-1','',
      'این تحویل روی مدل کاست ۲ و مرحلهٔ B ساخته شده است. شبیه‌سازی‌ها و آموزش‌ها واقعاً اجرا شده‌اند. **دقت بالای کل محصول و برتری RL احراز نشده است.**','',
      '## وضعیت در یک نگاه','',
      '| بررسی | نتیجهٔ اجرا | حدود تفسیر |','|---|---|---|',
      f'| شبکه/زمان دیوار، زمین و فیلم | {sum(x["passed_initial_targets"] for x in numerical)} از {sum(x["comparisons"] for x in numerical)} مقایسه از هدف اولیه عبور کرد | شبکهٔ هشت‌گرهی کاست و CFD کل دستگاه را تأیید نمی‌کند |',
      '| قطرات باران | ۳ شبکه × ۳ گام زمانی واقعی، با ۸۱۹۲ نمونه در هر اندازه | خطای نمونه‌گیری و محدودیت منبع ورودی باقی است |',
      '| آشفتگی | k–ε و SST، هندسه و شبکهٔ یکسان؛ دامنهٔ خالی هم جدا اجرا شد | SST ساختمان در ۱۲۰۰ تکرار و سپس ۸۰۰ تکرار با آرام‌سازی کمتر از دروازهٔ باقیمانده نگذشت؛ اختلاف آن برآورد معتبر خطای مدل نیست |',
      f'| PV آزمایشگاهی | خطای مطلق نسبی میانگین {pv[0]:.2f} تا {pv[1]:.2f}٪ در ۹ ماژول | فقط توان تابع تابش/دمای معلوم؛ نه مدل دمای PV جدید |',
      f'| باران تجربی | خطای میانگین مدل سادهٔ دیواری {d["rain"]["heldout_mean_abs_relative_percent"]:.1f}٪ | این تقریب برای ادعای دقت کافی نیست؛ CFD هنوز اعتبارسنجی نشده |',
      '| شبنم تجربی | مدل نرخ جمع‌آوری در دو اجرای ۱۲ساعته حدود +۸٫۸ و −۹٫۷٪ اختلاف تجمعی دارد | دمای صفحه/توان پلتیر/باد/بلندموج کافی نیست؛ تأیید انتقال حرارت کامل نیست |',
      f'| جانشین | MAE سطح PV={mae[0]:.3f} K و سطح میعان={mae[6]:.3f} K | تقریب حل‌گر شرطی موجود؛ نه جانشین CFD مستقل و تأییدشده |',
      f'| خروجی آب جانشین پس از بستن جرم | MAE={1000*assessment["after_mass_projection"]["drain_MAE_L"]:.3f} mL/h در برابر میانگین مرجع {1000*assessment["after_mass_projection"]["reference_mean_drain_L"]:.3f} mL/h | خطای آب برای تصمیم طراحی کافی نیست؛ پایداری موازنه معادل دقت شار نیست |',
      '| PPO | سه بذر × ۵۰۱۷۶ گام؛ انتخاب در ۲۰۲۲، آزمون در ۲۰۲۳ | ادعای برتری بر MPC/آستانه‌ای پشتیبانی نمی‌شود |','',
      '## راستی‌آزمایی و عدم‌قطعیت','',
      'برای هر جهت ثابت سطح، دیوار با ۲/۴/۸ سلول در هر لایه و زمین با ۱۶/۳۲/۶۴ سلول، در گام‌های ۱۲۰/۶۰/۳۰ ثانیه اجرا شد. فیلم آب/باران/برف با ۱۶/۳۲/۶۴ سلول و گام‌های ۲۰/۱۰/۵ ثانیه اجرا شد. مجموعاً ۴۵ مورد مرجع حرارتی/فیلم اجرا شده است. اهداف، حداکثر اختلاف ۲٪ یا حد مطلق ۰٫۰۰۰۱ kg/m²/h برای آب، ۰٫۰۲ Wh/m²/h برای انرژی، و ۰٫۲ K دمای سطح‌اند؛ معیار نسبی نزدیک صفر تنها به کار نرفته است.',
      'مقایسهٔ دمای سطح در این مطالعه دمای سطح میعان است؛ جزئیات دمای PV در آزمون‌های پایه و ارزیابی جانشین جدا گزارش می‌شود. ریزکردن دیوار، ریزکردن لایه‌های کاست نیست. GCI فقط در دادهٔ یکنواخت و با هشدار دربارهٔ ناحیهٔ مجانبی محاسبه می‌شود. عدد کوچکی برای GCI در دادهٔ نوسانی ساخته نشده است. جریان steady با تکرار SIMPLE حل می‌شود؛ گام زمانی آن زمان فیزیکی نیست.',
      'در ردیابی ۹حالت، فقط سه مقایسهٔ زمانی از شش مقایسهٔ مجاور از هدف ۲٪ عبور کردند؛ تغییرات حدود ۱٫۱ تا ۳٫۵٪ بودند. خطای نمونه‌گیری حدود ۱۲٪ و منبع ورودی محدود نیز باقی است. نتیجه، احراز استقلال برداشت باران نیست. جدول‌های دقیق شبکه/زمان در فایل tracking ثبت شده‌اند.',
      f'حساسیت کامل حل‌گر در {len(cases)} سناریو و دو اقلیم، همراه شش اجرای دو قانون انتقال جرم انجام شد. دما، RH، باد، تابش کوتاه/بلندموج، باران، جابه‌جایی یک‌ساعتهٔ بلندموج، میانگین‌گیری دوساعته، ترشوندگی، جمع‌آوری، پل حرارتی و توان موتور جدا تغییر کردند. دامنهٔ تغییر سناریوها فاصلهٔ اطمینان احتمالاتی نیست.','',
      '## دادهٔ تجربی و جداسازی','',
      'PV: شرایط تقریباً ۲۵/۵۰ درجه و تابش ≥۴۰۰ برای برازش نگه داشته شد؛ شرایط کامل دمای دیگر و تابش کم برای آزمون کنار ماند. ضرایب ماژول‌های آزمایشگاه به BOM نامعلوم پنل انتخابی منتقل نشدند.',
      'باران: رخداد ۱ برای ضرایب پناه‌گیری هر گیج، رخدادهای ۲ و ۳ فقط برای آزمون استفاده شد. مقدار تجمعی گیج‌ها تفاضل‌گیری/مقایسه شد؛ جمع ستون تجمعی به‌عنوان حجم باران به کار نرفت. B4 و B8 به استناد نبود ETOT معتبر در جدول خطای منبع کنار گذاشته شدند. اتلاف چسبندگی و آب باقی‌ماندهٔ گیج، بازهٔ یک‌طرفه است و CI ۹۵٪ نامیده نشده است.',
      'شبنم: فقط اجراهای ۱ تا ۷ شکل 4B برای برازش نرخ استفاده شدند. اجراهای ۸ و ۹ ابتدای دو اجرای شکل 4C هستند؛ تشخیص هم‌پوشانی در کد اجباری است. شکل 2B و S5 نیز یک آزمایش سرمایش هستند. نبود سری زمانی LW و باد و نبود اجرای مستقل سرمایش، دروازهٔ اعتبارسنجی حرارت را باز نگه می‌دارد. ولتاژ ۲٫۴ ولت پلتیر، توان سرمایش معلوم نیست.','',
      '## جانشین و کنترل','',
      f'آموزش جانشین: {sum(r.get("samples",0) for r in train["records"])} گذار در ۳۲ گروه آموزش. ۱۱۵۲ نمونهٔ انتخاب و ۲۳۰۴ نمونهٔ آزمون جدا هستند. ورودی شبکه ۴۹ ویژگی و مشاهدهٔ PPO پنجاه ویژگی دارد. جرم با تصویرکردن خروجی شبکه روی موجودی قابل‌دسترس بسته می‌شود؛ این تصحیح، درست‌بودن شار شبکه را تضمین نمی‌کند. دفتر انرژی جانشین ساعتی است و با توزیع زیرگام‌های حل‌گر اصلی تفاوت دارد؛ بازپخش خطای آن را آشکار می‌کند.',
      'خطای آبِ خام شبکه و پس از تصحیح موجودی جدا ثبت شده‌اند؛ تصحیح جرم در این اجرا خطای آب خروجی را افزایش داده است. خطای متوسط خروجی پس از تصحیح از میانگین خروجی مرجع بزرگ‌تر است. احتمال شبکه برای مجاز بودن رژیم کالیبره نیست: در امتیاز ≥۰٫۹، ۲۴ نمونهٔ خارج از دامنه از ۱۳۶ نمونه به‌اشتباه پذیرفته می‌شوند؛ فیلتر OOD جدا اعمال می‌شود. این تشخیص، حفاظت تأییدشدهٔ فیزیکی نیست.',
      'گرم‌سازی همهٔ ۵۶ پنجره جدا بازسازی شد و با نخستین ورودی ذخیره‌شدهٔ حل‌گر تطبیق داشت؛ هیچ بازنشانی به علت خطای حل‌گر در گرم‌سازی رخ نداده است. هش داده‌های گذار و وزن‌ها ثبت شده و بارگذاری وزن تغییرکرده رد می‌شود.',
      'شهرهای آموزش: سنگاپور، بندرعباس، بمبئی، فینیکس در ۲۰۲۰ و ۲۰۲۱. انتخاب: همان شهرها در ۲۰۲۲. آزمون: آن چهار شهر و رشت، دبی، سویا، استانبول در ۲۰۲۳. داده‌های ۲۰۲۴ قبلی برای آزمون جدید استفاده نشده‌اند. آموزش چهار پنجرهٔ فصلی و انتخاب/آزمون دو پنجره در بخش اول سال دارند؛ ارزیابی سالانه و اقلیم‌شناسی چندساله انجام نشده است. همهٔ فایل‌های سالانه موجودند.',
      'عمل در زمان t فقط هوا و وضعیت تا t را می‌بیند؛ محیط، هوای تحقق‌یافتهٔ t+1 را بعد از عمل وارد می‌کند. MPC فقط ماندگاری هوای فعلی با افق ۳ ساعت و پهنای جست‌وجوی ۳ دارد. هیچ پیش‌بینی صادرشدهٔ تاریخی دریافت نشده و بازتحلیل آینده به کنترل‌کننده داده نشده است.',
      'مساحت، ابعاد، هندسه، حالت خشک اولیه، دمای اولیه، باتری اولیهٔ ۱۰۰ Wh و ظرفیت ۲۰۰ Wh و منبع انرژی برای چهار کنترل برابرند. انرژی مصرفی موتور برابر نشده است؛ اختلاف آن، PV، کسری تأمین و SOC نهایی گزارش می‌شود. مقایسهٔ اصلی یک مقایسه با بودجه و منبع یکسان است، نه آزمایش با اجبار مصرف Wh یکسان. کنترل ثابت، PV در ۵۰ درجه است و بهترین کنترل ثابتِ ممکن برای هدف آب معرفی نشده است.',
      'پاداش فعلی آب، کسری انرژی، تغییر وضعیت و کمبود SOC نهایی را در نظر می‌گیرد؛ برای برق مازاد پاداش مستقلی ندارد. بنابراین کاهش شدید برق PV در PPO باید در کنار آب و پاداش دیده شود. وزن‌دهی یک هدف مشترک آب/برق و آزمایش با مصرف موتور دقیقاً همسان، پژوهش تازه با پروتکل و آزمون مستقل می‌خواهد؛ نتایج فعلی به‌عنوان آن آزمایش معرفی نمی‌شوند.',
      '## بازپخش چهار شهر روی حل‌گر اصلی','',
      'جدول زیر جمع چهار پنجرهٔ ۴۸ساعته است. سطرها ساعت خارج از دامنه دارند؛ آب ستون، خروجی شرطی برای عیب‌یابی است و عملکرد معتبر محصول نیست. PPO در این آزمون آب بیشتری از MPC یا کنترل آستانه‌ای تولید نکرده است.','',
      '| کنترل | آب شرطی L | PV ناخالص Wh | موتور Wh | ساعات خارج از دامنه |','|---|---:|---:|---:|---:|']
    for r in model_ranking:lines.append(f'| {r["controller"]} | {r["teacher_conditional_drain_L"]:.4f} | {r["teacher_PV_Wh"]:.1f} | {r["teacher_motor_Wh"]:.1f} | {r["teacher_unsupported_hours"]} |')
    lines+=['',
      'خطای تک‌گام، خطای چندگام با فرمان‌های یکسان، و بازپخش کنترل بسته سه فایل جدا دارند. بازپخش هنوز آزمایش مستقل فیزیکی نیست. بازه‌های bootstrap بر شهرها محاسبه شده‌اند؛ تنها یک سال آزمون و انتخاب پنجره‌های گرم، اجازهٔ تعبیر آن‌ها به بازهٔ عملکرد اقلیمی محصول را نمی‌دهد.','',
      '## فایل‌های قابل بررسی','',
      '- `results/research/release.json`: جمع‌بندی و همهٔ دروازه‌ها.',
      '- `results/research/C/`: شبکه/زمان، ردیابی، آشفتگی، موازنه‌ها، حساسیت و تشخیص Richardson.',
      '- `results/research/D/experiments.json` و `data/research/experiments/`: اندازه‌گیری، تقسیم، پیش‌بینی و منبع/هش.',
      '- `results/research/E/`: تقسیم داده، گذارها، وزن‌های جانشین، سه سیاست PPO، انتخاب، آزمون و بازپخش.',
      '- `docs/CDE-RUN-fa.md`: فرمان‌های اجرا و بازسازی.',
      '- `config/research/protocol.json`: پروتکل اولیه؛ تغییر آن برای پژوهش بعدی باید نسخهٔ تازه داشته باشد.','',
      '## برای مرحلهٔ بعد چه لازم است؟','',
      'کار فوری برای شما نصب و اجرای نمایش محلی است؛ برای این مرحله نیازی به ساخت نمونه نیست. پیش از ادعای دقت محصول، یک مورد مرجع مستقل با سری زمانی هم‌زمان دمای صفحه، رطوبت، باد، تابش بلندموج و دبی جمع‌آوری، و بازسازی واقعی CFD آرایهٔ نه‌مکعبی لازم است. اگر فقط تصاویر مقاله کافی باشند، عدم‌قطعیت دیجیتایز باید جدا افزوده شود. پیوندهای داده در سند منابع آمده‌اند؛ دادهٔ مفقود ساخته یا از نتیجهٔ مطلوب استنتاج نشده است.',
      'برای نتیجهٔ سالانه باید دوره‌های چندسالهٔ پیوسته و دورهٔ سرد با مدل کاست دارای یخ/برف واجد شرایط اجرا شوند. BOM واقعی PV، طیف جوی/سطح خیس، دوام پوشش، سینتیک قطره، پل حرارتی سه‌بعدی و بار باد گذرا همچنان حدود مرحلهٔ A/B هستند و C/D/E به‌تنهایی آن‌ها را حذف نمی‌کند.','']
    (ROOT/'docs/CDE-RESULTS-fa.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':'docs/CDE-RESULTS-fa.md','numerical_comparisons':numerical,'qualification':result['qualification']},indent=2))
if __name__=='__main__':main()

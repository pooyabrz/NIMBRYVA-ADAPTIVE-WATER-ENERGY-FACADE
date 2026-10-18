"""Reproducible Persian presentation report and exact quantitative figures."""
import argparse,html,json,math,base64
from pathlib import Path
import numpy as np
import pandas as pd
from assembly import ROOT
from presentation.metrics import calculate,OUT

LABEL={'fixed':'PV ثابت','emitter':'میعان ثابت','rule':'آستانه‌ای','mpc':'MPC','ppo':'PPO'}
CITY={c['id']:c['name_fa'] for c in json.loads((ROOT/'config/cities.json').read_text())}
ENGLISH={c['id']:c['name'] for c in json.loads((ROOT/'config/cities.json').read_text())}
PLOT=ROOT/'output/presentation/figures';PDF=ROOT/'output/pdf/Defensible-Report-fa.pdf';HTML=ROOT/'output/presentation/report-fa.html'

def fmt(x,dec=2):
    if x is None or not math.isfinite(float(x)):return 'نامشخص'
    return f'{x:,.{dec}f}'

def charts(d):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOT.mkdir(parents=True,exist_ok=True);z=[next(r for r in d['by_city'] if r['controller']=='rule' and r['city']==city['city']) for city in d['weather_coverage']];x=np.arange(len(z));a=np.array([np.nan if r['collected_rain_L_m2_h'] is None else r['collected_rain_L_m2_h']*1000 for r in z]);b=np.array([np.nan if r['collected_condensation_L_m2_h'] is None else r['collected_condensation_L_m2_h']*1000 for r in z])
    fig,axes=plt.subplots(1,2,figsize=(10.4,2.9),sharey=True)
    for ax,values,title,color in zip(axes,[a,b],['Collected rain','Collected condensation'],['#397ba6','#139a8f']):
        ax.barh(x,values,color=color);ax.set_yticks(x,[ENGLISH[r['city']]+' | '+str(r['hours'])+' h' for r in z],fontsize=8);ax.set_xlabel('mL / panel m² / scored h',fontsize=8);ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis();fig.suptitle('Frozen threshold rule | conditional 2025 windows | independent axis scales',fontsize=10);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(PLOT/('water-by-city.'+ext),dpi=200)
    plt.close(fig)
    kinds=list(d['summaries']['common_prefix']);m=d['summaries']['common_prefix'];x=np.arange(len(kinds));fig,ax=plt.subplots(figsize=(10.4,2.9))
    ax.bar(x-.2,[m[k]['pv_gross_Wh']/1000 for k in kinds],.4,label='Gross PV',color='#1b3652');ax.bar(x+.2,[m[k]['pv_useful_in_system_Wh']/1000 for k in kinds],.4,label='PV used in controls + retained PV stock',color='#139a8f')
    ax.set_xticks(x,kinds);ax.set_ylabel('PV energy (kWh)');ax.set_title('Same common prefixes | unused surplus is excluded from useful PV');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(PLOT/('energy-accounting.'+ext),dpi=200)
    plt.close(fig)

def sections(d):
    s=d['summaries']['common_prefix'];raw=d['summaries']['all_hours_diagnostic'];rule=s['rule'];new=json.loads((OUT/'evaluation.json').read_text());merge=json.loads((ROOT/'config/provenance/merge-record.json').read_text());N=len(new['episodes']);H=raw['rule']['hours'];h=rule['hours']
    pages=[]
    pages.append({'title':'نمای تطبیقی آب و انرژی','subtitle':'گزارش قابل ارائه - نسخه F1 - آزمون هوای ۲۰۲۵',
      'paragraphs':[
        'سطح شواهد: شبیه‌سازی غربالگری با مدل فیزیکی شرطی و سیاست‌های ثابت‌شده. عملکرد محصول ساخته‌نشده و دقت تجربی کل دستگاه هنوز احراز نشده است. برای RL ادعای بهبود ارائه نمی‌شود.',
        f'این نسخه روی فایل پیوست شما اعمال شد: {merge["base_files_identical"]} فایل پایه یکسان، ۱۰ فایل قبلی به‌روز و ۵۵۲ مسیر C/D/E افزوده شد. کد حسابداری و نتایج این گزارش نیز به همان پروژه اضافه شده‌اند.',
        f'ارزیابی تازه: {N} اجرای کنترل، هشت شهر، دو پنجره ۴۸ساعته برای هر شهر و پنج کنترل. هر کنترل {H} ساعت محاسبه دارد؛ {h} ساعتِ آغازین مشترک پیش از نخستین خروج از دامنه برای مقایسه نگه داشته شد. از ۱۶ پنجره، {len(d["complete_jointly_supported_groups"])} پنجره کاملاً در دامنهٔ مشترک بودند. این‌ها شروع‌های مستقل هستند و عملکرد یک سال پیوسته را نشان نمی‌دهند.',
        'مساحت مرجع هر پنل ۱٫۹۹۸۱۰۸ مترمربع و سطح فعال میعان ۱٫۶ مترمربع است. باتری اولیه ۱۰۰ Wh، ظرفیت ۲۰۰ Wh، سطح اولیه خشک و دمای اولیه برابر هوای همان زمان است. سرمایش فعال افزوده نشده است.'
      ],'table':(['شاخص کنترل آستانه‌ای در دامنهٔ مشترک','مقدار'],[
        ['آب باران جمع‌شده - L',fmt(rule['collected_rain_L'],4)],['آب میعان جمع‌شده - L',fmt(rule['collected_condensation_L'],4)],
        ['برق ناخالص PV - Wh',fmt(rule['pv_gross_Wh'])],['PV مصرف‌شده/ذخیره‌شده در سامانه - Wh',fmt(rule['pv_useful_in_system_Wh'])],
        ['انرژی مدل‌شدهٔ کنترل - Wh',fmt(rule['modeled_control_Wh'])],['تعداد حرکت واقعی',fmt(rule['changed'],0)]
      ]),'note':'اعداد جدول حاصل مدل‌اند؛ اندازه‌گیری مستقل محصول نیستند. خلاصهٔ اصلی، کنترل آستانه‌ای را به‌عنوان نمونهٔ تفسیر حسابداری نشان می‌دهد و انتخاب بهینه بر اساس آزمون نیست.'})
    waterrows=[]
    for k,r in s.items():waterrows.append([LABEL[k],fmt(r['rain_input_L'],3),fmt(r['condensed_L'],3),fmt(r['collected_rain_L'],3),fmt(r['collected_condensation_L'],3),fmt(r['evaporated_L'],3),fmt(r['rain_entry_loss_L']+r['uncollected_departure_L'],3),fmt(sum(r['stored_'+o+'_L'] for o in ['rain','condensation','initial']),3)])
    pages.append({'title':'آب قابل جمع‌آوری، با تفکیک منشأ','subtitle':f'همهٔ اعداد L هستند؛ جمع روی {h} ساعتِ آغازین مشترک برای هر کنترل',
      'paragraphs':['ورودی باران و تشکیل میعان، آب قابل برداشت نیستند. آب ذخیره‌شده روی سطوح، تبخیر و اتلاف مسیر جمع‌آوری از خروجی تخلیه جدا شده‌اند. ردیاب منفعل در هر زیرگام، منشأ آب را در سه مخزن دنبال می‌کند و حل فیزیکی را تغییر نمی‌دهد.',
        'برای تبخیر و لغزش، اختلاط کامل در هر مخزن فرض شده است. جهش CNF ابتدا به میعان نسبت داده می‌شود، چون قانون آن بر میعان تازه تعریف شده است. منشأ در اینجا مکانیزم ورود به سطح است؛ اختلاط گاز و بازچگالش مولکولی حل نشده‌اند.'],
      'table':(['کنترل','باران رسیده','میعان تشکیل‌شده','باران جمع‌شده','میعان جمع‌شده','تبخیر','تلفات','موجودی پایان'],waterrows),
      'paragraphs_after':['ستون تلفات شامل کاهش ورود باران با ضریب فرضی ۰٫۸۵ و اتلاف جمع‌آوری با بازده فرضی ۰٫۸۵ است. این‌ها اندازه‌گیری پاشش، تبخیر قطره در هوا یا ردیابی مجدد قطرات ثانویه نیستند. اجزای تبخیر، موجودی و تلفات برای هر منشأ در CSV جدا ثبت شده‌اند.',
        'موازنهٔ کنترل حجم: باران هندسی + میعان + موجودی آغاز = کاهش ورود + تبخیر + خروجی جمع‌شده + اتلاف مسیر + موجودی پایان. یک کیلوگرم آب تقریباً یک لیتر در نظر گرفته شده است.'],
      'note':'اگر مدلی در میانهٔ پنجره از دامنه خارج شود، امتیازدهی آن پنجره بعداً از سر گرفته نمی‌شود. دادهٔ خارج از دامنه برای عیب‌یابی نگه داشته شده است.'})
    energyrows=[]
    for k,r in s.items():energyrows.append([LABEL[k],fmt(r['pv_gross_Wh'],1),fmt(r['dc_available_Wh'],1),fmt(r['pv_useful_in_system_Wh'],1),fmt(r['surplus_Wh'],1),fmt(r['modeled_control_Wh'],1),fmt(r['initial_battery_Wh'],1),fmt(r['changed'],0)])
    ratiorows=[]
    for k,r in s.items():
        a=r['energy_water_ratios']['rain'];b=r['energy_water_ratios']['condensation']
        ratiorows.append([LABEL[k],fmt(a['whole_system_control_burden_Wh_per_L'],1) if a['denominator_resolved_at_design_floor'] else 'حل‌نشده نزدیک صفر',fmt(b['whole_system_control_burden_Wh_per_L'],1) if b['denominator_resolved_at_design_floor'] else 'حل‌نشده نزدیک صفر'])
    pages.append({'title':'برق، مصرف کنترل و تفسیر انرژی به آب','subtitle':'انرژی‌ها Wh هستند؛ سرمایش فعال در این طرح صفر است',
      'table':(['کنترل','PV ناخالص','پس از تبدیل DC','PV مفید سامانه','مازاد بی‌مصرف','کنترل مصرف‌شده','سهم باتری اولیه','حرکت'],energyrows),
      'paragraphs_after':['PV مفید سامانه = برق جدید PV مصرف‌شده توسط بارهای مدل‌شده + افزایش موجودی باتری با منشأ PV. برق اولیهٔ باتری، تلفات تبدیل/باتری و مازاد بی‌مصرف به آن اضافه نمی‌شوند. مازاد، برق فروخته‌شده یا مصرف‌شدهٔ ساختمان نیست؛ مبدل AC و مصرف‌کنندهٔ بیرونی مدل نشده‌اند.',
        'مصرف کنترل شامل موتور و فرض توان آماده‌به‌کار ۰٫۵ W است. توان واقعی پردازنده، حسگرها و ارتباطات اندازه‌گیری نشده و مجموع واقعی سخت‌افزار نامشخص است. سرمایش تابشی منفعل است؛ اگر سرمایش فعال افزوده شود، توان ورودی و برداشت گرما باید هم‌زمان به مدل افزوده شوند.'],
      'table2':(['کنترل','سربار کل کنترل / L باران','سربار کل کنترل / L میعان'],ratiorows),
      'note':'دو نسبت از یک صورت انرژی مشترک استفاده می‌کنند و جمع‌پذیر نیستند. نسبت باران انرژی تولید آب از بخار نیست؛ نسبت میعان SEC دستگاه سرمایشی نیست. نزدیک کف مطلق پیشنهادی مطالعه، نسبت قابل‌اتکا گزارش نمی‌شود؛ حساب اسمی با علامت هشدار در CSV موجود است.'})
    climaterows=[]
    for cid in [x['city'] for x in d['weather_coverage']]:
        r=next(x for x in d['by_city'] if x['city']==cid and x['controller']=='rule');cover=next(x for x in d['weather_coverage'] if x['city']==cid)
        climaterows.append([CITY[cid],str(r['hours']),fmt(r['collected_rain_L']*1000,2) if r['hours'] else 'دادهٔ قابل امتیاز ندارد',fmt(r['collected_condensation_L']*1000,2) if r['hours'] else 'دادهٔ قابل امتیاز ندارد',fmt(r['raw_outside_model_percent'],1),fmt(r['common_prefix_excluded_percent'],1),fmt(cover['annual_weather_outside_preselection_percent'],1)])
    pages.append({'title':'حساسیت اقلیمی و پوشش دامنه','subtitle':'کنترل آستانه‌ای؛ آب به mL و درصدها با مخرج صریح',
      'table':(['شهر','ساعت مشترک','باران جمع‌شده','میعان جمع‌شده','خارج از دامنه در ۹۶h','کنارگذاشته از مقایسه','هوای خارج از پیش‌انتخاب سال'],climaterows),
      'paragraphs_after':['در هر شهر دو پنجرهٔ ۴۸ساعته با هدف فصلی از پیش تعیین‌شده و نزدیک‌ترین دورهٔ گرم/بدون برف انتخاب شد. این انتخاب نمایندهٔ تصادفی کل سال نیست. تغییر ساعت مشترک ناشی از خروج هر یک از پنج کنترل است؛ بنابراین درصد کنارگذاری با درصد خروج کنترل آستانه‌ای متفاوت است.',
        'شرایط پیش‌انتخاب هوا: دمای هوا حداقل ۱۰ درجه، باد کمتر از ۱۰ m/s، نبود برف و RH حداکثر ۱۰۰٪. درصد سالانهٔ ستون آخر فقط همین شرایط هوا را می‌سنجد؛ درصد خروج ترمودینامیکی کل دستگاه در تمام سال محاسبه نشده است. رتبه‌بندی اقلیمی یا تولید سالانه از این پنجره‌ها استخراج نمی‌شود.']})
    ratios=[]
    for cid in [x['city'] for x in d['weather_coverage']]:
        r=next(x for x in d['by_city'] if x['city']==cid and x['controller']=='rule');values=[]
        for origin in ['rain','condensation']:
            item=r['energy_water_ratios'][origin];value=item['nominal_arithmetic_control_burden_Wh_per_L']
            values.append('تعریف‌نشده' if value is None else fmt(value/1000,3)+(' *' if not item['denominator_resolved_at_design_floor'] else ''))
        ratios.append([CITY[cid],str(r['hours']),fmt(r['modeled_control_Wh'],1),*values])
    pages.append({'title':'نرخ برداشت و انرژی به آب در هر شهر','subtitle':'کنترل آستانه‌ای؛ نسبت‌ها kWh/L هستند و تخصیص اختصاصی فرایند نیستند',
      'table':(['شهر','ساعت مقایسه','کنترل مصرف‌شده Wh','سربار / L باران','سربار / L میعان'],ratios),
      'figure':'water-by-city.png','paragraphs_after':['علامت * فقط حساب اسمی را نشان می‌دهد: آب جمع‌شده کمتر یا هم‌مرتبهٔ کف مطلق پیشنهادی ۰٫۰۰۰۱ kg/m²/h است و نسبت برای تصمیم طراحی قابل‌اتکا نیست. خروجی صفر با انرژی به آب صفر جایگزین نشده است. نمودار، نرخ متوسط در ساعت‌های همین پنجره‌هاست؛ مقدار روزانه یا سالانهٔ پیش‌بینی‌شده نیست.']})
    rawrows=[]
    for k,r in raw.items():rawrows.append([LABEL[k],str(r['hours']),fmt(r['outside_model_percent'],1),fmt(r['wet_emitter_dry_optics_approximation'],0),fmt(r['causal_wind_interlock'],0),fmt(r['action_blocked'],0),fmt(r['drive_utilization_max'],3),fmt(r['unserved_Wh'],3)])
    pages.append({'title':'حرکت، باد و اعتبار مقایسهٔ کنترل‌ها','subtitle':'آمار این جدول روی تمام ساعت‌های اجراشده است، حتی ساعت‌های عیب‌یابی',
      'table':(['کنترل','ساعت','خارج از دامنه %','ساعات تقریب طیف خشک','رد باد در تصمیم','رد در حل‌گر','بیشینه نسبت گشتاور','کسری انرژی Wh'],rawrows),
      'paragraphs_after':[f'مرز باد میانگین در غربالگری خشک این مدل حدود {d["wind_mean_limit_dry_screening_m_s"]:.2f} m/s است، در هوای مرجع با چگالی ۱٫۲۲۵ kg/m³، ضریب گاست ۱٫۵ و ضرایب فرضی بار. این عدد مجوز سازه یا تأیید موتور نیست. جرم آب، بار گذرا، اتصالات و میدان جریان نیاز به ارزیابی مستقل دارند. آزمون‌های بادِ کنترل‌شده برای چند جرم آب در wind-screening.csv موجودند.',
        'همهٔ کنترل‌ها هندسه، وضعیت شروع، مساحت، باتری و هوای یکسان دارند. مصرف موتور به اجبار برابر نشده و جدا گزارش شده است. کنترل ثابت میعان، ثابت PV، آستانه‌ای، MPC با ماندگاری هوا و PPO با سیاست منجمد مقایسه شدند. آیندهٔ بازتحلیل‌شده وارد مشاهده یا پیش‌بینی کنترل‌کننده نشده است.',
        'پس از نخستین خروج از دامنه یا کمبود برق کنترل، امتیازدهی پنجره متوقف می‌شود. خروجی «پنجره‌های کاملاً مشترکِ پشتیبانی‌شده» هم جدا ذخیره شده است. water_supported تنها دامنهٔ محاسباتی آب مایع را نشان می‌دهد و تأیید دقت طیفی، خیس‌شدن، باد یا محصول نیست.'],
      'figure':'energy-accounting.png'})
    ranges=[]
    for r in d['scenario_envelopes']:
        a=r['collected_condensation_L'];b=r['collected_rain_L'];ranges.append([CITY[r['city']],str(r['hours_per_run']),f'{a[0]*1000:.2f} تا {a[1]*1000:.2f}',f'{b[0]*1000:.2f} تا {b[1]*1000:.2f}',str(r['runs_with_unsupported_hours'])])
    interval=d['conditional_city_cluster_intervals'].get('ppo_minus_rule_condensation',{});ci=interval.get('city_cluster_resampling_95_interval_L_m2_h',[None,None])
    pages.append({'title':'عدم‌قطعیت و متن قابل‌دفاع برای ارائه','subtitle':'تفکیک خطای عددی، خطای فیزیک، سناریو و پراکندگی نمونه',
      'paragraphs':['بازهٔ عدم‌قطعیت کل فیزیکی قابل تعیین نشده است. مواد داخلی PV، طیف سطح خیس/جو، ترشوندگی/دوام و انتقال آشفته کاملاً شناسایی و اعتبارسنجی نشده‌اند. پوش حساسیت زیر دامنهٔ چند سناریوی فرضی است و فاصلهٔ اطمینان عملکرد محصول نیست.'],
      'table':(['شهر','ساعت هر سناریو','پوش میعان جمع‌شده mL','پوش باران جمع‌شده mL','سناریو دارای خروج'],ranges),
      'paragraphs_after':[
        f'بازهٔ بازنمونه‌گیری ۹۵٪ اختلاف نرخ میعان PPO و آستانه‌ای، در خوشه‌های شهر و آغازهای مشترک این آزمون: [{fmt(ci[0],7)}, {fmt(ci[1],7)}] L/m²/h. این بازه فقط پراکندگی نمونهٔ عددی شهرها را نشان می‌دهد و خطای فیزیکی یا اقلیم بلندمدت را پوشش نمی‌دهد.',
        'مرحله C: ۶۰ مقایسهٔ حرارتی/فیلم از اهداف اولیه گذشتند؛ استقلال باران در مطالعهٔ ۳ شبکه و ۳ گام احراز نشد. SST ساختمان در تلاش اصلی و آرام‌سازی کمتر همگرا نشد. مرحله D: خطای مدل توان PV حدود ۱٫۸ تا ۴٫۷٪؛ مدل سادهٔ باران حدود ۴۵٪؛ نرخ تجربی شبنم حدود ۹٪ اختلاف تجمعی داشت. این نتایج، اعتبارسنجی کامل محصول نیستند.',
        'متن پیشنهادی ارائه: «یک سامانهٔ محاسباتی قابل‌بازتولید برای مقایسهٔ نمای چرخان آب و انرژی ساخته‌ایم. برداشت آب را با منشأ، تلفات و موجودی و برق را با مصرف واقعی مدل‌شده و مازاد جدا می‌کنیم. سیاست‌ها روی هوای جدید ۲۰۲۵ آزموده شده‌اند. برای عملکرد محصول، تولید سالانه یا برتری RL ادعای تأییدشده نداریم.»',
        'داده و کد همراه: results/presentation/metrics.json، hourly-audited.csv.gz، city-metrics.csv، scenarios.json، wind-screening.csv؛ گزارش C/D/E و منابع در docs باقی‌اند. منبع هوا ERA5 از Open-Meteo و بلندموج NASA POWER است. داده‌های تجربی ETH و Sandia در بسته با منبع و مجوز نگه داشته شده‌اند.'
      ],'note':'گام بعدی برای بستن ادعای محصول: دادهٔ مرزی و آزمایش مستقل کامل، همگرایی مدل جریان/باران و شناسایی سخت‌افزار/مواد. افزودن گام‌های PPO به‌تنهایی این کمبودها را رفع نمی‌کند.'})
    return pages

def html_report(pages):
    def table(data):
        heads,rows=data;return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    parts=['<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><title>گزارش قابل دفاع نما</title><style>body{font-family:Tahoma,Arial,sans-serif;color:#163047;background:#edf3f5;line-height:1.95;margin:0}main{max-width:1150px;margin:auto}section{background:white;margin:26px 0;padding:32px 40px;border-top:5px solid #129388}h1{font-size:25px;margin:0}h2{font-size:14px;color:#527082;font-weight:normal}p{font-size:14px}table{width:100%;border-collapse:collapse;font-size:12px;margin:20px 0}th{background:#18364e;color:white}td,th{border-bottom:1px solid #d9e4e9;padding:9px;text-align:right}tr:nth-child(even){background:#f2f7f8}.note{border-right:4px solid #d99a32;background:#fff8e9;padding:12px;font-size:13px}img{width:100%;max-height:370px;object-fit:contain}@media print{body{background:white}section{break-after:page;border:0;margin:0;padding:10px}}</style><main>']
    for page in pages:
        parts+=['<section><h1>'+html.escape(page['title'])+'</h1><h2>'+html.escape(page['subtitle'])+'</h2>']
        parts+=['<p>'+html.escape(p)+'</p>' for p in page.get('paragraphs',[])]
        if 'table' in page:parts.append(table(page['table']))
        if 'table2' in page:parts.append(table(page['table2']))
        parts+=['<p>'+html.escape(p)+'</p>' for p in page.get('paragraphs_after',[])]
        if 'figure' in page:parts.append('<img alt="Scientific figure" src="data:image/png;base64,'+base64.b64encode((PLOT/page['figure']).read_bytes()).decode()+'">')
        if 'note' in page:parts.append('<div class="note">'+html.escape(page['note'])+'</div>')
        parts.append('</section>')
    parts.append('</main></html>');HTML.parent.mkdir(parents=True,exist_ok=True);HTML.write_text(''.join(parts),encoding='utf-8')

def pdf_report(pages):
    import arabic_reshaper
    from bidi.algorithm import get_display
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape,A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Flowable,Table,TableStyle,Spacer,PageBreak,Image
    font=ROOT/'assets/fonts';pdfmetrics.registerFont(TTFont('Persian',str(font/'DejaVuSans.ttf')));pdfmetrics.registerFont(TTFont('PersianBold',str(font/'DejaVuSans-Bold.ttf')))
    def visual(t):return get_display(arabic_reshaper.reshape(str(t)))
    class RTL(Flowable):
        def __init__(self,text,size=10.1,leading=16,color='#163047',bold=False):super().__init__();self.text=str(text);self.size=size;self.leading=leading;self.font='PersianBold' if bold else 'Persian';self.color=color
        def wrap(self,w,h):
            self.width=w;lines=[]
            for paragraph in self.text.split('\n'):
                line=''
                for word in paragraph.split():
                    candidate=(line+' '+word).strip()
                    if line and pdfmetrics.stringWidth(visual(candidate),self.font,self.size)>w:lines.append(line);line=word
                    else:line=candidate
                lines.append(line)
            self.lines=lines;self.height=len(lines)*self.leading+5;return w,self.height
        def draw(self):
            self.canv.setFillColor(colors.HexColor(self.color));self.canv.setFont(self.font,self.size)
            for i,line in enumerate(self.lines):self.canv.drawRightString(self.width,self.height-(i+1)*self.leading+3,visual(line))
    def tab(data):
        head,rows=data;n=len(head);width=748/n
        cells=[[RTL(x,8.2,12,'#ffffff',True) for x in reversed(head)]]+[[RTL(x,8.0,11.5) for x in reversed(row)] for row in rows]
        t=Table(cells,colWidths=[width]*n,hAlign='RIGHT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#18364e')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f6f7')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),3)]));return t
    PDF.parent.mkdir(parents=True,exist_ok=True);doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=46,rightMargin=46,topMargin=32,bottomMargin=30,title='Water facade - Defensible computational presentation',author='Water Facade Research Project')
    story=[]
    for j,page in enumerate(pages):
        if j:story.append(PageBreak())
        story+=[RTL(page['title'],19,25,bold=True),RTL(page['subtitle'],10,16,color='#39788a'),Spacer(1,7)]
        story += [RTL(p) for p in page.get('paragraphs',[])]
        if 'table' in page:story += [Spacer(1,6),tab(page['table']),Spacer(1,7)]
        story += [RTL(p,9.5,15) for p in page.get('paragraphs_after',[])]
        if 'table2' in page:story += [Spacer(1,5),tab(page['table2'])]
        if 'figure' in page:story += [Spacer(1,2),Image(str(PLOT/page['figure']),width=620,height=173)]
        if 'note' in page:story += [Spacer(1,5),RTL(page['note'],9,14,color='#91641a')]
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#c9d9df'));c.line(46,25,796,25);c.setFont('Persian',8);c.setFillColor(colors.HexColor('#527082'));c.drawString(46,13,'F1 | Conditional numerical evidence | 2025 weather holdout');c.drawRightString(796,13,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

def main():
    argparse.ArgumentParser().parse_args();d=calculate();charts(d);pages=sections(d);html_report(pages);pdf_report(pages)
    report={'metrics':'results/presentation/metrics.json','pdf':str(PDF.relative_to(ROOT)),'html':str(HTML.relative_to(ROOT)),
      'physical_product_qualification':False,'RL_improvement_claim':False,'active_cooling_added':False}
    (OUT/'presentation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

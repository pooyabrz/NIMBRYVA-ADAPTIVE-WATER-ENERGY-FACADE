"""Summarize completed runs and build an inspectable delivery manifest."""
import hashlib,importlib.metadata,json,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from assembly import ROOT,MODEL_VERSION


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    r=ROOT/'results/cassette-v2';s=pd.read_csv(r/'seasonal-summary-h48-dt60-lw0.csv')
    sensitivity=pd.read_csv(r/'parameter-scenarios-2024-07-h24.csv')
    verify=json.loads((r/'verification.json').read_text());audit=json.loads((r/'evidence-audit.json').read_text())
    records={
      'model_version':MODEL_VERSION,'status':'SCREENING_WITH_OPEN_DATA_AND_VALIDATION_GAPS',
      'created_utc':datetime.now(timezone.utc).isoformat(),'scientific_gate':audit['scientific_gate'],
      'checks':verify,'benchmark':{'cities':int(s.city.nunique()),'cases':len(s),
        'hours_each':48,'warmup_hours_each':24,'simulated_case_hours':len(s)*48,
        'scored_case_hours':int(s.scored_hours.sum()),'supported_liquid_regime_windows':int(s.window_water_supported.sum()),
        'unsupported_windows':int((~s.window_water_supported).sum()),
        'annual_yield_or_physical_validation_claimed':False,
        'max_mass_residual_kg':float(s.max_mass_residual_kg.max()),
        'max_battery_residual_Wh':float(s.max_battery_residual_Wh.max()),
        'max_solid_balance_residual_J_per_substep':float(s.max_thermal_residual_J.max()),
        'max_cavity_enthalpy_residual_W':float(s.max_cavity_enthalpy_residual_W.max())},
      'sensitivity':{'city':'singapore','hours_each':24,'warmup_hours_each':12,'scenario_count':len(sensitivity),
        'kind':'one-at-a-time conditional scenarios, no probabilistic confidence interval',
        'conditional_drain_L_range':[float(sensitivity.conditional_drain_L.min()),float(sensitivity.conditional_drain_L.max())]},
      'new_PPO_trained':False,'legacy_policy_compatible':False,'local_viewer_headless_check':json.loads((r/'viewer-check.json').read_text()),
      'unchanged_input_data':json.loads((r/'input-preservation.json').read_text()),
      'python':platform.python_version(),'packages':{x:importlib.metadata.version(x) for x in ['numpy','scipy','pandas','matplotlib','pvlib','gymnasium']},
      'config_sha256':digest(ROOT/'config/model.json')}
    (r/'release.json').write_text(json.dumps(records,indent=2)+'\n')
    # Do not rank unsupported or unvalidated water results as real performance.
    supported=int(s.window_water_supported.sum());fail=len(s)-supported
    text=f'''# نتایج بررسی نسخهٔ کاست ۲

نسخه: `{MODEL_VERSION}`. بررسی‌های عددی و نرم‌افزاری **{verify['tests']} مورد، همگی موفق**. این نتیجه اعتبارسنجی تجربی، CFD یا تأیید دستگاه نیست.

## اجرای دادهٔ واقعی

دادهٔ تمام سال ۲۰۲۴ هر ۱۲ شهر حفظ شده است. اجرای انجام‌شده برای ۱۵ ژوئیه، دو کنترل در هر شهر، هر مورد ۴۸ ساعت با ۲۴ ساعت گرم‌سازی است: **۲۴ مورد، ۱۱۵۲ ساعت محاسبه، ۵۷۶ ساعت امتیازدهی**. اجرای کامل چهار فصل و سالانه در این تحویل انجام نشده است. دادهٔ ورودی سالانه با نتیجهٔ شبیه‌سازی سالانه یکی نیست.

{supported} پنجره در محدودهٔ رژیم محاسباتی آب مایع بودند؛ {fail} پنجره کنار گذاشته شدند. لندن و ریکیاویک میعان روی جدار دیگر کانال داشتند که حل نشده؛ لاپاز وارد دامنهٔ یخ‌زدگی شد. مدل از ادامهٔ محاسبهٔ تشخیصی در این حالت‌ها، نتیجهٔ قابل امتیازدهی نمی‌سازد. برای مقایسهٔ سیاست‌ها فقط اشتراک پنجره‌های پشتیبانی‌شده قابل استفاده است.

| کنترل موازنه در اجرای ۱۲ شهر | بیشینهٔ باقیمانده |
|---|---:|
| جرم آب | {s.max_mass_residual_kg.max():.3g} کیلوگرم |
| باتری | {s.max_battery_residual_Wh.max():.3g} وات‌ساعت |
| انرژی جامدات در هر زیرگام | {s.max_thermal_residual_J.max():.3g} ژول |
| آنتالپی کانال | {s.max_cavity_enthalpy_residual_W.max():.3g} وات |

این‌ها خطای بستن معادلات تقریبی‌اند؛ **درصد خطای عملکرد واقعی نیستند**. حد خطای واقعی بدون اعتبارسنجی انتقال حرارت، اپتیک، جریان و جمع‌آوری قابل استخراج نیست.

خروجی مشروط آب در بسیاری از حالت‌های این پنجره صفر یا کوچک است. نتیجه‌های مثبت اغلب با علامت «سطح خارجی خیس ولی اپتیک خشک» همراه‌اند. بنابراین از جدول آب نمی‌توان ظرفیت واقعی یا برتری قطعی کنترل را نتیجه گرفت. فایل کامل `seasonal-summary-h48-dt60-lw0.csv` و رد زمانی هر مورد نگه‌داری شده است.

## حساسیت‌ها

برای سنگاپور {len(sensitivity)} سناریوی تک‌پارامتری، هرکدام ۲۴ ساعت با ۱۲ ساعت گرم‌سازی اجرا شد. خروجی مشروط جمع‌آوری در این مثال از {sensitivity.conditional_drain_L.min():.6g} تا {sensitivity.conditional_drain_L.max():.6g} لیتر تغییر کرد. این بازه از سناریوهای اختیاری، از جمله کارایی صفر جمع‌کننده و جمعیت قطرات، ساخته شده است؛ فاصلهٔ اطمینان یا کران عملکرد طبیعت نیست. سناریوهای طیف، همرفت و تخلیه می‌توانند علامت/مقدار تولید آب را تغییر دهند. تحلیل چندپارامتری هم‌بسته و حساسیت جهانی انجام نشده است.

همگرایی زمانی ۶۰، ۳۰ و ۱۵ ثانیه برای یک حالت شبِ مرطوب سه‌ساعته ثبت شده است؛ دمای نهایی تقریباً یکسان شد و آب جمع‌شده به‌ترتیب حدود ۰٫۰۳۰۲۸، ۰٫۰۳۰۳۰ و ۰٫۰۳۰۳۱ لیتر بود. در یک حالت تشخیصی جدا با باران ۵۰ میلی‌متر‌بر‌ساعت، اختلاف خروجی تخلیه بین ۶۰ و ۱۵ ثانیه حدود ۰٫۵۳٪ بود؛ همگرایی همهٔ شرایط و پارامترها اثبات نشده است. جزئیات در `timestep-convergence.json` و `storm-timestep-diagnostic.json` است.

همگرایی انتگرال طیفی، کنترل انتگرال سیاه‌جسم، و همگرایی ردیابی پرتو/نمونه‌برداری سطح هم بررسی شدند. این همگرایی‌ها به هندسهٔ تخت مجزا و اپتیک مفروض مربوط‌اند. کانال و پل حرارتی CFD/FE سه‌بعدی ندارند.

## کنترل کیفیت دادهٔ پوشش

میانگین سادهٔ دادهٔ Fig4B حدود ۲۸٫۶۴ و میانگین سادهٔ همهٔ نمونه‌های Fig4C حدود ۲۷٫۶۳ گرم‌بر‌مترمربع‌ساعت است. متن/کپشن مقاله برای Fig4C عدد ۲۸٫۱ گزارش کرده است. منشأ تفاوت پنجره/روش پردازش از دادهٔ موجود مشخص نیست. داده برای انطباق اجباری مقیاس نشده و برای کالیبراسیون تخلیهٔ کاست جدید استفاده نشده است؛ `coating-data-qc.json` را ببینید.

## بازتولید و محدودیت ادعا

نمایش محلی لایه‌ها، حرکت، کنترل‌ها و محاسبه در backend غیرتعاملی فقط برای بررسی برنامه اجرا شد؛ تصویر یا ویدئویی در پاسخ نمایش داده نشده است. PPO جدید آموزش ندیده؛ نسخهٔ قدیمی با ۲۷ ورودی حفظ شده و در مدل ۴۸ورودی رد می‌شود. libRadtran، CFD بیرونی/کانال، حل سه‌بعدی پل حرارتی، مدل قطره‌ای و اعتبارسنجی عمر/موتور نهایی اجرا نشده‌اند. پنج شکاف `evidence-audit.json` باز مانده‌اند و `audit --strict` به‌درستی کد خروج ۲ دارد.

گزارش کامل مشخصات، منابع و داده‌های تکمیلی لازم در `docs/ENGINEERING-fa.md` است. هیچ عدد خروجی به تولید سالانه یا آب آشامیدنی تعمیم داده نشده است.
'''
    (ROOT/'docs/RESULTS-fa.md').write_text(text,encoding='utf-8')
    files=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc' and path!=ROOT/'MANIFEST.json':
            files.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path)})
    (ROOT/'MANIFEST.json').write_text(json.dumps({'model_version':MODEL_VERSION,'scientific_gate':audit['scientific_gate'],'files':files},indent=2)+'\n')
    print(json.dumps({'tests':verify['tests'],'benchmark_cases':len(s),'scenario_count':len(sensitivity),'manifest_files':len(files)},indent=2))


if __name__=='__main__':main()

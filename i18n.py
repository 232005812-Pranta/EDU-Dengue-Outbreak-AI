"""
Offline English / Bangla UI strings (bonus feature). Pure local dictionary: no
translation API, no downloads. Bangla renders with the OS font (e.g. Nirmala UI on Windows).
"""
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

TIER = {
    "en": {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"},
    "bn": {"Low": "নিম্ন", "Medium": "মাঝারি", "High": "উচ্চ", "Critical": "জরুরি"},
}
ACTION_BN = {
    "Low": "নিয়মিত মশা নজরদারি ও জনসচেতনতা কার্যক্রম চালিয়ে যান।",
    "Medium": "লার্ভিসাইড প্রয়োগ করুন ও জমে থাকা ড্রেন পরিষ্কার করুন।",
    "High": "জরুরি ফগিং, হটস্পট নিয়ন্ত্রণ ও স্থানীয় ক্লিনিককে সতর্ক করুন।",
    "Critical": "জরুরি প্রস্তুতি নিন: ব্যাপক মশা নিধন ও হাসপাতালে বেড বরাদ্দ করুন।",
}

S = {
    "lang_label": ("🌐 Language / ভাষা", "🌐 Language / ভাষা"),
    "title": ("🦟 Urban Dengue Outbreak Early-Warning System (2–4 Week Forecast)",
              "🦟 শহুরে ডেঙ্গু প্রাদুর্ভাব আগাম সতর্কতা ব্যবস্থা (২–৪ সপ্তাহ আগে)"),
    "caption": ("Hyper-local ward risk engine modeling the 2-to-4-week Aedes aegypti breeding cycle, "
                "water-logging vulnerability, and sanitation scores across Bangladesh cities. "
                "Forecasts use ONLY data observed up to the reference week — no future weather or case data.",
                "এডিস মশার ২–৪ সপ্তাহের প্রজনন চক্র, জলাবদ্ধতা ও স্যানিটেশন বিবেচনায় ওয়ার্ড-ভিত্তিক ঝুঁকি পূর্বাভাস। "
                "শুধু রেফারেন্স সপ্তাহ পর্যন্ত জানা তথ্য ব্যবহার হয় — ভবিষ্যতের আবহাওয়া বা রোগীর তথ্য নয়।"),
    "sb_target": ("🎯 Target Area & Forecast Window", "🎯 এলাকা ও পূর্বাভাসের সময়"),
    "city": ("Select City", "সিটি নির্বাচন করুন"),
    "ward": ("Select Ward ID", "ওয়ার্ড নির্বাচন করুন"),
    "ref_week": ("Forecast Issued On (Reference Week)", "পূর্বাভাস দেওয়ার সপ্তাহ (রেফারেন্স)"),
    "ref_help": ("The last week of observed data. Pick the final week to see a genuine forecast into the unseen future.",
                 "যে সপ্তাহ পর্যন্ত তথ্য জানা। শেষ সপ্তাহ বাছলে সত্যিকারের ভবিষ্যৎ পূর্বাভাস দেখাবে।"),
    "horizon": ("Early-Warning Horizon (Weeks Ahead)", "কত সপ্তাহ আগের সতর্কতা"),
    "sb_weather": ("🌦️ Weather Scenario Simulator (What-If)", "🌦️ আবহাওয়া পরিস্থিতি সিমুলেটর (যদি-তবে)"),
    "sb_weather_cap": ("Adjust rainfall, temperature, or humidity to simulate climate shocks.",
                       "বৃষ্টি, তাপমাত্রা বা আর্দ্রতা বদলে প্রভাব দেখুন।"),
    "rain_mult": ("Recent Rainfall Multiplier (Last 4 Weeks)", "সাম্প্রতিক বৃষ্টির গুণক (গত ৪ সপ্তাহ)"),
    "temp_shift": ("Temperature Shift (°C)", "তাপমাত্রা পরিবর্তন (°C)"),
    "hum_shift": ("Relative Humidity Shift (%)", "আর্দ্রতা পরিবর্তন (%)"),
    "sb_ops": ("🚑 Operations Command (Phase 2)", "🚑 অপারেশন কমান্ড (ফেজ ২)"),
    "sb_ops_cap": ("Applies to all 60 wards for the reference week & horizon above.",
                   "উপরের সপ্তাহ ও সময়ের জন্য সব ৬০টি ওয়ার্ডে প্রযোজ্য।"),
    "stock": ("Available Regional Stock (% of demand)", "আঞ্চলিক মজুদ (চাহিদার %)"),
    "crews": ("Total Spraying Crews Available", "মোট স্প্রে টিম"),
    "alloc_rule": ("Scarcity allocation rule", "ঘাটতিতে বণ্টনের নিয়ম"),
    "alloc_help": ("weighted = everyone gets a share, high-risk/dense wards get more; "
                   "triage = strict priority order, a ward is fully served before the next.",
                   "weighted = সবাই কিছু পায়, ঝুঁকিপূর্ণ/ঘনবসতি ওয়ার্ড বেশি পায়; "
                   "triage = কঠোর অগ্রাধিকার, একটি ওয়ার্ড পুরো পেলে তবেই পরেরটি।"),
    "basis": ("Planning basis", "পরিকল্পনার ভিত্তি"),
    "basis_point": ("Expected cases", "সম্ভাব্য রোগী"),
    "basis_upper": ("90th pct (safety stock)", "৯০% সীমা (নিরাপত্তা মজুদ)"),
    "cap_mult": ("Bed Capacity Multiplier (stress test)", "বেড সক্ষমতা গুণক (স্ট্রেস টেস্ট)"),
    "efficacy": ("Crew efficacy (breeding reduction per crew)", "প্রতি টিমে প্রজনন হ্রাসের হার"),
    "kpi_target": ("+{h}W Target Week ({d})", "+{h} সপ্তাহ পরের সপ্তাহ ({d})"),
    "kpi_cases": ("{v:.1f} cases", "{v:.1f} জন রোগী"),
    "kpi_tier": ("Outbreak Risk Tier", "প্রাদুর্ভাবের ঝুঁকি"),
    "kpi_rain": ("Rainfall, Last 3 Weeks (feeds 2–4W breeding window)", "গত ৩ সপ্তাহের বৃষ্টি (২–৪ সপ্তাহের প্রজনন)"),
    "kpi_rain_delta": ("Cases this week: {c} | 4W avg {a:.1f}", "এই সপ্তাহে রোগী: {c} | ৪ সপ্তাহের গড় {a:.1f}"),
    "kpi_latency": ("Local Inference Latency", "লোকাল হিসাবের সময়"),
    "kpi_api": ("0 External API Calls", "০টি বাইরের API কল"),
    "action": ("**Recommended Public Health Action ({w}):** {a}", "**প্রস্তাবিত জনস্বাস্থ্য পদক্ষেপ ({w}):** {a}"),
    "tab1": ("📊 Ward Forecast & Weather Drivers", "📊 ওয়ার্ড পূর্বাভাস"),
    "tab2": ("🗺️ City-Wide Ward Risk Heatmap", "🗺️ সিটি হিটম্যাপ"),
    "tab3": ("📈 Chronological Validation (MAE / RMSE)", "📈 মডেল যাচাই"),
    "tab4": ("🚑 Resource Allocation", "🚑 সরবরাহ বণ্টন"),
    "tab5": ("🏥 Hospital Overflow Routing", "🏥 হাসপাতাল ওভারফ্লো"),
    "tab6": ("🦟 Vector Control Simulator", "🦟 মশা নিয়ন্ত্রণ সিমুলেটর"),
    "tab7": ("📨 Dispatch Notices", "📨 নির্দেশনা বার্তা"),
    "tab8": ("📍 Offline Ward Risk Map", "📍 অফলাইন ঝুঁকি মানচিত্র"),
    # Phase 2 tab texts
    "alloc_title": ("Constrained Resource Allocation — all 60 wards, target week {d} (+{h}W)",
                    "সীমিত সরবরাহ বণ্টন — ৬০টি ওয়ার্ড, লক্ষ্য সপ্তাহ {d} (+{h})"),
    "alloc_cap": ("Supply ratios learned from 2016–2023 data: saline {s:.3f}, kits {k:.3f}, beds {b:.3f} per case. "
                  "Each city depot holds {p}% of its demand; rule = {m}; priority = risk-tier weight × population density × case load.",
                  "২০১৬–২০২৩ তথ্য থেকে শেখা অনুপাত (প্রতি রোগীতে): স্যালাইন {s:.3f}, কিট {k:.3f}, বেড {b:.3f}। "
                  "প্রতিটি সিটির ডিপোতে চাহিদার {p}% মজুদ; নিয়ম = {m}; অগ্রাধিকার = ঝুঁকি × জনঘনত্ব × রোগীর সংখ্যা।"),
    "saline": ("Saline bags", "স্যালাইন ব্যাগ"),
    "kits": ("Diagnostic kits", "পরীক্ষার কিট"),
    "beds": ("Hospital beds", "হাসপাতাল বেড"),
    "crews_kpi": ("Spraying crews", "স্প্রে টিম"),
    "deficit_wards": ("Wards in supply deficit", "ঘাটতির ওয়ার্ড"),
    "deficit_delta": ("{n} deficit", "{n} ঘাটতি"),
    "fully_met": ("fully met", "সম্পূর্ণ পূরণ"),
    "hc": ("{n} High/Critical", "{n}টি উচ্চ/জরুরি"),
    "crit_alert": ("🚨 CRITICAL SUPPLY DEFICIT: {n} High/Critical-risk ward(s) cannot be 100% supplied — ",
                   "🚨 জরুরি সরবরাহ ঘাটতি: {n}টি উচ্চ/জরুরি ঝুঁকির ওয়ার্ডে চাহিদা পুরো পূরণ সম্ভব নয় — "),
    "partial_alert": ("⚠️ {n} lower-risk ward(s) are partially supplied.", "⚠️ {n}টি কম ঝুঁকির ওয়ার্ড আংশিক সরবরাহ পাচ্ছে।"),
    "ok_alert": ("✅ All ward supply demand is 100% fulfilled at the current stock level.",
                 "✅ বর্তমান মজুদে সব ওয়ার্ডের চাহিদা ১০০% পূরণ হচ্ছে।"),
    "city_only": ("Show selected city only", "শুধু নির্বাচিত সিটি দেখান"),
    "dl_alloc": ("⬇️ Download allocation plan (CSV)", "⬇️ বণ্টন পরিকল্পনা ডাউনলোড (CSV)"),
    "overflow_title": ("Hospital Surge & Intra-City Overflow Routing — target week {d}",
                       "হাসপাতালের চাপ ও সিটির ভেতরে রোগী স্থানান্তর — লক্ষ্য সপ্তাহ {d}"),
    "overflow_cap": ("Local dengue bed capacity is derived from history (city pool = 75th-pct monsoon bed load, "
                     "split across wards by population density). Overflow is routed only within the same city.",
                     "বেড সক্ষমতা অতীতের তথ্য থেকে নির্ণয় (বর্ষার ৭৫% চাহিদা, জনঘনত্ব অনুযায়ী ভাগ)। "
                     "রোগী শুধু একই সিটির ভেতরে স্থানান্তর হয়।"),
    "city_over": ("🚨 {c}: city bed pool over capacity by {n} bed(s) — escalate to divisional hospitals.",
                  "🚨 {c}: সিটির বেড {n}টি কম — বিভাগীয় হাসপাতালে জানান।"),
    "transfers": ("Transfer orders — {c}", "স্থানান্তর নির্দেশ — {c}"),
    "no_transfer": ("No intra-city transfers needed for this city and week.", "এই সপ্তাহে কোনো স্থানান্তর লাগবে না।"),
    "ledger": ("Ward bed ledger (net deficits & headroom)", "ওয়ার্ডের বেড হিসাব (ঘাটতি ও খালি বেড)"),
    "sim_title": ("What-If Vector Control Impact Simulator (model-driven counterfactual)",
                  "মশা নিয়ন্ত্রণ সিমুলেটর (মডেল-ভিত্তিক যদি-তবে)"),
    "plan_mode": ("Deployment plan", "টিম পাঠানোর পরিকল্পনা"),
    "auto": ("ROI-optimised (auto)", "সর্বোচ্চ ফলাফল (স্বয়ংক্রিয়)"),
    "manual": ("Manual", "নিজে বাছাই"),
    "sim_wards": ("Wards to deploy to", "যে ওয়ার্ডে টিম যাবে"),
    "sim_per": ("Crews per selected ward", "প্রতি ওয়ার্ডে টিম"),
    "crews_deployed": ("Crews deployed", "পাঠানো টিম"),
    "prevented": ("Cases prevented (wk +2…+4)", "প্রতিরোধিত রোগী (২–৪ সপ্তাহ)"),
    "beds_freed": ("Hospital bed-weeks freed", "খালি হওয়া বেড"),
    "saline_saved": ("Saline bags saved", "সাশ্রয়ী স্যালাইন"),
    "plan_table": ("Deployment plan ranked by ROI (cases prevented per crew)", "ROI অনুযায়ী পরিকল্পনা (প্রতি টিমে প্রতিরোধিত রোগী)"),
    "roi_table": ("Ward ROI ranking — first crew", "ওয়ার্ড ROI তালিকা — প্রথম টিম"),
    "dispatch_title": ("Multi-Channel Dispatch Notices (template-based, works offline)",
                       "নির্দেশনা বার্তা (টেমপ্লেট-ভিত্তিক, অফলাইনে চলে)"),
    "brief_dl": ("⬇️ Printable commissioner brief (HTML → Print → Save as PDF)",
                 "⬇️ কমিশনার ব্রিফ ডাউনলোড (HTML → Print → PDF)"),
    "sms_dl": ("⬇️ All SMS for this city (TXT)", "⬇️ এই সিটির সব SMS (TXT)"),
    "brief_preview": ("Preview commissioner brief", "কমিশনার ব্রিফ দেখুন"),
    # Map
    "map_title": ("Offline Ward Risk Map — target week {d} (+{h}W)", "অফলাইন ওয়ার্ড ঝুঁকি মানচিত্র — লক্ষ্য সপ্তাহ {d} (+{h})"),
    "map_cap": ("Drawn locally with no map tiles or internet. City positions are approximate; ward positions are a "
                "schematic layout (the dataset has no ward coordinates). Colour = risk tier, size = predicted cases, "
                "thick black ring = supply deficit, blue arrows = overflow patient transfers.",
                "ইন্টারনেট বা ম্যাপ টাইল ছাড়া লোকালি আঁকা। সিটির অবস্থান আনুমানিক, ওয়ার্ডের অবস্থান প্রতীকী "
                "(ডেটাসেটে ওয়ার্ডের স্থানাঙ্ক নেই)। রং = ঝুঁকি, আকার = সম্ভাব্য রোগী, "
                "মোটা কালো বৃত্ত = সরবরাহ ঘাটতি, নীল তীর = রোগী স্থানান্তর।"),
    "map_scope": ("Map scope", "মানচিত্রের পরিসর"),
    "all_cities": ("All 5 cities", "৫টি সিটি"),
    "sel_city": ("Selected city", "নির্বাচিত সিটি"),
    "m_cases": ("Predicted cases", "সম্ভাব্য রোগী"),
    "m_fulfil": ("Supply fulfilment", "সরবরাহ পূরণ"),
    "m_crews": ("Crews", "টিম"),
    "m_lon": ("Longitude (schematic)", "দ্রাঘিমা (প্রতীকী)"),
    "m_lat": ("Latitude (schematic)", "অক্ষাংশ (প্রতীকী)"),
}


def t(key: str, lang: str = "en", **fmt) -> str:
    en, bn = S[key]
    text = bn if lang == "bn" else en
    return text.format(**fmt) if fmt else text


def num(x, lang: str = "en") -> str:
    """Render digits in Bangla numerals when the UI is in Bangla."""
    return str(x).translate(BN_DIGITS) if lang == "bn" else str(x)

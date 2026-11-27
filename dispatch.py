"""
Phase 2 Requirement 4: template-based, multi-channel dispatch notices.

Pure string templates (no LLM, no network) -> identical output offline and online.
Channels: SMS (short, EN + BN), WhatsApp (formatted), printable HTML commissioner brief.
"""
import html
import os

import pandas as pd

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
TIER_BN = {"Low": "নিম্ন", "Medium": "মাঝারি", "High": "উচ্চ", "Critical": "জরুরি"}
TIER_COLOR = {"Low": "#16a34a", "Medium": "#ca8a04", "High": "#ea580c", "Critical": "#dc2626"}


def bn(x) -> str:
    return str(x).translate(BN_DIGITS)


def _ward_context(alloc_row, beds_row, transfers: pd.DataFrame, crews_plan: dict):
    w = alloc_row["ward_id"]
    out_t = transfers[transfers["from_ward"] == w] if len(transfers) else transfers
    in_t = transfers[transfers["to_ward"] == w] if len(transfers) else transfers
    return {
        "ward": w,
        "city": alloc_row["city"],
        "tier": alloc_row["risk_tier"],
        "target": pd.Timestamp(alloc_row["target_date"]).strftime("%d %b %Y"),
        "target_num": pd.Timestamp(alloc_row["target_date"]).strftime("%d/%m/%Y"),
        "horizon": int(alloc_row["horizon"]),
        "cases": float(alloc_row["predicted_cases"]),
        "lo": float(alloc_row["ci_lower"]),
        "hi": float(alloc_row["ci_upper"]),
        "saline": (int(alloc_row["saline_bags_needed_allocated"]), int(alloc_row["saline_bags_needed"])),
        "kits": (int(alloc_row["diagnostic_kits_needed_allocated"]), int(alloc_row["diagnostic_kits_needed"])),
        "beds": (int(beds_row["bed_capacity"]), int(beds_row["hospital_beds_required"])),
        "crews": int(crews_plan.get(w, alloc_row["crews_allocated"])),
        "crews_req": int(alloc_row["crews_required"]),
        "deficit": bool(alloc_row["supply_deficit"]),
        "out": [(r.to_ward, int(r.patients)) for r in out_t.itertuples()],
        "in": [(r.from_ward, int(r.patients)) for r in in_t.itertuples()],
        "unresolved": int(beds_row["net_unresolved_deficit"]),
        "action": alloc_row["recommended_action"],
    }


def sms_text(c: dict) -> dict:
    s_a, s_r = c["saline"]
    k_a, k_r = c["kits"]
    flag_en = "DEFICIT" if c["deficit"] else "OK"
    route_en = ""
    if c["out"]:
        route_en = " Route " + ",".join(f"{n}pt->{t}" for t, n in c["out"]) + "."
    en = (f"[DENGUE {c['tier'].upper()}] {c['ward']} +{c['horizon']}w ({c['target']}): "
          f"{c['cases']:.0f} cases. Saline {s_a}/{s_r}, Kits {k_a}/{k_r}, "
          f"Crews {c['crews']}. Supply {flag_en}.{route_en}")
    route_bn = ""
    if c["out"]:
        route_bn = " রোগী পাঠান: " + ", ".join(f"{bn(n)} জন→{t}" for t, n in c["out"]) + "।"
    bn_txt = (f"[ডেঙ্গু {TIER_BN[c['tier']]} ঝুঁকি] {c['ward']}: {bn(c['target_num'])} সপ্তাহে "
              f"সম্ভাব্য রোগী {bn(round(c['cases']))} জন। স্যালাইন {bn(s_a)}/{bn(s_r)}, "
              f"কিট {bn(k_a)}/{bn(k_r)}, স্প্রে টিম {bn(c['crews'])}টি।"
              f"{' সরবরাহ ঘাটতি!' if c['deficit'] else ''}{route_bn}")
    return {"en": en, "bn": bn_txt}


def whatsapp_text(c: dict) -> str:
    s_a, s_r = c["saline"]
    k_a, k_r = c["kits"]
    cap, req = c["beds"]
    lines = [
        f"*🦟 DENGUE DISPATCH — {c['ward']} ({c['city']})*",
        f"Risk tier: *{c['tier'].upper()}* | ঝুঁকি: *{TIER_BN[c['tier']]}*",
        f"Target week (+{c['horizon']}w): {c['target']}",
        f"Predicted cases: *{c['cases']:.1f}* (80% CI {c['lo']:.1f}–{c['hi']:.1f})",
        "",
        "*Supplies (allocated / required)*",
        f"• Saline bags: {s_a} / {s_r}{'  ⚠️' if s_a < s_r else ''}",
        f"• Diagnostic kits: {k_a} / {k_r}{'  ⚠️' if k_a < k_r else ''}",
        f"• Hospital beds: need {req}, local capacity {cap}",
        f"• Spraying crews assigned: {c['crews']} (SOP need {c['crews_req']})",
    ]
    if c["out"]:
        lines.append("*Overflow routing:* " + "; ".join(f"send {n} patient(s) → {t}" for t, n in c["out"]))
    if c["in"]:
        lines.append("*Receive overflow:* " + "; ".join(f"{n} from {f}" for f, n in c["in"]))
    if c["unresolved"]:
        lines.append(f"🚨 *{c['unresolved']} bed(s) still unresolved in city pool — escalate to DGHS*")
    if c["deficit"]:
        lines.append("⚠️ *SUPPLY DEFICIT: demand not fully met — request emergency restock*")
    lines += ["", f"Action: {c['action']}", "করণীয়: অবিলম্বে উপরের নির্দেশনা অনুযায়ী ব্যবস্থা নিন।"]
    return "\n".join(lines)


def commissioner_brief_html(city: str, alloc: pd.DataFrame, beds: pd.DataFrame, transfers: pd.DataFrame,
                            city_summary: pd.DataFrame, crews_plan: dict, meta: dict) -> str:
    a = alloc[alloc["city"] == city].sort_values("priority_rank")
    b = beds.set_index("ward_id")
    cs = city_summary[city_summary["city"] == city].iloc[0]
    rows = []
    for r in a.itertuples():
        br = b.loc[r.ward_id]
        crews = int(crews_plan.get(r.ward_id, r.crews_allocated))
        def cell(al, rq):
            style = ' style="color:#b91c1c;font-weight:700"' if al < rq else ""
            return f"<td{style}>{al} / {rq}</td>"
        rows.append(
            f"<tr><td>{r.priority_rank}</td><td>{html.escape(r.ward_id)}</td>"
            f"<td style='color:{TIER_COLOR[r.risk_tier]};font-weight:700'>{r.risk_tier}</td>"
            f"<td>{r.predicted_cases:.1f}<br><small>[{r.ci_lower:.1f}–{r.ci_upper:.1f}]</small></td>"
            + cell(r.saline_bags_needed_allocated, r.saline_bags_needed)
            + cell(r.diagnostic_kits_needed_allocated, r.diagnostic_kits_needed)
            + f"<td>{int(br.hospital_beds_required)} / {int(br.bed_capacity)}</td>"
            f"<td>{int(br.transferred_out)} out · {int(br.received_in)} in</td>"
            f"<td>{crews}</td>"
            f"<td>{'⚠️ DEFICIT' if r.supply_deficit else '✓'}</td></tr>"
        )
    t = transfers[transfers["city"] == city] if len(transfers) else transfers
    routes = "".join(f"<li>{n} patient(s): {html.escape(f)} → {html.escape(to)}</li>"
                     for f, to, n in zip(t.get("from_ward", []), t.get("to_ward", []), t.get("patients", [])))
    deficits = int(a["supply_deficit"].sum())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Commissioner Brief — {html.escape(city)}</title>
<style>
body{{font-family:'Noto Sans Bengali','Segoe UI',Arial,sans-serif;margin:24px;color:#111;background:#fff}}
h1{{color:#b91c1c;margin:0}} .meta{{color:#555;font-size:13px;margin:4px 0 14px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}} th,td{{border:1px solid #ccc;padding:5px 7px;text-align:left}}
th{{background:#fef2f2}} .kpi{{display:flex;gap:12px;margin:10px 0}} .kpi div{{border:1px solid #ddd;border-radius:8px;padding:8px 12px}}
.alert{{background:#fef2f2;border-left:5px solid #dc2626;padding:8px 12px;margin:10px 0}}
@media print{{button{{display:none}}}}
</style></head><body>
<button onclick="window.print()">🖨️ Print / Save as PDF</button>
<h1>Dengue Dispatch — Commissioner Brief: {html.escape(city)}</h1>
<div class="meta">Forecast issued {meta['origin']} · Target week +{meta['horizon']}w ({meta['target']}) ·
Regional stock {meta['stock_pct']}% · Planning basis: {meta['basis']} · Generated offline (deterministic templates)</div>
<div class="kpi"><div><b>{int(a['predicted_cases'].sum())}</b><br>predicted cases</div>
<div><b>{deficits}</b><br>wards with supply deficit</div>
<div><b>{cs['city_utilization_pct']}%</b><br>city bed utilisation</div>
<div><b>{int(cs['patients_rerouted'])}</b><br>patients rerouted</div>
<div><b>{int(cs['remaining_headroom'])}</b><br>beds headroom left</div></div>
{"<div class='alert'>🚨 City bed pool over capacity: " + str(int(cs['unresolved_deficit'])) + " bed(s) unresolved — escalate to DGHS / divisional hospitals.</div>" if cs['unresolved_deficit'] > 0 else ""}
{"<div class='alert'>⚠️ Critical supply deficit in " + str(deficits) + " ward(s): allocated stock below required demand.</div>" if deficits else ""}
<table><tr><th>#</th><th>Ward</th><th>Tier</th><th>Pred. cases</th><th>Saline alloc/req</th><th>Kits alloc/req</th>
<th>Beds req/cap</th><th>Overflow</th><th>Crews</th><th>Status</th></tr>{''.join(rows)}</table>
<h3>Overflow routing orders</h3><ul>{routes or '<li>No transfers needed.</li>'}</ul>
<p style="font-size:12px;color:#555">ঝুঁকিপূর্ণ ওয়ার্ডে অবিলম্বে স্প্রে টিম পাঠান, ঘাটতির তালিকা জেলা সিভিল সার্জনকে জানান।</p>
</body></html>"""


def build_ward_notices(alloc: pd.DataFrame, beds: pd.DataFrame, transfers: pd.DataFrame, crews_plan: dict) -> list:
    b = beds.set_index("ward_id")
    notices = []
    for _, r in alloc.iterrows():
        c = _ward_context(r, b.loc[r["ward_id"]], transfers, crews_plan)
        sms = sms_text(c)
        notices.append({"ward_id": c["ward"], "city": c["city"], "tier": c["tier"],
                        "deficit": c["deficit"], "sms_en": sms["en"], "sms_bn": sms["bn"],
                        "whatsapp": whatsapp_text(c)})
    return notices


def write_exports(out_dir: str, ops: dict, crews_plan: dict, meta: dict, top_n: int = 10) -> list:
    """Writes sample SMS / WhatsApp / printable HTML briefs under out_dir (default /exports)."""
    os.makedirs(out_dir, exist_ok=True)
    alloc, beds, transfers, city = ops["allocation"], ops["beds"], ops["transfers"], ops["city"]
    notices = build_ward_notices(alloc, beds, transfers, crews_plan)
    # sample = the 5 highest-priority wards + the 5 most important wards left in deficit
    top = notices[: top_n // 2]
    short = [n for n in notices if n["deficit"] and n not in top][: top_n - len(top)]
    focus = top + short
    written = []
    with open(os.path.join(out_dir, "sms_alerts.txt"), "w", encoding="utf-8") as f:
        for n in focus:
            f.write(n["sms_en"] + "\n" + n["sms_bn"] + "\n\n")
    with open(os.path.join(out_dir, "whatsapp_alerts.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(n["whatsapp"] for n in focus))
    written += ["sms_alerts.txt", "whatsapp_alerts.txt"]
    for c in sorted(alloc["city"].unique()):
        name = f"commissioner_brief_{c.replace(' ', '_')}.html"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(commissioner_brief_html(c, alloc, beds, transfers, city, crews_plan, meta))
        written.append(name)
    alloc.to_csv(os.path.join(out_dir, "allocation_plan.csv"), index=False)
    transfers.to_csv(os.path.join(out_dir, "overflow_transfers.csv"), index=False)
    written += ["allocation_plan.csv", "overflow_transfers.csv"]
    return written

"""올림픽공원 주차 만차 예측 & 유도 최적화 대시보드
실행: streamlit run app.py
"""
import json, pickle, datetime as dt
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go, plotly.express as px
from sim_core import ZONES, CAP, load_params, seasonal_factor, simulate
from opt_lp import build_demand, solve, occupancy, GATE, GATES

st.set_page_config(page_title="올림픽공원 주차 만차 예측·유도", layout="wide", page_icon="🅿️")

@st.cache_resource
def load():
    prof, durs = load_params()
    C = json.load(open("calib.json"))
    M = pickle.load(open("model1.pkl", "rb"))
    return prof, durs, C, M
prof, durs, C, M = load()

VENUES = {"KSPO DOME": ["P5", "P4", "P6", "P2"], "티켓링크 라이브 아레나(핸드볼)": ["P6", "P7", "P5", "P1A"],
          "올림픽홀": ["P2", "P1B", "P11", "P5"], "우리금융아트홀": ["P7", "P6", "P1A", "P2"]}

# ---------------- 사이드바 입력 ----------------
st.sidebar.header("📅 날짜·조건")
day = st.sidebar.date_input("방문 날짜", dt.date(2026, 5, 16))
weekend = day.weekday() >= 5
temp = st.sidebar.slider("낮(10~16시) 평균기온 ℃", -10, 38, 18)
rain = st.sidebar.checkbox("낮 시간 비 예보", False)
st.sidebar.header("🎤 공연")
has_ev = st.sidebar.checkbox("공연 있음", True)
venue = st.sidebar.selectbox("공연장", list(VENUES), disabled=not has_ev)
ev_hour = st.sidebar.slider("공연 시작 시각", 12, 20, 18, disabled=not has_ev)
ev_cars = st.sidebar.number_input("관객 차량 수(가정)", 100, 3000, 800, 100, disabled=not has_ev,
                                  help="KSPO DOME 1.5만석 × 자차 이용률 약 5% 가정. 실측값 확보 시 수정")
st.sidebar.header("🚦 유도 정책")
pol_zones = st.sidebar.multiselect("일반 차량 진입 제한 구역", ZONES, default=VENUES[venue][:2] if has_ev else [])
pol_ratio = st.sidebar.slider("제한 구역 일반 차량 상한(%)", 0, 100, 30, 5) / 100
pol_from = st.sidebar.slider("제한 시작 시각", 8, 20, 14)
n_rep = st.sidebar.slider("시뮬레이션 반복 횟수", 5, 30, 10)

event = dict(zone_pref=VENUES[venue], cars=int(ev_cars), start_hour=ev_hour) if has_ev else None
month = day.month
factor = seasonal_factor(month)

st.title("🅿️ 올림픽공원 주차 만차 예측 & 유도 최적화")
st.caption(f"{day:%Y-%m-%d} ({'주말' if weekend else '평일'}) · 계절계수 {factor:.2f} · "
           "데이터: 한국체육산업개발 입출차·주차장 혼잡도·월별주차현황·영문 공연정보 + Open-Meteo 기상")

# ---------------- ① 만차 위험도 ----------------
P = M["params"]
def risk(T, ev, rn, sat):
    z = P["Intercept"] + P["t2"] * (T - M["center"]) ** 2 + P["ev"] * ev + P["rain"] * rn + P["sat"] * sat
    return 1 / (1 + np.exp(-z))
c1, c2, c3 = st.columns([1, 1, 2])
if weekend:
    pr = risk(temp, int(has_ev), int(rain), int(day.weekday() == 5))
    c1.metric("주말 만차 발생 확률", f"{pr*100:.0f}%")
    lvl = "🔴 높음" if pr > .7 else ("🟠 보통" if pr > .4 else "🟢 낮음")
    c2.metric("위험 등급", lvl)
    Ts = np.arange(-10, 39)
    fig = go.Figure(go.Scatter(x=Ts, y=risk(Ts, int(has_ev), int(rain), int(day.weekday() == 5)) * 100, mode="lines"))
    fig.add_vline(x=temp, line_dash="dot"); fig.update_layout(height=200, margin=dict(t=10, b=10), xaxis_title="기온 ℃", yaxis_title="만차확률 %")
    c3.plotly_chart(fig, width='stretch')
else:
    c1.metric("주말 만차 확률 모델", "평일 미적용")
    c3.info("평일 만차는 2025년 기록상 공연일에만 발생(공연일 10%, 비공연일 0%). 아래 시뮬레이션으로 확인하세요.")

# ---------------- ② 시뮬레이션 ----------------
@st.cache_data(show_spinner="시뮬레이션 중…")
def run_sims(weekend, factor, event_key, policy_key, pol_from, n):
    ev = json.loads(event_key) if event_key else None
    pol = json.loads(policy_key) if policy_key else None
    out = []
    for sd in range(n):
        f = factor * np.random.default_rng(500 + sd).uniform(0.9, 1.1)
        r = simulate(prof, durs, weekend, f, attract=C["attract"], event=ev, seed=sd, share=C["share"],
                     shift=C["shift"] if weekend else 0, policy=pol, policy_from=pol_from)
        out.append(r)
    occ = sum(r["occ"] for r in out) / n
    ft = {z: [r["full_t"][z] for r in out if r["full_t"][z] is not None] for z in ZONES}
    summ = dict(rej=np.mean([r["rejected"] for r in out]),
                ev_hit=np.mean([r["ev_hit"] / max(r["ev_n"], 1) for r in out]) if ev else None,
                ev_rej=np.mean([r["ev_rej"] for r in out]) if ev else None,
                rej_t=np.concatenate([r["rej_t"] for r in out]))
    return occ, ft, summ

ek = json.dumps(event) if event else ""
pk = json.dumps({z: pol_ratio for z in pol_zones}) if pol_zones else ""
occ0, ft0, s0 = run_sims(weekend, factor, ek, "", 0, n_rep)
occ1, ft1, s1 = run_sims(weekend, factor, ek, pk, pol_from, n_rep) if pk else (occ0, ft0, s0)

tab1, tab2, tab3 = st.tabs(["📈 구역별 혼잡 예측", "🧮 최적 유도안 (Gurobi LP)", "⚖️ 정책 비교"])
with tab1:
    st.subheader("구역별 점유율 (무유도, 시뮬레이션 평균)")
    df = (occ0 / pd.Series(CAP) * 100).reset_index().melt(id_vars="index", var_name="구역", value_name="점유율")
    fig = px.line(df[(df["index"] >= 6) & (df["index"] <= 23)], x="index", y="점유율", color="구역")
    fig.add_hline(y=100, line_dash="dot"); fig.update_layout(height=420, xaxis_title="시각", yaxis_title="점유율 %")
    if event: fig.add_vline(x=ev_hour, line_color="red", annotation_text="공연 시작")
    st.plotly_chart(fig, width='stretch')
    tbl = pd.DataFrame({"구역": ZONES, "출입문": [GATE[z] for z in ZONES], "면수": [CAP[z] for z in ZONES],
                        "만차 확률": [len(ft0[z]) / n_rep for z in ZONES],
                        "예상 만차 시각(중앙값)": [f"{int(np.median(ft0[z]))}:{int(np.median(ft0[z])%1*60):02d}" if ft0[z] else "-" for z in ZONES]})
    st.dataframe(tbl.style.format({"만차 확률": "{:.0%}"}).background_gradient(subset=["만차 확률"], cmap="Reds"),
                 hide_index=True, width='stretch')
    st.caption(f"하루 평균 회차(전 구역 만차로 진입 실패) {s0['rej']:.0f}대 · P1A/P1B는 P1 562면 반분, P8~P10은 150면 가정")

with tab2:
    st.subheader("출입문→구역 최적 배분 (1시간 단위 LP)")
    T, park, evd, S, Sev = build_demand(prof, durs, C, month, weekend, event)
    base = solve(T, park, evd, S, Sev, ev_pref=tuple(event["zone_pref"][:2]) if event else ("P5", "P4"), guide=False)
    opt = solve(T, park, evd, S, Sev, ev_pref=tuple(event["zone_pref"][:2]) if event else ("P5", "P4"), guide=True)
    a, b, c, d = st.columns(4)
    a.metric("회차(무유도→유도)", f"{opt['R'].sum():.0f}대", f"{opt['R'].sum()-base['R'].sum():+.0f}", delta_color="inverse")
    if event:
        pref = [ZONES.index(z) for z in event["zone_pref"][:2]]
        nb, no = base["Y"][pref].sum(), opt["Y"][pref].sum()
        b.metric("공연 관객 근접구역 배정", f"{no:.0f}대", f"{no-nb:+.0f}")
        c.metric("공연 관객 회차", f"{opt['RE'].sum():.0f}대", f"{opt['RE'].sum()-base['RE'].sum():+.0f}", delta_color="inverse")
    X = opt["X"].sum(2)
    moved = sum(X[g, z] for g, gn in enumerate(GATES) for z, zn in enumerate(ZONES) if GATE[zn] != gn)
    d.metric("타 출입문 구역으로 재유도", f"{moved:.0f}대", f"{moved/park.sum()*100:.1f}% of 일반")
    occ, occ_ev = occupancy(opt, S, Sev, T)
    hrs = (T / 60).astype(int)
    heat = pd.DataFrame((occ + occ_ev) / np.array([CAP[z] for z in ZONES])[:, None] * 100, index=ZONES, columns=hrs)
    fig = px.imshow(heat, color_continuous_scale="RdYlGn_r", zmin=0, zmax=100, aspect="auto",
                    labels=dict(x="시각", y="구역", color="점유율%"))
    fig.update_layout(height=380); st.plotly_chart(fig, width='stretch')
    flow = pd.DataFrame(X, index=GATES, columns=ZONES).round(0)
    st.markdown("**출입문별 → 구역 배정량(대/일)**"); st.dataframe(flow, width='stretch')
    if event:
        pz = event["zone_pref"][:2]
        po = {z: [occ[ZONES.index(z), k] / CAP[z] for k in range(len(T))] for z in pz}
        stop = [int(hrs[k]) for k in range(1, len(T)) if all(po[z][k] < po[z][k-1] - 0.05 for z in pz) and hrs[k] >= 12]
        if stop: st.success(f"💡 권장 운영: {', '.join(pz)} 구역은 **{stop[0]}시경부터 일반 차량 신규 진입 제한**, 공연 관객용으로 확보")

with tab3:
    st.subheader("유도 정책 효과 (시뮬레이션)")
    if not pk:
        st.info("사이드바에서 진입 제한 구역을 선택하세요.")
    else:
        rows = [("무유도", s0["rej"], s0["ev_hit"], s0["ev_rej"]),
                (f"유도: {','.join(pol_zones)} {pol_ratio:.0%} 상한 ({pol_from}시~)", s1["rej"], s1["ev_hit"], s1["ev_rej"])]
        dd = pd.DataFrame(rows, columns=["정책", "총 회차(대)", "공연 관객 근접 주차율", "공연 관객 회차(대)"])
        st.dataframe(dd.style.format({"총 회차(대)": "{:.0f}", "공연 관객 근접 주차율": lambda v: "-" if v is None else f"{v:.0%}",
                                      "공연 관객 회차(대)": lambda v: "-" if v is None else f"{v:.1f}"}), hide_index=True, width='stretch')
        if event:
            ub = sum(CAP[z] for z in event["zone_pref"][:2]) / event["cars"]
            st.caption(f"근접 2개 구역 면수 합 ÷ 관객 차량 = 이론 상한 {min(ub,1):.0%}")
        h = pd.DataFrame({"무유도": np.histogram(s0["rej_t"], bins=range(6, 24))[0] / n_rep,
                          "유도": np.histogram(s1["rej_t"], bins=range(6, 24))[0] / n_rep}, index=range(6, 23))
        st.markdown("**시간대별 회차 대수**"); st.bar_chart(h)
        st.warning("총량이 부족한 날(봄·가을 주말)에는 구역 유도만으로 회차를 없앨 수 없음 → 대중교통 안내·사전예약 등 수요 분산 병행 필요")

with st.expander("ℹ️ 모델 설명·가정"):
    st.markdown(f"""
- **만차 위험도**: 2025년 주말(공휴일 제외) 92일 로지스틱 회귀. 낮 기온 {M['center']}℃에서 최대, LOO-AUC 0.68.
- **시뮬레이션**: 2024.7 입출차 21.5만 건의 시간대별 도착률·체류시간 분포 + 월별주차현황 계절계수. 혼잡도 로그(봄 주말)로 구역 선호도·유입비율 보정, 월별 만차구역 수 상관 0.70.
- **LP**: 출입문 단위 수요를 구역에 배분, 회차·이동거리·공연 관객 원거리 배정 비용 최소화 (Gurobi).
- **가정**: P1A/P1B = P1 562면 반분, P8~P10 각 150면, 관객 차량 대수는 입력값.
""")

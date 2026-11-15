"""2번: 시간대·구역별 주차 시뮬레이션 (이산사건, 분 단위)"""
import numpy as np, pandas as pd
ZONES=['P1A','P1B','P2','P4','P5','P6','P7','P8','P9','P10','P11']
CAP=dict(P1A=281,P1B=281,P2=320,P4=117,P5=176,P6=167,P7=137,P8=150,P9=150,P10=150,P11=190)  # P1 562 반분, P8~P10 가정 150
# --- 2024.7 입출차에서 파라미터 추출 ---
def load_params():
    p=pd.read_csv('data/inout_202407.csv',encoding='cp949')
    p['in']=pd.to_datetime(p['입차일시']); p['out']=pd.to_datetime(p['출차일시'])
    p['dur']=(p['out']-p['in']).dt.total_seconds()/60
    p=p[(p.dur>=5)&(p.dur<=12*60)]           # 5분 미만(통과·회차) / 12시간 초과(장기) 제외
    p['we']=p['in'].dt.dayofweek>=5; p['h']=p['in'].dt.hour
    nd={True:p[p.we]['in'].dt.date.nunique(), False:p[~p.we]['in'].dt.date.nunique()}
    prof={w:(p[p.we==w].groupby('h').size()/nd[w]).reindex(range(24),fill_value=0).values for w in (True,False)}
    durs={w:{h:p[(p.we==w)&(p.h==h)].dur.values for h in range(24)} for w in (True,False)}
    return prof,durs
def seasonal_factor(month):
    m=pd.read_csv('data/monthly.csv',encoding='cp949'); o=m[m['주차장소']=='올림픽공원']
    base=o[(o['연도']==2024)&(o['월']=='07월')]['주차대수'].iloc[0]
    v=o[o['월']==f'{month:02d}월']['주차대수'].mean()     # 2023~25 평균
    return v/base
def simulate(prof,durs,weekend=True,factor=1.0,attract=None,event=None,seed=0,cap=CAP,policy=None,share=1.0,shift=0,policy_from=0):
    """event: dict(zone_pref=list, cars=int, start_hour=float)  공연 관객 차량
       policy: dict(zone->max_fill_ratio) 유도정책(해당 비율 넘으면 일반 차량 신규 진입 차단)
       share: 2024.7 전체 입차 중 P1A~P11 11개 구역으로 가는 비율(보정 파라미터)"""
    import heapq
    rng=np.random.default_rng(seed)
    attract=attract or {z:1.0 for z in ZONES}
    lam=np.roll(prof[weekend],-int(shift))*factor*share   # shift: 도착 패턴을 n시간 앞당김(여름→봄 보정)
    ts=[];kinds=[]
    for h in range(24):
        n=rng.poisson(lam[h]); ts.extend(h*60+rng.uniform(0,60,n)); kinds.extend([0]*n)
    if event:
        n=rng.poisson(event['cars']); ts.extend(event['start_hour']*60-rng.gamma(3,30,n)); kinds.extend([1]*n)
    order=np.argsort(ts); ts=np.array(ts)[order]; kinds=np.array(kinds)[order]
    nz=len(ZONES); heaps=[[] for _ in range(nz)]; capv=np.array([cap[z] for z in ZONES],float)
    lim=capv*np.array([(policy or {}).get(z,1.0) for z in ZONES])
    w=np.array([attract[z]*cap[z] for z in ZONES],float)
    full_t=[None]*nz; rejected=0; served=0; rej_t=[]; ev_hit=0; ev_n=0; ev_rej=0
    G=24*6+1; grid=np.zeros((G,nz)); gi=0
    pref_idx=[ZONES.index(z) for z in event['zone_pref']] if event else []
    for t,k in zip(ts,kinds):
        for j in range(nz):
            hp=heaps[j]
            while hp and hp[0]<=t: heapq.heappop(hp)
        while gi<G and gi*10<=t:
            grid[gi]=[sum(1 for o in heaps[j] if o>gi*10) for j in range(nz)]; gi+=1
        cnt=np.array([len(hp) for hp in heaps])
        h=int(t//60)%24
        if k==1:
            dur=150; c=[j for j in pref_idx if cnt[j]<capv[j]]
            if not c: c=[j for j in range(nz) if cnt[j]<capv[j]]
            ev_n+=1
            if not c: rejected+=1; ev_rej+=1; rej_t.append(t); continue
            j=c[0] if c[0] in pref_idx else rng.choice(c,p=w[c]/w[c].sum())
            if j in pref_idx[:2]: ev_hit+=1
        else:
            d=durs[weekend][min(23,h+int(shift))]; d=d if len(d) else durs[weekend][12]; dur=rng.choice(d)
            c=np.where(cnt<(lim if t>=policy_from*60 else capv))[0]
            if len(c)==0: rejected+=1; rej_t.append(t); continue
            j=rng.choice(c,p=w[c]/w[c].sum())
        heapq.heappush(heaps[j],t+dur); served+=1
        if full_t[j] is None and len(heaps[j])>=capv[j]: full_t[j]=t/60
    while gi<G:
        grid[gi]=[sum(1 for o in heaps[j] if o>gi*10) for j in range(nz)]; gi+=1
    return dict(ev_hit=ev_hit,ev_n=ev_n,ev_rej=ev_rej,full_t=dict(zip(ZONES,full_t)),rejected=rejected,rej_t=np.array(rej_t)/60,served=served,
                occ=pd.DataFrame(grid,columns=ZONES,index=np.arange(G)*10/60))

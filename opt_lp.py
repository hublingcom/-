"""3번: 구역 유도 최적화 (Gurobi LP, 30분 단위 유체모형)
결정: 의도구역 d로 오는 차량을 시간대 t에 구역 z로 유도할 양 x[k,d,z,t]  (k=공원/공연)
목적: 회차(수용불가) 최소화 + 공연 관객 원거리 배정 최소화 + 공원 방문객 재유도 비용
"""
import json, numpy as np, pandas as pd, gurobipy as gp
from gurobipy import GRB
from sim_core import *
GATE=dict(P4='북2문',P5='북2문',P1B='동2문',P2='동2문',P11='동2문',P1A='남2문',P6='남2문',P7='남2문',P8='기타',P9='기타',P10='기타')
def cost(d,z):
    if d==z: return 0
    return 1 if GATE[d]==GATE[z] else 3
GATES=['북2문','동2문','남2문','기타']
def cost(g,z):  # 출입문 g로 들어온 차량을 구역 z로 보낼 때 이동 비용
    return 0 if GATE[z]==g else 2
def build_demand(prof,durs,C,month,weekend=True,event=None,dt=60,h0=6,h1=23):
    f=seasonal_factor(month); T=np.arange(h0*60,h1*60,dt)
    lam=np.roll(prof[weekend],-C['shift'])*f*C['share']
    w=pd.Series({z:C['attract'][z]*CAP[z] for z in ZONES}); w=w/w.sum()
    wg=w.groupby(GATE).sum().reindex(GATES)
    park=np.array([[lam[int(t//60)]*dt/60*wg[g] for t in T] for g in GATES])     # [gate,t]
    evd=np.zeros(len(T))
    if event:
        from scipy.stats import gamma
        for k,t in enumerate(T):
            lo=event['start_hour']*60-(t+dt); hi=event['start_hour']*60-t
            evd[k]=event['cars']*(gamma.cdf(max(hi,0),3,scale=30)-gamma.cdf(max(lo,0),3,scale=30))
    S={}
    for k,t in enumerate(T):
        d=durs[weekend][min(23,int(t//60)+C['shift'])]; d=d if len(d) else durs[weekend][12]
        S[k]=np.array([np.mean(d>l*dt+dt/2) for l in range(len(T))])   # 시간대 중간 기준 잔류확률
    Sev=np.array([1.0 if l*dt<150 else 0.0 for l in range(len(T))])
    return T,park,evd,S,Sev
def solve(T,park,evd,S,Sev,ev_pref=('P5','P4'),ev_weight=3.0,guide=True,cap=CAP,verbose=0):
    """guide=False: 출입문에 들어온 구역 안에서만 주차(타 출입문 구역 재유도 없음), 공연차량도 선착순"""
    m=gp.Model(); m.Params.OutputFlag=verbose
    K=range(len(T)); Z=range(len(ZONES)); G=range(len(GATES))
    x=m.addVars(len(GATES),len(ZONES),len(T),lb=0)
    for g in G:
        for z in Z:
            if not guide and GATE[ZONES[z]]!=GATES[g]:
                for t in K: x[g,z,t].ub=0
    y=m.addVars(len(ZONES),len(T),lb=0)        # 공연 차량 배정
    r=m.addVars(len(GATES),len(T),lb=0); re=m.addVars(len(T),lb=0)
    for g in G:
        for t in K: m.addConstr(x.sum(g,'*',t)+r[g,t]==park[g,t])
    for t in K: m.addConstr(y.sum('*',t)+re[t]==evd[t])
    for z in Z:
        for t in K:
            m.addConstr(gp.quicksum(x[g,z,s]*S[s][t-s] for g in G for s in range(t+1))
                        +gp.quicksum(y[z,s]*Sev[t-s] for s in range(t+1))<=cap[ZONES[z]])
    BIG=20; far=[i for i,z in enumerate(ZONES) if z not in ev_pref]
    obj=BIG*(r.sum()+ev_weight*re.sum())+gp.quicksum(cost(GATES[g],ZONES[z])*x[g,z,t] for g in G for z in Z for t in K)
    obj+=ev_weight*gp.quicksum(y[z,t]*(1 if GATE[ZONES[z]]=='북2문' or ZONES[z] in ev_pref else 3) for z in far for t in K)
    if not guide:   # 무유도 = 공원 차량을 먼저(시간순) 최대한 수용하고 남는 자리를 공연 차량이 사용
        obj+=-0.5*BIG*gp.quicksum(x[g,z,t]*(len(T)-t)/len(T) for g in G for z in Z for t in K)
    m.setObjective(obj,GRB.MINIMIZE); m.optimize()
    X=np.array([[[x[g,z,t].X for t in K] for z in Z] for g in G]); Y=np.array([[y[z,t].X for t in K] for z in Z])
    R=np.array([[r[g,t].X for t in K] for g in G]); RE=np.array([re[t].X for t in K])
    return dict(X=X,Y=Y,R=R,RE=RE,obj=m.ObjVal)
def occupancy(sol,S,Sev,T):
    K=len(T); occ=np.zeros((len(ZONES),K)); occ_ev=np.zeros((len(ZONES),K))
    for z in range(len(ZONES)):
        for t in range(K):
            occ[z,t]=sum(sol['X'][:,z,s].sum()*S[s][t-s] for s in range(t+1))
            occ_ev[z,t]=sum(sol['Y'][z,s]*Sev[t-s] for s in range(t+1))
    return occ,occ_ev
if __name__=='__main__':
    import time
    prof,durs=load_params(); C=json.load(open('calib.json'))
    ev=dict(zone_pref=['P5','P4','P6','P2'],cars=800,start_hour=18)
    T,park,evd,S,Sev=build_demand(prof,durs,C,5,True,ev)
    print('공원수요',park.sum().round(),'공연',evd.sum().round())
    for g in (False,True):
        t0=time.time(); sol=solve(T,park,evd,S,Sev,guide=g)
        Y=sol['Y'].sum(1); near=Y[ZONES.index('P5')]+Y[ZONES.index('P4')]
        X=sol['X'].sum(2); moved=sum(X[gi,zi] for gi,gn in enumerate(GATES) for zi,z in enumerate(ZONES) if GATE[z]!=gn)
        print('유도' if g else '무유도','회차 공원',sol['R'].sum().round(1),'공연',sol['RE'].sum().round(1),'| 공연차량 P4·P5',round(near,1),'/',round(evd.sum(),1),'| 타출입문 재유도',round(moved,1),f'{time.time()-t0:.1f}s')

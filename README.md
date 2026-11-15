# 올림픽공원 주차 만차 예측 & 유도 최적화 (Streamlit)
```
pip install -r requirements.txt
streamlit run app.py
```
- gurobipy는 pip 설치만 하면 무료 제한 라이선스(변수 2,000개)로 동작함 (이 모델은 약 1,100개)
- 파일: app.py(대시보드) / sim_core.py(시뮬레이션) / opt_lp.py(Gurobi LP) / calib.json(보정값) / model1.pkl(만차 위험도 로지스틱) / data/(입출차 2024.7, 월별주차현황)

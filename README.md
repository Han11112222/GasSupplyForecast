# GasSupplyForecast (Streamlit)

## 데이터 스키마
- 실적 엑셀(.xlsx): 열 이름 → 날짜 / 평균기온 / 공급량
- 시나리오 CSV: 열 이름 → 시나리오 / 월 / 평균기온
  (시나리오 값 = "연도" 문자열, 예: 2026)

## 로컬 실행
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py

## Streamlit Community Cloud 배포
- 본 레포를 깃허브에 push
- https://share.streamlit.io 에서 New app → repo 선택 → main branch → app.py 선택 → Deploy
# GasSupplyForecast

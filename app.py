import io, os, glob, platform, urllib.request
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb

# ========================
# 한글 폰트 (자동 다운로드 포함)
# ========================
def _ensure_korean_font_locally() -> str | None:
    """
    1) ./fonts 내 폰트가 있으면 우선 사용
    2) 없으면 NotoSansKR-Regular를 공식 저장소에서 다운로드 시도(OTF→TTF 순)
    """
    os.makedirs("fonts", exist_ok=True)
    local = [p for p in glob.glob("fonts/**/*", recursive=True)
             if p.lower().endswith((".ttf", ".otf", ".ttc"))]
    if local:
        return sorted(local)[0]

    urls = [
        "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansKR-Regular.otf",
        "https://github.com/notofonts/noto-cjk/raw/main/Sans/TTF/Korean/NotoSansKR-Regular.ttf",
    ]
    for url in urls:
        try:
            fname = os.path.join("fonts", os.path.basename(url))
            urllib.request.urlretrieve(url, fname)
            return fname
        except Exception:
            continue
    return None

def set_korean_font():
    chosen = None

    # 1) ./fonts 또는 자동 다운로드
    fp = _ensure_korean_font_locally()
    if fp and os.path.exists(fp):
        try:
            fm.fontManager.addfont(fp)
            chosen = fm.FontProperties(fname=fp).get_name()
        except Exception:
            chosen = None

    # 2) 시스템 설치 글꼴 후보
    if not chosen:
        for nm in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]:
            if any(f.name == nm for f in fm.fontManager.ttflist):
                chosen = nm
                break

    # 3) 전역 설정
    if chosen:
        mpl.rcParams["font.family"] = chosen
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    print(f"[Korean Font] 사용 폰트: {chosen if chosen else '기본(영문)'}")

set_korean_font()

# ========================
# 학습/예측 유틸
# ========================
def fit_one_model(model_name, base_model, X, y):
    if model_name == "3차 다항회귀":
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = base_model
        model.fit(X_poly, y)
        return model, poly
    else:
        model = base_model
        model.fit(X, y)
        return model, None

def predict_with(model_name, model, poly, X_new_2d):
    if model_name == "3차 다항회귀":
        return model.predict(poly.transform(X_new_2d))
    else:
        return model.predict(X_new_2d)

def calc_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) > 0 else np.nan
    mape = (np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0) if np.all(y_true != 0) else np.nan
    return r2, rmse, mape

def scenario_exists_for_year(scenario_df, year):
    return str(int(year)) in scenario_df["시나리오"].astype(str).unique()

def format_poly_equation(model, poly):
    if (model is None) or (poly is None):
        return None
    coefs = np.ravel(model.coef_)
    intercept = float(getattr(model, "intercept_", 0.0))
    include_bias = bool(poly.get_params().get("include_bias", True))
    if include_bias:
        a0 = intercept + (coefs[0] if len(coefs) > 0 else 0.0)
        a1 = coefs[1] if len(coefs) > 1 else 0.0
        a2 = coefs[2] if len(coefs) > 2 else 0.0
        a3 = coefs[3] if len(coefs) > 3 else 0.0
    else:
        a0 = intercept
        a1 = coefs[0] if len(coefs) > 0 else 0.0
        a2 = coefs[1] if len(coefs) > 1 else 0.0
        a3 = coefs[2] if len(coefs) > 2 else 0.0
    return f"3차식: y = {a3:.3e}·x³ + {a2:.3e}·x² + {a1:.3e}·x + {a0:.3e}"

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="도시가스 공급량 예측/검증", layout="wide")
st.title("도시가스 공급량 예측 · 검증 대시보드")

with st.sidebar:
    st.header("데이터 업로드")
    data_file = st.file_uploader("실적 엑셀 업로드(.xlsx)", type=["xlsx", "xls"])
    sc_file   = st.file_uploader("시나리오 CSV 업로드", type=["csv"])
    st.caption("※ 실적 파일 열 이름: **날짜 / 평균기온 / 공급량**\n"
               "※ 시나리오 열 이름: **시나리오 / 월 / 평균기온** (시나리오 값=연도)")

@st.cache_data
def load_data(data_file, sc_file):
    data = pd.read_excel(data_file)
    data = data[["날짜", "평균기온", "공급량"]].copy()
    data["날짜"] = pd.to_datetime(data["날짜"])
    data["Year"]  = data["날짜"].dt.year.astype(int)
    data["Month"] = data["날짜"].dt.month.astype(int)

    # CSV 인코딩 자동 시도
    try:
        scenario = pd.read_csv(sc_file, encoding="cp949")
    except UnicodeDecodeError:
        try:
            scenario = pd.read_csv(sc_file, encoding="utf-8-sig")
        except Exception:
            scenario = pd.read_csv(sc_file, encoding="utf-8")

    scenario["월"] = scenario["월"].astype(int)
    return data, scenario

if (data_file is not None) and (sc_file is not None):
    data, scenario_data = load_data(data_file, sc_file)

    min_year, max_year = int(data["Year"].min()), int(data["Year"].max())
    models = {
        "3차 다항회귀": LinearRegression(),
        "랜덤포레스트": RandomForestRegressor(random_state=42),
        "그레이디언트 부스팅": GradientBoostingRegressor(random_state=42),
        "아다부스트": AdaBoostRegressor(random_state=42),
        "LGBM": lgb.LGBMRegressor(random_state=42),
        "최근접이웃": KNeighborsRegressor(),
    }

    st.sidebar.header("예측/검증 설정")
    forecast_year = st.sidebar.selectbox(
        "예측연도(Y)", options=list(range(min_year+1, max_year+2)), index=(max_year+1 - (min_year+1))
    )

    # 학습 종료는 자동으로 Y-1 이하로 제한
    end_max = min(max_year, forecast_year - 1)
    train_start = st.sidebar.slider("학습 시작연도", min_year, end_max, max(min_year, end_max-4))
    train_end   = st.sidebar.slider("학습 종료연도(≤Y-1)", train_start, end_max, end_max)
    month_range = st.sidebar.slider("월 범위", 1, 12, (1, 12))
    sel_models  = st.sidebar.multiselect("모델 선택", list(models.keys()), default=list(models.keys()))
    show_avg    = st.sidebar.checkbox("예측영역에 실적(월평균) 보조선", value=False)
    show_tables = st.sidebar.checkbox("표 보기", value=True)

    st.sidebar.markdown("---")
    want_excel = st.sidebar.checkbox("엑셀로 결과 저장")
    out_name   = st.sidebar.text_input("엑셀 파일명", "forecast_backtest_report.xlsx")

    # -------- 예측 실행 --------
    if len(sel_models) == 0:
        st.warning("모델을 1개 이상 선택해줘.")
        st.stop()

    if not scenario_exists_for_year(scenario_data, forecast_year):
        st.error(f"시나리오 데이터에 '{forecast_year}' 항목이 없습니다.")
        st.stop()

    m1, m2 = month_range
    # 예측용 학습세트 (시작~Y-1)
    train_pred = data[(data["Year"] >= train_start) & (data["Year"] <= train_end)].dropna(subset=["공급량"])
    Xp, yp = train_pred[["평균기온"]].values, train_pred["공급량"].values

    trained_pred, r2_train_pred = {}, {}
    for name in sel_models:
        base = models[name]
        if name == "최근접이웃":
            n_neighbors = getattr(base, "n_neighbors", 5)
            if len(train_pred) < n_neighbors:
                st.info(f"[예측 SKIP] {name}: 표본 {len(train_pred)} < n_neighbors {n_neighbors}")
                continue
        mdl, poly = fit_one_model(name, base, Xp, yp)
        trained_pred[name] = (mdl, poly)
        r2_train_pred[name] = r2_score(yp, predict_with(name, mdl, poly, Xp)) if len(yp) > 1 else np.nan

    if len(trained_pred) == 0:
        st.error("예측용 학습에 성공한 모델이 없습니다.")
        st.stop()

    sdata = scenario_data[(scenario_data["월"] >= m1) & (scenario_data["월"] <= m2)]
    sdata = sdata[sdata["시나리오"].astype(str) == str(forecast_year)]

    preds_forecast_rows = []
    for name, (mdl, poly) in trained_pred.items():
        for _, row in sdata.iterrows():
            yhat = float(predict_with(name, mdl, poly, np.array([[float(row["평균기온"])]]))[0])
            preds_forecast_rows.append([
                int(row["월"]), str(forecast_year), float(row["평균기온"]), name,
                f"{train_start}~{train_end}", int(forecast_year), yhat
            ])
    preds_forecast = pd.DataFrame(preds_forecast_rows,
        columns=["Month","기온시나리오","평균기온","Model","학습기간","예측연도","예측공급량"]
    )

    # -------- 예측 그래프 --------
    fig, ax = plt.subplots(figsize=(11,5))
    for name, grp in preds_forecast.groupby("Model"):
        g = grp.sort_values("Month")
        ax.plot(g["Month"], g["예측공급량"], marker="o", linewidth=1.8, label=name)
    if show_avg:
        avg = data.groupby("Month", as_index=False)["공급량"].mean().rename(columns={"공급량":"실적(월평균)"})
        avg = avg[(avg["Month"]>=m1)&(avg["Month"]<=m2)]
        ax.plot(avg["Month"], avg["실적(월평균)"], linestyle="--", linewidth=2.2, label="실적(월평균)")

    ax.set_title(f"[예측] 예측연도:{forecast_year} / 시나리오:{forecast_year} / 월 {m1}~{m2} / 학습기간 {train_start}~{train_end}")
    ax.set_xlabel("월"); ax.set_ylabel("예측공급량")
    ax.grid(True, alpha=0.3); ax.set_xticks(range(m1, m2+1))
    ax.legend(loc="best", fontsize=9, ncol=2)
    # 3차 다항식 식 풋노트
    if "3차 다항회귀" in trained_pred:
        mdl, poly = trained_pred["3차 다항회귀"]
        eq = format_poly_equation(mdl, poly)
        if eq:
            fig.subplots_adjust(bottom=0.20)
            r2t = r2_train_pred.get("3차 다항회귀", np.nan)
            fig.text(0.5, 0.02, f"{eq}  |  학습 R²={r2t:.3f}", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig, use_container_width=True)

    if show_tables:
        st.subheader("예측 피벗")
        st.dataframe(
            preds_forecast.pivot_table(index="Month", columns="Model", values="예측공급량", aggfunc="mean").round(2)
        )

    # -------- 검증(backtest): 대상=Y-1, 학습=시작~Y-2 --------
    Ym1, Ym2 = (forecast_year-1), (forecast_year-2)
    train_bt_end = min(train_end, Ym2)
    if train_bt_end < train_start:
        st.info(f"[검증] 학습기간이 성립하지 않습니다. (시작={train_start}, 종료={train_bt_end})")
    else:
        train_bt = data[(data["Year"]>=train_start)&(data["Year"]<=train_bt_end)].dropna(subset=["공급량"])
        Xb, yb = train_bt[["평균기온"]].values, train_bt["공급량"].values
        trained_bt = {}
        for name in sel_models:
            base = models[name]
            if name == "최근접이웃":
                n_neighbors = getattr(base, "n_neighbors", 5)
                if len(train_bt) < n_neighbors:
                    st.info(f"[검증 SKIP] {name}: 표본 {len(train_bt)} < n_neighbors {n_neighbors}")
                    continue
            mdl, poly = fit_one_model(name, base, Xb, yb)
            trained_bt[name] = (mdl, poly)

        val_df = data[(data["Year"]==Ym1)&(data["Month"]>=m1)&(data["Month"]<=m2)].dropna(subset=["공급량","평균기온"])
        if val_df.empty:
            st.info(f"[검증] {Ym1}년 실제 데이터가 없습니다.")
        else:
            X_val, y_val = val_df[["평균기온"]].values, val_df["공급량"].values
            rows, preds_all = [], []
            for name,(mdl,poly) in trained_bt.items():
                yhat = predict_with(name, mdl, poly, X_val)
                r2, rmse, mape = calc_metrics(y_val, yhat)
                tmp = val_df[["Year","Month"]].copy()
                tmp["Model"] = name
                tmp["실제공급량"] = y_val
                tmp["예측공급량"] = yhat
                preds_all.append(tmp)
                rows.append([name, r2, rmse, mape])

            preds_val_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
            metrics_df   = pd.DataFrame(rows, columns=["Model","R2(검증)","RMSE","MAPE(%)"]).sort_values("R2(검증)", ascending=False)

            fig2, ax2 = plt.subplots(figsize=(11,5))
            gv = val_df.sort_values("Month")
            ax2.plot(gv["Month"], gv["공급량"], linestyle="--", marker="o", linewidth=3.0, label=f"실제 {Ym1}")
            best_model = metrics_df.iloc[0]["Model"] if not metrics_df.empty else None
            for name in metrics_df["Model"] if not metrics_df.empty else []:
                gpred = preds_val_df[preds_val_df["Model"]==name].sort_values("Month")
                lw = 2.2 if name == best_model else 1.5
                r2v = metrics_df.loc[metrics_df["Model"]==name, "R2(검증)"].values[0]
                ax2.plot(gpred["Month"], gpred["예측공급량"], marker="o", linewidth=lw, label=f"{name} (R²={r2v:.3f})")
            ax2.set_title(f"[검증] {Ym1}년 실제(점선) vs 예측 (학습기간 {train_start}~{train_bt_end})")
            ax2.set_xlabel("월"); ax2.set_ylabel("공급량")
            ax2.grid(True, alpha=0.3); ax2.set_xticks(range(m1, m2+1))
            ax2.legend(loc="best", fontsize=9, ncol=2)
            if "3차 다항회귀" in trained_bt:
                mdl_bt, poly_bt = trained_bt["3차 다항회귀"]
                eq_bt = format_poly_equation(mdl_bt, poly_bt)
                if eq_bt:
                    fig2.subplots_adjust(bottom=0.20)
                    r2_val = metrics_df.loc[metrics_df["Model"]=="3차 다항회귀","R2(검증)"]
                    r2_val = float(r2_val.iloc[0]) if len(r2_val)>0 else np.nan
                    fig2.text(0.5, 0.02, f"{eq_bt}  |  검증 R²={r2_val:.3f}", ha="center", va="bottom", fontsize=9)
            st.pyplot(fig2, use_container_width=True)

            if show_tables:
                st.subheader("검증 성능 요약")
                st.dataframe(metrics_df.reset_index(drop=True).round(4))
                if not preds_val_df.empty:
                    merged = preds_val_df.merge(val_df[["Month","공급량"]], on="Month", how="left", suffixes=("","_실제"))
                    pv_val = merged.pivot_table(index="Month", columns="Model", values="예측공급량", aggfunc="mean").round(2)
                    pv_val["실제"] = val_df.set_index("Month")["공급량"].round(2)
                    st.dataframe(pv_val)

            # ----- 엑셀 다운로드 -----
            if want_excel:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    preds_forecast.to_excel(writer, sheet_name=f"예측(Y={forecast_year})", index=False)
                    if not val_df.empty:
                        preds_val_df.to_excel(writer, sheet_name=f"검증(Y-1={forecast_year-1})_월별", index=False)
                        metrics_df.to_excel(writer, sheet_name=f"모델성능_검증", index=False)
                st.download_button("엑셀 다운로드", data=output.getvalue(),
                                   file_name=out_name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("왼쪽 사이드바에서 실적 엑셀과 시나리오 CSV를 업로드해 시작해줘.")

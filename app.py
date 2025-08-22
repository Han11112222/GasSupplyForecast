import io, os, glob, urllib.request, tempfile
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

# =====================================
# 🔤 한글 폰트: 자동 다운로드 + Matplotlib + 웹폰트 CSS
# =====================================
FONT_URLS = [
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/Korean/NotoSansKR-Regular.otf",
    "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansKR-Regular.otf",
    "https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/Sans/OTF/Korean/NotoSansKR-Regular.otf",
]

def _download_font_to_tmp() -> tuple[str | None, str | None]:
    os.makedirs(os.path.join(tempfile.gettempdir(), "fonts"), exist_ok=True)
    for url in FONT_URLS:
        try:
            local = os.path.join(tempfile.gettempdir(), "fonts", os.path.basename(url))
            urllib.request.urlretrieve(url, local)
            return local, url
        except Exception:
            continue
    return None, None

def apply_korean_font() -> str | None:
    chosen = None
    for nm in ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]:
        if any(f.name == nm for f in fm.fontManager.ttflist):
            chosen = nm
            break
    css_font_url = None
    if not chosen:
        path, css_font_url = _download_font_to_tmp()
        if path and os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                try:
                    fm._load_fontmanager(try_read_cache=False)
                except Exception:
                    try:
                        fm._rebuild()
                    except Exception:
                        pass
                chosen = fm.FontProperties(fname=path).get_name()
            except Exception:
                chosen = None
    if chosen:
        mpl.rcParams["font.family"] = chosen
        mpl.rcParams["font.sans-serif"] = [chosen]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    if css_font_url:
        st.markdown(
            f"""
            <style>
            @font-face {{
              font-family: 'AppKor';
              src: url('{css_font_url}') format('opentype');
              font-weight: normal; font-style: normal;
            }}
            html, body, [class*="css"] {{
              font-family: 'AppKor', {chosen if chosen else 'sans-serif'} !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    return chosen

KOREAN_FONT_NAME = apply_korean_font()
LEGEND_PROP = fm.FontProperties(family=mpl.rcParams.get("font.family"))

# =====================================
# ⚙️ 유틸
# =====================================
REQUIRED_ACTUAL   = ["날짜", "평균기온", "공급량"]
REQUIRED_SCENARIO = ["시나리오", "월", "평균기온"]

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

# ---- 숫자 포맷(지수표기 금지) ----
def _fmt_num(x: float) -> str:
    ax = abs(x)
    if ax >= 1e9: return f"{x:,.0f}"
    if ax >= 1e6: return f"{x:,.1f}"
    if ax >= 1e3: return f"{x:,.2f}"
    if ax >= 1:   return f"{x:,.3f}"
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"

def _signed(term: float, label: str | None) -> str:
    sign = "-" if term < 0 else "+"
    core = f"{_fmt_num(abs(term))}"
    return f"{sign} {core}{('·'+label) if label else ''}"

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
    eq = f"3차식: y = {_fmt_num(a3)}·x³ {_signed(a2,'x²')} {_signed(a1,'x')} {_signed(a0,None)}".rstrip()
    return eq

def validate_columns(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"[{label}] 필요한 컬럼 {required} 중 누락: {missing}\n\n현재 컬럼: {list(df.columns)}")
        st.stop()

def _read_any(src):
    def _try_csv(bio):
        for enc in ("cp949", "utf-8-sig", "utf-8"):
            try:
                return pd.read_csv(bio, encoding=enc)
            except Exception:
                bio.seek(0)
        raise
    if hasattr(src, "read") or isinstance(src, (bytes, bytearray)):
        bio = io.BytesIO(src if isinstance(src, (bytes, bytearray)) else src.read())
        try:
            bio.seek(0)
            return pd.read_excel(bio)
        except Exception:
            pass
        bio.seek(0)
        return _try_csv(bio)
    ext = os.path.splitext(str(src))[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(src)
    else:
        for enc in ("cp949", "utf-8-sig", "utf-8"):
            try:
                return pd.read_csv(src, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(src)

@st.cache_data
def load_data_mixed(actual_src, scenario_src, is_upload: bool):
    actual_df   = _read_any(actual_src)
    scenario_df = _read_any(scenario_src)

    validate_columns(actual_df, REQUIRED_ACTUAL, "실적 파일")
    validate_columns(scenario_df, REQUIRED_SCENARIO, "시나리오 파일")

    data = actual_df[["날짜", "평균기온", "공급량"]].copy()
    data["날짜"] = pd.to_datetime(data["날짜"])
    data["Year"]  = data["날짜"].dt.year.astype(int)
    data["Month"] = data["날짜"].dt.month.astype(int)

    scenario = scenario_df[["시나리오", "월", "평균기온"]].copy()
    scenario["월"] = scenario["월"].astype(int)
    scenario["시나리오"] = scenario["시나리오"].astype(str)
    return data, scenario

# =====================================
# 🖥️ Streamlit UI
# =====================================
st.set_page_config(page_title="도시가스 공급량 예측/검증", layout="wide")
st.title("도시가스 공급량 예측 · 검증 대시보드")
st.caption(f"한글 폰트 적용: {KOREAN_FONT_NAME if KOREAN_FONT_NAME else '다운로드 실패 → 기본 폰트'}")

DEFAULT_ACTUAL_PATH   = "data/실적.xlsx"
DEFAULT_SCENARIO_PATH = "data/기온시나리오.xlsx"

with st.sidebar:
    st.header("데이터 불러오기 방식")
    mode = st.radio("방식 선택", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    if mode == "Repo 내 파일 사용":
        repo_actual = sorted(glob.glob("data/*.xlsx") + glob.glob("data/*.xls"))
        repo_scn    = sorted(set(glob.glob("data/*.csv") + glob.glob("data/*.xlsx") + glob.glob("data/*.xls")))

        def pick_idx(options, target):
            try:
                return options.index(target)
            except ValueError:
                return 0 if options else 0

        if not repo_actual:
            st.error("data 폴더에 실적 엑셀 파일이 없네. 업로드 모드로 쓰면 돼.")
            st.stop()
        if not repo_scn:
            st.error("data 폴더에 시나리오 파일이 없네. data/기온시나리오.xlsx 또는 .csv 넣으면 돼.")
            st.stop()

        actual_path = st.selectbox("실적 파일(Excel)", options=repo_actual, index=pick_idx(repo_actual, DEFAULT_ACTUAL_PATH), key="actual_path")
        sc_path     = st.selectbox("시나리오 파일(CSV/Excel)", options=repo_scn, index=pick_idx(repo_scn, DEFAULT_SCENARIO_PATH), key="sc_path")
        data_input_ready = True
    else:
        st.header("데이터 업로드")
        data_file = st.file_uploader("실적 엑셀 업로드(.xlsx/.xls)", type=["xlsx", "xls"], key="up_actual")
        sc_file   = st.file_uploader("시나리오 업로드(CSV/Excel)", type=["csv", "xlsx", "xls"], key="up_scn")
        st.caption("※ 실적 파일 열: **날짜 / 평균기온 / 공급량**  ·  시나리오 열: **시나리오 / 월 / 평균기온** (시나리오 값=연도)")
        data_input_ready = (data_file is not None) and (sc_file is not None)

if not data_input_ready:
    st.info("왼쪽에서 파일을 선택(또는 업로드)하면 바로 처리할게.")
    st.stop()

# ===== 데이터 로드 =====
if mode == "Repo 내 파일 사용":
    data, scenario_data = load_data_mixed(st.session_state["actual_path"], st.session_state["sc_path"], is_upload=False)
else:
    data, scenario_data = load_data_mixed(st.session_state["up_actual"], st.session_state["up_scn"], is_upload=True)

min_year, max_year = int(data["Year"].min()), int(data["Year"].max())
current_year = pd.Timestamp.today().year

# 예측 가능한 연도 풀: (실적 첫해+1) ~ max(실적 마지막+3, 현재+3)
fy_min = min_year + 1
fy_max = max(max_year + 3, current_year + 3)
fy_options = list(range(fy_min, fy_max + 1))

# 기본 From/To: From = 실적 마지막+1, To = min(From+3, fy_max)
fy_from_default = min(max_year + 1, fy_max)
fy_to_default = min(fy_from_default + 3, fy_max)

models = {
    "3차 다항회귀": LinearRegression(),
    "랜덤포레스트": RandomForestRegressor(random_state=42),
    "그레이디언트 부스팅": GradientBoostingRegressor(random_state=42),
    "아다부스트": AdaBoostRegressor(random_state=42),
    "LGBM": lgb.LGBMRegressor(random_state=42),
    "최근접이웃": KNeighborsRegressor(),
}

# ===== 설정: 폼으로 묶어서 '공급량 예측 실행' 눌러야 반영 =====
with st.sidebar:
    st.header("예측/검증 설정")
    with st.form("run_form", clear_on_submit=False):
        st.caption("예측연도는 From~To(최대 3년 폭). 값 바꾸고 **공급량 예측 실행**을 누르면 반영돼.")

        fy_from = st.selectbox("예측연도 From", options=fy_options,
                               index=fy_options.index(fy_from_default), key="fy_from")
        # To 후보는 fy_from ~ fy_from+3 사이, 단 전체 상한은 fy_max
        to_candidates = [y for y in fy_options if fy_from <= y <= min(fy_from + 3, fy_max)]
        fy_to = st.selectbox("예측연도 To (≤ From+3)", options=to_candidates,
                             index=len(to_candidates)-1 if fy_to_default not in to_candidates else to_candidates.index(fy_to_default),
                             key="fy_to")

        # 학습 종료는 From-1을 넘지 않도록
        end_cap = min(max_year, fy_from - 1)
        tr_start = st.slider("학습 시작연도", min_year, end_cap, max(min_year, end_cap-4), key="tr_start")
        tr_end   = st.slider("학습 종료연도(≤From-1)", tr_start, end_cap, end_cap, key="tr_end")

        m1, m2   = st.slider("월 범위", 1, 12, (1, 12), key="mrange")
        sel_models = st.multiselect("모델 선택", list(models.keys()), default=list(models.keys()), key="sel_models")
        show_avg  = st.checkbox("예측영역에 실적(월평균) 보조선", value=False, key="show_avg")
        show_tbl  = st.checkbox("표 보기", value=True, key="show_tbl")

        st.markdown("---")
        want_excel = st.checkbox("엑셀로 결과 저장", key="want_excel")
        out_name   = st.text_input("엑셀 파일명", "forecast_backtest_report.xlsx", key="out_name")

        run_clicked = st.form_submit_button("공급량 예측 실행", use_container_width=True, type="primary")

if "init_run_done" not in st.session_state:
    run_clicked = True
    st.session_state["init_run_done"] = True

# ===== 공통 체크 =====
if len(st.session_state["sel_models"]) == 0:
    st.warning("모델을 1개 이상 선택해줘.")
    st.stop()

# ===== 실행 =====
if run_clicked:
    fy_from = st.session_state["fy_from"]
    fy_to   = st.session_state["fy_to"]
    year_list = [y for y in fy_options if fy_from <= y <= fy_to]  # 포함 범위
    train_start = st.session_state["tr_start"]
    train_end   = st.session_state["tr_end"]
    (m1, m2)    = st.session_state["mrange"]
    sel_models  = st.session_state["sel_models"]
    show_avg    = st.session_state["show_avg"]
    show_tables = st.session_state["show_tbl"]
    want_excel  = st.session_state["want_excel"]
    out_name    = st.session_state["out_name"]

    # 학습 세트 & 모델 학습 (한 번만)
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
        st.error("예측용 학습에 성공한 모델이 없어.")
        st.stop()

    # 엑셀 저장 준비
    excel_buf = io.BytesIO()
    writer = pd.ExcelWriter(excel_buf, engine="openpyxl") if want_excel else None

    # 연도별 예측/표/그래프
    for fy in year_list:
        if not scenario_exists_for_year(scenario_data, fy):
            st.warning(f"[예측 {fy}] 시나리오 데이터가 없어. 건너뛸게.")
            continue

        sdata = scenario_data[(scenario_data["월"] >= m1) & (scenario_data["월"] <= m2)]
        sdata = sdata[sdata["시나리오"].astype(str) == str(fy)]

        preds_rows = []
        for name, (mdl, poly) in trained_pred.items():
            for _, row in sdata.iterrows():
                yhat = float(predict_with(name, mdl, poly, np.array([[float(row["평균기온"])]]))[0])
                preds_rows.append([
                    int(row["월"]), str(fy), float(row["평균기온"]), name,
                    f"{train_start}~{train_end}", int(fy), yhat
                ])
        preds_forecast = pd.DataFrame(preds_rows,
            columns=["Month","기온시나리오","평균기온","Model","학습기간","예측연도","예측공급량"]
        )

        # --- 예측 그래프 ---
        fig, ax = plt.subplots(figsize=(11,5))
        for name, grp in preds_forecast.groupby("Model"):
            g = grp.sort_values("Month")
            ax.plot(g["Month"], g["예측공급량"], marker="o", linewidth=1.8, label=name)
        if show_avg:
            avg = data.groupby("Month", as_index=False)["공급량"].mean().rename(columns={"공급량":"실적(월평균)"})
            avg = avg[(avg["Month"]>=m1)&(avg["Month"]<=m2)]
            ax.plot(avg["Month"], avg["실적(월평균)"], linestyle="--", linewidth=2.2, label="실적(월평균)")

        ax.set_title(f"[예측] 예측연도:{fy} / 시나리오:{fy} / 월 {m1}~{m2} / 학습기간 {train_start}~{train_end}")
        ax.set_xlabel("월"); ax.set_ylabel("예측공급량")
        ax.grid(True, alpha=0.3); ax.set_xticks(range(m1, m2+1))
        ax.legend(loc="best", fontsize=9, ncol=2, prop=LEGEND_PROP)

        if "3차 다항회귀" in trained_pred:
            mdl, poly = trained_pred["3차 다항회귀"]
            eq = format_poly_equation(mdl, poly)
            if eq:
                fig.subplots_adjust(bottom=0.18)
                r2t = r2_train_pred.get("3차 다항회귀", np.nan)
                fig.text(0.5, 0.02, f"{eq}  |  학습 R²={r2t:.3f}", ha="center", va="bottom", fontsize=10, fontproperties=LEGEND_PROP)

        st.pyplot(fig, use_container_width=True)

        if show_tables:
            st.subheader(f"예측 피벗 (Y={fy})")
            st.dataframe(
                preds_forecast.pivot_table(index="Month", columns="Model", values="예측공급량", aggfunc="mean").round(2)
            )

        # --- 검증(backtest): 대상=Y-1 ---
        Ym1, Ym2 = (fy-1), (fy-2)
        train_bt_end = min(train_end, Ym2)
        if train_bt_end < train_start:
            st.info(f"[검증 Y={fy}] 학습기간이 성립하지 않아. (시작={train_start}, 종료={train_bt_end})")
        else:
            train_bt = data[(data["Year"]>=train_start)&(data["Year"]<=train_bt_end)].dropna(subset=["공급량"])
            Xb, yb = train_bt[["평균기온"]].values, train_bt["공급량"].values
            trained_bt = {}
            for name in sel_models:
                base = models[name]
                if name == "최근접이웃":
                    n_neighbors = getattr(base, "n_neighbors", 5)
                    if len(train_bt) < n_neighbors:
                        st.info(f"[검증 SKIP Y={fy}] {name}: 표본 {len(train_bt)} < n_neighbors {n_neighbors}")
                        continue
                mdl, poly = fit_one_model(name, base, Xb, yb)
                trained_bt[name] = (mdl, poly)

            val_df = data[(data["Year"]==Ym1)&(data["Month"]>=m1)&(data["Month"]<=m2)].dropna(subset=["공급량","평균기온"])
            if val_df.empty:
                st.info(f"[검증 Y={fy}] {Ym1}년 실제 데이터가 없어.")
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
                ax2.set_title(f"[검증] Y={fy} → 실제 {Ym1}(점선) vs 예측 (학습기간 {train_start}~{train_bt_end})")
                ax2.set_xlabel("월"); ax2.set_ylabel("공급량")
                ax2.grid(True, alpha=0.3); ax2.set_xticks(range(m1, m2+1))
                ax2.legend(loc="best", fontsize=9, ncol=2, prop=LEGEND_PROP)

                if "3차 다항회귀" in trained_bt:
                    mdl_bt, poly_bt = trained_bt["3차 다항회귀"]
                    eq_bt = format_poly_equation(mdl_bt, poly_bt)
                    if eq_bt:
                        fig2.subplots_adjust(bottom=0.18)
                        r2_val = metrics_df.loc[metrics_df["Model"]=="3차 다항회귀","R2(검증)"]
                        r2_val = float(r2_val.iloc[0]) if len(r2_val)>0 else np.nan
                        fig2.text(0.5, 0.02, f"{eq_bt}  |  검증 R²={r2_val:.3f}", ha="center", va="bottom", fontsize=10, fontproperties=LEGEND_PROP)
                st.pyplot(fig2, use_container_width=True)

                if show_tables:
                    st.subheader(f"검증 성능 요약 (Y={fy})")
                    st.dataframe(metrics_df.reset_index(drop=True).round(4))
                    if not preds_val_df.empty:
                        merged = preds_val_df.merge(val_df[["Month","공급량"]], on="Month", how="left", suffixes=("","_실제"))
                        pv_val = merged.pivot_table(index="Month", columns="Model", values="예측공급량", aggfunc="mean").round(2)
                        pv_val["실제"] = val_df.set_index("Month")["공급량"].round(2)
                        st.dataframe(pv_val)

        # 엑셀 시트 저장
        if want_excel and writer is not None:
            preds_forecast.to_excel(writer, sheet_name=f"예측(Y={fy})", index=False)

    # 엑셀 다운로드 버튼
    if want_excel and writer is not None:
        writer.close()
        st.download_button(
            "엑셀 다운로드",
            data=excel_buf.getvalue(),
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

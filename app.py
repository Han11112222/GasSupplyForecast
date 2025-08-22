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

# ============ NEW: 표 스타일/합계 도우미 ============
def add_total_row(df: pd.DataFrame, label: str = "합계") -> pd.DataFrame:
    """숫자열 합계를 마지막 행으로 추가."""
    out = df.copy()
    out.loc[label] = out.sum(numeric_only=True)
    return out

def style_thousands(df: pd.DataFrame, digits: int = 2) -> pd.io.formats.style.Styler:
    """천단위 콤마 포맷 적용(소수 digits자리)."""
    fmt = {col: f"{{:,.{digits}f}}" for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
    return df.style.format(fmt)

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

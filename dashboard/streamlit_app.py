import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import base64
import hashlib
from io import BytesIO
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Industrial Waste Intelligence",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8001"

WASTE_CLASSES = ["Metal", "Mixed Waste", "Paper-Cardboard", "Plastic", "Wood"]

CLASS_COLORS = {
    "Metal":           "#B0B0B0",
    "Mixed Waste":     "#E95500",
    "Paper-Cardboard": "#F7DC6F",
    "Plastic":         "#3498DB",
    "Wood":            "#8E5A2B",
}

FORECAST_CLASSES = ["Plastic", "Paper-Cardboard", "Mixed Waste", "Wood", "Metal"]

PAGES = [
    "Dashboard",
    "Image Analysis",
    "Analysis Records",
    "Trend Analysis",
    "Forecast",
    "Anomaly Detection",
    "AI Insights",
    "Ask AI",
]


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #1a1a2e;
        }
        [data-testid="stSidebar"] * {
            color: #e0e0e0 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #2a2a4a !important;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: #16213e;
            border: 1px solid #0f3460;
            border-radius: 8px;
            padding: 12px 16px;
        }
        [data-testid="stMetricValue"] {
            color: #e0e0e0 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #888 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem !important;
        }

        /* Headers */
        h1 { color: #E95500 !important; }
        h2, h3 { color: #e0e0e0 !important; }

        /* Main background */
        .main .block-container {
            background-color: #0d1117;
            padding-top: 2rem;
        }
        .stApp {
            background-color: #0d1117;
        }

        /* Text */
        p, li, span, label, div {
            color: #c0c0c0;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed #0f3460;
            border-radius: 8px;
            background-color: #16213e;
        }

        /* Primary button */
        .stButton > button[kind="primary"] {
            background-color: #E95500 !important;
            border: none !important;
            color: white !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #c44600 !important;
        }

        /* Expander */
        [data-testid="stExpander"] {
            border: 1px solid #0f3460 !important;
            border-radius: 4px !important;
            background-color: #16213e !important;
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            background-color: #16213e;
        }

        /* Divider */
        hr {
            border-color: #2a2a4a !important;
        }

        /* Selectbox / multiselect */
        [data-testid="stSelectbox"] > div,
        [data-testid="stMultiSelect"] > div {
            background-color: #16213e;
            border-color: #0f3460;
        }

        /* Multiselect tags — white background, dark text */
        [data-baseweb="tag"] {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
        }
        [data-baseweb="tag"] span {
            color: #1a1a2e !important;
        }
        [data-baseweb="tag"] svg {
            fill: #1a1a2e !important;
        }

        /* Alert boxes */
        [data-testid="stAlert"] {
            background-color: #16213e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_root():
    try:
        r = requests.get(f"{API_BASE}/", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


@st.cache_data(ttl=60)
def fetch_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


@st.cache_data(ttl=10)
def fetch_trend(class_name=None, date_from=None, date_to=None, limit=30):
    params = {}
    if class_name:
        params["class_name"] = class_name
    if date_from:
        params["date_from"] = str(date_from)
    if date_to:
        params["date_to"] = str(date_to)
    if not date_from and not date_to:
        params["limit"] = limit * 5 if not class_name else limit
    try:
        r = requests.get(f"{API_BASE}/trend", params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.exceptions.RequestException:
        return []


@st.cache_data(ttl=60)
def fetch_forecast(class_name=None):
    params = {}
    if class_name:
        params["class_name"] = class_name
    try:
        r = requests.get(f"{API_BASE}/forecast", params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.exceptions.RequestException:
        return []


@st.cache_data(ttl=60)
def fetch_anomaly():
    try:
        r = requests.get(f"{API_BASE}/anomaly", timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.exceptions.RequestException:
        return []


@st.cache_data(ttl=60)
def fetch_insights(insight_type=None):
    params = {}
    if insight_type:
        params["insight_type"] = insight_type
    try:
        r = requests.get(f"{API_BASE}/insights", params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.exceptions.RequestException:
        return []


def call_analyze(file_bytes, filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    try:
        r = requests.post(
            f"{API_BASE}/analyze",
            files={"file": (filename, file_bytes, mime)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        # Surface the API's error detail if available
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"_error": detail}
    except requests.exceptions.RequestException as e:
        return {"_error": str(e)}


def call_query(question):
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def call_save_record(result):
    """Save an /analyze result to the database via POST /records/save."""
    try:
        r = requests.post(
            f"{API_BASE}/records/save",
            json={
                "filename":            result.get("filename"),
                "composition":         result.get("composition", {}),
                "total_area_detected": result.get("total_area_detected", 0),
            },
            timeout=10,
        )
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def call_delete_last():
    try:
        r = requests.delete(f"{API_BASE}/records/last", timeout=10)
        r.raise_for_status()
        return True, r.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)


def call_delete_all():
    try:
        r = requests.delete(f"{API_BASE}/records/all", timeout=10)
        r.raise_for_status()
        return True, r.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)


def call_regenerate_forecast():
    try:
        r = requests.post(f"{API_BASE}/forecast/regenerate", timeout=30)
        r.raise_for_status()
        return True, r.json()
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return False, detail
    except requests.exceptions.RequestException as e:
        return False, str(e)


def call_generate_insights(report_type="daily_report", force=False):
    try:
        r = requests.post(
            f"{API_BASE}/insights/generate",
            params={"report_type": report_type, "force": int(force)},
            timeout=90,
        )
        r.raise_for_status()
        return True, r.json()
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return False, detail
    except requests.exceptions.RequestException as e:
        return False, str(e)


def call_check_anomalies():
    try:
        r = requests.post(f"{API_BASE}/anomaly/check", timeout=60)
        r.raise_for_status()
        return True, r.json()
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return False, detail
    except requests.exceptions.RequestException as e:
        return False, str(e)


@st.cache_data(ttl=5)
def fetch_records(date_from=None, date_to=None, limit=None):
    params = {}
    if date_from:
        params["date_from"] = str(date_from)
    if date_to:
        params["date_to"] = str(date_to)
    if limit:
        params["limit"] = limit
    try:
        r = requests.get(f"{API_BASE}/records", params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.exceptions.RequestException:
        return []


# ── Chart helpers ─────────────────────────────────────────────────────────────
def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.6)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#2a2a4a", linecolor="#2a2a4a"),
        yaxis=dict(gridcolor="#2a2a4a", linecolor="#2a2a4a"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0")),
    )
    return fig


def make_donut_chart(composition):
    labels = list(composition.keys())
    values = list(composition.values())
    colors = [CLASS_COLORS.get(lbl, "#888888") for lbl in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
            textinfo="percent+label",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5),
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
    )
    return apply_dark_theme(fig)


def make_bar_chart(composition):
    sorted_items = sorted(composition.items(), key=lambda x: x[1])
    classes = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [CLASS_COLORS.get(cls, "#888888") for cls in classes]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=classes,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#0d1117", width=1)),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="#e0e0e0"),
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 110], title="Composition (%)", ticksuffix="%"),
        yaxis=dict(title=""),
        margin=dict(t=20, b=20, l=10, r=60),
        height=max(200, len(composition) * 55),
    )
    return apply_dark_theme(fig)


def compute_trend_pct(data):
    """Normalize annotation_count per date to composition percentage."""
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    totals = df.groupby("date")["annotation_count"].transform("sum")
    df["composition_pct"] = (df["annotation_count"] / totals.replace(0, 1)) * 100
    return df


def make_trend_chart(data, selected_classes):
    df = compute_trend_pct(data)
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    for cls in selected_classes:
        cls_df = df[df["class_name"] == cls].sort_values("date")
        if cls_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=cls_df["date"],
                y=cls_df["composition_pct"],
                mode="lines+markers",
                name=cls,
                line=dict(color=CLASS_COLORS.get(cls, "#888888"), width=2),
                marker=dict(size=5),
                hovertemplate="%{x|%Y-%m-%d}: <b>%{y:.1f}%</b><extra>" + cls + "</extra>",
            )
        )
    fig.update_layout(
        xaxis=dict(title="Date", type="date", tickformat="%Y-%m-%d"),
        yaxis=dict(title="Composition (%)", range=[0, 100], ticksuffix="%"),
        hovermode="x unified",
        margin=dict(t=20, b=40, l=40, r=20),
        height=380,
    )
    return apply_dark_theme(fig)


def compute_forecast_pct(data):
    """Normalize predicted_count per forecast_date to composition percentages."""
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    totals = df.groupby("forecast_date")["predicted_count"].transform("sum")
    df["pct"] = (df["predicted_count"] / totals.replace(0, 1)) * 100
    return df


def make_forecast_chart(data, selected_classes):
    df = compute_forecast_pct(data)
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    for cls in selected_classes:
        cls_df = df[df["class_name"] == cls].sort_values("forecast_date")
        if cls_df.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=cls_df["forecast_date"],
                y=cls_df["pct"],
                name=cls,
                marker=dict(color=CLASS_COLORS.get(cls, "#888888")),
                text=[f"{v:.1f}%" for v in cls_df["pct"]],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#ffffff", size=11),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>" + cls + ": <b>%{y:.2f}%</b><extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Forecast Date"),
        yaxis=dict(title="Composition (%)", range=[0, 100], ticksuffix="%"),
        margin=dict(t=20, b=40, l=40, r=20),
        height=400,
        uniformtext=dict(minsize=9, mode="hide"),
    )
    return apply_dark_theme(fig)


def make_anomaly_chart(data):
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["anomaly_score"],
            mode="markers",
            marker=dict(color="#E95500", size=10, symbol="diamond"),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Score: %{y:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Anomaly Score (more negative = more anomalous)"),
        margin=dict(t=20, b=40, l=60, r=20),
        height=320,
    )
    return apply_dark_theme(fig)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#E95500;margin-bottom:0'>♻️ Industrial Waste</h2>"
            "<p style='color:#888;font-size:0.85em;margin-top:0'>Intelligence Platform</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio(
            "Navigation",
            options=PAGES,
            label_visibility="collapsed",
        )
        st.markdown("---")

        # Silent API status dot
        health = fetch_health()
        if health and health.get("status") == "ok":
            model_ok = health.get("model_ready", False)
            st.markdown(
                f"<span style='color:#2ecc71'>●</span> API connected"
                f"<br><span style='font-size:0.8em;color:#888'>"
                f"Model: {'ready' if model_ok else 'not loaded'}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#e74c3c'>●</span> API offline",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<br><p style='color:#555;font-size:0.7em'>Kinsei Sangyo Co., Ltd<br>"
            "Industrial Waste Monitoring</p>",
            unsafe_allow_html=True,
        )

    return page


# ── Page: Dashboard ───────────────────────────────────────────────────────────
def render_dashboard():
    st.title("Industrial Waste Intelligence")
    st.caption("Kinsei Sangyo — Real-time waste composition monitoring")

    root = fetch_root()
    health = fetch_health()

    if root is None or health is None:
        st.error(
            "Cannot connect to API at `http://127.0.0.1:8001`. "
            "Ensure the FastAPI server is running."
        )
        st.stop()

    # ── Status metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API Status", root.get("status", "—").upper())
    c2.metric("Version", root.get("version", "—"))
    c3.metric("Database", root.get("database", "—").capitalize())
    c4.metric("Model Ready", "Yes" if health.get("model_ready") else "No")

    st.markdown("---")

    # ── Quick data summary ────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Waste Trend (last 30 days)")
        today_dash   = pd.Timestamp.today().normalize()
        from_30_days = today_dash - pd.Timedelta(days=30)
        trend_data   = fetch_trend(date_from=from_30_days.date(), date_to=today_dash.date())
        if trend_data:
            fig = make_trend_chart(trend_data, WASTE_CLASSES)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data in the last 30 days.")

    with col_right:
        st.subheader("System Summary")

        anomaly_data  = fetch_anomaly()
        insights_data = fetch_insights()
        latest_records = fetch_records(limit=1)   # only need the most recent row

        # ── Dominant class from last-30-day trend already fetched ─────────────
        dominant_class = "—"
        dominant_pct   = 0.0
        if trend_data:
            df_dom       = pd.DataFrame(trend_data)
            class_totals = df_dom.groupby("class_name")["annotation_count"].sum()
            grand_total  = class_totals.sum()
            if grand_total > 0:
                dominant_class = class_totals.idxmax()
                dominant_pct   = round(float(class_totals.max()) / float(grand_total) * 100, 1)

        # ── Latest upload info ────────────────────────────────────────────────
        latest_display = "—"
        latest_file    = ""
        if latest_records:
            latest = latest_records[0]
            try:
                dt = pd.to_datetime(latest.get("analyzed_at", ""))
                latest_display = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                latest_display = str(latest.get("analyzed_at", "—"))[:16]
            latest_file = latest.get("filename", "")

        # ── Metrics ───────────────────────────────────────────────────────────
        dom_value = f"{dominant_class}  ({dominant_pct}%)" if dominant_class != "—" else "—"
        st.metric("🥇 Dominant Class (30d)", dom_value)
        st.metric("⚠️ Anomalous Days", len(anomaly_data))
        st.metric("💡 AI Insights", len(insights_data))
        st.metric("📤 Last Upload",
                  latest_display if latest_display != "—" else "No uploads yet")
        if latest_file:
            st.caption(f"📄 {latest_file}")


# ── Page: Image Analysis ──────────────────────────────────────────────────────
def render_image_analysis():
    st.title("Image Analysis")
    st.caption("Upload one or more waste facility images for AI-powered composition analysis")

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.session_state.pop("batch_results", None)
        st.markdown(
            "<div style='text-align:center;padding:40px;color:#555;"
            "border:2px dashed #2a2a4a;border-radius:8px'>"
            "📂 Drop one or more JPG / PNG images here to analyze waste composition"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Analyze any new files (skip already-cached ones) ─────────────────────
    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = {}

    cache = st.session_state["batch_results"]

    # Find files not yet analyzed
    new_files = []
    for f in uploaded_files:
        raw   = f.read()
        fhash = hashlib.md5(raw).hexdigest()
        if fhash not in cache:
            new_files.append((f.name, fhash, raw))

    if new_files:
        progress = st.progress(0, text="Analyzing images…")
        for i, (name, fhash, raw) in enumerate(new_files):
            progress.progress(
                int((i + 1) / len(new_files) * 100),
                text=f"Analyzing {name} ({i+1}/{len(new_files)})…",
            )
            result = call_analyze(raw, name)
            cache[fhash] = {
                "name":   name,
                "bytes":  raw,
                "result": result,
                "saved":  False,
            }
        progress.empty()

    # Keep only hashes that are still in the uploader (handles file removal)
    current_hashes = {hashlib.md5(f.read() or b"").hexdigest() for f in uploaded_files}
    # Note: f.read() after the loop above returns b"" because position is at end;
    # recalculate from cache keys that still match uploaded file names
    uploaded_names = {f.name for f in uploaded_files}
    stale = [h for h, v in cache.items() if v["name"] not in uploaded_names]
    for h in stale:
        del cache[h]

    # ── Batch summary table ───────────────────────────────────────────────────
    st.markdown(f"**{len(cache)} image(s) analyzed**")

    summary_rows = []
    for entry in cache.values():
        r    = entry["result"]
        failed = r is None or "_error" in (r or {})
        comp = r.get("composition", {}) if r and not failed else {}
        dominant = max(comp, key=comp.get) if comp else "—"
        summary_rows.append({
            "Filename":       entry["name"],
            "Status":         "❌ Failed" if failed else ("✅ Saved" if entry["saved"] else "⏳ Not saved"),
            "Dominant Class": dominant,
        })
    st.dataframe(
        pd.DataFrame(summary_rows),
        hide_index=True,
        use_container_width=True,
    )

    # ── Save All button ───────────────────────────────────────────────────────
    unsaved = [v for v in cache.values() if not v["saved"] and v["result"] and "_error" not in v["result"]]
    if unsaved:
        if st.button(f"💾 Save All to Records ({len(unsaved)} unsaved)", type="primary"):
            saved_count = 0
            failed = []
            prog = st.progress(0, text="Saving…")
            for i, entry in enumerate(unsaved):
                ok = call_save_record(entry["result"])
                if ok:
                    entry["saved"] = True
                    saved_count += 1
                else:
                    failed.append(entry["name"])
                prog.progress(int((i + 1) / len(unsaved) * 100))
            prog.empty()
            st.cache_data.clear()
            if failed:
                st.error(f"Saved {saved_count} — failed: {', '.join(failed)}")
            else:
                st.success(f"Saved {saved_count} record(s) to Analysis Records.", icon="✅")
    else:
        st.success("All images saved to Analysis Records.", icon="✅")

    # ── Per-image results in expanders ────────────────────────────────────────
    st.markdown("---")
    for entry in cache.values():
        name   = entry["name"]
        raw    = entry["bytes"]
        result = entry["result"]
        saved  = entry["saved"]

        has_error = result is None or "_error" in (result or {})
        label = f"{'✅ ' if saved else '❌ ' if has_error else ''}{name}"
        with st.expander(label, expanded=len(cache) == 1):
            if has_error:
                err_msg = (result or {}).get("_error", "API unavailable or model not loaded")
                st.error(f"**Analysis failed:** {err_msg}")
                continue

            composition = result.get("composition", {})

            # Images
            col_orig, col_annot = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                st.image(raw, use_container_width=True)
            with col_annot:
                st.markdown("**AI Segmentation**")
                img_data = base64.b64decode(result["annotated_image"])
                st.image(Image.open(BytesIO(img_data)), use_container_width=True)

            if not composition:
                st.warning("No waste detected.")
                continue

            # Charts
            col_donut, col_bar = st.columns(2)
            with col_donut:
                st.plotly_chart(
                    make_donut_chart(composition),
                    use_container_width=True,
                    key=f"donut_{name}",
                )
            with col_bar:
                st.plotly_chart(
                    make_bar_chart(composition),
                    use_container_width=True,
                    key=f"bar_{name}",
                )

            # Breakdown table
            df_comp = pd.DataFrame(
                [
                    {"Waste Class": k, "Composition (%)": round(v, 2)}
                    for k, v in sorted(
                        composition.items(), key=lambda x: x[1], reverse=True
                    )
                ]
            )
            st.dataframe(
                df_comp,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Composition (%)": st.column_config.NumberColumn(
                        "Composition (%)", format="%.2f%%"
                    )
                },
            )


# ── Page: Analysis Records ────────────────────────────────────────────────────
def render_analysis_records():
    st.title("Analysis Records")
    st.caption("Per-image waste composition records saved from dashboard uploads")

    # ── Manage Records (always visible) ──────────────────────────────────────
    st.subheader("Manage Records")
    st.caption("Changes here also update Trend Analysis automatically.")

    col_undo, col_del, col_spacer = st.columns([1, 1, 3])

    # ── Undo last save ────────────────────────────────────────────────────────
    with col_undo:
        if st.button("↩ Undo Last Save", use_container_width=True):
            st.session_state["confirm_delete_all"] = False
            ok, resp = call_delete_last()
            if ok:
                st.cache_data.clear()
                st.success(f"Deleted record saved on {resp.get('date', '')}.", icon="✅")
                st.rerun()
            else:
                st.error(f"Failed: {resp}")

    # ── Delete all ────────────────────────────────────────────────────────────
    with col_del:
        if not st.session_state.get("confirm_delete_all"):
            if st.button("🗑 Delete All Records", use_container_width=True):
                st.session_state["confirm_delete_all"] = True
                st.rerun()
        else:
            st.warning("Delete ALL records? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete all", type="primary", use_container_width=True):
                    ok, resp = call_delete_all()
                    st.session_state["confirm_delete_all"] = False
                    if ok:
                        st.cache_data.clear()
                        st.success(f"All records deleted.", icon="✅")
                        st.rerun()
                    else:
                        st.error(f"Failed: {resp}")
            with c2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["confirm_delete_all"] = False
                    st.rerun()

    st.markdown("---")

    # ── Date range filter ─────────────────────────────────────────────────────
    col_from, col_to, col_btn = st.columns([2, 2, 1])
    with col_from:
        date_from = st.date_input("From", value=None, key="rec_from")
    with col_to:
        date_to = st.date_input("To", value=None, key="rec_to")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear filter", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading records…"):
        data = fetch_records(
            date_from=date_from if date_from else None,
            date_to=date_to if date_to else None,
        )

    if not data:
        st.info("No records found. Upload and analyze images on the Image Analysis page.")
        return

    st.metric("Total Records", len(data))

    df = pd.DataFrame(data)
    df["analysis_date"] = pd.to_datetime(df["analysis_date"]).dt.date
    df["analyzed_at"]   = pd.to_datetime(df["analyzed_at"])

    # Rename columns for display
    display_df = df[[
        "filename", "analysis_date", "analyzed_at",
        "metal_pct", "mixed_waste_pct", "paper_cardboard_pct",
        "plastic_pct", "wood_pct", "total_area_detected",
    ]].rename(columns={
        "filename":            "Filename",
        "analysis_date":       "Date",
        "analyzed_at":         "Analyzed At",
        "metal_pct":           "Metal (%)",
        "mixed_waste_pct":     "Mixed Waste (%)",
        "paper_cardboard_pct": "Paper-Cardboard (%)",
        "plastic_pct":         "Plastic (%)",
        "wood_pct":            "Wood (%)",
        "total_area_detected": "Area Detected (%)",
    })

    pct_cols = ["Metal (%)", "Mixed Waste (%)", "Paper-Cardboard (%)", "Plastic (%)", "Wood (%)"]
    col_config = {
        "Filename":           st.column_config.TextColumn("Filename", width="medium"),
        "Date":               st.column_config.DateColumn("Date"),
        "Analyzed At":        st.column_config.DatetimeColumn("Analyzed At", format="YYYY-MM-DD HH:mm"),
        "Area Detected (%)":  st.column_config.NumberColumn("Area Detected (%)", format="%.2f%%"),
    }
    for col in pct_cols:
        col_config[col] = st.column_config.NumberColumn(col, format="%.2f%%")

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config=col_config,
    )

    # CSV export
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="analysis_records.csv",
        mime="text/csv",
    )


# ── Page: Trend Analysis ──────────────────────────────────────────────────────
def render_trend_analysis():
    st.title("Trend Analysis")
    st.caption("Historical waste annotation trends from the database")

    # ── Filters ───────────────────────────────────────────────────────────────
    col_cls, col_from, col_to, col_refresh, col_reset = st.columns([3, 1.5, 1.5, 0.8, 0.8])
    with col_cls:
        selected = st.multiselect(
            "Filter by waste class",
            options=WASTE_CLASSES,
            default=WASTE_CLASSES,
        )
    today_trend  = pd.Timestamp.today().normalize().date()
    default_from = today_trend - pd.Timedelta(days=30)
    with col_from:
        date_from = st.date_input("From", value=default_from, key="trend_from")
    with col_to:
        date_to = st.date_input("To", value=today_trend, key="trend_to")
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Refresh", type="primary", use_container_width=True, key="trend_refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset", use_container_width=True, key="trend_reset"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading trend data…"):
        data = fetch_trend(date_from=date_from, date_to=date_to)

    if not data:
        st.warning(
            "No trend data for this period. "
            "Upload and save images on the **Image Analysis** page first, "
            "then click **Refresh**."
        )
        return

    df = compute_trend_pct(data)
    df_filtered = df[df["class_name"].isin(selected)] if selected else df

    st.plotly_chart(make_trend_chart(data, selected), use_container_width=True)

    # Per-class avg composition metrics
    if selected:
        st.subheader("Average Composition")
        cols = st.columns(len(selected))
        for col, cls in zip(cols, selected):
            cls_df = df_filtered[df_filtered["class_name"] == cls]
            if not cls_df.empty:
                avg = cls_df["composition_pct"].mean()
                col.metric(cls, f"{avg:.1f}%")

    with st.expander("Raw Data"):
        display = df_filtered[["date", "class_name", "composition_pct", "image_count"]].copy()
        display = display.rename(columns={"composition_pct": "Composition (%)"})
        display["Composition (%)"] = display["Composition (%)"].round(2)
        st.dataframe(
            display.sort_values("date", ascending=False),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Composition (%)": st.column_config.NumberColumn(
                    "Composition (%)", format="%.2f%%"
                )
            },
        )


# ── Page: Forecast ────────────────────────────────────────────────────────────
def render_forecast():
    st.title("Forecast")
    st.caption("30-day ahead waste composition predictions")

    # ── Generate / refresh controls ───────────────────────────────────────────
    col_gen, col_info = st.columns([1, 3])
    with col_gen:
        if st.button("🔄 Generate Fresh Forecast", type="primary", use_container_width=True):
            with st.spinner("Computing 30-day forecast from latest data…"):
                ok, resp = call_regenerate_forecast()
            if ok:
                st.cache_data.clear()
                n = resp.get("forecasts_created", 0)
                st.success(f"Generated {n} forecast entries for the next 30 days!", icon="✅")
                st.rerun()
            else:
                st.error(f"Failed: {resp}")
    with col_info:
        st.info(
            "Click **Generate Fresh Forecast** to create 30-day predictions using your uploaded data. "
            "If the chart shows old dates (e.g. Sep 2024), those are from the original ML training pipeline — "
            "click the button to replace them with current forecasts."
        )

    st.markdown("---")

    selected = st.multiselect(
        "Select waste classes",
        options=FORECAST_CLASSES,
        default=FORECAST_CLASSES,
    )

    with st.spinner("Loading data…"):
        _all_forecast = fetch_forecast(class_name=None)
        today_str     = pd.Timestamp.today().strftime("%Y-%m-%d")
        today_records = fetch_records(date_from=today_str, date_to=today_str)

    # Show only future forecasts (≥ today) — strip old 2024 pipeline rows from chart
    forecast_data = [row for row in _all_forecast if row.get("forecast_date", "") >= today_str]

    # ── Today's actual from dashboard uploads ────────────────────────────────
    if today_records:
        df_today = pd.DataFrame(today_records)
        col_map  = {
            "metal_pct":           "Metal",
            "mixed_waste_pct":     "Mixed Waste",
            "paper_cardboard_pct": "Paper-Cardboard",
            "plastic_pct":         "Plastic",
            "wood_pct":            "Wood",
        }
        # Average across all today's images
        today_avg = {
            cls: round(df_today[col].mean(), 2)
            for col, cls in col_map.items()
            if col in df_today.columns
        }
        total = sum(today_avg.values())
        if total > 0:
            today_pct = {cls: round(v / total * 100, 2) for cls, v in today_avg.items()}
        else:
            today_pct = today_avg

        st.subheader("Today's Actual Composition")
        st.caption(f"Based on {len(df_today)} image(s) uploaded today ({today_str})")
        col_donut, col_bar = st.columns(2)
        with col_donut:
            st.plotly_chart(
                make_donut_chart(today_pct), use_container_width=True, key="today_donut"
            )
        with col_bar:
            st.plotly_chart(
                make_bar_chart(today_pct), use_container_width=True, key="today_bar"
            )

        st.markdown("---")

    # ── ML Forecast ───────────────────────────────────────────────────────────
    st.subheader("ML Forecast")
    if not forecast_data:
        st.info(
            "No forecast data yet. "
            "Click **🔄 Generate Fresh Forecast** above to create 30-day predictions from your uploaded data."
        )
    else:
        df = compute_forecast_pct(forecast_data)
        df_filtered = df[df["class_name"].isin(selected)] if selected else df

        st.plotly_chart(make_forecast_chart(forecast_data, selected), use_container_width=True)

        if "model_name" in df.columns:
            models_used = df["model_name"].dropna().unique().tolist()
            if models_used:
                st.caption(f"Forecast model(s): {', '.join(models_used)}")

        st.subheader("Exact Composition by Date")
        pivot = (
            df_filtered[["forecast_date", "class_name", "pct"]]
            .copy()
            .assign(forecast_date=lambda d: d["forecast_date"].dt.strftime("%Y-%m-%d"))
            .pivot(index="forecast_date", columns="class_name", values="pct")
            .round(2)
            .reset_index()
            .rename(columns={"forecast_date": "Date"})
        )
        pct_col_config = {
            col: st.column_config.NumberColumn(col, format="%.2f%%")
            for col in pivot.columns if col != "Date"
        }
        st.dataframe(
            pivot, hide_index=True, use_container_width=True, column_config=pct_col_config
        )


# ── Page: Anomaly Detection ───────────────────────────────────────────────────
def render_anomaly_detection():
    st.title("Anomaly Detection")
    st.caption("Days identified as statistically anomalous by Isolation Forest")

    # ── Check / refresh controls ──────────────────────────────────────────────
    col_check, col_info = st.columns([1, 3])
    with col_check:
        if st.button("🔍 Check for New Anomalies", type="primary", use_container_width=True):
            with st.spinner("Running Isolation Forest on all available data…"):
                ok, resp = call_check_anomalies()
            if ok:
                st.cache_data.clear()
                n     = resp.get("anomalies_found", 0)
                total = resp.get("total_days", 0)
                if resp.get("status") == "insufficient_data":
                    st.warning(resp.get("message", "Insufficient data for anomaly detection."))
                else:
                    st.success(f"Checked {total} days — found {n} anomalous day(s).", icon="✅")
                st.rerun()
            else:
                st.error(f"Check failed: {resp}")
    with col_info:
        st.info(
            "Runs Isolation Forest across all historical data. "
            "Re-check whenever new images are uploaded to detect new anomalies."
        )

    st.markdown("---")

    with st.spinner("Loading anomaly data…"):
        data = fetch_anomaly()

    if not data:
        st.success("✅ No anomalies detected in the database.")
        st.caption("All days show normal waste composition patterns.")
        return

    st.metric("Total Anomalous Days", len(data))
    st.caption(
        "Anomaly score: more negative = more anomalous (Isolation Forest decision function)."
    )

    st.subheader("Anomaly Score Timeline")
    st.plotly_chart(make_anomaly_chart(data), use_container_width=True)

    st.subheader("Anomaly Details")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df_display = df[["date", "anomaly_score", "annotation_count", "explanation"]].copy()
    df_display["anomaly_score"] = df_display["anomaly_score"].round(4)
    df_display = df_display.sort_values("anomaly_score")

    st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "anomaly_score": st.column_config.NumberColumn(
                "Anomaly Score", format="%.4f"
            ),
            "annotation_count": st.column_config.NumberColumn("Annotation Count"),
            "explanation": st.column_config.TextColumn("Explanation", width="large"),
        },
    )


# ── Page: AI Insights ─────────────────────────────────────────────────────────
def render_ai_insights():
    st.title("AI Insights")
    st.caption("LLM-generated operational reports — auto-updates daily at 18:00")

    # ── Auto-generate logic ───────────────────────────────────────────────────
    from datetime import datetime as _dt
    _now        = _dt.now()
    _today      = _now.date().isoformat()
    _weekday    = _now.weekday()           # 0=Monday … 6=Sunday
    _week_label = _now.strftime("%Y-W%W")  # unique per calendar week

    # Daily report: auto-generate at 18:00 every day
    _daily_key = f"auto_daily_{_today}"
    if _now.hour >= 18 and _daily_key not in st.session_state:
        st.session_state[_daily_key] = True
        _ok, _resp = call_generate_insights("daily_report", force=False)
        if _ok and isinstance(_resp, dict) and _resp.get("status") == "generated":
            st.cache_data.clear()
            st.rerun()

    # Weekly report: auto-generate every Monday (looks back at previous 7 days)
    _weekly_key = f"auto_weekly_{_week_label}"
    if _weekday == 0 and _weekly_key not in st.session_state:
        st.session_state[_weekly_key] = True
        _ok_w, _resp_w = call_generate_insights("weekly_report", force=False)
        if _ok_w and isinstance(_resp_w, dict) and _resp_w.get("status") == "generated":
            st.cache_data.clear()
            st.rerun()

    # ── Controls row ─────────────────────────────────────────────────────────
    col_filter, col_daily, col_weekly, col_refresh = st.columns([2, 1.1, 1.1, 0.7])
    with col_filter:
        insight_type_filter = st.selectbox(
            "Insight type",
            options=["All", "daily_report", "weekly_report", "anomaly_explanation"],
            index=0,
        )
    with col_daily:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 Generate Daily", use_container_width=True):
            with st.spinner("Generating daily report…"):
                ok, resp = call_generate_insights("daily_report", force=True)
            if ok:
                st.cache_data.clear()
                st.success("Daily report generated!", icon="✅")
                st.rerun()
            else:
                st.error(f"Failed: {resp}")
    with col_weekly:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📅 Generate Weekly", use_container_width=True):
            with st.spinner("Generating weekly report…"):
                ok, resp = call_generate_insights("weekly_report", force=True)
            if ok:
                st.cache_data.clear()
                st.success("Weekly report generated!", icon="✅")
                st.rerun()
            else:
                st.error(f"Failed: {resp}")
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Schedule info banner ──────────────────────────────────────────────────
    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    if _now.hour >= 18:
        st.caption(f"🕕 Daily auto-generated at 18:00 · 📅 Weekly auto-generated every Monday")
    else:
        st.caption(
            f"⏰ Next daily report: 18:00 today  ·  "
            f"📅 Next weekly report: {'today (Monday)' if _weekday == 0 else 'Monday'}"
        )

    selected_type = None if insight_type_filter == "All" else insight_type_filter

    with st.spinner("Loading insights…"):
        data = fetch_insights(insight_type=selected_type)

    if not data:
        st.info(
            "No insights yet. Click **📊 Generate Daily** to create today's report, "
            "or run `notebooks/05_llm.ipynb` to generate historical insights."
        )
        return

    st.markdown(f"Showing **{len(data)}** insight(s)")
    st.markdown("---")

    _BADGE_COLORS = {
        "daily_report":        "#27ae60",  # green
        "weekly_report":       "#3498DB",  # blue
        "anomaly_explanation": "#E95500",  # orange
    }
    _BADGE_LABELS = {
        "daily_report":        "📊 Daily Report",
        "weekly_report":       "📅 Weekly Report",
        "anomaly_explanation": "⚠️ Anomaly",
    }

    for item in data:
        itype   = item.get("insight_type", "unknown")
        idate   = item.get("insight_date", "")
        content = item.get("content", "")

        badge_color = _BADGE_COLORS.get(itype, "#888888")
        badge_label = _BADGE_LABELS.get(itype, itype)

        st.markdown(
            f'<span style="background:{badge_color};color:white;padding:3px 10px;'
            f'border-radius:4px;font-size:0.8em;font-weight:600">{badge_label}</span>'
            f'&nbsp;&nbsp;<span style="color:#888;font-size:0.85em">{idate}</span>',
            unsafe_allow_html=True,
        )

        # Daily & weekly reports: parse TREND / ANOMALY / ACTION sections
        if itype in ("daily_report", "weekly_report") and any(
            kw in content for kw in ("TREND", "ANOMALY", "ACTION")
        ):
            sections = {"TREND": "", "ANOMALY": "", "ACTION": ""}
            current  = None
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.upper().startswith("TREND"):
                    current = "TREND"
                    # Keep text on same line as the keyword (after "TREND: ")
                    rest = stripped[6:].lstrip(": ").strip()
                    if rest:
                        sections["TREND"] += rest + "\n"
                elif stripped.upper().startswith("ANOMALY"):
                    current = "ANOMALY"
                    rest = stripped[7:].lstrip(": ").strip()
                    if rest:
                        sections["ANOMALY"] += rest + "\n"
                elif stripped.upper().startswith("ACTION"):
                    current = "ACTION"
                    rest = stripped[6:].lstrip(": ").strip()
                    if rest:
                        sections["ACTION"] += rest + "\n"
                elif current:
                    sections[current] += line + "\n"

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**📈 Trend**")
                st.markdown(sections["TREND"].strip() or "—")
            with c2:
                st.markdown("**⚠️ Anomaly**")
                st.markdown(sections["ANOMALY"].strip() or "—")
            with c3:
                st.markdown("**✅ Action**")
                st.markdown(sections["ACTION"].strip() or "—")
        else:
            st.markdown(content)

        st.divider()


# ── Page: Ask AI ──────────────────────────────────────────────────────────────

_PLATFORM_GUIDE = """
## 📋 Platform Guide — Industrial Waste Intelligence

| Page | Description |
|------|-------------|
| **Dashboard** | Overview: API status, 30-day waste trend chart, system summary (anomaly count, insight count) |
| **Image Analysis** | Upload JPG/PNG images → AI segmentation (Mask R-CNN) → waste composition % per class |
| **Analysis Records** | Browse per-image results, filter by date range, export CSV, undo/delete saved records |
| **Trend Analysis** | Line chart of historical daily composition %; filter by class and date range |
| **Forecast** | ML-predicted composition for next 30 days; compares with today's actual uploads |
| **Anomaly Detection** | Identifies days with unusual waste composition using Isolation Forest |
| **AI Insights** | LLM-generated daily and weekly operational reports; auto-generates at 18:00 |
| **Ask AI** | This page — ask questions in English or Thai; queries live database + knowledge base |

**5 waste classes:** Metal · Mixed Waste · Paper-Cardboard · Plastic · Wood

**Workflow:** Upload images → Save to Records → View in Trend Analysis → Generate Insights
"""

_HELP_TRIGGERS = [
    "how to use", "how do i use", "how does this work", "how does it work",
    "what is this", "what does this", "what can i", "what are the pages",
    "explain the", "describe the", "tell me about",
    "dashboard page", "image analysis page", "records page", "trend analysis page",
    "forecast page", "anomaly page", "insights page", "ask ai page",
    "คืออะไร", "ใช้ยังไง", "อธิบาย", "แต่ละหน้า", "วิธีใช้", "platform guide",
    "help me understand", "guide",
]
_DATA_OVERRIDES = [
    "how much", "how many", "percent", "%", "today", "yesterday",
    "last week", "last month", "metal", "plastic", "wood", "mixed", "paper",
    "anomal", "forecast", "trend", "วันนี้", "เมื่อวาน", "สัปดาห์",
]


def _is_help_question(q: str) -> bool:
    q_lower = q.lower()
    has_help = any(kw in q_lower for kw in _HELP_TRIGGERS)
    has_data = any(kw in q_lower for kw in _DATA_OVERRIDES)
    return has_help and not has_data


def render_ask_ai():
    st.title("Ask AI")
    st.caption("Ask questions about waste data or platform usage — powered by live DB + RAG with Llama 3.2")

    with st.expander("📋 Platform Guide", expanded=False):
        st.markdown(_PLATFORM_GUIDE)

    with st.expander("💡 Suggested questions"):
        st.markdown(
            "**About current data:**\n"
            "- How many percent of Metal today?\n"
            "- What is the dominant waste type this week?\n"
            "- Were there any anomalies recently?\n\n"
            "**About trends & forecasts:**\n"
            "- What is the trend for Mixed Waste?\n"
            "- Which waste class is predicted to be highest next week?\n\n"
            "**About the platform:**\n"
            "- How do I use this platform?\n"
            "- What does the Trend Analysis page do?\n"
            "- What is ML Forecast?"
        )

    question = st.text_area(
        "Your question",
        placeholder="e.g. How many percent of Plastic today? / What does the Forecast page do?",
        height=100,
        key="ask_ai_input",
    )

    col_submit, _ = st.columns([1, 5])
    with col_submit:
        submit = st.button("Ask", type="primary", use_container_width=True)

    if submit and not question.strip():
        st.warning("Please enter a question before submitting.")
        return

    if submit and question.strip():
        q = question.strip()

        # Platform help questions → return local guide instantly (no API call)
        if _is_help_question(q):
            st.session_state["last_question"] = q
            st.session_state["last_answer"] = {
                "answer": _PLATFORM_GUIDE,
                "sources": ["platform_guide"],
            }
        else:
            with st.spinner("Querying live database + knowledge base… (may take a moment)"):
                result = call_query(q)

            if result is None:
                st.error(
                    "Query failed — API unavailable or Ollama not running. "
                    "Ensure Ollama is running locally with the `llama3.2:3b` model."
                )
                return

            st.session_state["last_question"] = q
            st.session_state["last_answer"]   = result

    # Display persisted answer
    if st.session_state.get("last_answer"):
        result = st.session_state["last_answer"]
        last_q = st.session_state.get("last_question", "")

        st.markdown("---")
        st.subheader("Answer")
        if last_q:
            st.markdown(f"**Q:** {last_q}")
        st.markdown(result.get("answer", "No answer returned."))

        sources = result.get("sources", [])
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**{i}.** {source}")


# ── Main entry point ──────────────────────────────────────────────────────────
inject_css()
page = render_sidebar()

if page == "Dashboard":
    render_dashboard()
elif page == "Image Analysis":
    render_image_analysis()
elif page == "Analysis Records":
    render_analysis_records()
elif page == "Trend Analysis":
    render_trend_analysis()
elif page == "Forecast":
    render_forecast()
elif page == "Anomaly Detection":
    render_anomaly_detection()
elif page == "AI Insights":
    render_ai_insights()
elif page == "Ask AI":
    render_ask_ai()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import random

st.set_page_config(
    page_title="抖音数据分析平台",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {text-align:center; padding:1.5rem 0; background:linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%); border-radius:15px; margin-bottom:2rem;}
    .main-header h1 {color:#0E1117; font-size:2.5rem; font-weight:800; margin:0;}
    .main-header p {color:#0E1117; opacity:0.8; font-size:1.1rem; margin:0.3rem 0 0 0;}
    .metric-card {background:#262730; border-radius:12px; padding:1.2rem; text-align:center; border:1px solid #3A3A4A; transition:transform 0.2s;}
    .metric-card:hover {transform:translateY(-3px); border-color:#FF6B6B;}
    .metric-card .label {color:#888; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;}
    .metric-card .value {color:#FAFAFA; font-size:2rem; font-weight:700; margin:0.3rem 0;}
    .metric-card .sub {color:#FF6B6B; font-size:0.8rem;}
    .section-title {font-size:1.5rem; font-weight:700; margin:1.5rem 0 1rem 0; padding-bottom:0.5rem; border-bottom:2px solid #FF6B6B; color:#FAFAFA;}
    .insight-box {background:#1E1E2E; border-left:4px solid #FF6B6B; border-radius:8px; padding:0.8rem 1.2rem; margin:0.5rem 0; color:#CCC;}
    .stTabs [data-baseweb="tab-list"] {gap:2px;}
    .stTabs [data-baseweb="tab"] {border-radius:8px 8px 0 0; padding:0.5rem 1rem;}
    .stTabs [aria-selected="true"] {background-color:#FF6B6B !important; color:#0E1117 !important;}
    div[data-testid="stSidebarUserContent"] {padding-top:1rem;}
    .st-emotion-cache-16txtl3 h1 {color:#FAFAFA;}
    .st-emotion-cache-1wivap2 {color:#FAFAFA;}
    [data-testid="stMetricValue"] {color:#FAFAFA;}
    [data-testid="stMetricDelta"] {color:#FF6B6B;}
</style>
""", unsafe_allow_html=True)


SAMPLE_SIZE = 30000
PLOT_SAMPLE = 5000
CITY_NAMES = {
    8.0: "北京", 21.0: "上海", 22.0: "广州", 45.0: "深圳", 31.0: "杭州",
    38.0: "南京", 80.0: "成都", 109.0: "武汉", 116.0: "长沙", 138.0: "郑州",
    169.0: "西安", 166.0: "重庆", 54.0: "苏州", 65.0: "天津", 121.0: "东莞",
    89.0: "宁波", 69.0: "厦门", 6.0: "沈阳", 40.0: "青岛", 30.0: "昆明",
    76.0: "大连", 46.0: "济南", 106.0: "哈尔滨", 202.0: "福州",
    13.0: "石家庄", 63.0: "合肥", 92.0: "南昌", 113.0: "贵阳",
    101.0: "兰州", 115.0: "温州", 19.0: "无锡", 16.0: "佛山",
    72.0: "南宁", 26.0: "常州", 5.0: "太原", 45.0: "珠海",
    203.0: "海口", 18.0: "呼和浩特", 213.0: "西宁", 330.0: "乌鲁木齐",
    292.0: "拉萨", 311.0: "银川", 323.0: "香港", 250.0: "台北",
}


@st.cache_data(show_spinner="正在加载数据(data*.csv)... 数据集约 170 万条记录，请稍候")
def load_data():
    import glob
    files = sorted(glob.glob("data*.csv"))
    dfs = [pd.read_csv(f, index_col=0) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df["user_city"] = df["user_city"].astype(float)
    df["item_city"] = df["item_city"].astype(float)
    df["city_name_user"] = df["user_city"].map(CITY_NAMES).fillna("其他")
    df["city_name_item"] = df["item_city"].map(CITY_NAMES).fillna("其他")
    return df


@st.cache_data(ttl=3600)
def get_sample(df, n=PLOT_SAMPLE):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


@st.cache_data(ttl=3600)
def get_user_agg(df):
    return df.groupby("uid").agg(
        浏览作品数=("item_id", "nunique"),
        总点赞数=("like", "sum"),
        观看完整数=("finish", "sum"),
        观看城市数=("user_city", "nunique"),
        活跃天数=("date", "nunique"),
    ).reset_index()


@st.cache_data(ttl=3600)
def get_author_agg(df):
    return df.groupby("author_id").agg(
        发布作品数=("item_id", "nunique"),
        作品平均时长=("duration_time", "mean"),
        总点赞数=("like", "sum"),
        总观看完整数=("finish", "sum"),
        去过城市数=("item_city", "nunique"),
        总浏览量=("item_id", "count"),
    ).reset_index()


@st.cache_data(ttl=3600)
def get_item_agg(df):
    return df.groupby("item_id").agg(
        浏览量=("uid", "count"),
        点赞量=("like", "sum"),
        观看完整数=("finish", "sum"),
        发布城市=("city_name_item", "first"),
        背景音乐=("music_id", "first"),
        时长=("duration_time", "first"),
    ).reset_index()


@st.cache_data(ttl=3600)
def get_daily_stats(df):
    return df.groupby("date").agg(
        浏览量=("uid", "count"),
        点赞数=("like", "sum"),
        观看完整数=("finish", "sum"),
        独立用户=("uid", "nunique"),
        独立作者=("author_id", "nunique"),
    ).reset_index()


data = load_data()
sample_df = get_sample(data, SAMPLE_SIZE)
user_agg = get_user_agg(data)
author_agg = get_author_agg(data)
item_agg = get_item_agg(data)
daily_stats = get_daily_stats(data)

with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🎵 抖音分析</h2>", unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "导航面板",
        ["📊 数据概览", "👤 用户分析", "🎬 作者分析", "📹 作品分析", "🔍 数据探索"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### ⏱ 时间筛选")
    date_range = st.date_input(
        "日期范围",
        value=(data["date"].min(), data["date"].max()),
        min_value=data["date"].min(),
        max_value=data["date"].max(),
    )
    if len(date_range) == 2:
        mask = (data["date"] >= pd.Timestamp(date_range[0])) & (
            data["date"] <= pd.Timestamp(date_range[1])
        )
        filtered = data[mask]
    else:
        filtered = data

    st.divider()
    st.markdown("### 📍 城市筛选")
    all_cities = sorted(data["city_name_user"].unique())
    selected_cities = st.multiselect(
        "选择城市", all_cities, default=[], placeholder="全部城市"
    )
    if selected_cities:
        filtered = filtered[filtered["city_name_user"].isin(selected_cities)]

    st.divider()
    st.markdown("### ⚙️ 采样设置")
    use_full = st.toggle("使用全量数据（较慢）", value=False)
    plot_n = PLOT_SAMPLE if not use_full else min(PLOT_SAMPLE * 3, len(filtered))

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#666;font-size:0.8rem;'>📊 抖音数据分析平台 v1.0</p>",
        unsafe_allow_html=True,
    )


def render_metric(label, value, delta=None, help_text=""):
    delta_str = f"↑ {delta}" if delta else None
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {f'<div class="sub">{help_text}</div>' if help_text else ''}
    </div>
    """,
        unsafe_allow_html=True,
    )


def show_overview():
    st.markdown(
        "<div class='main-header'><h1>🎵 抖音数据分析平台</h1><p>Douyin Data Analytics Platform</p></div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric("总记录数", f"{len(data):,}", help_text=f"筛选后: {len(filtered):,}")
    with col2:
        render_metric("独立用户", f"{data['uid'].nunique():,}")
    with col3:
        render_metric("独立作者", f"{data['author_id'].nunique():,}")
    with col4:
        render_metric("独立作品", f"{data['item_id'].nunique():,}")
    with col5:
        render_metric("覆盖城市", f"{data['user_city'].nunique():,}")

    st.markdown("<div class='section-title'>📈 每日趋势</div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["浏览量趋势", "点赞趋势", "用户 & 作者趋势"])
    with tab1:
        fig = px.line(
            daily_stats, x="date", y="浏览量",
            title="每日浏览量变化趋势",
            labels={"date": "日期", "浏览量": "浏览量"},
            template="plotly_dark", color_discrete_sequence=["#FF6B6B"],
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.line(
            daily_stats, x="date", y="点赞数",
            title="每日点赞数变化趋势",
            labels={"date": "日期", "点赞数": "点赞数"},
            template="plotly_dark", color_discrete_sequence=["#FFE66D"],
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=daily_stats["date"], y=daily_stats["独立用户"],
                       name="独立用户", line=dict(color="#4ECDC4")),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=daily_stats["date"], y=daily_stats["独立作者"],
                       name="独立作者", line=dict(color="#FF6B6B")),
            secondary_y=True,
        )
        fig.update_layout(
            title="每日独立用户与独立作者趋势",
            template="plotly_dark", hovermode="x unified",
        )
        fig.update_yaxes(title_text="独立用户", secondary_y=False)
        fig.update_yaxes(title_text="独立作者", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>🏙️ 用户城市分布 Top15</div>", unsafe_allow_html=True)
        city_counts = data["city_name_user"].value_counts().head(15)
        fig = px.bar(
            x=city_counts.values, y=city_counts.index,
            orientation="h", text=city_counts.values,
            labels={"x": "用户数", "y": "城市"},
            template="plotly_dark",
            color=city_counts.values, color_continuous_scale="Viridis",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, height=450, xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>🎵 作品城市分布 Top15</div>", unsafe_allow_html=True)
        city_item_counts = data["city_name_item"].value_counts().head(15)
        fig = px.bar(
            x=city_item_counts.values, y=city_item_counts.index,
            orientation="h", text=city_item_counts.values,
            labels={"x": "作品数", "y": "城市"},
            template="plotly_dark",
            color=city_item_counts.values, color_continuous_scale="Plasma",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, height=450, xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>🕐 小时活跃度热力图</div>", unsafe_allow_html=True)
        hourly = data.groupby(["H", "date"]).size().reset_index(name="count")
        hourly_avg = hourly.groupby("H")["count"].mean().reset_index()
        fig = px.bar(
            hourly_avg, x="H", y="count",
            labels={"H": "小时", "count": "平均活跃度"},
            template="plotly_dark",
            color="count", color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>📊 渠道分布</div>", unsafe_allow_html=True)
        channel_counts = data["channel"].value_counts().reset_index()
        channel_counts.columns = ["channel", "count"]
        fig = px.pie(
            channel_counts, values="count", names="channel",
            title="渠道分布",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.RdBu,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>💡 数据洞察</div>", unsafe_allow_html=True)
    total_likes = data["like"].sum()
    total_finish = data["finish"].sum()
    like_rate = total_likes / len(data) * 100
    finish_rate = total_finish / len(data) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='insight-box'>📌 <b>总点赞数:</b> {total_likes:,} 次<br>"
            f"<b>点赞率:</b> {like_rate:.2f}%</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='insight-box'>📌 <b>观看完整数:</b> {total_finish:,} 次<br>"
            f"<b>完播率:</b> {finish_rate:.2f}%</div>",
            unsafe_allow_html=True,
        )
    with col3:
        avg_duration = data["duration_time"].mean()
        st.markdown(
            f"<div class='insight-box'>📌 <b>平均视频时长:</b> {avg_duration:.1f}s<br>"
            f"<b>数据跨度:</b> {data['date'].min().date()} ~ {data['date'].max().date()}</div>",
            unsafe_allow_html=True,
        )


def show_user_analysis():
    st.markdown(
        "<div class='main-header' style='background:linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);'>"
        "<h1>👤 用户数据分析</h1><p>User Behavior Analysis</p></div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("独立用户", f"{user_agg['uid'].nunique():,}")
    with col2:
        avg_items = user_agg["浏览作品数"].mean()
        render_metric("人均浏览作品", f"{avg_items:.1f}")
    with col3:
        avg_likes = user_agg["总点赞数"].mean()
        render_metric("人均点赞", f"{avg_likes:.2f}")
    with col4:
        avg_finish = user_agg["观看完整数"].mean()
        render_metric("人均完整观看", f"{avg_finish:.2f}")

    st.markdown("<div class='section-title'>📊 用户行为分布</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "浏览作品数分布", "点赞数分布", "观看完整数分布", "活跃城市数分布",
    ])

    with tab1:
        fig = px.histogram(
            user_agg, x="浏览作品数", nbins=50,
            title="用户浏览作品数分布",
            labels={"浏览作品数": "浏览作品数", "count": "用户数"},
            template="plotly_dark",
            color_discrete_sequence=["#4ECDC4"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        p99 = user_agg["浏览作品数"].quantile(0.99)
        st.markdown(
            f"<div class='insight-box'>💡 99% 的用户浏览作品数不超过 {int(p99)} 个，"
            f"大部分用户为轻度浏览者</div>",
            unsafe_allow_html=True,
        )

    with tab2:
        like_dist = user_agg[user_agg["总点赞数"] > 0]
        fig = px.histogram(
            like_dist, x="总点赞数", nbins=50,
            title="用户点赞数分布（仅有点赞行为的用户）",
            labels={"总点赞数": "点赞数", "count": "用户数"},
            template="plotly_dark",
            color_discrete_sequence=["#FFE66D"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        like_rate_users = (user_agg["总点赞数"] > 0).sum() / len(user_agg) * 100
        st.markdown(
            f"<div class='insight-box'>💡 {like_rate_users:.1f}% 的用户有过点赞行为，"
            f"用户平均点赞 {user_agg['总点赞数'].mean():.2f} 次</div>",
            unsafe_allow_html=True,
        )

    with tab3:
        fig = px.histogram(
            user_agg, x="观看完整数", nbins=50,
            title="用户观看完整数分布",
            labels={"观看完整数": "完整观看数", "count": "用户数"},
            template="plotly_dark",
            color_discrete_sequence=["#FF6B6B"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        finish_rate_users = (user_agg["观看完整数"] > 0).sum() / len(user_agg) * 100
        st.markdown(
            f"<div class='insight-box'>💡 {finish_rate_users:.1f}% 的用户有过完整观看行为，"
            f"用户平均完整观看 {user_agg['观看完整数'].mean():.2f} 次</div>",
            unsafe_allow_html=True,
        )

    with tab4:
        fig = px.histogram(
            user_agg, x="观看城市数", nbins=30,
            title="用户观看城市数分布",
            labels={"观看城市数": "城市数", "count": "用户数"},
            template="plotly_dark",
            color_discrete_sequence=["#A78BFA"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        avg_cities = user_agg["观看城市数"].mean()
        st.markdown(
            f"<div class='insight-box'>💡 用户平均观看来自 {avg_cities:.1f} 个城市的作品，"
            f"跨城市内容消费活跃</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>🏆 用户排行榜</div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["浏览最多 Top20", "点赞最多 Top20", "最活跃用户 Top20"])

    with tab1:
        top_viewers = user_agg.nlargest(20, "浏览作品数")
        fig = px.bar(
            top_viewers, x="uid", y="浏览作品数",
            title="浏览作品数最多的用户",
            labels={"uid": "用户ID", "浏览作品数": "浏览作品数"},
            template="plotly_dark",
            color="浏览作品数", color_continuous_scale="Viridis",
            text="浏览作品数",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top_likers = user_agg.nlargest(20, "总点赞数")
        fig = px.bar(
            top_likers, x="uid", y="总点赞数",
            title="点赞数最多的用户",
            labels={"uid": "用户ID", "总点赞数": "点赞数"},
            template="plotly_dark",
            color="总点赞数", color_continuous_scale="Plasma",
            text="总点赞数",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        top_active = user_agg.nlargest(20, "活跃天数")
        fig = px.bar(
            top_active, x="uid", y="活跃天数",
            title="最活跃用户（活跃天数最多）",
            labels={"uid": "用户ID", "活跃天数": "活跃天数"},
            template="plotly_dark",
            color="活跃天数", color_continuous_scale="Reds",
            text="活跃天数",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>🌍 用户地理分析</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        city_user_counts = data["city_name_user"].value_counts().reset_index()
        city_user_counts.columns = ["城市", "用户数"]
        fig = px.treemap(
            city_user_counts.head(30),
            path=["城市"], values="用户数",
            title="用户城市分布 Treemap",
            template="plotly_dark",
            color="用户数", color_continuous_scale="Viridis",
        )
        fig.update_traces(textinfo="label+value")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 用户城市分布详情")
        st.dataframe(
            city_user_counts.style.background_gradient(cmap="Viridis", subset=["用户数"]),
            use_container_width=True, height=500,
        )

    st.markdown("<div class='section-title'>📈 用户行为相关性</div>", unsafe_allow_html=True)
    user_sample = user_agg.sample(n=min(2000, len(user_agg)), random_state=42)
    fig = px.scatter_matrix(
        user_sample,
        dimensions=["浏览作品数", "总点赞数", "观看完整数", "活跃天数"],
        title="用户行为多维散点矩阵",
        template="plotly_dark",
        color="总点赞数", color_continuous_scale="Viridis",
        opacity=0.5,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>🔍 用户查询</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        search_uid = st.number_input("输入用户ID查询", min_value=0, step=1, value=0)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if search_uid > 0 and st.button("🔍 查询", type="primary"):
            user_data = data[data["uid"] == search_uid]
            if len(user_data) > 0:
                st.success(f"找到用户 #{search_uid} 的 {len(user_data):,} 条记录")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    render_metric(
                        "浏览作品数", user_data["item_id"].nunique(),
                    )
                with col_b:
                    render_metric("总点赞数", int(user_data["like"].sum()))
                with col_c:
                    render_metric("完整观看", int(user_data["finish"].sum()))
                with col_d:
                    render_metric("活动天数", user_data["date"].nunique())

                st.dataframe(
                    user_data[["item_id", "author_id", "city_name_item",
                               "like", "finish", "duration_time", "date"]]
                    .head(100),
                    use_container_width=True,
                )
            else:
                st.warning(f"未找到用户 #{search_uid}")


def show_author_analysis():
    st.markdown(
        "<div class='main-header' style='background:linear-gradient(135deg, #A78BFA 0%, #6D28D9 100%);'>"
        "<h1>🎬 作者数据分析</h1><p>Creator Analytics</p></div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("独立作者", f"{author_agg['author_id'].nunique():,}")
    with col2:
        avg_items = author_agg["发布作品数"].mean()
        render_metric("人均发布作品", f"{avg_items:.1f}")
    with col3:
        avg_dur = author_agg["作品平均时长"].mean()
        render_metric("作品平均时长", f"{avg_dur:.1f}s")
    with col4:
        avg_cities = author_agg["去过城市数"].mean()
        render_metric("人均去过城市", f"{avg_cities:.1f}")

    st.markdown("<div class='section-title'>📊 作者创作分布</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "发布作品数分布", "作品时长分布", "创作活跃度", "城市覆盖分布",
    ])

    with tab1:
        fig = px.histogram(
            author_agg, x="发布作品数", nbins=50,
            title="作者发布作品数分布",
            labels={"发布作品数": "发布作品数", "count": "作者数"},
            template="plotly_dark",
            color_discrete_sequence=["#A78BFA"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        top_10_pct = author_agg["发布作品数"].quantile(0.9)
        st.markdown(
            f"<div class='insight-box'>💡 头部 10% 的作者发布作品数超过 {int(top_10_pct)} 个，"
            f"呈现明显的头部效应</div>",
            unsafe_allow_html=True,
        )

    with tab2:
        fig = px.histogram(
            author_agg, x="作品平均时长", nbins=50,
            title="作者作品平均时长分布",
            labels={"作品平均时长": "平均时长(s)", "count": "作者数"},
            template="plotly_dark",
            color_discrete_sequence=["#FF6B6B"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        author_daily = data.groupby(["author_id", "date"]).size().reset_index(name="count")
        author_activity = author_daily.groupby("author_id")["date"].nunique().reset_index()
        author_activity.columns = ["author_id", "活跃天数"]
        fig = px.histogram(
            author_activity, x="活跃天数", nbins=50,
            title="作者创作活跃度（活跃天数分布）",
            labels={"活跃天数": "活跃天数", "count": "作者数"},
            template="plotly_dark",
            color_discrete_sequence=["#4ECDC4"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = px.histogram(
            author_agg, x="去过城市数", nbins=30,
            title="作者去过城市数分布",
            labels={"去过城市数": "城市数", "count": "作者数"},
            template="plotly_dark",
            color_discrete_sequence=["#FFE66D"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<div class='insight-box'>💡 作者平均去过 {avg_cities:.1f} 个城市发布作品，"
            f"跨城创作者占比可观</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>🏆 作者排行榜</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "高产作者 Top20", "最长视频 Top20", "高赞作者 Top20", "多地作者 Top20",
    ])

    with tab1:
        top_prolific = author_agg.nlargest(20, "发布作品数")
        fig = px.bar(
            top_prolific, x="author_id", y="发布作品数",
            title="发布作品最多的作者",
            labels={"author_id": "作者ID", "发布作品数": "发布作品数"},
            template="plotly_dark",
            color="发布作品数", color_continuous_scale="Viridis",
            text="发布作品数",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top_duration = author_agg.nlargest(20, "作品平均时长")
        fig = px.bar(
            top_duration, x="author_id", y="作品平均时长",
            title="作品平均时长最长的作者",
            labels={"author_id": "作者ID", "作品平均时长": "平均时长(s)"},
            template="plotly_dark",
            color="作品平均时长", color_continuous_scale="Plasma",
            text=top_duration["作品平均时长"].round(1),
        )
        fig.update_traces(texttemplate="%{text}s", textposition="outside")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        top_liked_authors = author_agg.nlargest(20, "总点赞数")
        fig = px.bar(
            top_liked_authors, x="author_id", y="总点赞数",
            title="获得点赞最多的作者",
            labels={"author_id": "作者ID", "总点赞数": "总点赞数"},
            template="plotly_dark",
            color="总点赞数", color_continuous_scale="Reds",
            text="总点赞数",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        top_cities_author = author_agg.nlargest(20, "去过城市数")
        fig = px.bar(
            top_cities_author, x="author_id", y="去过城市数",
            title="去过最多城市的作者",
            labels={"author_id": "作者ID", "去过城市数": "城市数"},
            template="plotly_dark",
            color="去过城市数", color_continuous_scale="Teal",
            text="去过城市数",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>📈 作者创作分析</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        author_sample = author_agg.sample(n=min(2000, len(author_agg)), random_state=42)
        fig = px.scatter(
            author_sample, x="发布作品数", y="作品平均时长",
            size="总浏览量", color="总点赞数",
            title="发布作品数 vs 作品时长（气泡大小=浏览量）",
            labels={"发布作品数": "发布作品数", "作品平均时长": "平均时长(s)"},
            template="plotly_dark",
            color_continuous_scale="Viridis",
            hover_data=["author_id"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            author_sample, x="去过城市数", y="总点赞数",
            size="发布作品数", color="总浏览量",
            title="城市覆盖 vs 点赞数（气泡大小=发布作品数）",
            labels={"去过城市数": "去过城市数", "总点赞数": "总点赞数"},
            template="plotly_dark",
            color_continuous_scale="Plasma",
            hover_data=["author_id"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>📅 作者创作时间分析</div>", unsafe_allow_html=True)
    author_hourly = data.groupby(["author_id", "H"]).size().reset_index(name="count")
    author_hourly_avg = author_hourly.groupby("H")["count"].mean().reset_index()
    fig = px.line(
        author_hourly_avg, x="H", y="count",
        title="作者平均创作/发布小时分布",
        labels={"H": "小时", "count": "平均活跃度"},
        template="plotly_dark",
        markers=True, color_discrete_sequence=["#A78BFA"],
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    peak_hour = author_hourly_avg.loc[author_hourly_avg["count"].idxmax(), "H"]
    st.markdown(
        f"<div class='insight-box'>💡 作者最活跃时段为 {int(peak_hour)} 时，"
        f"建议在此时间段发布作品以获得更多曝光</div>",
        unsafe_allow_html=True,
    )


def show_item_analysis():
    st.markdown(
        "<div class='main-header' style='background:linear-gradient(135deg, #FFE66D 0%, #F59E0B 100%);'>"
        "<h1>📹 作品数据分析</h1><p>Content Performance Analytics</p></div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("独立作品", f"{item_agg['item_id'].nunique():,}")
    with col2:
        avg_views = item_agg["浏览量"].mean()
        render_metric("平均浏览量", f"{avg_views:.1f}")
    with col3:
        avg_likes = item_agg["点赞量"].mean()
        render_metric("平均点赞量", f"{avg_likes:.2f}")
    with col4:
        avg_dur = item_agg["时长"].mean()
        render_metric("平均时长", f"{avg_dur:.1f}s")

    st.markdown("<div class='section-title'>📊 作品表现分布</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "浏览量分布", "点赞量分布", "作品时长分布", "背景音乐分析",
    ])

    with tab1:
        fig = px.histogram(
            item_agg, x="浏览量", nbins=50,
            title="作品浏览量分布",
            labels={"浏览量": "浏览量", "count": "作品数"},
            template="plotly_dark",
            color_discrete_sequence=["#4ECDC4"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        p95 = item_agg["浏览量"].quantile(0.95)
        st.markdown(
            f"<div class='insight-box'>💡 95% 的作品浏览量不超过 {int(p95)} 次，"
            f"头部爆款作品占据大部分流量</div>",
            unsafe_allow_html=True,
        )

    with tab2:
        liked_items = item_agg[item_agg["点赞量"] > 0]
        fig = px.histogram(
            liked_items, x="点赞量", nbins=50,
            title="作品点赞量分布（有点赞的作品）",
            labels={"点赞量": "点赞量", "count": "作品数"},
            template="plotly_dark",
            color_discrete_sequence=["#FFE66D"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        like_pct = len(liked_items) / len(item_agg) * 100
        st.markdown(
            f"<div class='insight-box'>💡 仅 {like_pct:.1f}% 的作品获得过点赞，"
            f"点赞作品平均获得 {liked_items['点赞量'].mean():.2f} 赞</div>",
            unsafe_allow_html=True,
        )

    with tab3:
        fig = px.histogram(
            item_agg, x="时长", nbins=50,
            title="作品时长分布",
            labels={"时长": "时长(s)", "count": "作品数"},
            template="plotly_dark",
            color_discrete_sequence=["#FF6B6B"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<div class='insight-box'>💡 作品平均时长 {avg_dur:.1f}s，"
            f"中位数时长 {item_agg['时长'].median():.1f}s，短视频占据主流</div>",
            unsafe_allow_html=True,
        )

    with tab4:
        music_counts = item_agg["背景音乐"].value_counts().head(20)
        fig = px.bar(
            x=music_counts.values, y=music_counts.index.astype(str),
            orientation="h", text=music_counts.values,
            labels={"x": "作品数", "y": "音乐ID"},
            title="最热门的背景音乐 Top20",
            template="plotly_dark",
            color=music_counts.values, color_continuous_scale="Viridis",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>🏙️ 作品城市分析</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        city_item_counts = data["city_name_item"].value_counts().head(15)
        fig = px.bar(
            x=city_item_counts.values, y=city_item_counts.index,
            orientation="h", text=city_item_counts.values,
            labels={"x": "作品数", "y": "城市"},
            title="作品发布城市 Top15",
            template="plotly_dark",
            color=city_item_counts.values, color_continuous_scale="Plasma",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city_like_rate = data.groupby("city_name_item")["like"].mean().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=city_like_rate.values * 100, y=city_like_rate.index,
            orientation="h", text=city_like_rate.round(3) * 100,
            labels={"x": "点赞率(%)", "y": "城市"},
            title="城市点赞率 Top15",
            template="plotly_dark",
            color=city_like_rate.values, color_continuous_scale="Reds",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>📈 作品互动分析</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        item_sample = item_agg.sample(n=min(3000, len(item_agg)), random_state=42)
        fig = px.scatter(
            item_sample, x="浏览量", y="点赞量",
            size="观看完整数", color="时长",
            title="浏览量 vs 点赞量（气泡大小=完整观看数）",
            labels={"浏览量": "浏览量", "点赞量": "点赞量"},
            template="plotly_dark",
            color_continuous_scale="Viridis",
            hover_data=["item_id"],
            opacity=0.6,
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        duration_bins = pd.cut(item_agg["时长"], bins=20)
        dur_analysis = item_agg.groupby(duration_bins, observed=True).agg(
            平均点赞率=("点赞量", lambda x: x.sum() / len(x) * 100),
            作品数=("item_id", "count"),
        ).reset_index()
        dur_analysis["时长区间"] = dur_analysis["时长"].astype(str)
        fig = px.bar(
            dur_analysis, x="时长区间", y="平均点赞率",
            title="不同时长作品的点赞率对比",
            labels={"时长区间": "时长区间", "平均点赞率": "点赞率(%)"},
            template="plotly_dark",
            color="平均点赞率", color_continuous_scale="Reds",
            text=dur_analysis["平均点赞率"].round(2),
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>🎵 音乐影响力分析</div>", unsafe_allow_html=True)
    music_analysis = item_agg.groupby("背景音乐").agg(
        使用作品数=("item_id", "count"),
        平均点赞量=("点赞量", "mean"),
        平均浏览量=("浏览量", "mean"),
    ).reset_index()
    music_analysis = music_analysis[music_analysis["使用作品数"] >= 5].sort_values(
        "平均点赞量", ascending=False
    ).head(20)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            music_analysis, x="使用作品数", y="平均点赞量",
            size="平均浏览量", color="平均浏览量",
            title="音乐使用频率 vs 平均点赞量",
            labels={"使用作品数": "使用作品数", "平均点赞量": "平均点赞量"},
            template="plotly_dark",
            color_continuous_scale="Viridis",
            hover_data=["背景音乐"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_music = music_analysis.nlargest(10, "使用作品数")
        fig = px.bar(
            top_music, x=top_music["背景音乐"].astype(str), y="使用作品数",
            title="最受欢迎的背景音乐 Top10",
            labels={"背景音乐": "音乐ID", "使用作品数": "使用次数"},
            template="plotly_dark",
            color="平均点赞量", color_continuous_scale="Viridis",
            text="使用作品数",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>🕐 作品发布时间分析</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        item_hourly = data.groupby("H").agg(
            作品数=("item_id", "nunique"),
            点赞率=("like", "mean"),
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=item_hourly["H"], y=item_hourly["作品数"],
                   name="发布作品数", marker_color="#4ECDC4"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=item_hourly["H"], y=item_hourly["点赞率"] * 100,
                       name="点赞率", mode="lines+markers",
                       line=dict(color="#FF6B6B", width=3)),
            secondary_y=True,
        )
        fig.update_layout(
            title="各小时发布作品数及点赞率",
            template="plotly_dark", hovermode="x unified",
        )
        fig.update_yaxes(title_text="发布作品数", secondary_y=False)
        fig.update_yaxes(title_text="点赞率(%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        data["weekday"] = data["date"].dt.weekday
        weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday_stats = data.groupby("weekday").agg(
            作品数=("item_id", "nunique"),
            点赞率=("like", "mean"),
        ).reset_index()
        weekday_stats["weekday_name"] = weekday_stats["weekday"].map(
            lambda x: weekday_names[x]
        )
        fig = px.bar(
            weekday_stats, x="weekday_name", y="作品数",
            title="周内各天发布作品数",
            labels={"weekday_name": "星期", "作品数": "发布作品数"},
            template="plotly_dark",
            color="点赞率", color_continuous_scale="Viridis",
            text="作品数",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>🔍 作品查询</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        search_item = st.number_input("输入作品ID查询", min_value=0, step=1, value=0, key="item_search")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if search_item > 0 and st.button("🔍 查询作品", type="primary", key="item_search_btn"):
            item_data = data[data["item_id"] == search_item]
            if len(item_data) > 0:
                st.success(f"找到作品 #{search_item} 的 {len(item_data):,} 条浏览记录")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    render_metric("总浏览量", len(item_data))
                with col_b:
                    render_metric("总点赞数", int(item_data["like"].sum()))
                with col_c:
                    render_metric("完整观看", int(item_data["finish"].sum()))
                with col_d:
                    render_metric(
                        "发布城市",
                        item_data["city_name_item"].iloc[0],
                    )

                st.dataframe(
                    item_data[["uid", "author_id", "city_name_user",
                               "like", "finish", "duration_time", "date"]]
                    .head(100),
                    use_container_width=True,
                )
            else:
                st.warning(f"未找到作品 #{search_item}")


def show_data_explorer():
    st.markdown(
        "<div class='main-header' style='background:linear-gradient(135deg, #34D399 0%, #059669 100%);'>"
        "<h1>🔍 数据探索</h1><p>Raw Data Explorer</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>📋 数据预览</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        rows_to_show = st.slider("显示行数", 100, 10000, 1000, step=100)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 随机采样", use_container_width=True):
            st.rerun()

    display_df = filtered.sample(n=min(rows_to_show, len(filtered)), random_state=42)
    st.dataframe(
        display_df[["uid", "user_city", "item_id", "author_id", "item_city",
                     "like", "finish", "duration_time", "date", "H"]]
        .style.background_gradient(cmap="Viridis", subset=["like", "finish", "duration_time"]),
        use_container_width=True, height=500,
    )

    st.markdown("<div class='section-title'>📊 数据统计摘要</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 数值字段描述统计")
        desc_stats = filtered.describe()
        st.dataframe(desc_stats.style.background_gradient(cmap="Blues"), use_container_width=True)
    with col2:
        st.markdown("### 缺失值检查")
        missing = pd.DataFrame({
            "字段": filtered.columns,
            "非空数": filtered.count().values,
            "缺失数": filtered.isnull().sum().values,
            "缺失率(%)": (filtered.isnull().sum() / len(filtered) * 100).round(2).values,
        })
        st.dataframe(
            missing.style.background_gradient(cmap="Reds", subset=["缺失率(%)"]),
            use_container_width=True,
        )

    st.markdown("<div class='section-title'>📈 数据相关性热力图</div>", unsafe_allow_html=True)
    numeric_cols = ["like", "finish", "duration_time"]
    corr = filtered[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".3f", aspect="auto",
        title="数值字段相关性矩阵",
        template="plotly_dark",
        color_continuous_scale="RdBu_r",
        labels={"color": "相关系数"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>⬇️ 数据下载</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered.head(50000).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 下载 CSV (前 50,000 条)",
            data=csv,
            file_name="douyin_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.markdown(
            f"<div class='insight-box'>📌 数据集共 {len(data):,} 条记录<br>"
            f"当前筛选后 {len(filtered):,} 条<br>"
            f"字段: {', '.join(data.columns)}</div>",
            unsafe_allow_html=True,
        )


if page == "📊 数据概览":
    show_overview()
elif page == "👤 用户分析":
    show_user_analysis()
elif page == "🎬 作者分析":
    show_author_analysis()
elif page == "📹 作品分析":
    show_item_analysis()
elif page == "🔍 数据探索":
    show_data_explorer()

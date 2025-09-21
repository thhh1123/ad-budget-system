import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import tempfile, os, datetime, subprocess
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary  # type: ignore

# 页面配置
st.set_page_config(page_title="广告预算分配系统", layout="wide")
st.title("📊 广告预算分配与数据分析系统")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

@st.cache_data
def generate_sample_data():
    """生成模拟广告数据，活动天数限定为1-6天"""
    # 明确设置活动天数为1到6天
    days = list(range(1, 7))  # 1,2,3,4,5,6
    channels = ["渠道_1", "渠道_2", "渠道_3", "渠道_4", "渠道_5"]
    campaigns = ["活动_1", "活动_2"]
    
    data = []
    for campaign in campaigns:
        # 为每个活动生成完整的6天数据
        for day in days:
            for channel in channels:
                # 生成合理的广告数据
                impressions = np.random.randint(500, 15000)
                clicks = np.random.randint(5, max(6, int(impressions * 0.05)))  # 确保点击量合理
                cost = np.random.uniform(5, 100)
                orders = np.random.randint(0, min(1, int(clicks * 0.15)) + 1)  # 订单数不超过点击量的15%
                revenue = orders * np.random.uniform(50, 500)
                
                data.append([
                    campaign, day, channel, f"camp_{np.random.randint(1000,9999)}", 
                    f"group_{np.random.randint(1000,9999)}", revenue, orders, 
                    cost, impressions, clicks
                ])
    
    df = pd.DataFrame(
        data, 
        columns=["活动", "活动第几天", "渠道", "广告系列ID_h", "广告组ID_h", 
                "业绩", "订单", "花费", "曝光", "点击"]
    )
    
    # 确保活动第几天是整数类型
    df["活动第几天"] = df["活动第几天"].astype(int)
    
    return df
    

# ---------------- 通用清洗函数 ----------------
@st.cache_data
def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """返回清洗并衍生指标后的 DataFrame"""
    df = raw_df.copy()
    
    # 确保必须的列存在
    required_columns = ["活动第几天", "渠道", "业绩", "订单", "花费", "曝光", "点击"]
    for col in required_columns:
        if col not in df.columns:
            st.error(f"数据缺少必要的列：{col}")
            st.stop()
    
    # 缺失值处理
    for col in ["花费", "点击", "曝光", "业绩", "订单", "活动第几天"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.dropna(subset=["花费", "点击", "曝光", "业绩"])

    # 避免 0 除，计算关键指标
    df["CTR"] = np.where(df["曝光"] > 0, df["点击"] / df["曝光"], 0)
    df["CPC"] = np.where(df["点击"] > 0, df["花费"] / df["点击"], 0)
    df["CPM"] = np.where(df["曝光"] > 0, df["花费"] / df["曝光"] * 1000, 0)
    df["ROI"] = np.where(df["花费"] > 0, df["业绩"] / df["花费"], 0)
    df["CVR"] = np.where(df["点击"] > 0, df["订单"] / df["点击"], 0)
    
    # 统一保留2位小数
    df[["CTR", "CPC", "CPM", "ROI", "CVR"]] = df[["CTR", "CPC", "CPM", "ROI", "CVR"]].round(2)
    
    return df

# ---------------- 侧边栏数据选择 ----------------
with st.sidebar:
    st.subheader("📂 数据来源")
    data_source = st.radio("选择数据来源", ["上传数据", "使用模拟数据"])
    
    if data_source == "上传数据":
        uploaded = st.file_uploader("上传广告数据 Excel 或 CSV", type=["xlsx", "csv"])
        if uploaded is None:
            st.info("请上传包含以下列的数据：活动、活动第几天、渠道、广告系列ID_h、广告组ID_h、业绩、订单、花费、曝光、点击")
            st.stop()
        # 根据文件类型选择合适的读取方式
        if uploaded.name.endswith('.csv'):
            df = clean_data(pd.read_csv(uploaded))
        else:
            df = clean_data(pd.read_excel(uploaded))
    else:
        st.info("使用模拟数据进行演示，包含5个渠道和2个活动")
        df = clean_data(generate_sample_data())
    
    # 数据导出功能
    st.markdown("---")
    st.subheader("💾 数据导出")
    csv_full = df.to_csv(index=False).encode()
    st.download_button("导出清洗后完整数据", csv_full, "cleaned_data.csv", "text/csv")

# ---------------- 多维度筛选 ----------------
st.subheader("🔍 多维度数据筛选")
col1, col2, col3 = st.columns(3)

with col1:
    # 活动筛选
    if "活动" in df.columns:
        all_campaigns = df["活动"].unique()
        selected_campaigns = st.multiselect("选择活动", all_campaigns, default=all_campaigns)
    else:
        selected_campaigns = []

with col2:
    # 渠道筛选
    all_channels = df["渠道"].unique()
    selected_channels = st.multiselect("选择渠道", all_channels, default=all_channels)

with col3:
    # 活动天数筛选
    min_day = int(df["活动第几天"].min())
    max_day = int(df["活动第几天"].max())
    day_range = st.slider(
        "选择活动天数范围", 
        min_value=min_day, 
        max_value=max_day, 
        value=(min_day, max_day)
    )

# 应用筛选
filtered_df = df[df["渠道"].isin(selected_channels)]
if "活动" in df.columns and len(selected_campaigns) > 0:
    filtered_df = filtered_df[filtered_df["活动"].isin(selected_campaigns)]
filtered_df = filtered_df[(filtered_df["活动第几天"] >= day_range[0]) & 
                         (filtered_df["活动第几天"] <= day_range[1])]

# 业绩区间筛选
st.subheader("🎯 业绩区间筛选")
min_revenue = int(filtered_df["业绩"].min())
max_revenue = int(filtered_df["业绩"].max())

rev_range = st.slider(
    "选择业绩范围", 
    min_value=min_revenue, 
    max_value=max_revenue, 
    value=(min_revenue, max_revenue)
)

filtered_df = filtered_df[(filtered_df["业绩"] >= rev_range[0]) & (filtered_df["业绩"] <= rev_range[1])]

# ---------------- 指标卡片 ----------------
total_cost = filtered_df["花费"].sum()
total_revenue = filtered_df["业绩"].sum()
overall_roi = total_revenue / total_cost if total_cost else 0
total_orders = filtered_df["订单"].sum()
total_clicks = filtered_df["点击"].sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("总花费", f"{total_cost:,.0f} 元")
col2.metric("总业绩", f"{total_revenue:,.0f} 元")
col3.metric("整体 ROI", f"{overall_roi:.2f}")
col4.metric("总订单量", f"{total_orders:,.0f}")
col5.metric("总点击量", f"{total_clicks:,.0f}")

# ---------------- 渠道表现表 + 下载 ----------------
channel_summary = (
    filtered_df.groupby("渠道")
    .agg({
        "花费": "sum", 
        "订单": "sum", 
        "业绩": "sum", 
        "ROI": "mean",
        "CTR": "mean",
        "CVR": "mean"
    })
    .reset_index()
    .sort_values("ROI", ascending=False)
)

st.subheader("📈 渠道表现汇总")
st.dataframe(channel_summary.style.format({
    "ROI": "{:.2f}",
    "CTR": "{:.2%}",
    "CVR": "{:.2%}"
}))

# 下载按钮
csv = channel_summary.to_csv(index=False).encode()
st.download_button("下载渠道汇总表", csv, "channel_summary.csv", "text/csv")

# ---------------- 预算分配策略选择 ----------------
st.subheader("💰 预算分配设置")
budget_wan = st.number_input("输入总预算（万元）", min_value=1, value=100)
budget_yuan = budget_wan * 10_000

# 分配策略选择
strategy = st.selectbox("选择预算分配策略", [
    "ROI比例分配", 
    "平均分配", 
    "点击量优先分配"
])

# 避免 ROI 全 0
if channel_summary["ROI"].sum() <= 0 and strategy == "ROI比例分配":
    st.warning("所选渠道ROI均为0，自动切换为平均分配策略")
    strategy = "平均分配"

# 根据不同策略计算分配
if strategy == "ROI比例分配":
    channel_summary["建议分配"] = (
        budget_yuan * channel_summary["ROI"] / channel_summary["ROI"].sum()
    )
elif strategy == "平均分配":
    # 平均分配
    channel_summary["建议分配"] = budget_yuan / len(channel_summary)
else:  # 点击量优先
    # 基于历史点击量比例分配
    click_summary = filtered_df.groupby("渠道")["点击"].sum().reset_index()
    click_summary = click_summary.rename(columns={"点击": "总点击"})
    channel_summary = pd.merge(channel_summary, click_summary, on="渠道")
    channel_summary["建议分配"] = budget_yuan * channel_summary["总点击"] / channel_summary["总点击"].sum()

channel_summary["建议占比"] = channel_summary["建议分配"] / budget_yuan

# 显示分配结果
st.subheader(f"🎯 {strategy}结果")
st.dataframe(
    channel_summary[["渠道", "建议分配", "建议占比"]].style.format(
        {"建议分配": "{:,.0f} 元", "建议占比": "{:.1%}"}
    )
)

# 活动天数预算分配
days = sorted(filtered_df["活动第几天"].unique())
# 基于历史表现分配预算权重
daily_performance = filtered_df.groupby("活动第几天")["业绩"].sum().reset_index()
day_weights = daily_performance["业绩"] / daily_performance["业绩"].sum()
daily_budget = (budget_yuan * day_weights).astype(int)
daily_df = pd.DataFrame({"活动第几天": days, "预算": daily_budget})

st.subheader("📅 活动天数预算分配")
st.dataframe(daily_df.style.format({"预算": "{:,.0f} 元"}))

# 预算分配饼图
fig, ax = plt.subplots()
ax.pie(
    channel_summary["建议分配"],
    labels=channel_summary["渠道"],
    autopct="%.1f%%",
    startangle=90,
)
ax.set_title(f"{strategy}占比")
st.pyplot(fig)

# ---------------- 单渠道明细 ----------------
st.subheader("🔍 单渠道详情")
sel_chan = st.selectbox("选择渠道查看详情", filtered_df["渠道"].unique())
st.dataframe(filtered_df[filtered_df["渠道"] == sel_chan])

# ================= 线性规划预算优化 =================
@st.cache_data
def lp_allocate(budget_yuan, df_channel, min_roi=0.1, max_weight=0.6):
    """
    目标：最大化总业绩（GMV）
    约束：1.总花费<=预算 2.单渠道<=max_weight 3.ROI>=min_roi
    """
    ch = df_channel["渠道"].tolist()
    roi = dict(zip(ch, df_channel["ROI"]))
    spend = dict(zip(ch, df_channel["花费"]))
    n = len(ch)

    prob = LpProblem("BudgetOpt", LpMaximize)
    x = {i: LpVariable(f"x_{i}", lowBound=0, upBound=1) for i in ch}  # 投资比例

    # 目标：max ∑(x_i * spend_i * ROI_i) ≈ max ∑(x_i * 业绩_i)
    prob += lpSum([x[i] * spend[i] * roi[i] for i in ch])

    # 约束
    prob += lpSum([x[i] * spend[i] for i in ch]) <= budget_yuan          # 总预算
    for i in ch:
        prob += x[i] <= max_weight                                       # 单渠道上限
        prob += roi[i] * x[i] >= min_roi * x[i]                          # ROI 门槛

    prob.solve()
    res = pd.DataFrame({
        "渠道": ch,
        "投资比例": [x[i].value() for i in ch],
        "分配金额": [x[i].value() * spend[i] for i in ch]
    })
    return res

# ---------------- 线性规划参数设置与结果 ----------------
st.subheader("📊 线性规划最优分配")
with st.expander("设置线性规划参数", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_wan_lp = st.number_input("总预算（万元）", 1, 1000, 100)
    with col2:
        min_roi = st.number_input("最低 ROI 要求", 0.0, 1.0, 0.15)
    with col3:
        max_share = st.number_input("单渠道占比上限", 0.1, 1.0, 0.6)

budget_yuan_lp = budget_wan_lp * 10_000
opt_df = lp_allocate(budget_yuan_lp, channel_summary, min_roi, max_share)

# 线性规划结果展示
st.dataframe(opt_df.style.format({"投资比例": "{:.1%}", "分配金额": "{:,.0f} 元"}))

# 线性规划结果可视化
fig_lp = px.bar(
    opt_df.sort_values("分配金额", ascending=True), 
    x="分配金额", y="渠道", orientation="h",
    title="线性规划预算分配结果"
)
st.plotly_chart(fig_lp, use_container_width=True)

# ================= 转化漏斗 =================
def funnel(df):
    stage = ["曝光", "点击", "订单"]
    vals  = [df["曝光"].sum(), df["点击"].sum(), df["订单"].sum()]
    fig = go.Figure(go.Funnel(
        x=vals, 
        y=stage, 
        textinfo="value+percent initial"
    ))
    fig.update_layout(title="整体转化漏斗", height=350)
    return fig

st.subheader("🚦 转化漏斗分析")
st.plotly_chart(funnel(filtered_df), use_container_width=True)

# ---------------- 点击量 vs 订单量散点图 ----------------
st.subheader("📈 点击量与订单量关系")
scatter_df = filtered_df.groupby("渠道").agg({"点击":"sum", "订单":"sum"}).reset_index()
fig_scatter = px.scatter(
    scatter_df, 
    x="点击", 
    y="订单", 
    color="渠道", 
    hover_name="渠道", 
    size="订单", 
    title="点击量与订单量相关性分析"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- 花费与业绩关系分析 ----------------
st.subheader("📈 花费与业绩关系分析")
# 按活动天数聚合数据
spend_revenue_df = filtered_df.groupby("活动第几天").agg({
    "花费": "sum", 
    "业绩": "sum"
}).reset_index()

# 创建散点图并添加趋势线
fig_spend_rev = px.scatter(
    spend_revenue_df, 
    x="花费", 
    y="业绩", 
    trendline="ols",
    title="花费与业绩散点图及线性回归",
    labels={"花费": "总花费（元）", "业绩": "总业绩（元）"}
)

# 获取回归结果
results = px.get_trendline_results(fig_spend_rev)
r_squared = results.iloc[0]["px_fit_results"].rsquared

# 在图表中添加R²值
fig_spend_rev.add_annotation(
    x=0.5, y=1.05,
    text=f"R² = {r_squared:.4f}",
    showarrow=False,
    xref="paper", yref="paper",
    font=dict(size=14)
)

st.plotly_chart(fig_spend_rev, use_container_width=True)

# ---------------- 活动趋势分析 (基于活动天数) ----------------
st.subheader("📅 活动趋势分析 (基于活动天数)")

# 确保 '活动第几天' 为数值类型，以便正确排序
filtered_df['活动第几天'] = pd.to_numeric(filtered_df['活动第几天'])

# 按活动天数聚合数据
trend_df = filtered_df.groupby('活动第几天').agg({
    '花费': 'sum',
    '业绩': 'sum',
    '点击': 'sum',
    '订单': 'sum'
}).reset_index()

# 1. 业绩与订单趋势 (双Y轴)
fig_trend = px.line(trend_df, x='活动第几天', y='业绩', title='活动天数 vs 业绩/订单')
fig_trend.update_layout(yaxis_title='业绩')

# 添加订单量的次坐标轴
fig_trend.add_scatter(x=trend_df['活动第几天'], y=trend_df['订单'], 
                     mode='lines', name='订单', yaxis='y2', line=dict(color='orange'))
fig_trend.update_layout(
    yaxis2=dict(
        title='订单',
        overlaying='y',
        side='right'
    )
)
st.plotly_chart(fig_trend, use_container_width=True)

# 2. 多指标对比 (基于活动天数)
metrics = st.multiselect(
    "选择要对比的指标", 
    ["花费", "业绩", "点击", "订单"], 
    default=["花费", "业绩"]
)
if metrics:
    fig_multi = px.line(
        trend_df, 
        x="活动第几天", 
        y=metrics, 
        title="活动天数 vs 多指标趋势对比"
    )
    st.plotly_chart(fig_multi, use_container_width=True)

# ---------------- 渠道雷达图 ----------------
st.subheader("📊 渠道综合表现雷达图")
channels = st.multiselect(
    "选择要对比的渠道", 
    filtered_df["渠道"].unique(), 
    default=filtered_df["渠道"].unique()[:3]
)
if len(channels) >= 2:
    radar_df = channel_summary[channel_summary["渠道"].isin(channels)].copy()
    
    # 提取要展示的指标
    radar_data = {
        "渠道": radar_df["渠道"],
        "ROI": radar_df["ROI"] / radar_df["ROI"].max(),  # 归一化
        "点击率(CTR)": radar_df["CTR"],
        "转化率(CVR)": radar_df["CVR"]
    }
    radar_df = pd.DataFrame(radar_data)
    
    fig = px.line_polar(
        radar_df, 
        r="ROI", 
        theta="渠道", 
        line_close=True, 
        title="渠道ROI对比（归一化）"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- 渠道对比动态图 ----------------
st.subheader("📊 渠道指标对比")
metric_sel = st.selectbox("选择指标对比", ["ROI", "CTR", "CPC", "CVR"])
fig_bar = px.bar(
    channel_summary.sort_values(metric_sel, ascending=True),
    x=metric_sel, 
    y="渠道", 
    orientation="h",
    color=metric_sel, 
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- What-if 灵敏度分析 ----------------
with st.expander("🔧 What-if 灵敏度分析", expanded=False):
    roi_range = np.linspace(0.05, 0.3, 6)
    share_range = np.linspace(0.3, 0.8, 6)
    heatmap = np.array([
        [lp_allocate(budget_yuan_lp, channel_summary, r, s)["分配金额"].sum()
         for s in share_range] for r in roi_range
    ])
    fig_hm = px.imshow(
        heatmap, 
        x=np.round(share_range, 2), 
        y=np.round(roi_range, 2),
        labels=dict(x="单渠道上限", y="最低 ROI", color="总GMV"),
        aspect="auto", 
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# ---------------- 一键导出运行环境 ----------------
st.subheader("📦 系统资源导出")
col1, col2 = st.columns(2)
with col1:
    req = "\n".join(["streamlit", "plotly", "pulp", "pandas", "numpy", "openpyxl", "scipy"])
    st.download_button("下载依赖清单", req.encode(), "requirements.txt")

with col2:
    readme = "# 广告预算分配系统\n\n## 运行方式\n```bash\npip install -r requirements.txt\nstreamlit run ad_app.py\n```\n\n## 功能\n- 支持上传数据或使用模拟数据\n- 多维度数据筛选与分析\n- 三种预算分配策略（ROI比例/平均/点击量优先）\n- 线性规划优化预算分配\n- 丰富的数据可视化与报告导出"
    st.download_button("下载使用说明", readme.encode(), "README.md")

# ---------------- 文本报告 ----------------
def gen_report(opt_df, channel_summary, budget_yuan_lp, min_roi, max_share, filtered_df):
    total_allocated = opt_df["分配金额"].sum()
    top3 = opt_df.nlargest(3, "分配金额")
    
    lines = [
        "广告预算分配报告",
        "="*40,
        f"生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"总预算：{budget_yuan_lp:,.0f} 元",
        f"最低 ROI 要求：{min_roi:.1%}",
        f"单渠道占比上限：{max_share:.1%}",
        f"已分配总额：{total_allocated:,.0f} 元",
        "-"*40,
        "",
        "一、渠道预算分配明细",
        "-"*20
    ]
    
    for _, r in opt_df.sort_values("分配金额", ascending=False).iterrows():
        channel = r["渠道"]
        amount = r["分配金额"]
        ratio = r["投资比例"]
        roi = channel_summary.loc[channel_summary["渠道"] == channel, "ROI"].values[0]
        lines.append(f"{channel}：{amount:,.0f} 元（占比 {ratio:.1%}，ROI {roi:.2f}）")
    
    lines.extend([
        "",
        "二、活动天数预算分配计划",
        "-"*20
    ])
    
    days = sorted(filtered_df["活动第几天"].unique())
    daily_performance = filtered_df.groupby("活动第几天")["业绩"].sum().reset_index()
    day_weights = daily_performance["业绩"] / daily_performance["业绩"].sum()
    daily_budget = (budget_yuan_lp * day_weights).astype(int)
    for day, bud in zip(days, daily_budget):
        lines.append(f"第{day}天：{bud:,.0f} 元")
    
    lines.extend([
        "",
        "三、核心发现与建议",
        "-"*20,
        f"1. 重点投放渠道前三：{', '.join(top3['渠道'].tolist())}",
        f"2. 建议优先保证高ROI渠道的预算需求",
        f"3. 注意控制单渠道占比不超过 {max_share:.0%}",
        f"4. 建议持续监控渠道表现，及时调整预算分配"
    ])
    
    return "\n".join(lines)

if st.button("📄 生成分析报告"):
    report_txt = gen_report(opt_df, channel_summary, budget_yuan_lp, min_roi, max_share, filtered_df)
    st.download_button("下载报告", report_txt.encode(), "budget_report.txt")

# 侧边栏使用说明
with st.sidebar:
    st.markdown("---")
    st.info("""**使用指南**  
    1. 选择数据来源（上传或模拟）
    2. 使用多维度筛选器选择分析范围
    3. 设置总预算和分配策略
    4. 查看线性规划优化结果
    5. 通过可视化图表分析数据
    6. 下载报告和相关数据
    
    系统支持多种广告指标分析，帮助优化预算分配决策""")
    
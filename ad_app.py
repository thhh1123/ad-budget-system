import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import tempfile, os, datetime, subprocess
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary  # 用于线性规划

# ---------------- 页面基础配置 ----------------
# 设置页面标题和布局
st.set_page_config(page_title="广告预算分配系统", layout="wide")
st.title("📊 广告预算分配与数据分析系统")

# 配置matplotlib中文字体，确保图表中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题


# ---------------- 数据生成与清洗模块 ----------------
@st.cache_data  # 缓存数据生成结果，提升性能
def generate_sample_data():
    """
    生成模拟广告数据用于演示
    
    数据包含：
    - 2个活动（活动_1、活动_2）
    - 5个渠道（渠道_1至渠道_5）
    - 6天的活动数据
    - 核心指标：曝光、点击、花费、订单、业绩等
    """
    # 限定活动天数为1-6天
    days = list(range(1, 7))  # 1,2,3,4,5,6
    channels = ["渠道_1", "渠道_2", "渠道_3", "渠道_4", "渠道_5"]
    campaigns = ["活动_1", "活动_2"]
    
    data = []
    for campaign in campaigns:
        for day in days:
            for channel in channels:
                # 生成合理范围内的随机数据
                impressions = np.random.randint(500, 15000)  # 曝光量：500-15000
                clicks = np.random.randint(5, max(6, int(impressions * 0.05)))  # 点击量：不超过曝光量的5%
                cost = np.random.uniform(5, 100)  # 花费：5-100元
                orders = np.random.randint(0, min(1, int(clicks * 0.15)) + 1)  # 订单量：不超过点击量的15%
                revenue = orders * np.random.uniform(50, 500)  # 业绩：订单数*单品价值(50-500)
                
                data.append([
                    campaign, day, channel, f"camp_{np.random.randint(1000,9999)}", 
                    f"group_{np.random.randint(1000,9999)}", revenue, orders, 
                    cost, impressions, clicks
                ])
    
    # 构建DataFrame并指定列名
    df = pd.DataFrame(
        data, 
        columns=["活动", "活动第几天", "渠道", "广告系列ID_h", "广告组ID_h", 
                "业绩", "订单", "花费", "曝光", "点击"]
    )
    
    # 确保活动天数为整数类型
    df["活动第几天"] = df["活动第几天"].astype(int)
    
    return df


@st.cache_data  # 缓存清洗结果，避免重复计算
def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗与指标衍生处理
    
    处理步骤：
    1. 检查必要列是否存在
    2. 缺失值处理（数值列填充0）
    3. 计算核心指标（CTR、CPC等）
    4. 数据格式化（保留2位小数）
    
    参数:
        raw_df: 原始数据DataFrame
    返回:
        清洗后的DataFrame
    """
    df = raw_df.copy()  # 复制数据避免修改原数据
    
    # 检查必要列是否存在
    required_columns = ["活动第几天", "渠道", "业绩", "订单", "花费", "曝光", "点击"]
    for col in required_columns:
        if col not in df.columns:
            st.error(f"数据缺少必要的列：{col}")
            st.stop()  # 缺少必要列时终止程序
    
    # 缺失值处理：将数值列转换为数值类型并填充0
    for col in ["花费", "点击", "曝光", "业绩", "订单", "活动第几天"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 删除关键指标为空的行
    df = df.dropna(subset=["花费", "点击", "曝光", "业绩"])

    # 计算核心指标（避免除零错误）
    df["CTR"] = np.where(df["曝光"] > 0, df["点击"] / df["曝光"], 0)  # 点击率 = 点击/曝光
    df["CPC"] = np.where(df["点击"] > 0, df["花费"] / df["点击"], 0)  # 单次点击成本 = 花费/点击
    df["CPM"] = np.where(df["曝光"] > 0, df["花费"] / df["曝光"] * 1000, 0)  # 千次曝光成本
    df["ROI"] = np.where(df["花费"] > 0, df["业绩"] / df["花费"], 0)  # 投资回报率 = 业绩/花费
    df["CVR"] = np.where(df["点击"] > 0, df["订单"] / df["点击"], 0)  # 转化率 = 订单/点击
    
    # 统一保留2位小数
    df[["CTR", "CPC", "CPM", "ROI", "CVR"]] = df[["CTR", "CPC", "CPM", "ROI", "CVR"]].round(2)
    
    return df


# ---------------- 数据输入与筛选模块 ----------------
# 侧边栏：数据来源选择
with st.sidebar:
    st.subheader("📂 数据来源")
    data_source = st.radio("选择数据来源", ["上传数据", "使用模拟数据"])
    
    # 根据选择加载数据
    if data_source == "上传数据":
        uploaded = st.file_uploader("上传广告数据 Excel 或 CSV", type=["xlsx", "csv"])
        if uploaded is None:
            st.info("请上传包含以下列的数据：活动、活动第几天、渠道、广告系列ID_h、广告组ID_h、业绩、订单、花费、曝光、点击")
            st.stop()  # 未上传文件时终止程序
        # 根据文件类型读取数据
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


# 多维度筛选区域
st.subheader("🔍 多维度数据筛选")
col1, col2, col3 = st.columns(3)  # 分三列展示筛选条件

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

# 应用筛选条件
filtered_df = df[df["渠道"].isin(selected_channels)]  # 筛选渠道
if "活动" in df.columns and len(selected_campaigns) > 0:
    filtered_df = filtered_df[filtered_df["活动"].isin(selected_campaigns)]  # 筛选活动
filtered_df = filtered_df[(filtered_df["活动第几天"] >= day_range[0]) & 
                         (filtered_df["活动第几天"] <= day_range[1])]  # 筛选天数

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


# ---------------- 核心指标展示模块 ----------------
# 计算核心指标
total_cost = filtered_df["花费"].sum()  # 总花费
total_revenue = filtered_df["业绩"].sum()  # 总业绩
overall_roi = total_revenue / total_cost if total_cost else 0  # 整体ROI
total_orders = filtered_df["订单"].sum()  # 总订单量
total_clicks = filtered_df["点击"].sum()  # 总点击量

# 以卡片形式展示核心指标
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("总花费", f"{total_cost:,.0f} 元")
col2.metric("总业绩", f"{total_revenue:,.0f} 元")
col3.metric("整体 ROI", f"{overall_roi:.2f}")
col4.metric("总订单量", f"{total_orders:,.0f}")
col5.metric("总点击量", f"{total_clicks:,.0f}")


# ---------------- 渠道表现分析模块 ----------------
# 渠道表现汇总表
channel_summary = (
    filtered_df.groupby("渠道")
    .agg({
        "花费": "sum",  # 总花费
        "订单": "sum",  # 总订单
        "业绩": "sum",  # 总业绩
        "ROI": "mean",  # 平均ROI
        "CTR": "mean",  # 平均点击率
        "CVR": "mean"   # 平均转化率
    })
    .reset_index()
    .sort_values("ROI", ascending=False)  # 按ROI降序排列
)

st.subheader("📈 渠道表现汇总")
# 格式化显示数据
st.dataframe(channel_summary.style.format({
    "ROI": "{:.2f}",
    "CTR": "{:.2%}",
    "CVR": "{:.2%}"
}))

# 下载渠道汇总表
csv = channel_summary.to_csv(index=False).encode()
st.download_button("下载渠道汇总表", csv, "channel_summary.csv", "text/csv")


# ---------------- 预算分配策略模块 ----------------
st.subheader("💰 预算分配设置")
# 输入总预算（万元转换为元）
budget_wan = st.number_input("输入总预算（万元）", min_value=1, value=100)
budget_yuan = budget_wan * 10_000  # 转换为元

# 选择分配策略
strategy = st.selectbox("选择预算分配策略", [
    "ROI比例分配",  # 按渠道ROI比例分配
    "平均分配",      # 各渠道平均分配
    "点击量优先分配"  # 按历史点击量比例分配
])

# 特殊情况处理：ROI全为0时自动切换为平均分配
if channel_summary["ROI"].sum() <= 0 and strategy == "ROI比例分配":
    st.warning("所选渠道ROI均为0，自动切换为平均分配策略")
    strategy = "平均分配"

# 根据不同策略计算分配金额
if strategy == "ROI比例分配":
    # 按ROI比例分配：ROI越高，分配越多
    channel_summary["建议分配"] = (
        budget_yuan * channel_summary["ROI"] / channel_summary["ROI"].sum()
    )
elif strategy == "平均分配":
    # 平均分配：所有渠道分配相同金额
    channel_summary["建议分配"] = budget_yuan / len(channel_summary)
else:  # 点击量优先分配
    # 按历史点击量比例分配：点击量越高，分配越多
    click_summary = filtered_df.groupby("渠道")["点击"].sum().reset_index()
    click_summary = click_summary.rename(columns={"点击": "总点击"})
    channel_summary = pd.merge(channel_summary, click_summary, on="渠道")
    channel_summary["建议分配"] = budget_yuan * channel_summary["总点击"] / channel_summary["总点击"].sum()

# 计算建议占比
channel_summary["建议占比"] = channel_summary["建议分配"] / budget_yuan

# 显示分配结果
st.subheader(f"🎯 {strategy}结果")
st.dataframe(
    channel_summary[["渠道", "建议分配", "建议占比"]].style.format(
        {"建议分配": "{:,.0f} 元", "建议占比": "{:.1%}"}
    )
)

# 按活动天数分配预算
days = sorted(filtered_df["活动第几天"].unique())
# 基于历史业绩比例分配每日预算
daily_performance = filtered_df.groupby("活动第几天")["业绩"].sum().reset_index()
day_weights = daily_performance["业绩"] / daily_performance["业绩"].sum()  # 每天的业绩占比作为权重
daily_budget = (budget_yuan * day_weights).astype(int)  # 计算每日预算
daily_df = pd.DataFrame({"活动第几天": days, "预算": daily_budget})

st.subheader("📅 活动天数预算分配")
st.dataframe(daily_df.style.format({"预算": "{:,.0f} 元"}))

# 预算分配饼图可视化
fig, ax = plt.subplots()
ax.pie(
    channel_summary["建议分配"],
    labels=channel_summary["渠道"],
    autopct="%.1f%%",  # 显示百分比
    startangle=90,     # 起始角度
)
ax.set_title(f"{strategy}占比")
st.pyplot(fig)

# 单渠道详情查看
st.subheader("🔍 单渠道详情")
sel_chan = st.selectbox("选择渠道查看详情", filtered_df["渠道"].unique())
st.dataframe(filtered_df[filtered_df["渠道"] == sel_chan])


# ---------------- 线性规划优化模块 ----------------
@st.cache_data
def lp_allocate(budget_yuan, df_channel, min_roi=0.1, max_weight=0.6):
    """
    使用线性规划进行预算优化分配
    
    目标：最大化总业绩（GMV）
    约束条件：
    1. 总花费 ≤ 预算
    2. 单渠道预算占比 ≤ max_weight（上限）
    3. 渠道ROI ≥ min_roi（最低要求）
    
    参数:
        budget_yuan: 总预算（元）
        df_channel: 渠道汇总数据（包含ROI和花费）
        min_roi: 最低ROI要求
        max_weight: 单渠道最大占比
    返回:
        包含渠道、投资比例和分配金额的DataFrame
    """
    # 提取渠道列表和关键指标
    ch = df_channel["渠道"].tolist()
    roi = dict(zip(ch, df_channel["ROI"]))  # 渠道ROI字典
    spend = dict(zip(ch, df_channel["花费"]))  # 渠道历史花费字典

    # 创建线性规划问题（最大化目标）
    prob = LpProblem("BudgetOpt", LpMaximize)
    # 定义变量：各渠道的投资比例（0-1之间）
    x = {i: LpVariable(f"x_{i}", lowBound=0, upBound=1) for i in ch}

    # 目标函数：最大化总业绩（≈ 花费 * ROI * 投资比例）
    prob += lpSum([x[i] * spend[i] * roi[i] for i in ch])

    # 约束条件
    prob += lpSum([x[i] * spend[i] for i in ch]) <= budget_yuan  # 总预算约束
    for i in ch:
        prob += x[i] <= max_weight  # 单渠道占比上限
        prob += roi[i] * x[i] >= min_roi * x[i]  # ROI最低要求

    # 求解线性规划
    prob.solve()
    
    # 整理结果
    res = pd.DataFrame({
        "渠道": ch,
        "投资比例": [x[i].value() for i in ch],  # 求解得到的投资比例
        "分配金额": [x[i].value() * spend[i] for i in ch]  # 计算分配金额
    })
    return res


# 线性规划参数设置与结果展示
st.subheader("📊 线性规划最优分配")
with st.expander("设置线性规划参数", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_wan_lp = st.number_input("总预算（万元）", 1, 1000, 100)
    with col2:
        min_roi = st.number_input("最低 ROI 要求", 0.0, 1.0, 0.15)
    with col3:
        max_share = st.number_input("单渠道占比上限", 0.1, 1.0, 0.6)

# 转换预算单位（万元→元）
budget_yuan_lp = budget_wan_lp * 10_000
# 执行线性规划优化
opt_df = lp_allocate(budget_yuan_lp, channel_summary, min_roi, max_share)

# 展示优化结果
st.dataframe(opt_df.style.format({"投资比例": "{:.1%}", "分配金额": "{:,.0f} 元"}))

# 可视化线性规划结果
fig_lp = px.bar(
    opt_df.sort_values("分配金额", ascending=True), 
    x="分配金额", y="渠道", orientation="h",
    title="线性规划预算分配结果"
)
st.plotly_chart(fig_lp, use_container_width=True)


# ---------------- 数据可视化分析模块 ----------------
# 转化漏斗分析
def funnel(df):
    """生成转化漏斗图"""
    stage = ["曝光", "点击", "订单"]  # 转化阶段
    vals  = [df["曝光"].sum(), df["点击"].sum(), df["订单"].sum()]  # 各阶段数值
    fig = go.Figure(go.Funnel(
        x=vals, 
        y=stage, 
        textinfo="value+percent initial"  # 显示数值和占初始值的百分比
    ))
    fig.update_layout(title="整体转化漏斗", height=350)
    return fig

st.subheader("🚦 转化漏斗分析")
st.plotly_chart(funnel(filtered_df), use_container_width=True)


# 点击量与订单量关系散点图
st.subheader("📈 点击量与订单量关系")
scatter_df = filtered_df.groupby("渠道").agg({"点击":"sum", "订单":"sum"}).reset_index()
fig_scatter = px.scatter(
    scatter_df, 
    x="点击", 
    y="订单", 
    color="渠道", 
    hover_name="渠道", 
    size="订单",  # 点的大小由订单量决定
    title="点击量与订单量相关性分析"
)
st.plotly_chart(fig_scatter, use_container_width=True)


# 花费与业绩关系分析
st.subheader("📈 花费与业绩关系分析")
# 按活动天数聚合数据
spend_revenue_df = filtered_df.groupby("活动第几天").agg({
    "花费": "sum", 
    "业绩": "sum"
}).reset_index()

# 创建散点图并添加趋势线（线性回归）
fig_spend_rev = px.scatter(
    spend_revenue_df, 
    x="花费", 
    y="业绩", 
    trendline="ols",  # 普通最小二乘法回归
    title="花费与业绩散点图及线性回归",
    labels={"花费": "总花费（元）", "业绩": "总业绩（元）"}
)

# 获取回归结果，计算R²值（拟合优度）
results = px.get_trendline_results(fig_spend_rev)
r_squared = results.iloc[0]["px_fit_results"].rsquared

# 在图表中添加R²值标注
fig_spend_rev.add_annotation(
    x=0.5, y=1.05,
    text=f"R² = {r_squared:.4f}",  # R²越接近1，拟合效果越好
    showarrow=False,
    xref="paper", yref="paper",  # 相对坐标（0-1范围）
    font=dict(size=14)
)

st.plotly_chart(fig_spend_rev, use_container_width=True)


# 活动趋势分析（基于活动天数）
st.subheader("📅 活动趋势分析 (基于活动天数)")

# 确保'活动第几天'为数值类型，以便正确排序
filtered_df['活动第几天'] = pd.to_numeric(filtered_df['活动第几天'])

# 按活动天数聚合数据
trend_df = filtered_df.groupby('活动第几天').agg({
    '花费': 'sum',
    '业绩': 'sum',
    '点击': 'sum',
    '订单': 'sum'
}).reset_index()

# 1. 业绩与订单趋势（双Y轴图表）
fig_trend = px.line(trend_df, x='活动第几天', y='业绩', title='活动天数 vs 业绩/订单')
fig_trend.update_layout(yaxis_title='业绩')

# 添加订单量的次坐标轴（右侧Y轴）
fig_trend.add_scatter(x=trend_df['活动第几天'], y=trend_df['订单'], 
                     mode='lines', name='订单', yaxis='y2', line=dict(color='orange'))
fig_trend.update_layout(
    yaxis2=dict(
        title='订单',
        overlaying='y',  # 与主Y轴重叠
        side='right'     # 显示在右侧
    )
)
st.plotly_chart(fig_trend, use_container_width=True)

# 2. 多指标对比（可自定义选择指标）
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


# 渠道雷达图（综合表现对比）
st.subheader("📊 渠道综合表现雷达图")
channels = st.multiselect(
    "选择要对比的渠道", 
    filtered_df["渠道"].unique(), 
    default=filtered_df["渠道"].unique()[:3]  # 默认选择前3个渠道
)
if len(channels) >= 2:  # 至少选择2个渠道才显示
    radar_df = channel_summary[channel_summary["渠道"].isin(channels)].copy()
    
    # 提取并归一化指标（便于雷达图对比）
    radar_data = {
        "渠道": radar_df["渠道"],
        "ROI": radar_df["ROI"] / radar_df["ROI"].max(),  # ROI归一化（除以最大值）
        "点击率(CTR)": radar_df["CTR"],
        "转化率(CVR)": radar_df["CVR"]
    }
    radar_df = pd.DataFrame(radar_data)
    
    # 生成雷达图
    fig = px.line_polar(
        radar_df, 
        r="ROI", 
        theta="渠道", 
        line_close=True, 
        title="渠道ROI对比（归一化）"
    )
    st.plotly_chart(fig, use_container_width=True)


# 渠道指标对比条形图
st.subheader("📊 渠道指标对比")
metric_sel = st.selectbox("选择指标对比", ["ROI", "CTR", "CPC", "CVR"])
fig_bar = px.bar(
    channel_summary.sort_values(metric_sel, ascending=True),
    x=metric_sel, 
    y="渠道", 
    orientation="h",  # 水平条形图
    color=metric_sel,  # 按指标值着色
    color_continuous_scale="Viridis"  # 颜色渐变方案
)
st.plotly_chart(fig_bar, use_container_width=True)


# What-if灵敏度分析（参数影响分析）
with st.expander("🔧 What-if 灵敏度分析", expanded=False):
    # 定义参数范围
    roi_range = np.linspace(0.05, 0.3, 6)  # ROI从0.05到0.3，共6个点
    share_range = np.linspace(0.3, 0.8, 6)  # 单渠道上限从0.3到0.8，共6个点
    
    # 生成热力图数据（不同参数组合下的总GMV）
    heatmap = np.array([
        [lp_allocate(budget_yuan_lp, channel_summary, r, s)["分配金额"].sum()
         for s in share_range] for r in roi_range
    ])
    
    # 绘制热力图
    fig_hm = px.imshow(
        heatmap, 
        x=np.round(share_range, 2), 
        y=np.round(roi_range, 2),
        labels=dict(x="单渠道上限", y="最低 ROI", color="总GMV"),
        aspect="auto", 
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ---------------- 系统资源与报告导出模块 ----------------
st.subheader("📦 系统资源导出")
col1, col2 = st.columns(2)
with col1:
    # 导出依赖清单
    req = "\n".join(["streamlit", "plotly", "pulp", "pandas", "numpy", "openpyxl", "scipy"])
    st.download_button("下载依赖清单", req.encode(), "requirements.txt")

with col2:
    # 导出使用说明
    readme = "# 广告预算分配系统\n\n## 运行方式\n```bash\npip install -r requirements.txt\nstreamlit run ad_app.py\n```\n\n## 功能\n- 支持上传数据或使用模拟数据\n- 多维度数据筛选与分析\n- 三种预算分配策略（ROI比例/平均/点击量优先）\n- 线性规划优化预算分配\n- 丰富的数据可视化与报告导出"
    st.download_button("下载使用说明", readme.encode(), "README.md")


# 生成分析报告
def gen_report(opt_df, channel_summary, budget_yuan_lp, min_roi, max_share, filtered_df):
    """生成广告预算分配分析报告文本"""
    total_allocated = opt_df["分配金额"].sum()
    top3 = opt_df.nlargest(3, "分配金额")  # 预算最高的3个渠道
    
    # 报告内容构建
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
    
    # 添加渠道分配明细
    for _, r in opt_df.sort_values("分配金额", ascending=False).iterrows():
        channel = r["渠道"]
        amount = r["分配金额"]
        ratio = r["投资比例"]
        roi = channel_summary.loc[channel_summary["渠道"] == channel, "ROI"].values[0]
        lines.append(f"{channel}：{amount:,.0f} 元（占比 {ratio:.1%}，ROI {roi:.2f}）")
    
    # 添加每日预算分配
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
    
    # 添加核心发现与建议
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

# 生成并下载报告
if st.button("📄 生成分析报告"):
    report_txt = gen_report(opt_df, channel_summary, budget_yuan_lp, min_roi, max_share, filtered_df)
    st.download_button("下载报告", report_txt.encode(), "budget_report.txt")


# 侧边栏使用指南
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


# ---------------- 算法效果评估模块 ----------------
def evaluate_algorithm(opt_result, heuristic_results, total_budget):
    """
    评估线性规划算法效果
    
    参数:
        opt_result: 线性规划结果（包含分配金额和预估业绩）
        heuristic_results: 其他启发式策略的结果
        total_budget: 总预算
    返回:
        包含评估指标的字典
    """
    # 计算线性规划的GMV
    opt_gmv = opt_result["预估业绩"].sum()
    # 计算资源利用率（实际分配金额/总预算）
    resource_util = opt_result["分配金额"].sum() / total_budget
    # 计算ROI达标率（满足最低ROI要求的渠道比例）
    roi_compliance = sum(opt_result["渠道"].apply(
        lambda x: channel_summary[channel_summary["渠道"] == x]["ROI"].values[0] >= min_roi
    )) / len(opt_result)
    
    # 计算相对启发式策略的提升幅度
    improvements = {}
    for name, res in heuristic_results.items():
        heuristic_gmv = res["预估业绩"].sum()
        improvements[name] = (opt_gmv - heuristic_gmv) / heuristic_gmv * 100  # 提升百分比
    
    return {
        "总GMV": opt_gmv,
        "资源利用率": resource_util,
        "ROI达标率": roi_compliance,
        "相对启发式策略提升(%)": improvements
    }


# 评估结果展示
st.subheader("📊 算法效果评估")

# 1. 为线性规划结果计算预估业绩
opt_df_with_gmv = opt_df.copy()
roi_dict = dict(zip(channel_summary["渠道"], channel_summary["ROI"]))  # 渠道ROI字典
opt_df_with_gmv["预估业绩"] = opt_df_with_gmv["渠道"].map(roi_dict) * opt_df_with_gmv["分配金额"]

# 2. 准备启发式策略的结果（以平均分配为例）
heuristic_avg = channel_summary[["渠道"]].copy()
heuristic_avg["分配金额"] = budget_yuan_lp / len(heuristic_avg)  # 平均分配金额
heuristic_avg["预估业绩"] = heuristic_avg["渠道"].map(roi_dict) * heuristic_avg["分配金额"]  # 计算预估业绩

heuristic_results = {
    "平均分配": heuristic_avg
}

# 3. 调用评估函数
metrics = evaluate_algorithm(opt_df_with_gmv, heuristic_results, budget_yuan_lp)

# 4. 展示评估结果
st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "数值"}))


# ---------------- 算法效率测试模块 ----------------
import time
def generate_test_data(n_channels):
    """
    生成算法效率测试数据
    
    参数:
        n_channels: 渠道数量
    返回:
        包含渠道、点击、花费、ROI的DataFrame
    """
    channels = [f"渠道{i}" for i in range(1, n_channels + 1)]
    clicks = np.random.randint(1000, 10000, n_channels)  # 随机点击量
    costs = np.random.randint(100, 1000, n_channels)     # 随机花费
    rois = np.random.uniform(0.1, 0.5, n_channels)       # 随机ROI（0.1-0.5）

    df = pd.DataFrame({
        "渠道": channels,
        "点击": clicks,
        "花费": costs,
        "ROI": rois
    })
    return df


def run_efficiency_evaluation():
    """运行算法效率评估，测试不同渠道数量下的性能"""
    st.subheader("⚡ 算法效率评估")
    st.write("测试不同渠道数量下算法的运行效率，帮助评估算法性能。")
    st.markdown("---")
    
    # 测试参数配置
    with st.expander("测试参数设置", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            # 选择要测试的渠道数量
            test_sizes = st.multiselect(
                "选择要测试的渠道数量",
                options=[5, 10, 20, 50, 100, 200],
                default=[5, 10, 20],
                key="channel_sizes"
            )
        
        with col2:
            # 测试用预算设置
            test_budget = st.number_input(
                "测试用预算（元）",
                min_value=10000,
                max_value=10000000,
                value=1000000,
                step=100000,
                key="test_budget"
            )
    
    # 开始测试按钮
    st.markdown("")
    start_test = st.button(
        "🚀 开始效率测试",
        type="primary",
        use_container_width=True,
        key="start_test_btn"
    )
    
    # 执行测试
    if start_test:
        if not test_sizes:
            st.error("请至少选择一个渠道数量进行测试！")
            return
        
        with st.spinner("正在进行效率测试，请稍候..."):
            results = []
            # 按渠道数量从小到大测试
            for n in sorted(test_sizes):
                test_data = generate_test_data(n)  # 生成测试数据
                
                # 记录运行时间
                start_time = time.time()
                try:
                    # 执行线性规划算法
                    result = lp_allocate(
                        budget_yuan=test_budget,
                        df_channel=test_data,
                        min_roi=0.15,
                        max_weight=0.6
                    )
                    status = "成功"
                except Exception as e:
                    status = f"失败: {str(e)[:20]}..."  # 捕获异常信息
                
                elapsed_time = time.time() - start_time  # 计算耗时
                results.append({
                    "渠道数量": n,
                    "运行时间(秒)": round(elapsed_time, 4),
                    "状态": status
                })
            
            # 展示测试结果
            st.success("效率测试完成！")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # 绘制运行时间趋势图
            fig = px.line(
                results_df[results_df["状态"] == "成功"],
                x="渠道数量",
                y="运行时间(秒)",
                title="运行时间随渠道数量变化趋势",
                markers=True,
                color_discrete_sequence=["#2196F3"]
            )
            fig.update_layout(
                xaxis_title="渠道数量",
                yaxis_title="运行时间(秒)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


# 运行效率测试模块
if __name__ == "__main__":
    run_efficiency_evaluation()
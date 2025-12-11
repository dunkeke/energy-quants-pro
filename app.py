import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta  # [修复] 补全核心时间库
import warnings

# 忽略部分 sklearn 警告
warnings.filterwarnings("ignore")

# ==========================================
# PART 1: 核心策略类 (逻辑对齐 Deepnote)
# ==========================================

class HMMStandardStrategy:
    """经典 HMM 策略: 低波(0)做多，高波(2)做空，中波(1)空仓"""
    def __init__(self, n_components=3, iter_num=1000, window_size=21):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def generate_signals(self, df):
        df = df.copy()
        # 计算收益率和波动率
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Ret'].rolling(window=self.window_size).std()
        df.dropna(inplace=True)
        
        if len(df) < 100: return df
        
        # 特征缩放 (关键步骤，解决数值不稳定)
        X = df[['Log_Ret', 'Volatility']].values * 100.0
        
        try:
            model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.iter_num, random_state=42, tol=0.01, min_covar=0.01)
            model.fit(X)
        except:
            st.warning("HMM 模型训练失败 (Standard)")
            return df

        hidden_states = model.predict(X)
        
        # 状态排序 (按波动率)
        state_vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_stats = sorted(list(enumerate(state_vol_means)), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        # 硬编码信号
        df['Signal'] = 0
        df.loc[df['Regime'] == 0, 'Signal'] = 1   # 低波 -> Long
        df.loc[df['Regime'] == self.n_components-1, 'Signal'] = -1 # 高波 -> Short
        
        return df

class HMMAdaptiveStrategy:
    """
    自适应贝叶斯策略 (与 Deepnote 版本逻辑完全对齐)
    核心：不假设 State 0 一定涨，而是计算 Posterior Expected Return
    """
    def __init__(self, n_components=3, iter_num=1000, window_size=21):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def generate_signals(self, df):
        df = df.copy()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Ret'].rolling(window=self.window_size).std()
        df.dropna(inplace=True)
        
        if len(df) < 100: return df
        
        X = df[['Log_Ret', 'Volatility']].values * 100.0
        
        try:
            model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.iter_num, random_state=42, tol=0.01, min_covar=0.01)
            model.fit(X)
        except:
            st.warning("HMM 模型训练失败 (Adaptive)")
            return df
        
        hidden_states = model.predict(X)
        
        # 排序
        state_vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_stats = sorted(list(enumerate(state_vol_means)), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        # 后验概率
        posterior_probs = model.predict_proba(X)
        sorted_probs = np.zeros_like(posterior_probs)
        for old_i, new_i in mapping.items():
            sorted_probs[:, new_i] = posterior_probs[:, old_i]
            
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        # --- 贝叶斯核心逻辑 ---
        
        # 1. Priors: 计算每个状态的历史平均收益
        state_means = []
        for i in range(self.n_components):
            mean_ret = df[df['Regime'] == i]['Log_Ret'].mean()
            state_means.append(mean_ret)
        state_means = np.array(state_means)
        
        # 2. Transition: 重映射转移矩阵
        new_transmat = np.zeros_like(model.transmat_)
        for i in range(self.n_components):
            for j in range(self.n_components):
                new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
                
        # 3. Expectation: 计算下一日的期望收益
        # Next_Probs = Current_Probs * Trans_Mat
        next_probs = np.dot(sorted_probs, new_transmat)
        df['Bayes_Exp_Ret'] = np.dot(next_probs, state_means)
        
        # 4. Signal: 基于期望收益的正负
        threshold = 0.0003 # 3bps 阈值，过滤噪音
        df['Signal'] = 0
        df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
        df.loc[df['Bayes_Exp_Ret'] < -threshold, 'Signal'] = -1
        
        return df

class SpreadArbStrategy:
    """统计套利策略 (Pairs Trading)"""
    def __init__(self, window_size=20, z_threshold=1.5):
        self.window_size = window_size
        self.z_threshold = z_threshold

    def generate_signals(self, df_a, df_b):
        # 确保索引对齐
        data = pd.DataFrame(index=df_a.index)
        data['Price_A'] = df_a['Close']
        data['Price_B'] = df_b['Close'] # 自动对齐 index
        data.dropna(inplace=True)
        
        if len(data) < 50: return data

        # 计算价差
        data['Spread'] = data['Price_A'] - data['Price_B']
        data['Spread_Mean'] = data['Spread'].rolling(self.window_size).mean()
        data['Spread_Std'] = data['Spread'].rolling(self.window_size).std()
        
        # 避免除以零
        data['Z_Score'] = (data['Spread'] - data['Spread_Mean']) / (data['Spread_Std'] + 1e-8)
        
        data['Signal'] = 0
        data.loc[data['Z_Score'] > self.z_threshold, 'Signal'] = -1 # 卖价差
        data.loc[data['Z_Score'] < -self.z_threshold, 'Signal'] = 1 # 买价差
        
        # 合成收益
        ret_a = np.log(data['Price_A'] / data['Price_A'].shift(1)).fillna(0)
        ret_b = np.log(data['Price_B'] / data['Price_B'].shift(1)).fillna(0)
        data['Spread_Ret_Raw'] = ret_a - ret_b
        
        return data

# ==========================================
# PART 2: 回测引擎 (增强版：修复 NaN 问题)
# ==========================================

class BacktestEngine:
    def __init__(self, initial_capital=100000, transaction_cost=0.0002):
        self.initial_capital = initial_capital
        self.cost = transaction_cost

    def run(self, df, ret_col='Log_Ret'):
        df = df.copy()
        
        # 1. 信号滞后 (避免未来函数)
        df['Position'] = df['Signal'].shift(1).fillna(0)
        
        # 2. 计算成本
        trades = df['Position'].diff().abs().fillna(0)
        fees = trades * self.cost
        
        # 3. 策略收益 (填充 NaN 以防计算断裂)
        # 确保收益率列没有 NaN
        df[ret_col] = df[ret_col].fillna(0)
        
        df['Strategy_Ret'] = (df['Position'] * df[ret_col]) - fees
        
        # 4. 净值曲线 (起点归一化)
        # 从第一天非0数据开始计算
        df['Equity_Curve'] = self.initial_capital * (1 + df['Strategy_Ret']).cumprod()
        df['Benchmark_Curve'] = self.initial_capital * (1 + df[ret_col]).cumprod()
        
        return df

    def calculate_metrics(self, df):
        # 检查数据有效性
        if df.empty or 'Equity_Curve' not in df.columns:
            return self._empty_metrics()
            
        equity = df['Equity_Curve']
        ret = df['Strategy_Ret']
        
        # 修复 NaN% 问题：确保首尾有值
        if len(equity) < 2: return self._empty_metrics()
        
        start_val = equity.iloc[0]
        end_val = equity.iloc[-1]
        
        if start_val == 0 or np.isnan(start_val): start_val = self.initial_capital
        
        total_ret = (end_val / start_val) - 1
        
        # 年化计算
        days = (equity.index[-1] - equity.index[0]).days
        if days > 0:
            cagr = (1 + total_ret) ** (365 / days) - 1
        else:
            cagr = 0
            
        # 波动率与夏普
        vol = ret.std() * np.sqrt(252)
        sharpe = (ret.mean() * 252) / (vol + 1e-8) if vol > 0 else 0
        
        # 回撤
        roll_max = equity.cummax()
        dd = (equity - roll_max) / (roll_max + 1e-8)
        max_dd = dd.min()
        
        # 胜率
        active_days = df[df['Position'] != 0]
        if len(active_days) > 0:
            win_rate = len(active_days[active_days['Strategy_Ret'] > 0]) / len(active_days)
        else:
            win_rate = 0
            
        return {
            "Total Return": f"{total_ret*100:.2f}%",
            "CAGR": f"{cagr*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd*100:.2f}%",
            "Win Rate": f"{win_rate*100:.1f}%"
        }
        
    def _empty_metrics(self):
        return {k: "N/A" for k in ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Win Rate"]}

# ==========================================
# PART 3: Streamlit UI
# ==========================================

st.set_page_config(page_title="能源量化实验室 Pro", layout="wide", page_icon="⚡")

st.title("⚡ Energy Quant Lab: HMM & Arbitrage System (Pro)")
st.markdown("### 专业的能源市场量化回测与信号平台")

# 侧边栏
st.sidebar.header("⚙️ 策略控制台")

strategy_type = st.sidebar.selectbox(
    "选择策略类型",
    ["HMM 自适应贝叶斯 (Adaptive)", "HMM 经典模型 (Standard)", "统计套利 (Pairs Trading)"]
)

tickers = {
    "Brent Crude": "BZ=F", 
    "WTI Crude": "CL=F", 
    "Natural Gas (HH)": "NG=F", 
    "Dutch TTF": "TTF=F"
}

if "套利" in strategy_type:
    col1, col2 = st.sidebar.columns(2)
    asset_a = col1.selectbox("资产 A (Long)", list(tickers.keys()), index=0)
    asset_b = col2.selectbox("资产 B (Short)", list(tickers.keys()), index=1)
    ticker = f"{asset_a} vs {asset_b}"
else:
    asset = st.sidebar.selectbox("选择交易标的", list(tickers.keys()))
    ticker = tickers[asset]

# 默认时间：过去2年
default_start = datetime.now() - timedelta(days=365*2)
start_date = st.sidebar.date_input("回测开始", default_start)
end_date = st.sidebar.date_input("回测结束", datetime.now())

run_btn = st.sidebar.button("🚀 运行回测", type="primary")

if run_btn:
    engine = BacktestEngine(initial_capital=100000)
    
    with st.spinner(f"正在对 {ticker} 进行量化回测..."):
        try:
            if "套利" in strategy_type:
                df_a = yf.download(tickers[asset_a], start=start_date, end=end_date, progress=False, auto_adjust=True)
                df_b = yf.download(tickers[asset_b], start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                # MultiIndex 修复
                if isinstance(df_a.columns, pd.MultiIndex): df_a.columns = df_a.columns.get_level_values(0)
                if isinstance(df_b.columns, pd.MultiIndex): df_b.columns = df_b.columns.get_level_values(0)

                if df_a.empty or df_b.empty:
                    st.error("数据获取失败，请检查时间范围或网络。")
                else:
                    strat = SpreadArbStrategy()
                    df_res = strat.generate_signals(df_a, df_b)
                    
                    if len(df_res) > 0:
                        df_bt = engine.run(df_res, ret_col='Spread_Ret_Raw')
                        
                        metrics = engine.calculate_metrics(df_bt)
                        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                        kpi1.metric("总回报", metrics['Total Return'])
                        kpi2.metric("年化收益", metrics['CAGR'])
                        kpi3.metric("夏普比率", metrics['Sharpe Ratio'])
                        kpi4.metric("最大回撤", metrics['Max Drawdown'])
                        kpi5.metric("胜率", metrics['Win Rate'])
                        
                        st.divider()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="Strategy Equity", line=dict(color='cyan', width=2)))
                        fig.update_layout(title="套利策略净值曲线", height=400, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Z_Score'], name="Z-Score", line=dict(color='yellow')))
                        fig2.add_hline(y=1.5, line_dash="dash", line_color="red")
                        fig2.add_hline(y=-1.5, line_dash="dash", line_color="green")
                        fig2.update_layout(title="价差 Z-Score 监控", height=300, template="plotly_dark")
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("有效交易数据不足。")

            else:
                df = yf.download(tickers[asset], start=start_date, end=end_date, progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                if df.empty:
                    st.error("数据获取失败。")
                else:
                    if "自适应" in strategy_type:
                        strat = HMMAdaptiveStrategy()
                    else:
                        strat = HMMStandardStrategy()
                        
                    df_res = strat.generate_signals(df)
                    
                    if 'Signal' in df_res.columns:
                        df_bt = engine.run(df_res, ret_col='Log_Ret')

                        metrics = engine.calculate_metrics(df_bt)
                        
                        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                        kpi1.metric("总回报", metrics['Total Return'])
                        kpi2.metric("年化收益", metrics['CAGR'])
                        kpi3.metric("夏普比率", metrics['Sharpe Ratio'])
                        kpi4.metric("最大回撤", metrics['Max Drawdown'])
                        kpi5.metric("胜率", metrics['Win Rate'])
                        
                        st.divider()

                        tab1, tab2 = st.tabs(["📈 资金曲线 & 信号", "🔬 详细数据"])
                        
                        with tab1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="Strategy Equity", line=dict(color='cyan', width=2)))
                            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Benchmark_Curve'], name="Buy & Hold", line=dict(color='gray', dash='dot')))
                            
                            buy_sig = df_bt[df_bt['Signal'] == 1]
                            sell_sig = df_bt[df_bt['Signal'] == -1]
                            
                            fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig['Equity_Curve'], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=8), name='Buy Signal'))
                            fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig['Equity_Curve'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=8), name='Sell Signal'))
                            
                            fig.update_layout(title="策略净值曲线", height=500, template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            if 'Regime' in df_bt.columns:
                                fig2 = go.Figure()
                                # 价格线
                                fig2.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], name="Price", line=dict(color='white', width=1)))
                                
                                colors = ['#00ff00', '#ffff00', '#ff0000']
                                regime_names = ["Low Vol (0)", "Mid Vol (1)", "High Vol (2)"]
                                for i in range(3):
                                    mask = df_bt['Regime'] == i
                                    # 用散点覆盖来显示颜色
                                    fig2.add_trace(go.Scatter(x=df_bt.index[mask], y=df_bt['Close'][mask], mode='markers', marker=dict(size=3, color=colors[i]), name=regime_names[i]))
                                    
                                fig2.update_layout(title="HMM 市场体制识别 (Regime Detection)", height=400, template="plotly_dark")
                                st.plotly_chart(fig2, use_container_width=True)

                        with tab2:
                            st.dataframe(df_bt.tail(100).sort_index(ascending=False))
                    else:
                        st.warning("模型未能生成有效信号 (可能是数据量不足)。")

        except Exception as e:
            st.error(f"运行出错: {e}")

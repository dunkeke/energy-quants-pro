import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# 忽略部分 sklearn 警告
warnings.filterwarnings("ignore")

# ==========================================
# PART 1: 核心策略类 (原 strategies.py)
# ==========================================

class HMMStandardStrategy:
    """经典 HMM 策略: 低波做多，高波做空"""
    def __init__(self, n_components=3, iter_num=1000, window_size=21):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def generate_signals(self, df):
        df = df.copy()
        # 特征准备
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Ret'].rolling(window=self.window_size).std()
        df.dropna(inplace=True)
        
        if len(df) < 100: return df
        
        X = df[['Log_Ret', 'Volatility']].values * 100.0
        
        # 训练模型
        try:
            model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.iter_num, random_state=42, tol=0.01, min_covar=0.01)
            model.fit(X)
        except:
            return df

        hidden_states = model.predict(X)
        
        # 状态排序 (按波动率)
        vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_idx = np.argsort(vol_means)
        mapping = {old: new for new, old in enumerate(sorted_idx)}
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        # 信号生成 (硬编码规则)
        df['Signal'] = 0
        df.loc[df['Regime'] == 0, 'Signal'] = 1   # 低波 -> Long
        df.loc[df['Regime'] == self.n_components-1, 'Signal'] = -1 # 高波 -> Short
        
        return df

class HMMAdaptiveStrategy:
    """自适应贝叶斯策略: 基于状态历史收益决定方向"""
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
            return df
        
        # 排序
        hidden_states = model.predict(X)
        vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_idx = np.argsort(vol_means)
        mapping = {old: new for new, old in enumerate(sorted_idx)}
        
        # 概率与期望
        posterior_probs = model.predict_proba(X)
        sorted_probs = np.zeros_like(posterior_probs)
        for old_i, new_i in mapping.items():
            sorted_probs[:, new_i] = posterior_probs[:, old_i]
            
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        # 贝叶斯预测 (Priors -> Transition -> Posterior Exp)
        state_means = []
        for i in range(self.n_components):
            mean_ret = df[df['Regime'] == i]['Log_Ret'].mean()
            state_means.append(mean_ret)
        
        new_transmat = np.zeros_like(model.transmat_)
        for i in range(self.n_components):
            for j in range(self.n_components):
                new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
                
        next_probs = np.dot(sorted_probs, new_transmat)
        df['Bayes_Exp_Ret'] = np.dot(next_probs, state_means)
        
        # 自适应信号 (根据期望收益正负)
        df['Signal'] = 0
        df.loc[df['Bayes_Exp_Ret'] > 0.0003, 'Signal'] = 1
        df.loc[df['Bayes_Exp_Ret'] < -0.0003, 'Signal'] = -1
        
        return df

class SpreadArbStrategy:
    """统计套利策略: 基于价差均值回归"""
    def __init__(self, window_size=20, z_threshold=1.5):
        self.window_size = window_size
        self.z_threshold = z_threshold

    def generate_signals(self, df_a, df_b):
        # 对齐数据
        data = pd.DataFrame()
        data['Price_A'] = df_a['Close']
        data['Price_B'] = df_b['Close']
        data.dropna(inplace=True)
        
        if len(data) < 50: return data

        # 计算价差与Z-Score
        data['Spread'] = data['Price_A'] - data['Price_B']
        data['Spread_Mean'] = data['Spread'].rolling(self.window_size).mean()
        data['Spread_Std'] = data['Spread'].rolling(self.window_size).std()
        data['Z_Score'] = (data['Spread'] - data['Spread_Mean']) / data['Spread_Std']
        
        # 信号生成 (均值回归)
        data['Signal'] = 0
        data.loc[data['Z_Score'] > self.z_threshold, 'Signal'] = -1 # 卖价差
        data.loc[data['Z_Score'] < -self.z_threshold, 'Signal'] = 1 # 买价差
        
        # 计算合成收益 (假设等权重对冲)
        ret_a = np.log(data['Price_A'] / data['Price_A'].shift(1))
        ret_b = np.log(data['Price_B'] / data['Price_B'].shift(1))
        
        data['Spread_Ret_Raw'] = ret_a - ret_b # 基础价差收益
        
        return data

# ==========================================
# PART 2: 回测引擎 (原 backtest_engine.py)
# ==========================================

class BacktestEngine:
    def __init__(self, initial_capital=100000, transaction_cost=0.0002):
        self.initial_capital = initial_capital
        self.cost = transaction_cost

    def run(self, df, ret_col='Log_Ret'):
        df = df.copy()
        # 仓位滞后一天 (T日信号 T+1日执行)
        df['Position'] = df['Signal'].shift(1).fillna(0)
        
        # 计算成本
        trades = df['Position'].diff().abs()
        fees = trades * self.cost
        
        # 策略收益
        df['Strategy_Ret'] = (df['Position'] * df[ret_col]) - fees
        
        # 净值
        df['Equity_Curve'] = self.initial_capital * (1 + df['Strategy_Ret']).cumprod()
        df['Benchmark_Curve'] = self.initial_capital * (1 + df[ret_col]).cumprod()
        
        return df

    def calculate_metrics(self, df):
        equity = df['Equity_Curve']
        ret = df['Strategy_Ret']
        
        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
        
        days = (equity.index[-1] - equity.index[0]).days
        if days > 0:
            cagr = (1 + total_ret) ** (365 / days) - 1
        else:
            cagr = 0
            
        vol = ret.std() * np.sqrt(252)
        sharpe = (ret.mean() * 252) / (vol) if vol > 0 else 0
        
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        max_dd = dd.min()
        
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

# ==========================================
# PART 3: Streamlit UI (主程序)
# ==========================================

# 页面配置
st.set_page_config(page_title="能源量化实验室", layout="wide", page_icon="⚡")

st.title("⚡ Energy Quant Lab: HMM & Arbitrage System")
st.markdown("### 专业的能源市场量化回测与信号平台 (Single-File Version)")

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

# 使用 datetime.date 对象
start_date = st.sidebar.date_input("回测开始", pd.to_datetime("2022-01-01").date())
end_date = st.sidebar.date_input("回测结束", pd.to_datetime("today").date())

run_btn = st.sidebar.button("🚀 运行回测", type="primary")

if run_btn:
    engine = BacktestEngine(initial_capital=100000)
    
    with st.spinner("正在量化计算中..."):
        try:
            # 1. 数据获取与策略执行
            if "套利" in strategy_type:
                # 获取数据
                df_a = yf.download(tickers[asset_a], start=start_date, end=end_date, progress=False, auto_adjust=True)
                df_b = yf.download(tickers[asset_b], start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                # 兼容性处理 MultiIndex
                if isinstance(df_a.columns, pd.MultiIndex): df_a.columns = df_a.columns.get_level_values(0)
                if isinstance(df_b.columns, pd.MultiIndex): df_b.columns = df_b.columns.get_level_values(0)

                if len(df_a) == 0 or len(df_b) == 0:
                    st.error("数据获取失败，请检查时间范围或网络。")
                else:
                    strat = SpreadArbStrategy()
                    df_res = strat.generate_signals(df_a, df_b)
                    
                    if len(df_res) > 0:
                        df_bt = engine.run(df_res, ret_col='Spread_Ret_Raw')
                        
                        # 结果展示
                        metrics = engine.calculate_metrics(df_bt)
                        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                        kpi1.metric("总回报", metrics['Total Return'])
                        kpi2.metric("年化收益", metrics['CAGR'])
                        kpi3.metric("夏普比率", metrics['Sharpe Ratio'])
                        kpi4.metric("最大回撤", metrics['Max Drawdown'])
                        kpi5.metric("胜率", metrics['Win Rate'])
                        
                        st.divider()
                        
                        # 绘图
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="Strategy Equity", line=dict(color='cyan', width=2)))
                        # Benchmark for spread is just holding the spread (often 0 return if mean reverting)
                        # fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Benchmark_Curve'], name="Benchmark", line=dict(color='gray', dash='dot')))
                        fig.update_layout(title="套利策略净值曲线", height=400, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Z-Score 图
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Z_Score'], name="Z-Score", line=dict(color='yellow')))
                        fig2.add_hline(y=1.5, line_dash="dash", line_color="red")
                        fig2.add_hline(y=-1.5, line_dash="dash", line_color="green")
                        fig2.update_layout(title="价差 Z-Score 监控", height=300, template="plotly_dark")
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("有效交易数据不足。")

            else:
                # 单标的策略
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                if len(df) == 0:
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
                            
                            # 体制图
                            if 'Regime' in df_bt.columns:
                                fig2 = go.Figure()
                                colors = ['#00ff00', '#ffff00', '#ff0000']
                                for i in range(3):
                                    mask = df_bt['Regime'] == i
                                    fig2.add_trace(go.Scatter(x=df_bt.index[mask], y=df_bt['Close'][mask], mode='markers', marker=dict(size=3, color=colors[i]), name=f"Regime {i}"))
                                fig2.update_layout(title="HMM 市场体制识别", height=300, template="plotly_dark")
                                st.plotly_chart(fig2, use_container_width=True)

                        with tab2:
                            st.dataframe(df_bt.tail(100).sort_index(ascending=False))
                    else:
                        st.warning("模型未能生成有效信号 (可能是数据量不足)。")

        except Exception as e:
            st.error(f"运行出错: {e}")
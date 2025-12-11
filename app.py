import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# 忽略部分 sklearn 警告
warnings.filterwarnings("ignore")

# ==========================================
# PART 1: 核心策略类 (Strategies)
# ==========================================

class HMMStandardStrategy:
    """经典 HMM 策略: 低波(0)做多，高波(2)做空，中波(1)空仓"""
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
            st.warning("HMM 模型训练失败 (Standard)")
            return df

        hidden_states = model.predict(X)
        
        # 状态排序 (按波动率)
        state_vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_stats = sorted(list(enumerate(state_vol_means)), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        # 信号生成
        df['Signal'] = 0
        df.loc[df['Regime'] == 0, 'Signal'] = 1   # 低波 -> Long
        df.loc[df['Regime'] == self.n_components-1, 'Signal'] = -1 # 高波 -> Short
        
        # 辅助信息：置信度 (简化版，直接用Regime代替)
        df['Signal_Strength'] = "N/A" # 经典模型不计算置信度
        
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
        
        # 记录每个状态的概率用于展示
        for i in range(self.n_components):
            df[f'Prob_S{i}'] = sorted_probs[:, i]
        
        # 贝叶斯预测
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
        
        # 信号生成
        threshold = 0.0003
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
        data = pd.DataFrame(index=df_a.index)
        data['Price_A'] = df_a['Close']
        data['Price_B'] = df_b['Close']
        data.dropna(inplace=True)
        
        if len(data) < 50: return data

        data['Spread'] = data['Price_A'] - data['Price_B']
        data['Spread_Mean'] = data['Spread'].rolling(self.window_size).mean()
        data['Spread_Std'] = data['Spread'].rolling(self.window_size).std()
        
        data['Z_Score'] = (data['Spread'] - data['Spread_Mean']) / (data['Spread_Std'] + 1e-8)
        
        data['Signal'] = 0
        data.loc[data['Z_Score'] > self.z_threshold, 'Signal'] = -1 # 卖价差 (做空 Spread)
        data.loc[data['Z_Score'] < -self.z_threshold, 'Signal'] = 1 # 买价差 (做多 Spread)
        
        ret_a = np.log(data['Price_A'] / data['Price_A'].shift(1)).fillna(0)
        ret_b = np.log(data['Price_B'] / data['Price_B'].shift(1)).fillna(0)
        data['Spread_Ret_Raw'] = ret_a - ret_b
        
        return data

# ==========================================
# PART 2: 回测引擎 (Backtest Engine)
# ==========================================

class BacktestEngine:
    def __init__(self, initial_capital=100000, transaction_cost=0.0002):
        self.initial_capital = initial_capital
        self.cost = transaction_cost

    def run(self, df, ret_col='Log_Ret'):
        df = df.copy()
        df['Position'] = df['Signal'].shift(1).fillna(0)
        trades = df['Position'].diff().abs().fillna(0)
        fees = trades * self.cost
        
        df[ret_col] = df[ret_col].fillna(0)
        df['Strategy_Ret'] = (df['Position'] * df[ret_col]) - fees
        
        df['Equity_Curve'] = self.initial_capital * (1 + df['Strategy_Ret']).cumprod()
        df['Benchmark_Curve'] = self.initial_capital * (1 + df[ret_col]).cumprod()
        return df

    def calculate_metrics(self, df):
        if df.empty or 'Equity_Curve' not in df.columns or len(df) < 2:
            return self._empty_metrics()
            
        equity = df['Equity_Curve']
        ret = df['Strategy_Ret']
        
        start_val = equity.iloc[0] if equity.iloc[0] > 0 else self.initial_capital
        total_ret = (equity.iloc[-1] / start_val) - 1
        
        days = (equity.index[-1] - equity.index[0]).days
        cagr = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
        vol = ret.std() * np.sqrt(252)
        sharpe = (ret.mean() * 252) / (vol + 1e-8) if vol > 0 else 0
        
        roll_max = equity.cummax()
        dd = (equity - roll_max) / (roll_max + 1e-8)
        max_dd = dd.min()
        
        active_days = df[df['Position'] != 0]
        win_rate = len(active_days[active_days['Strategy_Ret'] > 0]) / len(active_days) if len(active_days) > 0 else 0
            
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
# PART 3: 信号解读与展示 (UI Helper)
# ==========================================

def display_signal_panel(df, strategy_type):
    """
    智能信号驾驶舱
    """
    last = df.iloc[-1]
    sig = last['Signal']
    
    st.markdown("### 🚦 实时交易信号驾驶舱")
    
    # 1. 信号大卡片
    col_sig, col_reason = st.columns([1, 2])
    
    with col_sig:
        if sig == 1:
            st.success("## 🟢 强力做多\n**LONG SIGNAL**")
        elif sig == -1:
            st.error("## 🔴 强力卖出\n**SHORT SIGNAL**")
        else:
            st.warning("## ⚪ 空仓观望\n**WAIT / CASH**")
            
    # 2. 深度逻辑解读
    with col_reason:
        st.markdown("#### 🤖 策略逻辑分析")
        
        if "自适应" in strategy_type:
            prob_0 = last.get('Prob_S0', 0) * 100
            prob_2 = last.get('Prob_S2', 0) * 100
            exp_ret = last.get('Bayes_Exp_Ret', 0) * 100
            
            regime_desc = "低波动 (通常利多)" if last['Regime'] == 0 else ("高波动 (风险极大)" if last['Regime'] == 2 else "震荡过渡期")
            
            msg = f"""
            - **当前体制**: State {int(last['Regime'])} ({regime_desc})
            - **概率置信度**: State 0 (牛): **{prob_0:.1f}%** | State 2 (熊): **{prob_2:.1f}%**
            - **贝叶斯期望**: 下一日预期收益为 **{exp_ret:.4f}%**
            """
            if sig == 1:
                msg += "\n\n💡 **结论**: 市场虽有波动，但数学期望收益显著为正，建议**持有或加仓**。"
            elif sig == -1:
                msg += "\n\n💡 **结论**: 高波动伴随负收益预期，系统检测到**崩盘风险**，建议清仓。"
            else:
                msg += "\n\n💡 **结论**: 预期收益微弱，不足以覆盖交易成本，建议**观望**。"
            st.info(msg)
            
        elif "套利" in strategy_type:
            z_score = last.get('Z_Score', 0)
            spread = last.get('Spread', 0)
            
            msg = f"""
            - **当前价差**: {spread:.2f}
            - **偏离度 (Z-Score)**: **{z_score:.2f} σ** (标准差)
            """
            if sig == 1:
                msg += "\n\n💡 **结论**: 价差过度收缩 (Z < -1.5)，统计学上大概率将**反弹扩大**。建议：买入价差组合。"
            elif sig == -1:
                msg += "\n\n💡 **结论**: 价差过度扩张 (Z > 1.5)，统计学上大概率将**回归均值**。建议：卖出价差组合。"
            else:
                msg += "\n\n💡 **结论**: 价差处于合理区间 (-1.5 ~ 1.5)，无明显套利机会。"
            st.info(msg)
            
        else: # Standard
            regime = int(last['Regime'])
            msg = f"- **当前体制**: State {regime}"
            if regime == 0: msg += " (低波稳健期) -> **做多**"
            elif regime == 2: msg += " (高波恐慌期) -> **做空**"
            else: msg += " (震荡期) -> **空仓**"
            st.info(msg)

# ==========================================
# PART 4: Streamlit UI 主程序
# ==========================================

st.set_page_config(page_title="能源量化终端 Pro+", layout="wide", page_icon="⚡")

st.title("⚡ Energy Quant Lab: HMM & Arbitrage System (Pro+)")
st.markdown("### 专业的能源市场量化回测与信号平台")

# 侧边栏
st.sidebar.header("⚙️ 策略控制台")
strategy_type = st.sidebar.selectbox("选择策略类型", ["HMM 自适应贝叶斯 (Adaptive)", "HMM 经典模型 (Standard)", "统计套利 (Pairs Trading)"])

tickers = {"Brent Crude": "BZ=F", "WTI Crude": "CL=F", "Natural Gas (HH)": "NG=F", "Dutch TTF": "TTF=F"}

if "套利" in strategy_type:
    col1, col2 = st.sidebar.columns(2)
    asset_a = col1.selectbox("资产 A (Long)", list(tickers.keys()), index=0)
    asset_b = col2.selectbox("资产 B (Short)", list(tickers.keys()), index=1)
    ticker = f"{asset_a} vs {asset_b}"
else:
    asset = st.sidebar.selectbox("选择交易标的", list(tickers.keys()))
    ticker = tickers[asset]

start_date = st.sidebar.date_input("回测开始", datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("回测结束", datetime.now())

if st.sidebar.button("🚀 运行分析", type="primary"):
    engine = BacktestEngine(initial_capital=100000)
    
    with st.spinner(f"正在计算 {ticker} 的量化信号..."):
        try:
            if "套利" in strategy_type:
                df_a = yf.download(tickers[asset_a], start=start_date, end=end_date, progress=False, auto_adjust=True)
                df_b = yf.download(tickers[asset_b], start=start_date, end=end_date, progress=False, auto_adjust=True)
                # 兼容性处理
                if isinstance(df_a.columns, pd.MultiIndex): df_a.columns = df_a.columns.get_level_values(0)
                if isinstance(df_b.columns, pd.MultiIndex): df_b.columns = df_b.columns.get_level_values(0)

                if df_a.empty or df_b.empty:
                    st.error("数据获取失败。")
                else:
                    strat = SpreadArbStrategy()
                    df_res = strat.generate_signals(df_a, df_b)
                    if len(df_res) > 0:
                        # 1. 信号驾驶舱 (最优先展示)
                        display_signal_panel(df_res, strategy_type)
                        st.divider()
                        
                        # 2. 回测结果
                        df_bt = engine.run(df_res, ret_col='Spread_Ret_Raw')
                        metrics = engine.calculate_metrics(df_bt)
                        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                        kpi1.metric("总回报", metrics['Total Return'])
                        kpi2.metric("年化收益", metrics['CAGR'])
                        kpi3.metric("夏普比率", metrics['Sharpe Ratio'])
                        kpi4.metric("最大回撤", metrics['Max Drawdown'])
                        kpi5.metric("胜率", metrics['Win Rate'])
                        
                        # 3. 图表
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="Strategy Equity", line=dict(color='cyan', width=2)))
                        fig.update_layout(title="套利净值曲线", height=400, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Z_Score'], name="Spread Z-Score", line=dict(color='yellow')))
                        fig2.add_hline(y=1.5, line_dash="dash", line_color="red")
                        fig2.add_hline(y=-1.5, line_dash="dash", line_color="green")
                        fig2.update_layout(title="价差 Z-Score 监控", height=300, template="plotly_dark")
                        st.plotly_chart(fig2, use_container_width=True)
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
                        # 1. 信号驾驶舱
                        display_signal_panel(df_res, strategy_type)
                        st.divider()
                        
                        # 2. 回测结果
                        df_bt = engine.run(df_res, ret_col='Log_Ret')
                        metrics = engine.calculate_metrics(df_bt)
                        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                        kpi1.metric("总回报", metrics['Total Return'])
                        kpi2.metric("年化收益", metrics['CAGR'])
                        kpi3.metric("夏普比率", metrics['Sharpe Ratio'])
                        kpi4.metric("最大回撤", metrics['Max Drawdown'])
                        kpi5.metric("胜率", metrics['Win Rate'])
                        
                        # 3. 图表
                        tab1, tab2 = st.tabs(["📈 净值与信号", "🔬 详细数据"])
                        with tab1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="Strategy Equity", line=dict(color='cyan', width=2)))
                            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Benchmark_Curve'], name="Buy & Hold", line=dict(color='gray', dash='dot')))
                            
                            buy_sig = df_bt[df_bt['Signal'] == 1]
                            sell_sig = df_bt[df_bt['Signal'] == -1]
                            fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig['Equity_Curve'], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Buy Signal'))
                            fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig['Equity_Curve'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))
                            
                            fig.update_layout(title="策略净值曲线", height=500, template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            if 'Regime' in df_bt.columns:
                                fig2 = go.Figure()
                                fig2.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], name="Price", line=dict(color='white', width=1)))
                                colors = ['#00ff00', '#ffff00', '#ff0000']
                                for i in range(3):
                                    mask = df_bt['Regime'] == i
                                    fig2.add_trace(go.Scatter(x=df_bt.index[mask], y=df_bt['Close'][mask], mode='markers', marker=dict(size=3, color=colors[i]), name=f"Regime {i}"))
                                fig2.update_layout(title="HMM 市场体制识别", height=300, template="plotly_dark")
                                st.plotly_chart(fig2, use_container_width=True)
                        with tab2:
                            st.dataframe(df_bt.tail(100).sort_index(ascending=False))
                    else:
                        st.warning("信号生成失败，可能是数据量不足。")

        except Exception as e:
            st.error(f"运行出错: {e}")

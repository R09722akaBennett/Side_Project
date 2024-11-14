import logging
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 設置頁面配置
st.set_page_config(page_title="預算優化分析", layout="wide")

# 初始數據
@st.cache_data
def load_initial_data():
    data = {
        '年': [2022]*12 + [2023]*12,
        '月': list(range(1, 13))*2,
        '變現收益': [
            # 2022年
            1665937, 1513545, 1831731, 1937624, 1874419, 1723995,
            1978887, 1998035, 1746071, 1331042, 1258247, 1121431,
            # 2023年
            1059160, 999901, 1076458, 943998, 1077483, 1162024,
            1073448, 1023352, 848734, 749857, 749430, 792460
        ],
        '投遞金額（google ads）': [
            # 2022年
            704177, 762384, 812837, 904768, 1013294, 1217421,
            1328718, 1530757, 1547773, 1548895, 1452694, 1095080,
            # 2023年
            897250, 842486, 1036517, 1154801, 1042375, 1263188,
            727369, 494465, 382925, 353211, 509009, 506131
        ],
        '新使用者': [
            # 2022年
            260617, 230443, 243010, 251661, 263704, 286982,
            301807, 294228, 279265, 280226, 274812, 241128,
            # 2023年
            273338, 226604, 247551, 200626, 196865, 199755,
            106915, 151460, 147381, 124078, 108805, 90858
        ],
        '活躍使用者': [
            # 2022年
            1487546, 1468368, 1464235, 1402852, 1386879, 1369241,
            1356332, 1364901, 1347618, 1294489, 1287219, 1199877,
            # 2023年
            1262118, 1188010, 1221980, 1135310, 1116841, 1099087,
            944065, 969298, 946241, 892729, 823957, 759620
        ],
        '月留存率': [
            # 2022年
            0.8119, 0.8119, 0.8402, 0.7921, 0.8092, 0.7971,
            0.7810, 0.7838, 0.7718, 0.7533, 0.7779, 0.7187,
            # 2023年
            0.8509, 0.7247, 0.8379, 0.7265, 0.8070, 0.8078,
            0.6772, 0.9135, 0.8200, 0.7877, 0.7840, 0.7899
        ],
        'ARRPU': [
            # 2022年
            1.12, 1.03, 1.25, 1.38, 1.35, 1.26,
            1.46, 1.46, 1.30, 1.03, 0.98, 0.93,
            # 2023年
            0.84, 0.84, 0.88, 0.83, 0.96, 1.06,
            1.14, 1.06, 0.90, 0.84, 0.91, 1.04
        ]
    }
    df = pd.DataFrame(data)
    df['年'] = df['年'].astype(str)
    return df

def prepare_data(df):
    """準備數據"""
    df['年月'] = pd.to_datetime(df['年'].astype(str) + '-' + df['月'].astype(str), format='%Y-%m')
    df['ROI'] = df['變現收益'] / df['投遞金額（google ads）']
    
    # 創建滯後特徵
    df['投遞金額_lag1'] = df['投遞金額（google ads）'].shift(1)
    df['新使用者_lag1'] = df['新使用者'].shift(1)
    df['活躍使用者_lag1'] = df['活躍使用者'].shift(1)
    
    # 創建季節性特徵
    df['季度'] = df['月'].map(lambda x: (x-1)//3 + 1)
    
    return df.dropna()

def train_models(df_clean):
    """訓練模型"""
    # 準備特徵
    X_new_users = df_clean[['投遞金額（google ads）', '投遞金額_lag1', '月', '季度']]
    y_new_users = df_clean['新使用者']
    
    X_revenue = df_clean[['新使用者', '活躍使用者', 'ARRPU', '月', '季度']]
    y_revenue = df_clean['變現收益']
    
    # 標準化
    scaler_X_new = StandardScaler()
    scaler_X_rev = StandardScaler()
    
    X_new_users_scaled = scaler_X_new.fit_transform(X_new_users)
    X_revenue_scaled = scaler_X_rev.fit_transform(X_revenue)
    
    # 訓練模型
    model_new_users = RandomForestRegressor(random_state=42)
    model_revenue = RandomForestRegressor(random_state=42)
    
    model_new_users.fit(X_new_users_scaled, y_new_users)
    model_revenue.fit(X_revenue_scaled, y_revenue)
    
    return (model_new_users, model_revenue), (scaler_X_new, scaler_X_rev)

def predict_revenue(investment_plan, base_data, models, scalers):
    """預測收益"""
    results = []
    prev_investment = base_data['投遞金額（google ads）']
    prev_users = base_data['新使用者']
    prev_active_users = base_data['活躍使用者']
    prev_arrpu = base_data['ARRPU']
    
    model_new_users, model_revenue = models
    scaler_X_new, scaler_X_rev = scalers
    
    for month in range(1, 13):
        # 預測新使用者
        X_new = np.array([[
            investment_plan[month-1],
            prev_investment,
            month,
            (month-1)//3 + 1
        ]])
        X_new_scaled = scaler_X_new.transform(X_new)
        predicted_new_users = model_new_users.predict(X_new_scaled)[0]
        
        # 預測變現收益
        X_rev = np.array([[
            predicted_new_users,
            prev_active_users,
            prev_arrpu,
            month,
            (month-1)//3 + 1
        ]])
        X_rev_scaled = scaler_X_rev.transform(X_rev)
        predicted_revenue = model_revenue.predict(X_rev_scaled)[0]
        
        roi = predicted_revenue / investment_plan[month-1]
        
        results.append({
            '月份': month,
            '投資金額': investment_plan[month-1],
            '預測新使用者': predicted_new_users,
            '預測收益': predicted_revenue,
            '預測ROI': roi
        })
        
        prev_investment = investment_plan[month-1]
        prev_users = predicted_new_users
        
    return pd.DataFrame(results)

def plot_results(results, title):
    """繪製結果圖表"""
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('投資金額', '預測收益', 'ROI', '累計收益'))
    
    # 投資金額
    fig.add_trace(
        go.Scatter(x=results['月份'], y=results['投資金額'],
                  name='投資金額', mode='lines+markers'),
        row=1, col=1
    )
    
    # 預測收益
    fig.add_trace(
        go.Scatter(x=results['月份'], y=results['預測收益'],
                  name='預測收益', mode='lines+markers'),
        row=1, col=2
    )
    
    # ROI
    fig.add_trace(
        go.Scatter(x=results['月份'], y=results['預測ROI'],
                  name='ROI', mode='lines+markers'),
        row=2, col=1
    )
    
    # 累計收益
    fig.add_trace(
        go.Scatter(x=results['月份'], y=results['預測收益'].cumsum(),
                  name='累計收益', mode='lines+markers'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, width=1200, title_text=title)
    return fig


def main():
    st.title('廣告變現收益預測')
    df = load_initial_data()
    monthly_budget = []  # 初始化每月預算列表
    st.sidebar.subheader("請設置每月預算")
    monthly_budget = [float(st.sidebar.text_input(f'第 {month} 月預算', value='800000')) for month in range(1, 13)]
    
    
    # 選擇數據來源
    data_source = st.radio("選擇數據來源", ("使用預設資料", "上傳 CSV 文件"))
    
    if data_source == "上傳 CSV 文件":
        uploaded_file = st.file_uploader("上傳 CSV 文件", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            expected_columns = ['年', '月', '變現收益', '投遞金額（google ads）', '新使用者', '活躍使用者', '月留存率', 'ARRPU']
            if not all(col in df.columns for col in expected_columns):
                st.error("上傳的文件格式不正確，請確保包含以下列: " + ", ".join(expected_columns))
                return
            else:
                st.subheader("上傳的數據")
                st.dataframe(df, use_container_width=True)  
    else:
        # 載入預設數據
        df = load_initial_data()
        st.subheader("預設數據")
        st.dataframe(df, use_container_width=True)  
    
    df_prepared = prepare_data(df)
    
    # 訓練模型
    models, scalers = train_models(df_prepared)
    
    # 準備預測
    base_data = df_prepared.iloc[-1]  
    
    # 預測結果
    st.subheader("預測結果")
    results = predict_revenue(monthly_budget, base_data, models, scalers)
    st.dataframe(results, use_container_width=True) 
       # 下載按鈕
    csv = results.to_csv(index=False)
    st.download_button(
        label="下載預測結果",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

    # 顯示主要指標
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("預計總投資", f"{results['投資金額'].sum():,.0f}")
    with col2:
        st.metric("預計總收益", f"{results['預測收益'].sum():,.0f}")
    with col3:
        st.metric("平均ROI", f"{results['預測ROI'].mean():.2f}")
    with col4:
        st.metric("ROI標準差", f"{results['預測ROI'].std():.2f}")
    
    # 顯示圖表
    st.plotly_chart(plot_results(results, "預測結果"))

  
    
 
if __name__ == "__main__":
    main()
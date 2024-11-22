# import logging
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_absolute_percentage_error
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import warnings
# warnings.filterwarnings('ignore')

# # 設置頁面配置
# st.set_page_config(page_title="預算優化分析", layout="wide")

# # 初始數據
# @st.cache_data
# def load_initial_data():
#     data = {
#         '年': [2022]*12 + [2023]*12,
#         '月': list(range(1, 13))*2,
#         '變現收益': [
#             # 2022年
#             1665937, 1513545, 1831731, 1937624, 1874419, 1723995,
#             1978887, 1998035, 1746071, 1331042, 1258247, 1121431,
#             # 2023年
#             1059160, 999901, 1076458, 943998, 1077483, 1162024,
#             1073448, 1023352, 848734, 749857, 749430, 792460
#         ],
#         '投遞金額（google ads）': [
#             # 2022年
#             704177, 762384, 812837, 904768, 1013294, 1217421,
#             1328718, 1530757, 1547773, 1548895, 1452694, 1095080,
#             # 2023年
#             897250, 842486, 1036517, 1154801, 1042375, 1263188,
#             727369, 494465, 382925, 353211, 509009, 506131
#         ],
#         '新使用者': [
#             # 2022年
#             260617, 230443, 243010, 251661, 263704, 286982,
#             301807, 294228, 279265, 280226, 274812, 241128,
#             # 2023年
#             273338, 226604, 247551, 200626, 196865, 199755,
#             106915, 151460, 147381, 124078, 108805, 90858
#         ],
#         '活躍使用者': [
#             # 2022年
#             1487546, 1468368, 1464235, 1402852, 1386879, 1369241,
#             1356332, 1364901, 1347618, 1294489, 1287219, 1199877,
#             # 2023年
#             1262118, 1188010, 1221980, 1135310, 1116841, 1099087,
#             944065, 969298, 946241, 892729, 823957, 759620
#         ],
#         '月留存率': [
#             # 2022年
#             0.8119, 0.8119, 0.8402, 0.7921, 0.8092, 0.7971,
#             0.7810, 0.7838, 0.7718, 0.7533, 0.7779, 0.7187,
#             # 2023年
#             0.8509, 0.7247, 0.8379, 0.7265, 0.8070, 0.8078,
#             0.6772, 0.9135, 0.8200, 0.7877, 0.7840, 0.7899
#         ],
#         'ARRPU': [
#             # 2022年
#             1.12, 1.03, 1.25, 1.38, 1.35, 1.26,
#             1.46, 1.46, 1.30, 1.03, 0.98, 0.93,
#             # 2023年
#             0.84, 0.84, 0.88, 0.83, 0.96, 1.06,
#             1.14, 1.06, 0.90, 0.84, 0.91, 1.04
#         ]
#     }
#     df = pd.DataFrame(data)
#     df['年'] = df['年'].astype(str)
#     return df

# def prepare_data(df):
#     """準備數據"""
#     df['年月'] = pd.to_datetime(df['年'].astype(str) + '-' + df['月'].astype(str), format='%Y-%m')
#     df['ROI'] = df['變現收益'] / df['投遞金額（google ads）']
    
#     # 創建滯後特徵
#     df['投遞金額_lag1'] = df['投遞金額（google ads）'].shift(1)
#     df['新使用者_lag1'] = df['新使用者'].shift(1)
#     df['活躍使用者_lag1'] = df['活躍使用者'].shift(1)
    
#     # 創建季節性特徵
#     df['季度'] = df['月'].map(lambda x: (x-1)//3 + 1)
    
#     return df.dropna()

# def train_models(df_clean):
#     """訓練模型"""
#     # 準備特徵
#     X_new_users = df_clean[['投遞金額（google ads）', '投遞金額_lag1', '月', '季度']]
#     y_new_users = df_clean['新使用者']
    
#     X_revenue = df_clean[['新使用者', '活躍使用者', 'ARRPU', '月', '季度']]
#     y_revenue = df_clean['變現收益']
    
#     # 標準化
#     scaler_X_new = StandardScaler()
#     scaler_X_rev = StandardScaler()
    
#     X_new_users_scaled = scaler_X_new.fit_transform(X_new_users)
#     X_revenue_scaled = scaler_X_rev.fit_transform(X_revenue)
    
#     # 訓練模型
#     model_new_users = RandomForestRegressor(random_state=42)
#     model_revenue = RandomForestRegressor(random_state=42)
    
#     model_new_users.fit(X_new_users_scaled, y_new_users)
#     model_revenue.fit(X_revenue_scaled, y_revenue)
    
#     return (model_new_users, model_revenue), (scaler_X_new, scaler_X_rev)

# def get_prediction_interval(model, X, y_mean, percentile=95):
#     """計算預測的信賴區間"""
#     predictions = []
#     for estimator in model.estimators_:
#         predictions.append(estimator.predict(X))
#     predictions = np.array(predictions)
    
#     # 計算指定百分位數的信賴區間
#     lower_bound = np.percentile(predictions, (100 - percentile) / 2, axis=0)
#     upper_bound = np.percentile(predictions, 100 - (100 - percentile) / 2, axis=0)
    
#     return lower_bound, upper_bound

# def predict_revenue(investment_plan, base_data, models, scalers):
#     """預測收益，考慮用戶細分"""
#     results = []
#     prev_investment = base_data['投遞金額（google ads）']
#     prev_active_users = base_data['活躍使用者']
#     prev_arrpu = base_data['ARRPU']
#     retention_rate = base_data['月留存率']
    
#     # 業務邏輯常數
#     NEW_USER_ACTIVE_RATE = 0.4  # 新用戶中活躍用戶的比例
    
#     model_new_users, model_revenue = models
#     scaler_X_new, scaler_X_rev = scalers
    
#     for month in range(1, 13):
#         current_investment = investment_plan[month-1]
        
#         # 預測新使用者
#         if current_investment == 0:
#             predicted_new_users = 0
#             new_users_lower = 0
#             new_users_upper = 0
#         else:
#             X_new = np.array([[
#                 current_investment,
#                 prev_investment,
#                 month,
#                 (month-1)//3 + 1
#             ]])
#             X_new_scaled = scaler_X_new.transform(X_new)
#             predicted_new_users = max(0, model_new_users.predict(X_new_scaled)[0])
            
#             # 計算新用戶預測的信賴區間
#             new_users_lower, new_users_upper = get_prediction_interval(
#                 model_new_users, X_new_scaled, predicted_new_users
#             )
#             new_users_lower = max(0, new_users_lower[0])
#             new_users_upper = max(0, new_users_upper[0])
        
#         # 計算新用戶中的活躍用戶
#         predicted_new_active_users = predicted_new_users * NEW_USER_ACTIVE_RATE
#         new_active_users_lower = new_users_lower * NEW_USER_ACTIVE_RATE
#         new_active_users_upper = new_users_upper * NEW_USER_ACTIVE_RATE
        
#         # 計算預期活躍用戶數（考慮留存的舊用戶 + 新活躍用戶）
#         predicted_old_active_users = prev_active_users * retention_rate
#         predicted_active_users = predicted_old_active_users + predicted_new_active_users
        
#         # 估算活躍用戶的信賴區間
#         old_users_std = predicted_old_active_users * 0.1  # 假設10%的標準差
#         active_users_lower = (predicted_old_active_users - 1.96 * old_users_std) + new_active_users_lower
#         active_users_upper = (predicted_old_active_users + 1.96 * old_users_std) + new_active_users_upper
        
#         # 預測變現收益
#         X_rev = np.array([[
#             predicted_new_users,
#             predicted_active_users,
#             prev_arrpu,
#             month,
#             (month-1)//3 + 1
#         ]])
#         X_rev_scaled = scaler_X_rev.transform(X_rev)
#         predicted_revenue = max(0, model_revenue.predict(X_rev_scaled)[0])
        
#         # 計算收益預測的信賴區間
#         revenue_lower, revenue_upper = get_prediction_interval(
#             model_revenue, X_rev_scaled, predicted_revenue
#         )
#         revenue_lower = max(0, revenue_lower[0])
#         revenue_upper = max(0, revenue_upper[0])
        
#         # 計算ROI
#         roi = predicted_revenue / current_investment if current_investment > 0 else 0
#         roi_lower = revenue_lower / current_investment if current_investment > 0 else 0
#         roi_upper = revenue_upper / current_investment if current_investment > 0 else 0
        
#         results.append({
#             '月份': month,
#             '投資金額': current_investment,
#             '預測新使用者': predicted_new_users,
#             '預測新活躍使用者': predicted_new_active_users,
#             '新使用者下界': new_users_lower,
#             '新使用者上界': new_users_upper,
#             '預測舊活躍用戶': predicted_old_active_users,
#             '預測總活躍用戶': predicted_active_users,
#             '活躍用戶下界': active_users_lower,
#             '活躍用戶上界': active_users_upper,
#             '預測收益': predicted_revenue,
#             '收益下界': revenue_lower,
#             '收益上界': revenue_upper,
#             '預測ROI': roi,
#             'ROI下界': roi_lower,
#             'ROI上界': roi_upper
#         })
        
#         # 更新下個月的基礎數據
#         prev_investment = current_investment
#         prev_active_users = predicted_active_users
        
#     return pd.DataFrame(results)


# def get_model_formula(model, feature_names, scaler):
#     """獲取模型的近似線性公式"""
#     # 獲取特徵重要性
#     importances = model.feature_importances_
    
#     # 獲取一個樣本點進行預測
#     sample_point = np.zeros((1, len(feature_names)))
    
#     # 計算每個特徵的單位變化對預測的影響
#     feature_effects = []
#     for i in range(len(feature_names)):
#         test_point1 = sample_point.copy()
#         test_point2 = sample_point.copy()
        
#         # 使用標準差為1的變化
#         test_point1[0, i] = -1
#         test_point2[0, i] = 1
        
#         # 進行預測
#         pred1 = model.predict(test_point1)[0]
#         pred2 = model.predict(test_point2)[0]
        
#         # 計算效應
#         effect = (pred2 - pred1) / 2
#         feature_effects.append(effect)
    
#     # 生成公式字符串
#     formula_parts = []
#     for name, importance, effect in zip(feature_names, importances, feature_effects):
#         if importance > 0.05:  # 只顯示重要性大於5%的特徵
#             coefficient = effect
#             if coefficient > 0:
#                 formula_parts.append(f"+{coefficient:.2f}×{name}")
#             else:
#                 formula_parts.append(f"{coefficient:.2f}×{name}")
    
#     formula = "預測值 = " + " ".join(formula_parts)
#     return formula, dict(zip(feature_names, importances))

# def plot_feature_importance(feature_importance, title):
#     """繪製特徵重要性圖表"""
#     features = list(feature_importance.keys())
#     importance_values = list(feature_importance.values())
    
#     fig = go.Figure(data=[
#         go.Bar(x=features, y=importance_values)
#     ])
    
#     fig.update_layout(
#         title=title,
#         xaxis_title="特徵",
#         yaxis_title="重要性",
#         height=400
#     )
    
#     return fig

# def plot_results(results, title):
#     """繪製結果圖表，加入用戶細分視圖"""
#     # 創建子圖，指定類型
#     fig = make_subplots(
#         rows=3, cols=2,
#         specs=[
#             [{"type": "xy"}, {"type": "xy"}],
#             [{"type": "xy"}, {"type": "xy"}],
#             [{"type": "domain"}, {"type": "xy"}]  # 將第三行第一列設為 domain 類型，用於繪製餅圖
#         ],
#         subplot_titles=(
#             '投資金額與預測收益', '預測新使用者',
#             '活躍用戶組成', '預測ROI',
#             '新舊活躍用戶比例', '月度趨勢'
#         )
#     )
    
#     # 投資金額與預測收益 (row=1, col=1)
#     fig.add_trace(
#         go.Scatter(x=results['月份'], y=results['投資金額'],
#                   name='投資金額', mode='lines+markers', line=dict(color='blue')),
#         row=1, col=1
#     )
    
#     fig.add_trace(
#         go.Scatter(x=results['月份'], y=results['預測收益'],
#                   name='預測收益', mode='lines+markers', line=dict(color='red')),
#         row=1, col=1
#     )
    
#     # 收益信賴區間
#     fig.add_trace(
#         go.Scatter(x=list(results['月份']) + list(results['月份'])[::-1],
#                   y=list(results['收益上界']) + list(results['收益下界'])[::-1],
#                   fill='toself', fillcolor='rgba(255,0,0,0.2)',
#                   line=dict(color='rgba(255,0,0,0)'),
#                   name='收益95%信賴區間'),
#         row=1, col=1
#     )
    
#     # 預測新使用者 (row=1, col=2)
#     fig.add_trace(
#         go.Scatter(x=results['月份'], y=results['預測新使用者'],
#                   name='預測新使用者', mode='lines+markers', line=dict(color='green')),
#         row=1, col=2
#     )
    
#     # 添加新使用者信賴區間
#     fig.add_trace(
#         go.Scatter(x=list(results['月份']) + list(results['月份'])[::-1],
#                   y=list(results['新使用者上界']) + list(results['新使用者下界'])[::-1],
#                   fill='toself', fillcolor='rgba(0,255,0,0.2)',
#                   line=dict(color='rgba(0,255,0,0)'),
#                   name='新使用者95%信賴區間'),
#         row=1, col=2
#     )
    
#     # 活躍用戶組成堆疊圖 (row=2, col=1)
#     fig.add_trace(
#         go.Bar(x=results['月份'], y=results['預測新活躍使用者'],
#                name='新活躍用戶', marker_color='lightgreen'),
#         row=2, col=1
#     )
    
#     fig.add_trace(
#         go.Bar(x=results['月份'], y=results['預測舊活躍用戶'],
#                name='舊活躍用戶', marker_color='darkgreen'),
#         row=2, col=1
#     )
    
#     # ROI (row=2, col=2)
#     fig.add_trace(
#         go.Scatter(x=results['月份'], y=results['預測ROI'],
#                   name='預測ROI', mode='lines+markers', line=dict(color='orange')),
#         row=2, col=2
#     )
    
#     # ROI信賴區間
#     fig.add_trace(
#         go.Scatter(x=list(results['月份']) + list(results['月份'])[::-1],
#                   y=list(results['ROI上界']) + list(results['ROI下界'])[::-1],
#                   fill='toself', fillcolor='rgba(255,165,0,0.2)',
#                   line=dict(color='rgba(255,165,0,0)'),
#                   name='ROI95%信賴區間'),
#         row=2, col=2
#     )
    
#     # 新舊活躍用戶比例餅圖 (row=3, col=1)
#     total_new_active = results['預測新活躍使用者'].sum()
#     total_old_active = results['預測舊活躍用戶'].sum()
#     fig.add_trace(
#         go.Pie(labels=['新活躍用戶', '舊活躍用戶'],
#                values=[total_new_active, total_old_active],
#                name='活躍用戶組成',
#                marker_colors=['lightgreen', 'darkgreen']),
#         row=3, col=1
#     )
    
#     # 活躍用戶月度趨勢 (row=3, col=2)
#     fig.add_trace(
#         go.Scatter(x=results['月份'], 
#                   y=results['預測總活躍用戶'],
#                   name='總活躍用戶',
#                   mode='lines+markers',
#                   line=dict(color='purple')),
#         row=3, col=2
#     )
    
#     # 更新布局
#     fig.update_layout(
#         height=1200,
#         width=1200,
#         title_text=title,
#         showlegend=True,
#         barmode='stack'
#     )
    
#     # 更新子圖的橫軸標題
#     fig.update_xaxes(title_text="月份", row=1, col=1)
#     fig.update_xaxes(title_text="月份", row=1, col=2)
#     fig.update_xaxes(title_text="月份", row=2, col=1)
#     fig.update_xaxes(title_text="月份", row=2, col=2)
#     fig.update_xaxes(title_text="月份", row=3, col=2)
    
#     # 更新子圖的縱軸標題
#     fig.update_yaxes(title_text="金額", row=1, col=1)
#     fig.update_yaxes(title_text="新使用者數", row=1, col=2)
#     fig.update_yaxes(title_text="活躍用戶數", row=2, col=1)
#     fig.update_yaxes(title_text="ROI", row=2, col=2)
#     fig.update_yaxes(title_text="活躍用戶數", row=3, col=2)
    
#     return fig

# def main():
#     st.title('廣告變現收益預測')
#     df = load_initial_data()
#     monthly_budget = []  # 初始化每月預算列表
#     st.sidebar.subheader("請設置每月預算")
#     default_budgets = {1: 496675.0, 2: 646544.0, 3: 631547.0, 4: 730672.0, 5: 1192148.0, 6: 813243.0, 7: 782203.0, 8: 780915.0, 9: 780966.0, 10: 794793.0, 11: 794793.0, 12: 794793.0}
#     monthly_budget = [float(st.sidebar.text_input(f'第 {month} 月預算', value=str(default_budgets[month]))) for month in range(1, 13)]

    
#     # 選擇數據來源
#     data_source = st.radio("選擇數據來源", ("使用預設資料", "上傳 CSV 文件"))
    
#     if data_source == "上傳 CSV 文件":
#         uploaded_file = st.file_uploader("上傳 CSV 文件", type=["csv"])
        
#         if uploaded_file is not None:
#             df = pd.read_csv(uploaded_file)
#             expected_columns = ['年', '月', '變現收益', '投遞金額（google ads）', '新使用者', '活躍使用者', '月留存率', 'ARRPU']
#             if not all(col in df.columns for col in expected_columns):
#                 st.error("上傳的文件格式不正確，請確保包含以下列: " + ", ".join(expected_columns))
#                 return
#             else:
#                 st.subheader("上傳的數據")
#                 st.dataframe(df, use_container_width=True)  
#     else:
#         # 載入預設數據
#         df = load_initial_data()
#         st.subheader("預設數據")
#         st.dataframe(df, use_container_width=True)  
    
#     df_prepared = prepare_data(df)
    
#     # 訓練模型
#     models, scalers = train_models(df_prepared)
    
#     # 準備預測
#     base_data = df_prepared.iloc[-1]  
    
#     # 預測結果
#     st.subheader("預測結果")
#     results = predict_revenue(monthly_budget, base_data, models, scalers)
#     st.dataframe(results, use_container_width=True) 
#        # 下載按鈕
#     csv = results.to_csv(index=False)
#     st.download_button(
#         label="下載預測結果",
#         data=csv,
#         file_name="prediction_results.csv",
#         mime="text/csv"
#     )
#     st.header("模型解釋")
    
#     # 獲取新用戶模型的公式和特徵重要性
#     new_users_formula, new_users_importance = get_model_formula(
#         models[0], 
#         ['投遞金額', '上月投遞金額', '月份', '季度'],
#         scalers[0]
#     )
    
#     # 獲取收益模型的公式和特徵重要性
#     revenue_formula, revenue_importance = get_model_formula(
#         models[1],
#         ['新使用者數', '活躍使用者數', 'ARPPU', '月份', '季度'],
#         scalers[1]
#     )
    
#     # 顯示模型公式
#     st.subheader("預測模型近似公式")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("**新用戶預測模型：**")
#         st.write(new_users_formula)
#         st.plotly_chart(plot_feature_importance(new_users_importance, "新用戶模型特徵重要性"))
    
#     with col2:
#         st.markdown("**收益預測模型：**")
#         st.write(revenue_formula)
#         st.plotly_chart(plot_feature_importance(revenue_importance, "收益模型特徵重要性"))


#     # 顯示主要指標
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("預計總投資", f"{results['投資金額'].sum():,.0f}")
#     with col2:
#         st.metric("預計總收益", f"{results['預測收益'].sum():,.0f}")
#     with col3:
#         st.metric("平均ROI", f"{results['預測ROI'].mean():.2f}")
#     with col4:
#         st.metric("ROI標準差", f"{results['預測ROI'].std():.2f}")
    
#     # 顯示圖表
#     st.plotly_chart(plot_results(results, "預測結果"))

  
    
 
# if __name__ == "__main__":
#     main()



import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 設置頁面配置
st.set_page_config(page_title="廣告變現收益預測", layout="wide")

# 初始數據
@st.cache_data
def load_initial_data():
    data = {
        'date': ['2023-07-01','2023-08-01','2023-09-01','2023-10-01','2023-11-01','2023-12-01','2024-01-01','2024-02-01','2024-03-01','2024-04-01','2024-05-01','2024-06-01'],
        'cost': [
            727369, 494465, 382925, 353211, 509009, 506131,
            496675, 636544, 631547, 730672, 1192148, 813243
        ],
        'revenue': [
            1073448, 1023352, 848734, 749857, 749430, 792460,
            848311, 793719, 840042, 779230, 777497, 738903
        ]
    }
    
    return pd.DataFrame(data)

def prepare_data(df):
    """準備Prophet所需的數據格式"""
    # 確保日期格式正確
    df['date'] = pd.to_datetime(df['date'])
    
    # 準備Prophet數據
    prophet_df = df.rename(columns={
        'date': 'ds',
        'revenue': 'y'
    }).copy()
    
    # 添加cost作為regressor
    prophet_df['cost'] = df['cost']
    
    # 確保數據按日期排序
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    return prophet_df

def train_model(df_prepared):
    """訓練Prophet模型"""
    model = Prophet(
        seasonality_mode='additive',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        # changepoint_prior_scale=0.001,  # 減小趨勢變化的靈活性
        # seasonality_prior_scale=0.1     # 適度的季節性
    )
    
    # 添加月度季節性
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=3  # 減小order以避免過擬合
    )
    
    # 添加cost作為regressor
    model.add_regressor('cost', standardize=False)
    
    # 訓練模型
    model.fit(df_prepared)
    
    return model

def predict_revenue(model, future_costs, periods):
    """使用Prophet模型進行預測"""
    # 創建未來日期DataFrame
    future_dates = model.make_future_dataframe(
        periods=periods,
        freq='MS'
    )
    
    # 獲取模型的訓練數據中的成本
    historical_costs = model.history['cost']
    
    # 組合歷史成本和未來成本
    future_dates['cost'] = pd.concat([
        historical_costs,
        pd.Series(future_costs[:periods])  # 確保只使用需要的預測期數的成本
    ]).reset_index(drop=True)
    
    # 確保沒有NaN值
    if future_dates['cost'].isna().any():
        raise ValueError("成本數據中存在缺失值，請確保所有時間點都有對應的成本數據")
    
    # 進行預測
    forecast = model.predict(future_dates)
    
    # 只獲取預測期間的結果
    forecast_results = forecast.tail(periods)
    
    # 準備結果DataFrame
    results = pd.DataFrame({
        'date': range(1, periods + 1),
        'cost': future_costs[:periods],
        'predict_revenue': forecast_results['yhat'],
        'lower_bound': forecast_results['yhat_lower'],
        'upper_bound': forecast_results['yhat_upper']
    })
    
    # 計算ROI
    results['roi'] = results['predict_revenue'] / results['cost']
    results['roi_lower'] = results['predict_revenue'] / results['cost']
    results['roi_upper'] = results['predict_revenue'] / results['cost']
    
    # 新增預測日期
    results['date'] = pd.to_datetime(future_dates['ds']).dt.strftime('%Y-%m-%d') 
    

    return results, forecast

def plot_results(results, forecast, title):
    """繪製預測結果圖表"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '投資金額與預測收益',
            '預測ROI',
            '趨勢分解',
            '月度季節性'
        )
    )
    
    # 投資金額與預測收益
    fig.add_trace(
        go.Scatter(
            x=results['date'],
            y=results['cost'],
            name='投資金額',
            mode='lines+markers',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=results['date'],
            y=results['predict_revenue'],
            name='預測收益',
            mode='lines+markers',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # 收益信賴區間
    fig.add_trace(
        go.Scatter(
            x=list(results['date']) + list(results['date'])[::-1],
            y=list(results['upper_bound']) + list(results['lower_bound'])[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
            name='收益95%信賴區間'
        ),
        row=1, col=1
    )
    
    # ROI
    fig.add_trace(
        go.Scatter(
            x=results['date'],
            y=results['roi'],
            name='預測ROI',
            mode='lines+markers',
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    # ROI信賴區間
    fig.add_trace(
        go.Scatter(
            x=list(results['date']) + list(results['date'])[::-1],
            y=list(results['roi_upper']) + list(results['roi_lower'])[::-1],
            fill='toself',
            fillcolor='rgba(0,255,0,0.2)',
            line=dict(color='rgba(0,255,0,0)'),
            name='ROI95%信賴區間'
        ),
        row=1, col=2
    )
    
    # 趨勢
    fig.add_trace(
        go.Scatter(
            x=list(range(len(forecast))),  # 將range轉換為列表
            y=forecast['trend'],
            name='趨勢',
            line=dict(color='purple')
        ),
        row=2, col=1
    )

    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text=title,
        showlegend=True
    )
    
    # 更新軸標籤
    fig.update_xaxes(title_text="月份", row=1, col=1)
    fig.update_xaxes(title_text="月份", row=1, col=2)
    fig.update_xaxes(title_text="時間序列", row=2, col=1)
    fig.update_xaxes(title_text="時間序列", row=2, col=2)
    
    fig.update_yaxes(title_text="金額", row=1, col=1)
    fig.update_yaxes(title_text="ROI", row=1, col=2)
    fig.update_yaxes(title_text="趨勢值", row=2, col=1)
    fig.update_yaxes(title_text="季節性", row=2, col=2)
    
    return fig

def main():
    st.title('收入預測分析')
    
    # 設置側邊欄的預算輸入
    st.sidebar.subheader("請設置每月預算")
    default_budgets = {1: 782203.0, 2: 780915.0, 3: 780966.0, 4: 794793.0, 5: 794793.0, 6: 794793.0, 7: 794793.0, 8: 794793.0, 9: 794793.0, 10: 794793.0, 11: 794793.0, 12: 794793.0}
    monthly_budget = [
        float(st.sidebar.text_input(
            f'第 {month} 月預算',
            value=str(default_budgets[month])
        ))
        for month in range(1, 13)
    ]
    
    # 選擇數據來源
    data_source = st.radio("選擇數據來源", ("使用預設資料", "上傳 CSV 文件"))
    
    if data_source == "上傳 CSV 文件":
        uploaded_file = st.file_uploader("上傳 CSV 文件", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            expected_columns = ['date', 'cost', 'revenue']
            if not all(col in df.columns for col in expected_columns):
                st.error("上傳的文件格式不正確，請確保包含以下列: " + ", ".join(expected_columns))
                return
            df['date'] = pd.to_datetime(df['date'])
            st.subheader("上傳的數據")
            st.dataframe(df)
    else:
        df = load_initial_data()
        df_display = df.rename(columns={
        'date': '預測日期',
        'cost': '歷史投遞金額',
        'revenue': '歷史變現收益'})
        st.subheader("預設數據")
        st.dataframe(df_display, use_container_width=True)
    
    # 準備數據並訓練模型
    df_prepared = prepare_data(df)
    model = train_model(df_prepared)
    
    # 預測結果
    st.subheader("預測結果")
    results, forecast = predict_revenue(model, monthly_budget, 12)
    df_results_display = results.rename(columns={
        'date': '日期',
        'cost': '預期投遞金額',
        'lower_bound': '預測下限',
        'predict_revenue': '預測變現收益',
        'upper_bound': '預測上限',
        'roi_lower': '預計ROAS下限',
        'roi': '預計ROAS',
        'roi_upper': '預計ROAS上限',
        })
    st.dataframe(df_results_display, use_container_width=True)
    
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
        st.metric("預計總投資", f"{results['cost'].sum():,.0f}")
    with col2:
        st.metric("預計總收益", f"{results['predict_revenue'].sum():,.0f}")
    with col3:
        st.metric("平均ROI", f"{results['roi'].mean():.2f}")
    with col4:
        st.metric("ROI標準差", f"{results['roi'].std():.2f}")
    
    # 顯示圖表
    st.plotly_chart(plot_results(results, forecast, "預測結果分析"))

if __name__ == "__main__":
    main()
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
def load_initial_data(dataset_name="kdan_android"):
    if dataset_name == "kdan_android":
        data = {
            'date': ['2022-01-01','2022-02-01','2022-03-01','2022-04-01','2022-05-01','2022-06-01','2022-07-01','2022-08-01','2022-09-01','2022-10-01','2022-11-01','2022-12-01','2023-01-01','2023-02-01','2023-03-01','2023-04-01','2023-05-01','2023-06-01','2023-07-01','2023-08-01','2023-09-01','2023-10-01','2023-11-01','2023-12-01'],
            'cost': [
                704177, 762384, 812837, 904768, 1013294, 1217421,
                1328718, 1530757, 1547773, 1548895, 1452694, 1095080,
                897250, 842486, 1036517, 1154801, 1042375, 1263188,
                727369, 494465, 382925, 353211, 509009, 506131
            ],
            'active_user': [
                1487546, 1468368, 1464235, 1402852, 1386879, 1369241, 1356332, 1364901, 1347618, 1294489, 1287219, 1199877, 1262118, 1188010, 1221980, 1135310, 1116841, 1099087, 944065, 969298, 946241, 892729, 823957, 759620
            ],
            'revenue': [
                1665937, 1513545, 1831731, 1937624, 1874419, 1723995,
                1979887, 1998035, 1746071, 1331042, 1258247, 1121431,
                1059160, 999901, 1076458, 943998, 1077483, 1162024,
                1073448, 1023352, 848734, 749857, 749430, 792460
            ]
        }
    elif dataset_name == "cs_android":
        data = {
    'date': [
        '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01',
        '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01',
        '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01',
        '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01'
    ],
    'cost': [
        411166.0, 459678.0, 467154.0, 358090.0, 321809.0, 375642.0,
        516304.0, 389143.0, 325003.0, 286079.0, 356050.0, 293915.0,
        112422.0, 109266.0, 113934.0, 139135.0, 129609.0, 141700.0,
        172153.0, 198878.0, 159169.0, 173970.0, 194594.0, 181865.0
    ],
    'active_user': [
        780453.0, 869452.0, 938582.0, 794291.0, 794872.0, 751335.0,
        747692.0, 769719.0, 790245.0, 778229.0, 798234.0, 698742.0,
        618280.0, 586202.0, 583201.0, 561235.0, 548519.0, 512903.0,
        500318.0, 496642.0, 482258.0, 489265.0, 476973.0, 450954.0
    ],
    'revenue': [
        233243.0, 265104.0, 345975.0, 307883.0, 309512.0, 296876.0,
        307495.0, 285060.0, 276729.0, 132227.0, 174468.0, 141109.0,
        95627.0, 70130.0, 65817.0, 53972.0, 54981.0, 102854.0,
        115993.0, 104805.0, 103678.0, 94063.0, 91066.0, 56464.0
    ]
}
    else:
        raise ValueError("Unknown dataset name")

    return pd.DataFrame(data)

def prepare_future_regressor(historical_data, column_name, forecast_periods, months=6):
    """
    使用最後6個月的平均值作為未來預測值，這種方法對於波動的活躍用戶數是合理的
    """
    last_6_months = historical_data[column_name].tail(months)
    mean_value = last_6_months.mean()
    
    # 增加一些診斷信息
    std_value = last_6_months.std()
    cv = std_value / mean_value  # 變異係數
    
    # print(f"\n{column_name} 預測診斷：")
    # print(f"過去{months}個月平均值: {mean_value:.2f}")
    # print(f"標準差: {std_value:.2f}")
    # print(f"變異係數: {cv:.2f}")
    # print(f"最大值: {last_6_months.max():.2f}")
    # print(f"最小值: {last_6_months.min():.2f}")
    
    return [mean_value] * forecast_periods



def prepare_data(df):
    """準備Prophet所需的數據格式"""
    # 確保日期格式正確
    df['date'] = pd.to_datetime(df['date'])
    
    # 確保其他列為浮點數
    df['revenue'] = df['revenue'].astype(float)
    df['cost'] = df['cost'].astype(float)
    df['active_user'] = df['active_user'].astype(float)
        
    # 準備Prophet數據
    prophet_df = df.rename(columns={
        'date': 'ds',
        'revenue': 'y'
    }).copy()
    
    # 添加regressors
    prophet_df['cost'] = df['cost']
    prophet_df['active_user'] = df['active_user']
    
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    min_cap = max(0, df['revenue'].min() * 0.5)  
    max_cap = df['revenue'].max() * 1.5          
    prophet_df['floor'] = min_cap
    prophet_df['cap'] = max_cap
    
    return prophet_df
def get_model_parameters(model_name="model_kdan_android"):
    """取得不同模型的參數配置"""
    model_configs = {
        "model_kdan_android": {
            "model_params": {
                'seasonality_mode': 'multiplicative',
                'growth': 'logistic',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.0005,
                'seasonality_prior_scale': 0.01,
                'interval_width': 0.67,
                'n_changepoints': 6
            },
            "seasonality_params": {
                'period': 30.5,
                'fourier_order': 2,
                'prior_scale': 0.01
            },
            "regressor_params": {
                'cost': {
                    'prior_scale': 0.25,
                    'mode': 'additive'
                },
                'active_user': {
                    'prior_scale': 0.95,
                    'mode': 'additive'
                }
            }
        },
        "custom": {
            "model_params": {
                'seasonality_mode': 'multiplicative',
                'growth': 'logistic',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.001,
                'seasonality_prior_scale': 0.1,
                'interval_width': 0.95,
                'n_changepoints': 10
            },
            "seasonality_params": {
                'period': 30.5,
                'fourier_order': 3,
                'prior_scale': 0.1
            },
            "regressor_params": {
                'cost': {
                    'prior_scale': 0.5,
                    'mode': 'additive'
                },
                'active_user': {
                    'prior_scale': 0.5,
                    'mode': 'additive'
                }
            }
        },
        "model_cs_android": {
            "model_params": {
                'seasonality_mode': 'multiplicative',
                'growth': 'logistic',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.0003,
                'seasonality_prior_scale': 0.005,
                'interval_width': 0.58,
                'n_changepoints': 8
            },
            "seasonality_params": {
                'period': 30.5,
                'fourier_order': 3,
                'prior_scale': 0.02
            },
            "regressor_params": {
                'cost': {
                    'prior_scale': 0.4,
                    'mode': 'multiplicative'
                },
                'active_user': {
                    'prior_scale': 0.85,
                    'mode': 'multiplicative'
                }
            }
        }
    }
    
    return model_configs.get(model_name)

def customize_model_parameters(st, base_params):
    """允許使用者自定義模型參數"""
    st.markdown("### 模型參數設置")
    
    with st.expander("基本參數設置"):
        model_params = base_params["model_params"].copy()
        model_params['seasonality_mode'] = st.selectbox(
            "季節性模式",
            ['multiplicative', 'additive'],
            index=0 if model_params['seasonality_mode'] == 'multiplicative' else 1
        )
        model_params['growth'] = st.selectbox(
            "成長模式",
            ['logistic', 'linear', 'flat'],
            index=0 if model_params['growth'] == 'logistic' else 1
        )
        model_params['changepoint_prior_scale'] = st.number_input(
            "變點先驗尺度",
            min_value=0.0001,
            max_value=0.5,
            value=float(model_params['changepoint_prior_scale']),
            format='%f'
        )
        model_params['seasonality_prior_scale'] = st.number_input(
            "季節性先驗尺度",
            min_value=0.01,
            max_value=10.0,
            value=float(model_params['seasonality_prior_scale']),
            format='%f'
        )
        model_params['interval_width'] = st.slider(
            "預測區間寬度",
            min_value=0.5,
            max_value=0.95,
            value=float(model_params['interval_width'])
        )
        model_params['n_changepoints'] = st.slider(
            "變點數量",
            min_value=1,
            max_value=20,
            value=int(model_params['n_changepoints'])
        )
    
    with st.expander("季節性參數設置"):
        seasonality_params = base_params["seasonality_params"].copy()
        seasonality_params['fourier_order'] = st.slider(
            "傅立葉階數",
            min_value=1,
            max_value=10,
            value=int(seasonality_params['fourier_order'])
        )
        seasonality_params['prior_scale'] = st.number_input(
            "季節性先驗尺度",
            min_value=0.01,
            max_value=10.0,
            value=float(seasonality_params['prior_scale']),
            format='%f'
        )
    
    with st.expander("Regressor 參數設置"):
        regressor_params = base_params["regressor_params"].copy()
        for regressor in ['cost', 'active_user']:
            st.markdown(f"#### {regressor} 設置")
            regressor_params[regressor]['prior_scale'] = st.number_input(
                f"{regressor} 先驗尺度",
                min_value=0.01,
                max_value=10.0,
                value=float(regressor_params[regressor]['prior_scale']),
                format='%f',
                key=f"{regressor}_prior_scale"
            )
            regressor_params[regressor]['mode'] = st.selectbox(
                f"{regressor} 模式",
                ['additive', 'multiplicative'],
                index=0 if regressor_params[regressor]['mode'] == 'additive' else 1,
                key=f"{regressor}_mode"
            )
    
    return {
        "model_params": model_params,
        "seasonality_params": seasonality_params,
        "regressor_params": regressor_params
    }

def train_model(df_prepared, model_params, seasonality_params, regressor_params):
    """使用指定參數訓練Prophet模型"""
    model = Prophet(**model_params)
    
    # 添加月度季節性
    model.add_seasonality(
        name='monthly',
        **seasonality_params
    )
    
    # 添加regressors
    for regressor_name, params in regressor_params.items():
        if regressor_name in df_prepared.columns:
            model.add_regressor(regressor_name, **params)
    
    # 訓練模型
    model.fit(df_prepared)
    
    return model


def predict_revenue(model, future_costs, periods, historical_data):
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
    
    # 確保active_user的未來預測值也被添加
    future_user_values = prepare_future_regressor(historical_data, 'active_user', len(future_dates))  # Use the length of future_dates

    future_dates['active_user'] = future_user_values  # Assign the correct length

    # 添加 floor 和 cap 列
    min_cap = max(0, historical_data['revenue'].min() * 0.5)  
    max_cap = historical_data['revenue'].max() * 1.5          
    future_dates['floor'] = min_cap
    future_dates['cap'] = max_cap

    # 確保沒有NaN值
    if future_dates['cost'].isna().any() or future_dates['active_user'].isna().any():
        raise ValueError("成本或活躍用戶數據中存在缺失值，請確保所有時間點都有對應的數據")
    
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
    
    results['roi_lower'] = round(results['predict_revenue'] / results['cost'],2)
    results['roi_upper'] = round(results['upper_bound'] / results['cost'],2)
    
    results['predict_revenue'] = round(results['predict_revenue'],2)
    results['upper_bound'] = round(results['upper_bound'],2)
    # 新增預測日期
    results['date'] = pd.to_datetime(future_dates['ds']).dt.strftime('%Y-%m') 
    print(results)
    return results, forecast


# def calculate_diagnostics(forecast, historical_data):
#     """計算預測診斷指標"""
#     # 只取歷史數據部分的預測
#     historical_predictions = forecast[forecast['ds'].isin(historical_data['date'])]
#     actual_values = historical_data['revenue'].values
#     predicted_values = historical_predictions['yhat'].values
    
#     # 計算各種誤差指標
#     mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
#     rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
#     mae = np.mean(np.abs(actual_values - predicted_values))
    
#     # 計算預測區間覆蓋率
#     in_interval = np.sum((actual_values >= historical_predictions['yhat_lower']) & 
#                         (actual_values <= historical_predictions['yhat_upper']))
#     coverage = in_interval / len(actual_values) * 100
    
#     return {
#         'MAPE': round(mape, 2),
#         'RMSE': round(rmse, 2),
#         'MAE': round(mae, 2),
#         'Coverage': round(coverage, 2)
#     }

# def main():
#     st.title('收入預測分析')
    
#     # 設置側邊欄的預算輸入
#     st.sidebar.subheader("選擇預設預算")
#     budget_choice = st.sidebar.radio(
#         "選擇預算類型",
#         ["kdan_android", "cs_android"]
#     )

#     # Define default budgets for each type
#     default_budgets_kdan = {
#         1: 496675.0, 
#         2: 646544.0, 
#         3: 631547.0, 
#         4: 730672.0, 
#         5: 1192148.0, 
#         6: 813243.0, 
#         7: 782203.0, 
#         8: 780915.0, 
#         9: 780966.0, 
#         10: 794793.0, 
#         11: 794793.0, 
#         12: 794793.0}
#     default_budgets_cs = {
#         1: 26584.0, 
#         2: 27790.0, 
#         3: 33841.0, 
#         4: 43701.0, 
#         5: 38353.0, 
#         6: 33748.0, 
#         7: 46247.0, 
#         8: 47568.0, 
#         9: 49241.0, 
#         10: 60630.0,
#         11: 60630.0,
#         12: 60630.0
#     }

#     st.sidebar.subheader("選擇數據集")
#     dataset_choice = st.sidebar.radio(
#         "選擇數據集",
#         ["kdan_android", "cs_android"]
#     )
    
#     # Set default budgets based on dataset choice
#     if dataset_choice == "kdan_android":
#         default_budgets = default_budgets_kdan
#     else:
#         default_budgets = default_budgets_cs

#     # Set monthly budgets based on the selected default
#     monthly_budget = [
#         float(st.sidebar.text_input(
#             f'第 {month} 月預算',
#             value=str(default_budgets[month])
#         ))
#         for month in range(1, 13)
#     ]
#     df = load_initial_data(dataset_choice)
    
#     # 選擇數據來源
#     data_source = st.radio("選擇數據來源", ("使用預設資料", "上傳 CSV 文件"))
#     if data_source == "上傳 CSV 文件":
#         uploaded_file = st.file_uploader("上傳 CSV 文件", type=["csv"])
#         if uploaded_file is not None:
#             df = pd.read_csv(uploaded_file)
#             expected_columns = ['date', 'cost', 'revenue', 'active_user']
#             if not all(col in df.columns for col in expected_columns):
#                 st.error("上傳的文件格式不正確，請確保包含以下列: " + ", ".join(expected_columns))
#                 return
#             df['date'] = pd.to_datetime(df['date'])
#             st.subheader("上傳的數據")
#             df_display = df.rename(columns={
#                 'date': '預測日期',
#                 'cost': '歷史投遞金額',
#                 'active_user': '活躍用戶數',
#                 'revenue': '歷史變現收益'
#             })
#             df_display['預測日期'] = pd.to_datetime(df_display['預測日期']).dt.strftime('%Y-%m')
#             st.dataframe(df_display, use_container_width=True)

#             st.subheader("年度統計資料")
#             df['year'] = df['date'].dt.year

#             yearly_stats = pd.DataFrame({
#                 '指標': ['總額', '平均', '標準差', '最小值', '最大值'],
#                 '變現收益': [
#                     df.groupby('year')['revenue'].sum().round(2),
#                     df.groupby('year')['revenue'].mean().round(2),
#                     df.groupby('year')['revenue'].std().round(2),
#                     df.groupby('year')['revenue'].min().round(2),
#                     df.groupby('year')['revenue'].max().round(2)
#                 ],
#                 '投遞金額': [
#                     df.groupby('year')['cost'].sum().round(2),
#                     df.groupby('year')['cost'].mean().round(2),
#                     df.groupby('year')['cost'].std().round(2),
#                     df.groupby('year')['cost'].min().round(2),
#                     df.groupby('year')['cost'].max().round(2)
#                 ],
#                 '活躍用戶': [
#                     df.groupby('year')['active_user'].mean().round(2),
#                     df.groupby('year')['active_user'].mean().round(2),
#                     df.groupby('year')['active_user'].std().round(2),
#                     df.groupby('year')['active_user'].min().round(2),
#                     df.groupby('year')['active_user'].max().round(2)
#                 ]
#             }).set_index('指標')

#             st.dataframe(yearly_stats, use_container_width=True)

#             st.subheader("年度變化趨勢")
#             yearly_trends = df.groupby('year').agg({
#                 'revenue': 'sum',
#                 'cost': 'sum',
#                 'active_user': 'mean'
#             }).reset_index()

#             yearly_trends['revenue_pct'] = yearly_trends['revenue'].pct_change() * 100
#             yearly_trends['cost_pct'] = yearly_trends['cost'].pct_change() * 100
#             yearly_trends['active_user_pct'] = yearly_trends['active_user'].pct_change() * 100

#             trend_table = pd.DataFrame({
#                 '年份': yearly_trends['year'],
#                 '變現收益變化率(%)': yearly_trends['revenue_pct'].round(2),
#                 '投遞金額變化率(%)': yearly_trends['cost_pct'].round(2),
#                 '活躍用戶變化率(%)': yearly_trends['active_user_pct'].round(2)
#             })

#             st.dataframe(trend_table, use_container_width=True)

#     else:
#         dataset_choice = st.radio("選擇預設數據集", ("kdan_android", "cs_android"))
#         df = load_initial_data(dataset_name=dataset_choice)
#         df['date'] = pd.to_datetime(df['date'])
        
#         df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
#         df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
#         df['active_user'] = pd.to_numeric(df['active_user'], errors='coerce')

#         df_display = df.rename(columns={
#             'date': '預測日期',
#             'cost': '歷史投遞金額',
#             'active_user': '活躍用戶數',
#             'revenue': '歷史變現收益'
#         })
#         df_display['預測日期'] = pd.to_datetime(df_display['預測日期']).dt.strftime('%Y-%m')  
#         st.subheader("預設數據")
#         st.dataframe(df_display, use_container_width=True)

#         # 按年份統計資料
#         # 按年份統計資料
#         st.subheader("年度統計資料")
#         df['year'] = df['date'].dt.strftime('%Y')

#         # 計算統計量並轉置
#         yearly_stats = pd.DataFrame({
#         '指標': ['總額', '平均', '標準差', '最小值', '最大值'],
#         '2022年變現收益': [
#             df[df['year']=='2022']['revenue'].sum().round(2),
#             df[df['year']=='2022']['revenue'].mean().round(2),
#             df[df['year']=='2022']['revenue'].std().round(2),
#             df[df['year']=='2022']['revenue'].min().round(2),
#             df[df['year']=='2022']['revenue'].max().round(2)
#         ],
#         '2023年變現收益': [
#             df[df['year']=='2023']['revenue'].sum().round(2),
#             df[df['year']=='2023']['revenue'].mean().round(2),
#             df[df['year']=='2023']['revenue'].std().round(2),
#             df[df['year']=='2023']['revenue'].min().round(2),
#             df[df['year']=='2023']['revenue'].max().round(2)
#         ],
#         '2022年投遞金額': [
#             df[df['year']=='2022']['cost'].sum().round(2),
#             df[df['year']=='2022']['cost'].mean().round(2),
#             df[df['year']=='2022']['cost'].std().round(2),
#             df[df['year']=='2022']['cost'].min().round(2),
#             df[df['year']=='2022']['cost'].max().round(2)
#         ],
#         '2023年投遞金額': [
#             df[df['year']=='2023']['cost'].sum().round(2),
#             df[df['year']=='2023']['cost'].mean().round(2),
#             df[df['year']=='2023']['cost'].std().round(2),
#             df[df['year']=='2023']['cost'].min().round(2),
#             df[df['year']=='2023']['cost'].max().round(2)
#         ],
#         '2022年活躍用戶': [
#             df[df['year']=='2022']['active_user'].mean().round(2),  # 活躍用戶用平均值替代總和
#             df[df['year']=='2022']['active_user'].mean().round(2),
#             df[df['year']=='2022']['active_user'].std().round(2),
#             df[df['year']=='2022']['active_user'].min().round(2),
#             df[df['year']=='2022']['active_user'].max().round(2)
#         ],
#         '2023年活躍用戶': [
#             df[df['year']=='2023']['active_user'].mean().round(2),  # 活躍用戶用平均值替代總和
#             df[df['year']=='2023']['active_user'].mean().round(2),
#             df[df['year']=='2023']['active_user'].std().round(2),
#             df[df['year']=='2023']['active_user'].min().round(2),
#             df[df['year']=='2023']['active_user'].max().round(2)
#         ]
#         }).set_index('指標')

#         st.dataframe(yearly_stats, use_container_width=True)
        
#         # 計算年度變化趨勢
#         st.subheader("年度變化趨勢")
#         yearly_trends = df.groupby('year').agg({
#             'revenue': 'sum',
#             'cost': 'sum',
#             'active_user': 'mean'
#         }).reset_index()
        
#         # 計算變化率
#         yearly_trends['revenue_pct'] = yearly_trends['revenue'].pct_change() * 100
#         yearly_trends['cost_pct'] = yearly_trends['cost'].pct_change() * 100
#         yearly_trends['active_user_pct'] = yearly_trends['active_user'].pct_change() * 100
        
#         # 創建趨勢表格
#         trend_table = pd.DataFrame({
#             '年份': yearly_trends['year'],
#             '變現收益變化率(%)': yearly_trends['revenue_pct'].round(2),
#             '投遞金額變化率(%)': yearly_trends['cost_pct'].round(2),
#             '活躍用戶變化率(%)': yearly_trends['active_user_pct'].round(2)
#         })
        
#         st.dataframe(trend_table, use_container_width=True)
    
#     st.sidebar.subheader("模型設置")
#     model_choice = st.sidebar.radio(
#         "選擇模型參數",
#         ["model_kdan_android", "model_cs_android"]
#     )
    
#     if model_choice == "model_kdan_android":
#         selected_params = get_model_parameters("model_kdan_android")
#     else:
#         selected_params = get_model_parameters("model_cs_android")
    
#     # 準備數據並訓練模型
#     df_prepared = prepare_data(df)
#     model = train_model(
#         df_prepared, 
#         selected_params["model_params"],
#         selected_params["seasonality_params"],
#         selected_params["regressor_params"]
#     )
    
#     # 顯示選擇的模型參數
#     st.subheader("使用的模型參數")
#     params_df = pd.DataFrame([
#         {'參數類型': param_type, '參數名稱': param_name, '參數值': str(param_value)}
#         for param_type, params in [
#             ('基本參數', selected_params["model_params"].items()),
#             ('季節性參數', selected_params["seasonality_params"].items()),
#             ('Cost參數', selected_params["regressor_params"]['cost'].items()),
#             ('Active User參數', selected_params["regressor_params"]['active_user'].items())
#         ]
#         for param_name, param_value in params
#     ])
#     st.dataframe(params_df, use_container_width=True)
    

#     # 預測結果
#     st.subheader("預測結果")
    
#     results, forecast = predict_revenue(model, monthly_budget, 12, df)
#     results_display = results[['date', 'cost', 'predict_revenue', 'upper_bound', 'roi_lower', 'roi_upper']].rename(columns={
#         'date': '預測日期',
#         'cost': '預期預算',
#         'predict_revenue':'預測下限',
#         'upper_bound':'預測上限',
#         'roi_lower': 'ROAS下限',
#         'roi_upper': "ROAS上限"})   
#     metrics = [
#         ("未來12個月預測總預算", results['cost'].sum().round(0)),
#         ("未來12個月預測總收益（下限）", results['predict_revenue'].sum().round(0)),
#         ("未來12個月預測總收益（上限）", results['upper_bound'].sum().round(0)),
#         ("未來12個月預測總ROAS（下限）", results['roi_lower'].mean().round(2)),
#         ("未來12個月預測總ROAS（上限）", results['roi_upper'].mean().round(2))
#     ]
#     col1, col2, col3, col4, col5 = st.columns(5)
#     col1.metric("未來12個月預測總預算", int(metrics[0][1]))
#     col2.metric("未來12個月預測總收益（下限）", int(metrics[1][1]))
#     col3.metric("未來12個月預測總收益（上限）", int(metrics[2][1]))
#     col4.metric("未來12個月預測總ROAS（下限）", float(metrics[3][1]))
#     col5.metric("未來12個月預測總ROAS（上限）", float(metrics[4][1]))

#     st.dataframe(results_display, use_container_width=True) 
#     # # 顯示診斷信息
#     # diagnostics = calculate_diagnostics(forecast, df)
    
#     # # 使用列來顯示診斷指標
#     # col1, col2, col3, col4 = st.columns(4)
#     # col1.metric("MAPE (%)", diagnostics['MAPE'])
#     # col2.metric("RMSE", int(diagnostics['RMSE']))
#     # col3.metric("MAE", int(diagnostics['MAE']))
#     # col4.metric("預測區間覆蓋率 (%)", diagnostics['Coverage'])
#     future_active_user = prepare_future_regressor(df, 'active_user', 12)
#     st.metric("預測活躍用戶數",int(future_active_user[0].round(0)))
#     st.write("使用最近6個月平均活躍用戶為估值")
    
#     components = ['trend', 'seasonal', 'cost', 'active_user']
#     component_importance = {}
#     for comp in components:
#         if comp in forecast.columns:
#             component_importance[comp] = abs(forecast[comp]).mean()
#     st.write("各變數平均影響力：")
#     st.bar_chart(component_importance)
#         # 1. 收入預測圖
#     # 1. 收入預測圖
#     st.markdown("### 收入預測圖")
#     st.markdown("此圖顯示了歷史收入數據以及未來預測的收入範圍。紅色虛線表示預測的下限，橙色虛線表示預測的上限，紅色區域顯示了預測的信賴區間(0.95)。")
    
#     fig1 = make_subplots(rows=1, cols=1, subplot_titles=("收入預測圖",))
#     fig1.add_trace(
#         go.Scatter(x=df['date'], y=df['revenue'], name='歷史收入', line=dict(color='blue', width=2)),
#     )
#     fig1.add_trace(
#           go.Scatter(
#             x=df['date'], 
#             y=df['active_user'], 
#             name='活躍使用者', 
#             line=dict(color='green', width=2),
#             hovertemplate='活躍使用者: %{y:.2f}<br>日期: %{x}<extra></extra>'
#         ),
#     )
#     fig1.add_trace(
#         go.Scatter(
#             x=results['date'], 
#             y=results['predict_revenue'], 
#             name='預測下限', 
#             line=dict(color='red', dash='dash', width=2),
#             hovertemplate='預測下限: %{y:.2f}<br>日期: %{x}<extra></extra>'
#         ),
#     )
#     fig1.add_trace(
#         go.Scatter(
#             x=results['date'], 
#             y=results['upper_bound'], 
#             name='預測上限', 
#             line=dict(color='orange', dash='dash', width=2),
#             hovertemplate='預測上限: %{y:.2f}<br>日期: %{x}<extra></extra>'
#         ),
#     )
#     fig1.add_trace(
#         go.Scatter(
#             x=pd.concat([results['date'], results['date'][::-1]]), 
#             y=pd.concat([results['upper_bound'], results['predict_revenue'][::-1]]), 
#             fill='toself', 
#             fillcolor='rgba(255,0,0,0.2)', 
#             name='預測區間', 
#             line=dict(color='rgba(255,0,0,0)'),
#             hovertemplate='預測區間: %{y:.2f}<br>日期: %{x}<extra></extra>'
#         )
#     )
#     fig1.update_layout(
#         title='收入預測與歷史數據比較',
#         xaxis_title='日期',
#         yaxis_title='收入',
#         legend_title='圖例',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig1)

#     # 2. ROAS 分析
#     st.markdown("### ROAS 分析")
#     st.markdown("此圖顯示了預測的投資回報率（ROAS）。ROAS 是預測收入與預算的比率，幫助評估投資的效益。")
    
#     fig2 = make_subplots(rows=1, cols=1, subplot_titles=("ROAS 分析",))
#     fig2.add_trace(
#         go.Scatter(x=results['date'], y=results['roi_lower'], name='預測 ROAS', line=dict(color='purple', width=2),
#                    hovertemplate='日期: %{x}<br>預測 ROAS: %{y:.2f}<extra></extra>'),
#     )
#     fig2.update_layout(
#         title='預測 ROAS 分析',
#         xaxis_title='日期',
#         yaxis_title='ROAS',
#         legend_title='圖例',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig2)

#     # 3. 趨勢分解
#     st.markdown("### 趨勢分解")
#     st.markdown("此圖顯示了預測模型中的趨勢成分，幫助理解收入隨時間的變化趨勢。")
    
#     fig3 = make_subplots(rows=1, cols=1, subplot_titles=("趨勢分解",))
#     fig3.add_trace(
#         go.Scatter(x=forecast['ds'], y=forecast['trend'], name='趨勢', line=dict(color='blue', width=2)),
#     )
#     fig3.update_layout(
#         title='趨勢分解',
#         xaxis_title='日期',
#         yaxis_title='趨勢',
#         legend_title='圖例',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig3)

# if __name__ == "__main__":
#     main()

def main():
    st.title('廣告變現收益預測')
    
    # 設置側邊欄的預算和數據集選擇
    st.sidebar.subheader("選擇數據集和預算")
    dataset_choice = st.sidebar.radio(
        "選擇數據集",
        ["kdan_android", "cs_android"]
    )

    # Define default budgets for each type
    default_budgets_kdan = {
        1: 496675.0, 
        2: 646544.0, 
        3: 631547.0, 
        4: 730672.0, 
        5: 1192148.0, 
        6: 813243.0, 
        7: 782203.0, 
        8: 780915.0, 
        9: 780966.0, 
        10: 794793.0, 
        11: 794793.0, 
        12: 794793.0
    }
    default_budgets_cs = {
        1: 26584.0, 
        2: 27790.0, 
        3: 33841.0, 
        4: 43701.0, 
        5: 38353.0, 
        6: 33748.0, 
        7: 46247.0, 
        8: 47568.0, 
        9: 49241.0, 
        10: 60630.0,
        11: 60630.0,
        12: 60630.0
    }

    # Set default budgets based on dataset choice
    if dataset_choice == "kdan_android":
        default_budgets = default_budgets_kdan
    else:
        default_budgets = default_budgets_cs

    # Set monthly budgets based on the selected default
    monthly_budget = [
        float(st.sidebar.text_input(
            f'第 {month} 月預算',
            value=str(default_budgets[month])
        ))
        for month in range(1, 13)
    ]
    
    # Load initial data based on dataset choice
    df = load_initial_data(dataset_choice)
    
    # 選擇數據來源
    data_source = st.radio("選擇數據來源", ("使用預設資料", "上傳 CSV 文件"))
    if data_source == "上傳 CSV 文件":
        uploaded_file = st.file_uploader("上傳 CSV 文件", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            expected_columns = ['date', 'cost', 'revenue', 'active_user']
            if not all(col in df.columns for col in expected_columns):
                st.error("上傳的文件格式不正確，請確保包含以下列: " + ", ".join(expected_columns))
                return
            df['date'] = pd.to_datetime(df['date'])
            st.subheader("上傳的數據")
            df_display = df.rename(columns={
                'date': '日期',
                'cost': '歷史投遞金額',
                'active_user': '歷史活躍用戶數',
                'revenue': '歷史變現收益'
            })
            df_display['預測日期'] = pd.to_datetime(df_display['預測日期']).dt.strftime('%Y-%m')
            st.dataframe(df_display, use_container_width=True)

            st.subheader("年度統計資料")
            df['year'] = df['date'].dt.year

            yearly_stats = pd.DataFrame({
                '指標': ['總額', '平均', '標準差', '最小值', '最大值'],
                '變現收益': [
                    df.groupby('year')['revenue'].sum().round(2),
                    df.groupby('year')['revenue'].mean().round(2),
                    df.groupby('year')['revenue'].std().round(2),
                    df.groupby('year')['revenue'].min().round(2),
                    df.groupby('year')['revenue'].max().round(2)
                ],
                '投遞金額': [
                    df.groupby('year')['cost'].sum().round(2),
                    df.groupby('year')['cost'].mean().round(2),
                    df.groupby('year')['cost'].std().round(2),
                    df.groupby('year')['cost'].min().round(2),
                    df.groupby('year')['cost'].max().round(2)
                ],
                '活躍用戶': [
                    df.groupby('year')['active_user'].mean().round(2),
                    df.groupby('year')['active_user'].mean().round(2),
                    df.groupby('year')['active_user'].std().round(2),
                    df.groupby('year')['active_user'].min().round(2),
                    df.groupby('year')['active_user'].max().round(2)
                ]
            }).set_index('指標')

            st.dataframe(yearly_stats, use_container_width=True)

            st.subheader("年度變化趨勢")
            yearly_trends = df.groupby('year').agg({
                'revenue': 'sum',
                'cost': 'sum',
                'active_user': 'mean'
            }).reset_index()

            yearly_trends['revenue_pct'] = yearly_trends['revenue'].pct_change() * 100
            yearly_trends['cost_pct'] = yearly_trends['cost'].pct_change() * 100
            yearly_trends['active_user_pct'] = yearly_trends['active_user'].pct_change() * 100

            trend_table = pd.DataFrame({
                '年份': yearly_trends['year'],
                '變現收益變化率(%)': yearly_trends['revenue_pct'].round(2),
                '投遞金額變化率(%)': yearly_trends['cost_pct'].round(2),
                '活躍用戶變化率(%)': yearly_trends['active_user_pct'].round(2)
            })

            st.dataframe(trend_table, use_container_width=True)

    else:
        df['date'] = pd.to_datetime(df['date'])
        
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df['active_user'] = pd.to_numeric(df['active_user'], errors='coerce')

        df_display = df.rename(columns={
            'date': '日期',
            'cost': '歷史投遞金額',
            'active_user': '歷史活躍用戶數',
            'revenue': '歷史變現收益'
        })
        st.dataframe(df_display, use_container_width=True)

    # 選擇模型參數
    st.sidebar.subheader("選擇模型參數")
    model_param_choice = st.sidebar.radio(
        "選擇模型參數",
        ["kdan_android", "cs_android"]
    )

    # Prepare data and train model
    df_prepared = prepare_data(df)
    selected_params = get_model_parameters(f"model_{model_param_choice}")
    model = train_model(
        df_prepared, 
        selected_params["model_params"],
        selected_params["seasonality_params"],
        selected_params["regressor_params"]
    )
    
    # Display selected model parameters
    st.subheader("使用的模型參數")
    params_df = pd.DataFrame([
        {'參數類型': param_type, '參數名稱': param_name, '參數值': str(param_value)}
        for param_type, params in [
            ('基本參數', selected_params["model_params"].items()),
            ('季節性參數', selected_params["seasonality_params"].items()),
            ('Cost參數', selected_params["regressor_params"]['cost'].items()),
            ('Active User參數', selected_params["regressor_params"]['active_user'].items())
        ]
        for param_name, param_value in params
    ])
    st.dataframe(params_df, use_container_width=True)
    
    # 預測結果
    st.subheader("預測結果")
    
    results, forecast = predict_revenue(model, monthly_budget, 12, df)
    results_display = results[['date', 'cost', 'predict_revenue', 'upper_bound', 'roi_lower', 'roi_upper']].rename(columns={
        'date': '預測日期',
        'cost': '預期預算',
        'predict_revenue':'預測下限',
        'upper_bound':'預測上限',
        'roi_lower': 'ROAS下限',
        'roi_upper': "ROAS上限"})   
    metrics = [
        ("未來12個月預測總預算", results['cost'].sum().round(0)),
        ("未來12個月預測總收益（下限）", results['predict_revenue'].sum().round(0)),
        ("未來12個月預測總收益（上限）", results['upper_bound'].sum().round(0)),
        ("未來12個月預測總ROAS（下限）", results['roi_lower'].mean().round(2)),
        ("未來12個月預測總ROAS（上限）", results['roi_upper'].mean().round(2))
    ]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("未來12個月預測總預算", int(metrics[0][1]))
    col2.metric("未來12個月預測總收益（下限）", int(metrics[1][1]))
    col3.metric("未來12個月預測總收益（上限）", int(metrics[2][1]))
    col4.metric("未來12個月預測總ROAS（下限）", float(metrics[3][1]))
    col5.metric("未來12個月預測總ROAS（上限）", float(metrics[4][1]))

    st.dataframe(results_display, use_container_width=True) 
    # # 顯示診斷信息
    # diagnostics = calculate_diagnostics(forecast, df)
    
    # # 使用列來顯示診斷指標
    # col1, col2, col3, col4 = st.columns(4)
    # col1.metric("MAPE (%)", diagnostics['MAPE'])
    # col2.metric("RMSE", int(diagnostics['RMSE']))
    # col3.metric("MAE", int(diagnostics['MAE']))
    # col4.metric("預測區間覆蓋率 (%)", diagnostics['Coverage'])
    future_active_user = prepare_future_regressor(df, 'active_user', 12)
    st.metric("預測活躍用戶數",int(future_active_user[0].round(0)))
    st.write("使用最近6個月平均活躍用戶為估值")
    
    components = ['trend', 'seasonal', 'cost', 'active_user']
    component_importance = {}
    for comp in components:
        if comp in forecast.columns:
            component_importance[comp] = abs(forecast[comp]).mean()
    st.write("各變數平均影響力：")
    st.bar_chart(component_importance)
    
    # 1. 收入預測圖
    st.markdown("### 收入預測圖")
    st.markdown("此圖顯示了歷史收入數據以及未來預測的收入範圍。紅色虛線表示預測的下限，橙色虛線表示預測的上限，紅色區域顯示了預測的信賴區間(0.95)。")
    
    fig1 = make_subplots(rows=1, cols=1, subplot_titles=("收入預測圖",))
    fig1.add_trace(
        go.Scatter(x=df['date'], y=df['revenue'], name='歷史收入', line=dict(color='blue', width=2)),
    )
    fig1.add_trace(
          go.Scatter(
            x=df['date'], 
            y=df['active_user'], 
            name='活躍使用者', 
            line=dict(color='green', width=2),
            hovertemplate='活躍使用者: %{y:.2f}<br>日期: %{x}<extra></extra>'
        ),
    )
    fig1.add_trace(
        go.Scatter(
            x=results['date'], 
            y=results['predict_revenue'], 
            name='預測下限', 
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='預測下限: %{y:.2f}<br>日期: %{x}<extra></extra>'
        ),
    )
    fig1.add_trace(
        go.Scatter(
            x=results['date'], 
            y=results['upper_bound'], 
            name='預測上限', 
            line=dict(color='orange', dash='dash', width=2),
            hovertemplate='預測上限: %{y:.2f}<br>日期: %{x}<extra></extra>'
        ),
    )
    fig1.add_trace(
        go.Scatter(
            x=pd.concat([results['date'], results['date'][::-1]]), 
            y=pd.concat([results['upper_bound'], results['predict_revenue'][::-1]]), 
            fill='toself', 
            fillcolor='rgba(255,0,0,0.2)', 
            name='預測區間', 
            line=dict(color='rgba(255,0,0,0)'),
            hovertemplate='預測區間: %{y:.2f}<br>日期: %{x}<extra></extra>'
        )
    )
    fig1.update_layout(
        title='收入預測與歷史數據比較',
        xaxis_title='日期',
        yaxis_title='收入',
        legend_title='圖例',
        template='plotly_white'
    )
    st.plotly_chart(fig1)

    # 2. ROAS 分析
    st.markdown("### ROAS 分析")
    st.markdown("此圖顯示了預測的投資回報率（ROAS）。ROAS 是預測收入與預算的比率，幫助評估投資的效益。")
    
    fig2 = make_subplots(rows=1, cols=1, subplot_titles=("ROAS 分析",))
    fig2.add_trace(
        go.Scatter(x=results['date'], y=results['roi_lower'], name='預測 ROAS', line=dict(color='purple', width=2),
                   hovertemplate='日期: %{x}<br>預測 ROAS: %{y:.2f}<extra></extra>'),
    )
    fig2.update_layout(
        title='預測 ROAS 分析',
        xaxis_title='日期',
        yaxis_title='ROAS',
        legend_title='圖例',
        template='plotly_white'
    )
    st.plotly_chart(fig2)

    # 3. 趨勢分解
    st.markdown("### 趨勢分解")
    st.markdown("此圖顯示了預測模型中的趨勢成分，幫助理解收入隨時間的變化趨勢。")
    
    fig3 = make_subplots(rows=1, cols=1, subplot_titles=("趨勢分解",))
    fig3.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['trend'], name='趨勢', line=dict(color='blue', width=2)),
    )
    fig3.update_layout(
        title='趨勢分解',
        xaxis_title='日期',
        yaxis_title='趨勢',
        legend_title='圖例',
        template='plotly_white'
    )
    st.plotly_chart(fig3)

if __name__ == "__main__":
    main()
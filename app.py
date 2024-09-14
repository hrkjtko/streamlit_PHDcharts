import time
import streamlit as st

import pandas as pd
from pandas import json_normalize
import requests


from io import BytesIO
import numpy as np

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import datetime


url = st.secrets["API_URL"]

response = requests.get(url)
data = response.json()

df = pd.DataFrame(data['経過'])

parameters = ['月齢', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', 'CA', '後頭部対称率', 'CVAI', 'CI']
df[parameters] = df[parameters].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

df_h = pd.DataFrame(data['ヘルメット'])
df_h = df_h[(df_h['ダミーID'] != '') & (df_h['ヘルメット'] != '')]

treated_patients = df_h['ダミーID'].unique()
df_first = df[df['治療ステータス'] == '治療前'].drop_duplicates('ダミーID')

df_tx = df[df['ダミーID'].isin(treated_patients)]
df_tx_pre_last = df_tx[df_tx['治療ステータス'] == '治療前'].drop_duplicates('ダミーID', keep='last')

df_tx_pre_last['治療前月齢'] = df_tx_pre_last['月齢']

df_tx_pre_last['治療前PSRレベル'] = ''
df_tx_pre_last['治療前PSRレベル'] = df_tx_pre_last['治療前PSRレベル'].mask(df_tx_pre_last['後頭部対称率']>=90, 'レベル1')
df_tx_pre_last['治療前PSRレベル'] = df_tx_pre_last['治療前PSRレベル'].mask(df_tx_pre_last['後頭部対称率']<90, 'レベル2')
df_tx_pre_last['治療前PSRレベル'] = df_tx_pre_last['治療前PSRレベル'].mask(df_tx_pre_last['後頭部対称率']<85, 'レベル3')
df_tx_pre_last['治療前PSRレベル'] = df_tx_pre_last['治療前PSRレベル'].mask(df_tx_pre_last['後頭部対称率']<80, 'レベル4')

df_tx_pre_last['治療前短頭症'] = ''
df_tx_pre_last['治療前短頭症'] = df_tx_pre_last['治療前短頭症'].mask(df_tx_pre_last['短頭率']>126, '長頭')
df_tx_pre_last['治療前短頭症'] = df_tx_pre_last['治療前短頭症'].mask(df_tx_pre_last['短頭率']<=126, '正常')
df_tx_pre_last['治療前短頭症'] = df_tx_pre_last['治療前短頭症'].mask(df_tx_pre_last['短頭率']<106, '軽症')
df_tx_pre_last['治療前短頭症'] = df_tx_pre_last['治療前短頭症'].mask(df_tx_pre_last['短頭率']<100, '重症')

#経過も利用する場合
df_tx_post =  df_tx[df_tx['治療ステータス'] == '治療後']

df_tx_pre_age = df_tx_pre_last[['ダミーID', '月齢']]
df_tx_pre_age = df_tx_pre_age.rename(columns = {'月齢':'治療前月齢'})

df_tx_post = pd.merge(df_tx_post, df_tx_pre_age, on='ダミーID', how='left')

df_tx_post['治療期間'] = df_tx_post['月齢'] - df_tx_post['治療前月齢']
df_period = df_tx_post[['ダミーID', '治療期間']]

df_tx_pre_last['治療期間'] = 0

df_tx_post = pd.merge(df_tx_post, df_tx_pre_last[['ダミーID', '治療前PSRレベル', '治療前短頭症']], on='ダミーID', how='left')

df_tx_pre_post = pd.concat([df_tx_pre_last, df_tx_post])

df_tx_pre_post = pd.merge(df_tx_pre_post, df_h, on='ダミーID', how='left')

# Streamlitアプリのページ設定
st.set_page_config(page_title='重症度の分布とヘルメットの種類', layout='wide')

#治療率ありでパラメータごとにヒストグラムを作成（go.Barを利用）
def hist(parameter='短頭率', df_first=df_first):
  import plotly.graph_objects as go

  all_number = len(df_first['ダミーID'].unique())

  df_first[parameter] = pd.to_numeric(df_first[parameter], errors='coerce')
  df_first[parameter] = df_first[parameter].round()

  df_first_tx = df_first[df_first['ダミーID'].isin(treated_patients)]
  tx_number = len(df_first_tx['ダミーID'].unique())
  tx_rate = round((tx_number/all_number)*100, 1)

  treated = []
  untreated = []
  all = []
  tx_rates=[]

  min = int(df_first[parameter].min())
  max_para = int(df_first[parameter].max())

  for i in list(range(min, max_para)):
    tx_n = df_first_tx[df_first_tx[parameter] == i][parameter].count()
    all_n = df_first[df_first[parameter] == i][parameter].count()
    untx_n = all_n-tx_n
    if all_n > 0:
      rate = (tx_n/all_n)*100
      rate = round(rate, 1)
    else:
      rate = ''

    treated.append(round(tx_n, 1))
    untreated.append(round(untx_n, 1))
    all.append(round(all_n, 1)) #不要？
    tx_rates.append(rate)

  x=list(range(min, max_para))

  y=[0, max(all)]

  fig = go.Figure(go.Bar(x=x, y=treated, name='治療あり', marker_color='blue')) #opacity=0.8
  fig.add_trace(go.Bar(x=x, y=untreated, name='治療なし',  marker_color='cyan', text=tx_rates)) #opacity=0.4
  fig.update_traces(textfont_size=12, textfont_color='black',
                    #textangle=0,
                    textposition="outside", cliponaxis=False)

  if parameter == '短頭率':
    limits=list({106, 126} & set(x))
  elif parameter in ['前頭部対称率', '後頭部対称率']:
    limits=list({80, 85, 90} & set(x))
  elif parameter == 'CA':
    limits=list({6, 9, 13, 17} & set(x))
  elif parameter == 'CVAI':
    limits=list({5, 7, 10, 14} & set(x))
  elif parameter == 'CI':
    limits=list({80, 94, 101} & set(x))

  for i in range(len(limits)):
    fig.add_trace(go.Line(x=[limits[i],limits[i]], y=y, mode='lines', marker_color='pink', line=dict(dash='dot'), name=str(limits[i])))

  if all_number >= 1000:
    all_number = str(all_number)
    digits = len(all_number)
    all_number = all_number[:digits-3] + ',' + all_number[digits-3:]
  else:
    all_number = str(all_number)

  fig.update_layout(width=1600, height=900,
      plot_bgcolor='white',
      title_text=parameter+'の分布（全'+all_number+'人で'+str(tx_rate)+'％が治療）',
      xaxis_title_text=parameter,
      yaxis_title_text='人数',
      barmode='stack'
      )

  st.plotly_chart(fig)

def show_helmet_proportion():
  # 色をカスタマイズ
  colors = ['red', 'green', 'blue']

  # ヘルメットの種類ごとに行の数を集計
  counts = df_h['ヘルメット'].value_counts().reset_index()
  counts.columns = ['ヘルメット', '数']

  # 円グラフ作成
  fig = px.pie(counts, names='ヘルメット', values='数', color_discrete_sequence=colors)
  fig.update_layout(width=900, title='ヘルメットの種類の内訳')

  # Streamlitアプリにグラフを表示
  st.plotly_chart(fig)

def animate_BI_PSR(df0, df):
  category_orders={'治療前PSRレベル':['レベル1', 'レベル2', 'レベル3', 'レベル4'], '治療前短頭症':['軽症', '重症', '正常', '長頭']}
  colors = [
    '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8C33', '#33FFF1', '#8C33FF', '#FF5733', '#57FF33', '#5733FF',
    '#FF3357', '#33FFA1', '#FFA133', '#33FF8C', '#FF338C', '#8CFF33', '#A1FF33', '#338CFF', '#A133FF', '#33A1FF'
  ]

  #df0 = df.drop_duplicates('ダミーID', keep='first')
  
  df1 = df.drop_duplicates('ダミーID', keep='last')

  common_patients = set(df1['ダミーID'].unique()).intersection(set(df0['ダミーID'].unique()))

  df = pd.concat([df0, df1])
  df = df[df['ダミーID'].isin(common_patients)]

  #複数のヘルメットを使用している患者を除外
  helmet_counts = df.groupby('ダミーID')['ヘルメット'].nunique()
  common_patients = helmet_counts[helmet_counts > 1].index.tolist()

  df = df[~df['ダミーID'].isin(common_patients)]

  fig = px.scatter(df, x='短頭率', y='後頭部対称率', color='治療前PSRレベル', symbol='治療前短頭症', facet_col = 'ヘルメット',
                   hover_data=['ダミーID', '治療期間', '治療前月齢', 'ヘルメット'] + parameters, category_orders=category_orders, animation_frame='治療ステータス', animation_group='ダミーID', color_discrete_sequence=colors)
  i=0
  for i in range(len(df['ヘルメット'].unique())):
    #短頭率の正常範囲
    fig.add_trace(go.Scatter(x=[106, 106], y=[50, 100], mode='lines', line=dict(color='gray', dash = 'dot'), name='短頭率正常下限'), row=1, col=i+1)
    fig.add_trace(go.Scatter(x=[126, 126], y=[50, 100], mode='lines', line=dict(color='gray', dash = 'dot'), name='短頭率正常上限'), row=1, col=i+1)

    #対称率の正常範囲
    fig.add_trace(go.Scatter(x=[90, 150], y=[90, 90], mode='lines', line=dict(color='gray', dash = 'dot'), name='後頭部対称率正常下限'), row=1, col=i+1)

  fig.update_xaxes(range = [df['短頭率'].min()-2,df['短頭率'].max()+2])
  fig.update_yaxes(range = [df['後頭部対称率'].min()-2,100])

  #width = 800*(i+1)
  width = 800*len(df['ヘルメット'].unique())

  fig.update_layout(height=800, width=width, title='短頭率と後頭部対称率の治療前後の変化')

  for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split('=')[-1]

  st.plotly_chart(fig)

parameters = ['短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'CI']

for parameter in parameters:
  hist(parameter)

show_helmet_proportion()

with st.form(key='filter_form'):
  st.write('患者を絞ってグラフを作成します')

  # スライダーで範囲を指定
  min_age, max_age = st.slider(
      '月齢の範囲を選択してください',
      min_value=int(df_tx_pre_post['治療前月齢'].min()),
      max_value=int(df_tx_pre_post['治療前月齢'].max()),
      value=(int(df_tx_pre_post['治療前月齢'].min()), int(df_tx_pre_post['治療前月齢'].max()))
  )

  min_value, max_value = st.slider(
      '治療期間の範囲を選択してください',
      min_value=int(df_tx_pre_post['治療期間'].min()),
      max_value=int(df_tx_pre_post['治療期間'].max()),
      value=(int(df_tx_pre_post['治療期間'].min()), int(df_tx_pre_post['治療期間'].max()))
  )

  st.write('ヘルメットを選択してください')

  # チェックボックスを作成
  filter_pass0 = st.checkbox('アイメット')
  filter_pass1 = st.checkbox('クルム')
  filter_pass2 = st.checkbox('クルムフィット')

  submit_button = st.form_submit_button(label='実行')

# 「実行」ボタンを作成
#if st.button('実行'):
if submit_button:
  filtered_df0 = df_tx_pre_post[df_tx_pre_post['治療ステータス'] == '治療後']
  # スライダーで選択された範囲でデータをフィルタリング
  filtered_df = df_tx_pre_post[(df_tx_pre_post['治療前月齢'] >= min_age) & (df_tx_pre_post['治療前月齢'] <= max_age)]
  filtered_df = filtered_df[(filtered_df['治療期間'] >= min_value) & (df_tx_pre_post['治療期間'] <= max_value)]

  filtered_df0 = df_tx_pre_post[df_tx_pre_post['治療ステータス'] == '治療前']

  # チェックボックスの状態に応じてデータをフィルタリング
  if not filter_pass0:
      filtered_df = filtered_df[filtered_df['ヘルメット'] != 'アイメット']
      filtered_df0 = filtered_df0[filtered_df0['ヘルメット'] != 'アイメット']
  if not filter_pass1:
      filtered_df = filtered_df[filtered_df['ヘルメット'] != 'クルム']
      filtered_df0 = filtered_df0[filtered_df0['ヘルメット'] != 'クルム']
  if not filter_pass2:
      filtered_df = filtered_df[filtered_df['ヘルメット'] != 'クルムフィット']
      filtered_df0 = filtered_df0[filtered_df0['ヘルメット'] != 'クルムフィット']

  treated_patients = filtered_df['ダミーID'].unique()
  filtered_df = filtered_df[filtered_df['ダミーID'].isin(treated_patients)]
  filtered_df0 = filtered_df0[filtered_df0['ダミーID'].isin(treated_patients)]

  animate_BI_PSR(filtered_df0, filtered_df)
else:
    st.write('実行ボタンを押すとグラフが作成されます')

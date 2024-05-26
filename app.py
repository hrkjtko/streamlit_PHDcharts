import time

import streamlit as st

from io import BytesIO
import numpy as np
import pandas as pd
from pandas import json_normalize
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import datetime
import requests

#あらかじめ作成しておいたCSVファイルをデータフレームに
# ファイルのURL
url = st.secrets["URL"]

# URLからファイルをダウンロード
response = requests.get(url)
data = BytesIO(response.content)

# ダウンロードしたデータをPandasのDataFrameに読み込む
df = pd.read_csv(data)
df = df.drop('Unnamed: 0', axis=1) #「Unnamed: 0」はどこで発生した？
id = '診察券ID'
df = df.drop_duplicates([id, '診察日'])

#日付をdatetime型に
for i in ['診察日', '初診時診察日', '治療前診察日', '最終診察日']:
  df[i] = pd.to_datetime(df[i])

#最終診察日が2024年3月までの症例（卒業済みと見込まれる症例）に限定する場合、以下を追加する
df = df[df['最終診察日'] < '2024/4/1']

#治療例の抽出（経過観察期間なし）
tx_list = list(df[df['診療ステータス'] == '治療後'][id].unique())

dftx = df[(df['診療ステータス'] == '治療前') + (df['診療ステータス'] == '治療後') + (df['診療ステータス'] == '最終')]

#調整回数のデータが無いものを治療前として扱う処理
dftx['調整回数'].mask(dftx['調整回数'].isnull(), '治療前', inplace=True)
dftx['調整回数'] = dftx['調整回数'].astype(str)

dftx['治療後月齢'] = dftx['月齢（概算）']+dftx['全治療期間（月数）']

parameters = ['頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI']
for parameter in parameters:
  dftx[parameter+'変化量'] = dftx['最終'+parameter] - dftx[parameter]

#経過観察症例の抽出
co_list = list(df[df['診療ステータス'] == '再診前'][id].unique())

dfco = df[df[id].isin(co_list)]
dfco = dfco[(dfco['診療ステータス']  != '治療後') & (dfco['診療ステータス']  != '最終')] #治療後の計測データを除去

dfco = dfco.sort_values([id, '診察日']).reset_index() #indexカラムが発生する

#経過観察期間の追記
for index in range(len(dfco)):
  pre_date = list(dfco[dfco[id] == dfco.loc[index, id]]['診察日'])[0]
  dfco.loc[index, '経過観察期間'] = dfco.loc[index, '診察日'] - pre_date

dfco['経過観察期間'] = dfco['経過観察期間']/pd.Timedelta(1,"D") #経過観察期間（日数）をfloat型に
dfco['経過観察期間'] = dfco['経過観察期間'].astype(int) #経過観察期間（日数）をint型に
dfco['経過観察期間（月数）'] = dfco['経過観察期間']/30.4375 #経過観察期間を月数（概算）float型に

#経過観察期間の追記
for index in range(len(dfco)):
  pre_date = list(dfco[dfco[id] == dfco.loc[index, id]]['診察日'])[0]
  dfco.loc[index, '経過観察期間'] = dfco.loc[index, '診察日'] - pre_date

#データフレームの抽出をシンプルにした 2022/12/11

#全患者の初診時のパラメータのデータフレームを作成
df_first = df.drop_duplicates(subset=id) #重複を削除

#最終診察日が2024年3月までの症例（卒業済みと見込まれる症例）に限定する場合、以下を追加する
df = df[df['最終診察日'] < '2024/4/1']

#治療を受けた患者の治療前のパラメータのデータフレームの作成
dftx_pre = dftx[dftx['診療ステータス'] == '治療前']

#経過観察された患者の経過観察前のパラメータのデータフレームの作成
dfco_pre = dfco[dfco['診療ステータス'] == '再診前']

#Zスコアの計算
parameters = ['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', 'CI', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'APR']

for df in [df_first, dftx_pre, dfco_pre]:
  for i in parameters:
    df['z_'+i] = (df[i] - df[i].mean())/df[i].std()

#df_firstでは10行がNaNとなる

qurum_members = dftx[dftx['ヘルメット種類'] == 'クルム'][id].unique()


#類似症例の経過のプロットと患者を合わせて表示する関数の定義　通院回数が増えるにつれて線を伸ばす？
def tx_plot(dfpt, dftx=dftx, n=10, mo_weight=1):
  id = '診察券ID'
  dfpt['APR'] = dfpt['前頭部対称率']/dfpt['後頭部対称率']
  dftx_pre = dftx[dftx['診療ステータス'] == '治療前']
  #return(dftx_pre)
  dftx_pre['APR'] = dftx_pre['前頭部対称率']/dftx_pre['後頭部対称率']

  #最大人数
  #n = 30

  parameters = ['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'APR']
  dfpt_temp = dfpt[parameters]
  #st.write(dfpt.dtypes)

  #症例のZスコアの計算
  dftx_pre_para = dftx_pre[parameters]
  dfpt_z = (dfpt-dftx_pre_para.mean())/dftx_pre_para.std()
  dfpt_z = dfpt_z.reset_index()

  #重みをデータフレームでまとめて計算
  dfpt_w = 10**abs(dfpt_z)

  #月齢の重みを他の重みの最大値にする
  if dfpt_w['月齢（概算）'][0] < dfpt_w.T.max()[0]:
    dfpt_w['月齢（概算）'] = dfpt_w.T.max()[0]
    dfpt_w['月齢（概算）'] = dfpt_w['月齢（概算）']*mo_weight

  #重み付き二乗和の計算
  dftx_pre['w_delta'] = 0

  for parameter in parameters:
    dftx_pre['w_delta'] += dfpt_w[parameter][0]*abs(df_first['z_'+parameter] - dfpt_z[parameter][0])**2

  #ランキングを作成 11/27追加
  rank = list(dftx_pre.sort_values('w_delta')[id])

  similar_patients=rank[:n] #人数

  #空のデータフレームを作成
  dftxn = pd.DataFrame()

  #ランク上位から追加　うまく機能しなかった？
  for id in similar_patients: #人数
    df_temp = dftx[dftx[id] == id]
    dftxn = pd.concat([dftxn, df_temp])

  #グラフの配置を4×2の表で定義
  para_table=[['頭囲', '短頭率'],
              ['前頭部対称率', '後頭部対称率'],
              ['CA', 'CVAI'],
              ['前後径', '左右径']]

  fig = make_subplots(rows=len(para_table), cols=len(para_table[0]), subplot_titles=(sum(para_table,[]))) #sum(,[])で行列を1行に cf. https://note.nkmk.me/python-list-flatten/

  for i in range(len(para_table)):
    for j in range(len(para_table[i])):
      fig.add_trace(go.Scatter(x=dfpt['月齢（概算）'],y=dfpt[para_table[i][j]], name='お子様'+'_'+str(para_table[i][j]), marker=dict(size=10, color='green')), row=i+1, col=j+1)

  #IDごとの色分け
  list_colors = px.colors.qualitative.Alphabet #Dark24 cf. https://oeconomicus.jp/2021/07/plotly-color-scale/
  c = 0

  #print(rank)

  xmin = dftxn['月齢（概算）'].min() - 0.1
  xmax = dftxn['月齢（概算）'].max() + 0.1
  #print(xmin, xmax)

  top_id = ''

  for id in dftxn[id].unique()[:10]:
    dftxn_temp = dftxn[dftxn[id] == id]
    if (dftxn_temp[parameters].iloc[0,:] == dfpt_temp.iloc[0,:]).all():
      top_id = id
      print(top_id)

  if len(top_id)>0:
    print("クエリが患者群に含まれています")

    length = len(dftxn_temp)
    for r in range(length):
      for id in dftxn[id].unique()[:10]: #必要に応じて表示数を調整
        dftxn_temp = dftxn[dftxn[id] == id]

        if id in qurum_members: #2023/6/1 クルムは星になるようにした
          marker_symbol='star'
        else:
          marker_symbol='circle'

        if id == top_id:
          dftxn_temp_temp = dftxn_temp.iloc[0:r, :]
          for i in range(len(para_table)):
            for j in range(len(para_table[i])):
              fig.add_trace(go.Scatter(x=dftxn_temp_temp['月齢（概算）'],
                                      y=dftxn_temp_temp[para_table[i][j]],
                                      name=str(id)+'_'+str(para_table[i][j]+'_'+str(r)),
                                      #marker=dict(color=list_colors[c]),
                                      marker=dict(size=10, color='green'),
                                      marker_symbol=marker_symbol), row=i+1, col=j+1)
        else:
          #print("クエリは見つかりませんでした")
          for i in range(len(para_table)):
            for j in range(len(para_table[i])):
              fig.add_trace(go.Scatter(x=dftxn_temp['月齢（概算）'],
                                      y=dftxn_temp[para_table[i][j]],
                                      name=str(id)+'_'+str(para_table[i][j]),
                                      marker=dict(color=list_colors[c]),
                                      marker_symbol=marker_symbol), row=i+1, col=j+1)
          c += 1
          st.write(c)

      fig.update_layout(width=1000,height=1000)

      for i in range(len(para_table)):
        for j in range(len(para_table[i])):
          fig.update_xaxes(title='月齢', range = [xmin, xmax], row=i+1, col=j+1)

      fig.update_traces(showlegend=False)

      #fig.show()
      st.plotly_chart(fig)

      #fig.write_image("similar_patients_"+str(r)+".png")

  else:
    for id in dftxn[id].unique()[:10]: #必要に応じて表示数を調整
      dftxn_temp = dftxn[dftxn[id] == id]

      if id in qurum_members: #2023/6/1 クルムは星になるようにした
        marker_symbol='star'
      else:
        marker_symbol='circle'

      for i in range(len(para_table)):
        for j in range(len(para_table[i])):
          fig.add_trace(go.Scatter(x=dftxn_temp['月齢（概算）'],
                                  y=dftxn_temp[para_table[i][j]],
                                  name=str(id)+'_'+str(para_table[i][j]),
                                  marker=dict(color=list_colors[c]),
                                  marker_symbol=marker_symbol), row=i+1, col=j+1)
      c += 1

    fig.update_layout(width=1000,height=1000)

    for i in range(len(para_table)):
      for j in range(len(para_table[i])):
        fig.update_xaxes(title='月齢', range = [xmin, xmax], row=i+1, col=j+1)

    fig.update_traces(showlegend=False)
    
    #fig.show()
    #st.pyplot(fig)
    st.plotly_chart(fig)

#全初診患者から類似症例を抽出する関数を定義

def tx_rate(dfpt, df_first=df_first, n=30):
  id = '診察券ID'
  dfpt['APR'] = dfpt['前頭部対称率']/dfpt['後頭部対称率']

  parameters = ['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'APR']

  #最大人数
  #n = 30

  #症例のZスコアの計算
  df_first_para = df_first[parameters].reset_index()
  dfpt_z = (dfpt-df_first_para.mean())/df_first_para.std()

  #重みをデータフレームでまとめて計算
  dfpt_w = 10**abs(dfpt_z).reset_index()

  #月齢の重みを他の重みの最大値にする
  if dfpt_w['月齢（概算）'][0] < dfpt_w.T.max()[0]:
    dfpt_w['月齢（概算）'][0] = dfpt_w.T.max()[0]

  #重み付き二乗和の計算
  df_first['w_delta'] = 0

  for parameter in parameters:
    df_first['w_delta'] += dfpt_w[parameter].iloc[0]*abs(df_first['z_'+parameter] - dfpt_z[parameter].iloc[0])**2

  #最適人数の導出（集団の平均が症例に最も近づく人数）
  d=10000 #平均との誤差スコアの初期値
  N = 1 #2023/1/22追記
  for i in range(1,n):
    #重み付き二乗和の順に患者リストを作成
    ply = list(df_first.sort_values('w_delta')[id])[:i] #patients like you

    #治療患者から類似症例を抽出
    dftxn = dftx[dftx[id].isin(ply)]

    #全症例からi人を抽出
    dfalln = df_first.sort_values(['w_delta'])[:i]

    #平均
    dfn = dfalln.describe()[1:2][parameters].reset_index() #indexをmeanから0に直す

    #平均との差の二乗和
    df_delta=(dfpt-dfn)**2
    sum_delta = df_delta.sum(axis=1)[0]

    #平均との誤差スコアを更新
    if d > sum_delta:
      d = sum_delta
      #スコア更新時の人数を記録
      N = i

  #print(str(N)+'人のときの平均との誤差スコア：'+str(round(d, 2)))
  st.write(str(N)+'人のときの平均との誤差スコア：'+str(round(d, 2)))

  #人数が少なかった場合、10人以上で再計算
  if N < 10:
    d=10000
    for i in range(10, n):
      #重み付き二乗和の順に患者リストを作成
      ply = list(df_first.sort_values('w_delta')[id])[:i] #patients like you

      #治療患者から類似症例を抽出
      dftxn = dftx[dftx[id].isin(ply)]

      #全症例からi人を抽出
      dfalln = df_first.sort_values(['w_delta'])[:i]

      #平均
      dfn = dfalln.describe()[1:2][parameters].reset_index() #indexをmeanから0に直す

      #平均との差の二乗和
      df_delta=(dfpt-dfn)**2
      sum_delta = df_delta.sum(axis=1)[0]

      #平均との誤差スコアを更新
      if d > sum_delta:
        d = sum_delta
        #スコア更新時の人数を記録
        N = i

    #print(str(N)+'人のときの平均との誤差スコア：'+str(round(d, 2)))
    st.write(str(N)+'人のときの平均との誤差スコア：'+str(round(d, 2)))


  #平均との誤差スコアが小さかったN症例のIDをピックアップ
  ply = list(df_first.sort_values('w_delta')[id])[:N] #patients like you
  dftxn = dftx[dftx[id].isin(ply)]
  dfcon = dfco[dfco[id].isin(ply)]

  #治療率を計算
  outcome_list = list(df_first.sort_values(['w_delta'])[:N]['診療ステータス'])

  #治療患者の人数
  ntx = outcome_list.count('治療前')

  #再診患者の人数
  if '再診前' in outcome_list:
    nco = outcome_list.count('再診前')

  #print('治療率: '+str(round(ntx/N*100,1)) +'% ('+str(ntx)+'人)')
  st.write('治療率: '+str(round(ntx/N*100,1)) +'% ('+str(ntx)+'人)')

  dfallN = dfalln[:N]

  #平均の治療期間、通院期間を表示する
  dfallNgr = dfallN[dfallN['診療ステータス'] == '治療前']
  #print('平均治療期間: '+str(round(dfallNgr['全治療期間（月数）'].mean(), 1))+' ±'+str(round(dfallNgr['全治療期間（月数）'].std(), 1))+'か月')
  #st.write('平均治療期間: '+str(round(dfallNgr['全治療期間（月数）'].mean(), 1))+' ±'+str(round(dfallNgr['全治療期間（月数）'].std(), 1))+'か月')
  #print('平均通院回数: '+str(round(dfallNgr['通院回数'].mean()+1, 1))+' ±'+str(round(dfallNgr['通院回数'].std(), 1))+'回') #再診も含まれる？
  #st.write('平均通院回数: '+str(round(dfallNgr['通院回数'].mean()+1, 1))+' ±'+str(round(dfallNgr['通院回数'].std(), 1))+'回')

  if '再診前' in outcome_list:
    #print('再診率: '+str(round(nco/N*100,1)) +'% ('+str(nco)+'人)')
    st.write('再診率: '+str(round(nco/N*100,1)) +'% ('+str(nco)+'人)')

  #N人の平均・標準偏差を表示　差の大きい指標も表示
  d_para = ''

  #表示パラメータを定義
  displayed_parameters = ['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI']
  #displayed_parameters = parameters.copy().remove('APR')
  for i in displayed_parameters:
    #平均との差
    difference = abs(dfallN.describe()[1:3][displayed_parameters][i].iloc[0]-dfpt[i].iloc[0])

    #標準偏差
    sd = dfallN.describe()[1:3][displayed_parameters][i][1]

    #print(i, difference, sd)

    if difference > sd:
      d_para=d_para+i+' '

  if d_para != '':
    #print('1SD離れているもの：'+d_para)
    st.write('1SD離れているもの：'+d_para)

  d_para = ''
  for i in displayed_parameters:
    #平均との差 2023/1/15 追記修正
    difference = abs(dfallN.describe()[1:3][displayed_parameters][i].iloc[0]-dfpt[i].iloc[0])

    #標準偏差
    sd = dfallN.describe()[1:3][displayed_parameters][i][1]

    if difference > 2*sd:
      d_para=d_para+i+' '
  if d_para != '':
    #print('2SD離れているもの：'+d_para)
    st.write('2SD離れているもの：'+d_para)

  df_ms = dfallN.describe().iloc[[1,-1]][displayed_parameters] #append()で項目を増やす場合、別のリストを作成した方が良い（parametersを使い回せなくなる）
  df_ms.loc['mean', id] = '平均'
  df_ms.loc['std', id] = '標準偏差'


  displayed_parameters = [id, '月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', '診療ステータス']
  dfallN[id] = dfallN[id].astype(str)
  dfpt[id] = 'お子様'
  df_result=pd.concat([dfpt.drop('APR', axis=1), dfallN[displayed_parameters], df_ms])
  df_result = df_result.fillna('-')
  df_result = df_result.set_index(id)
  displayed_parameters = ['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', '診療ステータス']
  #return(df_result[displayed_parameters])
  df_result = df_result.round(1)
  st.dataframe(df_result[displayed_parameters])
  #st.table(df_result[displayed_parameters])


#治療を受けた患者から類似症例を抽出する関数の定義
def similar_pts(dfpt, min=5):
  id = '診察券ID'
  parameters = ['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'APR']
  dfpt[id] = 'お子様'
  dfpt['APR'] = dfpt['前頭部対称率']/dfpt['後頭部対称率']
  #print(dfpt)
  #dfpt=dfpt[['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'APR', '診察券ID']]
  dftx_pre = dftx[dftx['診療ステータス'] == '治療前']
  dftx_pre['APR'] = dftx_pre['前頭部対称率']/dftx_pre['後頭部対称率']

  #最大人数
  n = 30

  #症例のZスコアの計算
  dftx_pre_para = dftx_pre[parameters]

  dfpt_temp = dfpt[['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'APR']]
  dfpt_z = (dfpt_temp-dftx_pre_para.mean())/dftx_pre_para.std()

  #重みをデータフレームでまとめて計算
  dfpt_w = 10**abs(dfpt_z)
  dfpt_w = dfpt_w[['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'APR']]

  if dfpt_w['月齢（概算）'].iloc[0] < dfpt_w.T.max().iloc[0]:
    dfpt_w['月齢（概算）'] = dfpt_w.T.max().iloc[0]

  #月齢の重みを他の重みの最大値にする
  if dfpt_w['月齢（概算）'].iloc[0] < dfpt_w.T.max().iloc[0]:
    w_m = dfpt_w.T.max().iloc[0]

  count = len(dftx_pre[id].unique())

  #print('探索対象は', count, '人です')
  st.write('探索対象は', count, '人です')

  #重み付き二乗和の計算
  dftx_pre['w_delta'] = 0

  for parameter in parameters:
    dftx_pre['w_delta'] += dfpt_w[parameter].iloc[0]*abs(df_first['z_'+parameter] - dfpt_z[parameter].iloc[0])**2

  #最適人数の導出
  d=10000
  for i in range(min, n): #最低人数を調整可
    ply = list(dftx_pre.sort_values('w_delta')[id])[:i] #patients like you
    dftxn = dftx[dftx[id].isin(ply)]
    dfpren = dftx_pre.sort_values(['w_delta'])[:i] #prepost？
    dfn = dfpren.describe()[1:2][parameters].reset_index()
    df_delta=(dfpt_temp-dfn)**2
    sum_delta = df_delta.sum(axis=1).iloc[0]
    if d > sum_delta:
      d = sum_delta
      N = i
  #print(str(N)+'人のとき'+str(round(d, 2)))
  st.write(str(N)+'人のときの平均との誤差スコア：'+str(round(d, 2)))

  #平均との誤差スコアが小さかったN症例のIDをピックアップ
  ply = list(dftx_pre.sort_values('w_delta')[id])[:N] #patients like you
  dftxn = dftx[dftx[id].isin(ply)]
  #dftxn = dftxn[dftxn['全治療期間m'] != 0] #dftxnから経過観察症例を除外 10/29追記 必要？

  dfpren = dftx_pre.sort_values(['w_delta'])[[id, '月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'APR', '全治療期間（月数）',#'通院回数',
                                                  '最終頭囲', '最終短頭率', '最終前頭部対称率', '最終後頭部対称率', '最終CA', '最終CVAI' #なぜか最終の値が初診の値と同じ
                                                  ]][:N]

  #print('平均治療期間：'+str(round(dfpren['全治療期間（月数）'].mean(), 1))+' ±'+str(round(dfpren['全治療期間（月数）'].std(), 1))+' か月')
  st.write('平均治療期間：'+str(round(dfpren['全治療期間（月数）'].mean(), 1))+' ±'+str(round(dfpren['全治療期間（月数）'].std(), 1))+' か月')
  #print('平均通院回数：'+str(round(dfpren['通院回数'].mean() + 1, 1))+' ±'+str(round(dfpren['通院回数'].std(), 1))+'回')
  #st.write('平均通院回数：'+str(round(dfpren['通院回数'].mean() + 1, 1))+' ±'+str(round(dfpren['通院回数'].std(), 1))+'回')

  d_para = ''

  for i in parameters:
    difference = abs(dfpren.describe()[1:3][parameters][i].iloc[0]-dfpt_temp[i].iloc[0])
    sd = dfpren.describe()[1:3][parameters][i].iloc[1]
    if difference > sd:
      d_para=d_para+i+' '

  if d_para != '':
    #print('1SD離れているもの：'+d_para)
    st.write('1SD離れているもの：'+d_para)

  d_para = ''

  for i in parameters:
    difference = abs(dfpren.describe()[1:3][parameters][i].iloc[0]-dfpt_temp[i].iloc[0])
    sd = dfpren.describe()[1:3][parameters][i].iloc[1]
    if difference > 2*sd:
      d_para=d_para+i+' '

  if d_para != '':
    #print('2SD離れているもの：'+d_para)
    st.write('2SD離れているもの：'+d_para)

  parameters = ['頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI']
  for parameter in parameters:
    dfpren[parameter+'変化量'] = dfpren['最終'+parameter] - dfpren[parameter]

  df_ms = dfpren.describe()[1:3] #.drop(id, axis=1) #[parameters] #append()で項目を増やす場合、別のリストを作成した方が良い（parametersを使い回せなくなる）
  df_ms.loc['mean', id] = '平均'
  df_ms.loc['std', id] = '標準偏差'

  df_result=pd.concat([dfpt.drop('APR', axis=1), dfpren, df_ms])
  df_result['治療前月齢'] = df_result['月齢（概算）']
  df_result['治療後月齢'] = df_result['月齢（概算）']+df_result['全治療期間（月数）']
  df_result['治療期間'] = df_result['全治療期間（月数）']

  df_result_show = df_result[[id, '治療前月齢', '治療後月齢', '治療期間', #'通院回数',
                    '頭囲', '最終頭囲', '頭囲変化量',
                    '短頭率', '最終短頭率', '短頭率変化量',
                    '前頭部対称率', '最終前頭部対称率', '前頭部対称率変化量',
                    '後頭部対称率', '最終後頭部対称率', '後頭部対称率変化量',
                    'CA', '最終CA', 'CA変化量',
                    'CVAI', '最終CVAI', 'CVAI変化量']]

  #for parameter in parameters:
    #df_result_show = df_result_show.rename(columns={'最終'+parameter:'最終', parameter+'変化量':'変化量'})

  df_result_show = df_result_show.round(1)
  df_result_show = df_result_show.fillna('-')
  df_result_show = df_result_show.set_index(id)

  #return(df_result_show)
  #st.table(df_result_show)
  st.dataframe(df_result_show)

#ランダムな初期患者を取得する関数
@st.cache_data
def get_random_pt():
  import random
  random_float = int(random.random()*10)/10
  random_id = df_first[id].unique()[int(len(df_first[id].unique())*random_float)]
  dfpt = df_first[df_first[id] == random_id]
  return(dfpt)

#初診患者のパラメータの入力
#import random
#random_float = int(random.random()*10)/10
#random_id = df_first[id].unique()[int(len(df_first[id].unique())*random_float)]
#dfpt = df_first[df_first[id] == random_id]
dfpt = get_random_pt()

# 今日の日付を取得し、pandasのTimestampオブジェクトに変換
#d = pd.to_datetime(datetime.date.today())

# Timestampを文字列に変換（フォーマット：YYYYMMDD）
#formatted_date = d.strftime('%Y%m%d')
#date_as_int = int(formatted_date)  # 文字列を整数に変換
#random_id =  list(df_first[id].unique())[date_as_int**2]
#dfpt = df_first[df_first[id] == random_id]


#生年月日
bd = pd.to_datetime(datetime.date.today()) - pd.Timedelta(days=dfpt['月齢（概算）'].iloc[0]*30.4375)

#bd = st.text_input('生年月日(YYYY-MM-DD)',value = '2024-01-01')
bd = st.text_input('生年月日(YYYY-MM-DD)',value = bd.date())
bd =  pd.to_datetime(bd)

d = pd.to_datetime(datetime.date.today()) #今日の日付
d =  st.text_input('基準日', value = str(d)[:10]) #基準日を変えたい場合はこちらに入力
d =  pd.to_datetime(d)

ap = dfpt['前後径'].iloc[0]
#ap =  float(st.text_input('前後径', value = 136))
ap =  float(st.text_input('前後径', value = int(ap)))

lr = dfpt['左右径'].iloc[0]
#lr =  float(st.text_input('左右径', value = 130))
lr =  float(st.text_input('左右径', value = int(lr)))

c = dfpt['頭囲'].iloc[0]
#c =  float(st.text_input('頭囲', value = 420))
c =  float(st.text_input('頭囲', value = int(c)))

bi = dfpt['短頭率'].iloc[0]
#bi =  float(st.text_input('短頭率', value = 105))
bi =  float(st.text_input('短頭率', value = int(bi)))

asr = dfpt['前頭部対称率'].iloc[0]
#asr =  float(st.text_input('前頭部対称率', value = 97))
asr =  float(st.text_input('前頭部対称率', value = int(asr)))

psr = dfpt['後頭部対称率'].iloc[0]
#psr =  float(st.text_input('後頭部対称率', value = 91))
psr =  float(st.text_input('後頭部対称率', value = int(psr)))

ca = dfpt['CA'].iloc[0]
#ca =  float(st.text_input('CA', value = 2))
ca =  float(st.text_input('CA', value = int(ca)))

cvai = dfpt['CVAI'].iloc[0]
#cvai =  float(st.text_input('CVAI', value = 1))
cvai =  float(st.text_input('CVAI', value = int(cvai)))

m = (d-bd)/pd.Timedelta(30.4375,"D") #月齢

if st.button('実行'):
  #if 100 not in [ap, lr, c, bi, asr, psr, ca, cvai]:
  dfpt = pd.DataFrame(data={'月齢（概算）':[m],
                            '前後径':[ap],
                            '左右径':[lr],
                            '頭囲':[c],
                            '短頭率':[bi],
                            '前頭部対称率':[asr],
                            '後頭部対称率':[psr],
                            'CA':[ca],
                            'CVAI':[cvai]})
    #全体
    #similar_members = tx_plot(dfpt, n=500)
    #n=500で10秒
  tx_plot(dfpt, n=500)

  st.write('治療率を計算中・・・')
  tx_rate(dfpt)

  st.write('治療結果を計算中・・・')
  similar_pts(dfpt)

  #else:
    #st.write('値を入力してください。')

if st.button('その他のランダムな患者で実行'):
  #初診患者のパラメータの入力
  import random
  random_float = int(random.random()*10)/10
  random_id = df_first[id].unique()[int(len(df_first[id].unique())*random_float)]
  dfpt = df_first[df_first[id] == random_id]
  bd = pd.to_datetime(datetime.date.today()) - pd.Timedelta(days=dfpt['月齢（概算）'].iloc[0]*30.4375)
  #bd = bd.date()
  m = (d-bd)/pd.Timedelta(30.4375,"D")
  #dfpt['月齢（概算）'] = float(m)

  parameters = ['月齢（概算）', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI']
  dfpt = dfpt[parameters]

  #st.write(dfpt.dtypes)

  tx_plot(dfpt, n=500)
  st.write('治療率を計算中・・・')
  tx_rate(dfpt)
  st.write('治療結果を計算中・・・')
  similar_pts(dfpt)

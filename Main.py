import os
import sys
import tensorflow as tf
import pandas as pd
import Preprocess
import Environment
import DQN
import Trainer
import numpy as np
import matplotlib.pyplot as plt

global _symbol
global _split_ratio
global _epochs

_symbol = "AAPL"
_split_ratio = 0.8
_epochs = 5

data_ex = pd.read_csv("apple.csv") #애플 주식데이터 2016년 3월 ~ 2018년 3월

# 훈련용, 검증용 데이터 분리
train, test = Preprocess.data_preprocess(data_ex, _split_ratio)

# 훈련 및 검증 환경 생성
env_train = Environment.StockEnv(train)
env_test = Environment.StockEnv(test)

runner = Trainer.Runner()
# DQN을 훈련시키고, 훈련된 모델로 돌아가 최종 결과를 업데이트
trained_model = runner.trainer(_symbol, env_train, _epochs)

# 훈련 된 trained_Q를 사용하여 테스트 데이터를 분석하고 예측 된 최종 거래 행동을 제공
fortune, act, reward, cash = runner.tester(env_test, trained_model)
print("fortune:{},act:{},reward:{},cash:{}".format(fortune[-1], act[-1], reward[-1], cash[-1]))

# 결과 시각화 
close_price = test[5:, 3] #검증용 데이터의 종가 정보를 가져옴

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(len(close_price)), close_price)
ax.set_xlim((0, len(close_price)))
ax.set_ylim((np.min(close_price), np.max(close_price)))
ax.set_xlabel("Steps")
ax.set_ylabel("Close Price")
ax.set_title('Trade Point predicted by DQN Trader')

# 검증용 데이터의 종가 정보에 따른 강화학습 모델의 매수/매도 행동을 확인
# 빨간색: 매수
# 초록색: 매도

for i in range(len(act)):
    if act[i] == 1:
        ax.scatter(x=i, y=close_price[i], c='r', marker='o', linewidths=0, label='Buy')
    if act[i] == 2:
        ax.scatter(x=i, y=close_price[i], c='g', marker='o', linewidths=0, label='Sell')
        
        









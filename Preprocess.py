import numpy as np 
import pandas as pd


def data_preprocess(data, split_ratio):

	"""
	data: 활용 데이터
	split_ratio: 훈련 데이터의 비율
	"""
	_data = data
	_split_ratio = split_ratio

	# 데이터의 행 수
	n_rows = _data.shape[0]
    
	# 데이터의 열 수
	n_cols = _data.shape[1]
    
	# 개시 가격 데이터 로드
	data_c_open = _data[['Open']]
    
	# 다음날의 개시가격 로드
	tradePrice = np.array(data_c_open[1:])
	_data = _data[:-1]
    
	# 행수가 동일하면 데이터를 접합
    
	if len(tradePrice) == len(_data):
		# 데이터의 개시가격, 종가, 최저가격, 최고가격, 거래량 가져오기
		_data = _data.loc[:,['Open','High','Low','Close','Volume']] 
        
		# 사용 당일, 즉 당일의 개시일 및 거래가격 
		_data['tradePrice'] = tradePrice
        
		# 초기 현금 100,000원으로 설정
		_data['cash'] = 100000.
        
		# 초기 주식량 0으로 설정
		_data['stockValue'] = 0.
        
		# 처리된 데이터를 배열로 변환
		_data = np.array(_data)

		# 훈련 데이터의 수
		n_train = int(np.round(_split_ratio * n_rows))
        
		# 검증 데이터의 수
		n_test = n_rows - n_train

		# 훈련 데이터 셋
		train = _data[:n_train]
        
		# 검증 데이터 셋
		test = _data[-n_test:]

		return train, test

	else:
		return None, None


	
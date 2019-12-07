from tensorforce.environments import Environment
import pandas as pd 
import numpy as np 

# 상태의 단계
N_TIMESTEPS = 5
# 매번 거래되는 주식 수
ORDER_SIZE = 10

class StockEnv(Environment):
    
    def __init__(self, data):
        
        # 과거 20일의 주식 데이터와 그날의 개시가격을 포함하여 주가 데이터를 가져옴(총 21일)
        self.xdata = data

        # 초기화 환경
        self.reset() 
        
    def __str__(self):
        return 'StockEnvironment'

    def close(self):
        """
        환경을 닫음. 추후 다른 메소드 호출이 가능하지 않음
        """
        print('stock over')

    def seed(self, seed):
        """
        환경의 난수를 지정된 값으로 설정(Seed = None인 경우 현재 시간임)
        """
        return None

    def reset(self):
        """
        환경을 재설정하고 새 에피소드를 설정
        """
        # 샘플수 설정
        self.sample_size = len(self.xdata)
        # 훈련 초기화
        self.step_counter = 0  
        # 상태 단계 초기화
        self.n_timesteps = N_TIMESTEPS
        # 단일 거래 번호 초기화
        self.order_size = ORDER_SIZE
        # 초기화 된 action 수
        self.n_actions = len(self.actions)
        # 보상 초기화
        self.reward = 0.
        # 상태 초기화
        self.current_state = self.states 
        self.next_states = self.xdata[self.step_counter + 1: self.step_counter + self.n_timesteps + 1]
        
        # 거래 당일 개시가격(21일)
        self.tradePrice = self.states[-1, 5]
        
        # 기록 마감 가격(20일), 주식 가치 측정을 위함
        self.cal_value_price = self.states[-1, 3]
        
        # 현재주식
        self.stock_amount = 0. # 보유한 주식 수량 초기화
        self.stock_value = 0. # 주식값 초기화

        # 현재 현금
        self.cash_hold = 100000. 
        
        # 현재 재산의 총 가치 초기화 （현재현금 + 주식가치)
        self.current_value = self.cash_hold + self.stock_value 
        
        # 과거 재산의 총가치 초기화
        self.past_value = 100000. 
        
        # 현재 최종 상태에 있는지 여부
        self.done = False 
        

    def execute(self, action):
        """
        행동을 실행하고 다음 상태 및 보상을 관찰
        """
        # 과거 재산을 현재 재산으로 업데이트
        self.past_value = self.current_value
        
        # 주식 가치 계산
        self.cal_value_price = self.states[-1, 3]
        
        # 거래 주문
        self.tradePrice = self.states[-1, 5]
        
        if action == 1: # 구매
            # 현금 총액 업데이트
            self.cash_hold = self.cash_hold - self.tradePrice * self.order_size
            
            # 주식 수량 업데이트
            self.stock_amount = self.stock_amount + self.order_size
            
        elif action == 2: # 판매
            # 현금 총액 업데이트
            self.cash_hold = self.cash_hold + self.tradePrice * self.order_size
            
            # 주식 수량 업데이트
            self.stock_amount = self.stock_amount - self.order_size  
        # elif action == 0: # 보류
        # 돈의 가치를 고려하고 현금의 총가치를 업데이트
        self.cash_hold = 0.9997 * self.cash_hold 


        # 상태 업데이트
        if self.step_counter + self.n_timesteps + 1 < self.sample_size:
            self.done = False
            self.next_states = self.xdata[self.step_counter + 1 : self.step_counter + self.n_timesteps + 1]
            # 주식 가치를 계산하는데 사용된느 다음날의 종가를 계산
            self.stock_value = self.next_states[-1, 3] * self.stock_amount
            # 당일 현금 및 재고 값 업데이트
            self.xdata[self.step_counter + self.n_timesteps, 6] = self.cash_hold
            self.xdata[self.step_counter + self.n_timesteps, 7] = self.stock_value  
        else:
            self.done = True
            self.next_states = self.states
            # 당일의 종가가 계산되지 않았으므로, 거래 개시 시점의 주가를 계산함 
            self.stock_value = self.next_states[-1, 5] * self.stock_amount
            # 당일 현금 및 재고 값 업데이트
            self.xdata[self.step_counter + self.n_timesteps, 6] = self.cash_hold
            self.xdata[self.step_counter + self.n_timesteps, 7] = self.stock_value

        # 자본의 총액 계산(현금 + 주식 가치)
        self.current_value = self.cash_hold + self.stock_value
        
        # 보상 규칙
        # 다음과 같은 상황에선느 최대 패널티를 내며 주식 거래를 일찍 종료：
        #       1. 현금 < 0; 
        #       2. 자본금 < 70,000
        #       3. 현금이 총 자본금의 30% 미만으로 유지될 때
        #       4. 주식 구매시 현금이 충분하지 않을 때
        
        if self.cash_hold <= 0 or self.current_value <= 70000. or (self.cash_hold <= (0.3 * self.current_value)):  
            self.reward = self.reward - 1.

        if self.stock_amount < 0 and ((-1 * self.stock_amount) > (0.7 * self.cash_hold / self.states[-1,5])):
            self.reward = self.reward - 1.

        # 보상 벌칙
        self.reward = self.reward + 1. * (self.current_value - self.past_value) / self.past_value
            
        # 학습 시간 업데이트
        self.step_counter = self.step_counter + 1
        

        return self.next_states, self.done, self.reward 
    
    
    @property
    def states(self):
        """
        상태공간을 반환, 여러 상태가 동시에 존재하는 경우 하위 그룹으로 포함될 수 있음
        """
        # 현재 시점에서 20개 이상 데이터를 취할수 있는 경우 현재 상태를 반환
        if self.step_counter + self.n_timesteps < self.sample_size:
            states = self.xdata[self.step_counter : self.step_counter + self.n_timesteps]
            return states
        # 현재 시점에서 20개 이상 데이터를 취할수 없는 경우 이전 상태를 반환
        else:
            # print("No More Data.")
            self.done = True
            states = self.xdata[self.step_counter-1 : self.step_counter + self.n_timesteps-1]
            return states

    @property
    def actions(self):
        """
        action 공간을 반환, 여러 작업이 동시에 진행되는 경우 하위 action이 포함될 수 있음
        """
        # 세가지 경우( 0:보류, 1:구매, 2:판매 )
        actions = [0, 1, 2] 
        
        return actions


    @staticmethod
    def from_spec(spec, kwargs):
        """
        환경을 기록
        """
        env = tensorforce.util.get_object(
            obj=spec,
            predefined_objects=tensorforce.environments.environments,
            kwargs=kwargs
        )
        assert isinstance(env, Environment)
        return env
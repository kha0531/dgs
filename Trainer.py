import Environment
import DQN

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model


# 보상 감소 비율
GAMMA = 0.9

# EPSILON 기본값 설정
EPSILON = 1.0

# 최소 EPSILON 설정
EPSILON_MIN = 0.01

# EPSILON 감소 비율
EPSILON_DECAY = 0.995

class Runner:
    def __init__(self):
        # 모델을 사용하기 전 이전 모델이 차지한 메모리를 삭제
        K.clear_session()
        tf.reset_default_graph()
        
        # 보상 감소 비율 초기화
        self.gamma = GAMMA
        # epsilon 초기화
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN  
        self.epsilon_decay = EPSILON_DECAY  
        self.train_success = False
        
        # Runner가 훈련을 마친 뒤, 테스트를 위해 최종 훈련된 모델의 이름을 반환
        self.model_name = ''
        
    def trainer(self, symbol, env, epochs):
        # 액세스 환경
        self.env = env
        
        # DQN Agent 초기화
        self.dqn_agent = DQN.DQN(env=self.env)
        
        # epoch 초기화
        self.epochs = epochs
        # epoch 당 훈련 단계 수 초기화
        self.epoch_len = self.env.sample_size - self.env.n_timesteps

        # 훈련 시작
        for epoch in range(self.epochs):
            
            # 훈련 환경 초기화
            self.env.reset()
            
            # 현재상태 가져오기
            cur_state = self.env.states
            
            # 자본 기록 초기화
            fortune = list()
            
            # 현금 기록 초기화
            cash = list()
            
            # action 기록 초기화
            act = list()
            
            # 보상기록 초기화
            re = list()
            
            for step in range(self.epoch_len):
                
                # 현재 상태에 따라 action을 선택
                action = self.dqn_agent.act(cur_state)
                
                # action 실행에 따른 상태 반환
                new_state, done, reward = self.env.execute(action)

                # 반환된 action을 action 기록에 추가
                act.append(action)
                
                # 반환된 보상을 보상 기록에 추가
                re.append(reward)
                
                # 반환된 현금을 현금 기록에 추가
                cash.append(new_state[-1,6])
                
                # 현재 상태의 총 자본 가치를 기록하고 자본 기록에 추가
                _fortune = new_state[-1,6] + new_state[-1,7]
                fortune.append(_fortune)

                # 기억 메모리에 저장
                self.dqn_agent.remember(cur_state, action, reward, new_state, done)
                self.dqn_agent.replay() 
                if step > 20:
                    
                    # 모델 매개변수 업데이트
                    self.dqn_agent.target_train() 
                    
                # 다음 상태 가져오기
                cur_state = new_state

                # 종료상태인 경우 훈련을 종류하고 그렇지 않으면 훈련을 계속
                if done:
                    # 모델 훈련 완료시 모델을 저장
                    if fortune[-1] >= 120000. and cash[-1] >= 0.:
                        self.train_success = True
                        self.model_name = "success-model-{}-{}.h5".format(symbol, epoch)
                        self.dqn_agent.save_model(self.model_name)
                        
                    # 모델 훈련 미완료시 모델 제외
                    else:
                        self.train_success = False
                        self.model_name = "train-model-{}-{}.h5".format(symbol, epoch)
                        self.dqn_agent.save_model(self.model_name)
                    break
            
            print("Epoch {}: Fortune-{}, Cash-{}, Reward-{}".format(str(epoch), fortune[-1], cash[-1], re[-1]))    
            
        # 모든 훈려을 마친 후 모델명 변환   
        return self.model_name
                       

    def tester(self, env, model_name):
        
        # 검증 데이터를 기반으로 환경 초기화
        self.env = env
        
        # epoch당 훈련 단계 수 초기화
        self.epoch_len = self.env.sample_size - self.env.n_timesteps
        
        # 모델 이름 초기화
        self.model_name = model_name
        
        # 모델 이름을 기반으로 모델 불러오기
        self.test_model = load_model(self.model_name)

        # 검증 환경 초기화
        self.env.reset()
        
        # 현재상태 가져오기
        cur_state = self.env.states
        
        # 자본 기록 초기화
        fortune = list()
        
        # 현금 기록 초기화
        cash = list()
        
        # action 기록 초기화
        act = list()
        
        # 보상 기록 초기화
        re = list()
        
        # 검증 시작
        for step in range(self.epoch_len):
            
            # epsilon 업데이트
            self.epsilon *= self.epsilon_decay
            
            # epsilon과 epsilon_min 중 큰 값으로 epsilon 업데이트
            self.epsilon = max(self.epsilon_min, self.epsilon)
            
            # 발생시킨 난수가 epsilon보다 작으면, 무작위로 action을 선택
            if np.random.random() < self.epsilon:
                action = np.random.randint(0, self.env.n_actions)
                
            # 발생시킨 난수가 epsilon보다 크거나 같으면 검증 모델에 따라 action을 선택
            else:
                action = np.argmax(self.test_model.predict(cur_state)[0])

            # action 실행에 따른 상태 반환
            new_state, done, reward = self.env.execute(action)

            # 반환된 action을 action 기록에 추가
            act.append(action)
            
            # 반환된 보상을 보상 기록에 추가
            re.append(reward)
            
            # 반환된 현금을 현금 기록에 추가
            cash.append(new_state[-1,6])          
            
            # 현재 상태의 총 자본 가치를 기록하고 자본 기록에 추가  
            _fortune = new_state[-1,6] + new_state[-1,7]
            fortune.append(_fortune)

            # 다음 상태 가져오기
            cur_state = new_state

            # 검증 종료         
            if done:
                break
        
        print("Test Result: Fortune-{}, Cash-{}, Reward-{}".format(fortune[-1], cash[-1], re[-1]))          

        # 결과 반환
        return fortune, act, re, cash

                
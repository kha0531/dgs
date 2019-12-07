import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam


# 파라미터 설정
BATCH_SIZE = 64
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TAU = 0.05 


class DQN:
    def __init__(self, env):
        # 학습환경 설정
        self.env = env

        # 초기화 매개변수 설정
        self.memory  = deque(maxlen=2000)  # 메모리 대기열, 최대 메모리는 2000으로 설정
        self.gamma = GAMMA  
        self.epsilon = EPSILON 
        self.epsilon_min = EPSILON_MIN  
        self.epsilon_decay = EPSILON_DECAY  
        self.learning_rate = LEARNING_RATE  
        self.tau = TAU 

        # 평가 네트워크 생성
        self.eval_model = self.create_model()
        # 차겟 네트워크 생성
        self.target_model = self.create_model()

    def create_model(self):
        # 모델 생성
        model   = Sequential()

        # 입력데이터의 차원으로 shape[0]은 행의 수, shape[1]은 열의 수를 의미
        state_shape  = self.env.states.shape

        # 첫번째 히든 레이어 설정
        model.add(Dense(32, input_dim=state_shape[1], activation="relu"))
        model.add(Dropout(0.5))

        # 두번째 히든 레이어 설정
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))

        # 세번째 히든 레이어 설정
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))

        # 출력 레이어 설정(soft-max)
        model.add(Dense(self.env.n_actions, activation="linear"))

        # 네트워크 모델 생성
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        
        return model

    def act(self, state):
        # epsilon 없데이트
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.env.n_actions)
        return np.argmax(self.eval_model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        # 메모리 설정
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        # 배치 수 설정
        batch_size = BATCH_SIZE
        if len(self.memory) < batch_size: 
            return
        samples = random.sample(self.memory, batch_size)
        # 랜덤 샘플링
        for sample in samples:
            state, action, reward, new_state, done = sample
            # 타겟 모델을 사용해 예측하고 예측된 결과를 대상에 저장
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.eval_model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        # 평가 네트워크의 가중치 매개변수를 가져옴
        eval_weights = self.eval_model.get_weights()
        # 타겟 네트워크의 가중치 매개변수를 가져옴
        target_weights = self.target_model.get_weights()
        # 타겟 네트워크의 가중치 매개변수를 하나씩 업데이트
        for i in range(len(target_weights)):
            # 평가 및 타겟 네트워크의 가중치 매개변수값을 tau로 조정하여 업데이트
            target_weights[i] = eval_weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        # 업데이트된 가중치 매개변수를 타겟 네트워크로 설정
        self.target_model.set_weights(target_weights)
        
    def save_model(self, fn):
        # 모델 저장
        self.eval_model.save(fn)
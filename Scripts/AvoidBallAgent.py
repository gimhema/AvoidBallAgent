import tensorflow as tf
import unreal_engine as ue
import numpy as np
import random
from collections import deque
from TFPluginAPI import TFPluginAPI
import dqn

from typing import List

#인풋:
#에이전트 이전위치(y좌표: float)
#에이전트 위치 (y좌표: float)
#각 제네레이터별 공 발사 여부(발사했을경우 1.0, 아니면 0.0으로)
#예) 2번에서 공 발사할경우
#0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 로 설정 

#아웃풋:
#좌로이동
#우로이동

# 반복문은 블루프린트에서 돌린다.
# 함수 하나하나를 self값의 인자로 설정하여 BP에서 접근하는것을 생각해본다



class AVoidBallAgent(TFPluginAPI):
    def set_dqn_info(self, type):
        self.INPUT_SIZE = 10
        self.OUTPUT_SIZE = 2
        self.DISCOUNT_RATE = 0.99
        self.REPLAY_MEMORY = 50000
        self.BATCH_SIZE = 64
        self.TARGET_UPDATE_FREQUENCY = 5
        self.MAX_EPISODES = 5000
        self.episode = 200
        self.state = np.zeros(10)
        self.next_state = np.zeros(10)
        self.action = 1
        self.reward = 0.0
        pass
    
    def replay_train(self, mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:
        ue.log("Execution Replay Train Event")
        states = np.vstack([x[0] for x in train_batch])
        actions = np.array([x[1] for x in train_batch])
        rewards = np.array([x[2] for x in train_batch])
        next_states = np.vstack([x[3] for x in train_batch])
        done = np.array([x[4] for x in train_batch])

        X = states

        Q_target = rewards + self.DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1)*~done

        y = mainDQN.predict(states)
        ue.log("Replay Train debug")
        ue.log(str(np.arange(len(X))))
        ue.log(str(actions))

        y[np.arange(len(X)), actions] = Q_target

        return mainDQN.update(X, y)

    def get_copy_var_ops(self, *, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
        ue.log("Execution Copy array Event")
        op_holder = []

        src_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=src_scope_name)
        dest_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))
        
        return op_holder

    def game_play_start(self, type):
        self.replay_buffer = deque(maxlen=self.REPLAY_MEMORY)
        self.last_100_game_reward  = deque(maxlen=100)
        self.sess = tf.compat.v1.Session()
        
        self.mainDQN = dqn.DQN(self.sess, self.INPUT_SIZE, self.OUTPUT_SIZE, name="main")
        self.targetDQN = dqn.DQN(self.sess, self.INPUT_SIZE, self.OUTPUT_SIZE, name="target")
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.copy_ops = self.get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        self.sess.run(self.copy_ops)
        # 이 다음부턴 BP로 반복문 통제
        pass

    def begin_episode(self, type):
        # for문에서 시작하는 함수
         # 이 값은 BP에서의 for문 횟수와 일치해야한다.
        ue.log("Execution Begin Episode Event")
        self.e = 1. / ((self.episode / 10) + 1)
        self.done = False
        self.step_count = 0
        pass
    
    def rewardPenalty(self, type):
        ue.log("Execution Penalty Event")
        self.reward = -1
        self.done = True
        self.episode -= 1
        pass

    def saveExperience(self, type):
        self.replay_buffer.append((self.state, self.action, self.reward, self.next_state, self.done))
        if len(self.replay_buffer) > self.BATCH_SIZE:
            minibatch = random.sample(self.replay_buffer, self.BATCH_SIZE)
            loss, _ = self.replay_train(self.mainDQN, self.targetDQN, minibatch)
        
        if self.step_count & self.TARGET_UPDATE_FREQUENCY == 0:
            self.sess.run(self.copy_ops)
        
        self.state = self.next_state
        self.step_count += 1

    def saveReward_last_100_game(self, type):
        self.last_100_game_reward.append(self.step_count)
        
        if len(self.last_100_game_reward) == self.last_100_game_reward.maxlen:
            avg_reward = np.mean(self.last_100_game_reward)
            if avg_reward > 199:
                pass
        pass


    def onJsonInput(self, jsonInput):
        # BP while문에서 돌려야할 함수
        # Json으로 전달된 상태값에 대한 액션값을 리턴한다
        ue.log("Execution Json Input Event")
        ue.log(str(jsonInput['isNowAction'][0]))
        if(str(jsonInput['isNowAction'][0]) == "t"):
            ue.log("Create State obj")
            state_1 = float(jsonInput['previousLocation'])
            state_2 = float(jsonInput['presentLocation'])
            state_3 = float(jsonInput['ballGenerator1Loc'])
            state_4 = float(jsonInput['ballGenerator2Loc'])
            state_5 = float(jsonInput['ballGenerator3Loc'])
            state_6 = float(jsonInput['ballGenerator4Loc'])
            state_7 = float(jsonInput['ballGenerator5Loc'])
            state_8 = float(jsonInput['ballGenerator6Loc'])
            state_9 = float(jsonInput['ballGenerator7Loc'])
            state_10 = float(jsonInput['ballGenerator8Loc'])
            self.state = np.array([state_1,state_2,state_3,state_4,state_5,state_6,state_7,state_8,state_9,state_10], dtype='f')
            
            if(np.random.rand() < self.e):
                dice = random.randrange(0,2)
                if(dice == 0):
                    self.action = 1 # 좌로이동
                elif(dice == 1):
                    self.action = 0 # 우로이동
            # action값은 좌,우 이동 둘중에 하나를 선택한다
            else:
                self.action = np.argmax(self.mainDQN.predict(self.state))
            return self.action # self.action값을 리턴한다.

        elif(str(jsonInput['isNowAction'][0]) == "f"):
            ue.log("Create Next State obj")
            state_1 = float(jsonInput['previousLocation'])
            state_2 = float(jsonInput['presentLocation'])
            state_3 = float(jsonInput['ballGenerator1Loc'])
            state_4 = float(jsonInput['ballGenerator2Loc'])
            state_5 = float(jsonInput['ballGenerator3Loc'])
            state_6 = float(jsonInput['ballGenerator4Loc'])
            state_7 = float(jsonInput['ballGenerator5Loc'])
            state_8 = float(jsonInput['ballGenerator6Loc'])
            state_9 = float(jsonInput['ballGenerator7Loc'])
            state_10 = float(jsonInput['ballGenerator8Loc'])
            self.next_state = np.array([state_1,state_2,state_3,state_4,state_5,state_6,state_7,state_8,state_9,state_10], dtype='f')
            return self.action

    def onBeginTraining(self):
        ue.log("starting avoid agent training")
        self.INPUT_SIZE = 10
        self.OUTPUT_SIZE = 2
        self.DISCOUNT_RATE = 0.99
        self.REPLAY_MEMORY = 50000
        self.BATCH_SIZE = 64
        self.TARGET_UPDATE_FREQUENCY = 5
        self.MAX_EPISODES = 5000
        self.episode = 200
        self.state = np.zeros(10)
        self.next_state = np.zeros(10)
        self.action = 1
        self.reward = 0.0
        self.done = False
        self.step_count = 0


        self.replay_buffer = deque(maxlen=self.REPLAY_MEMORY)
        self.last_100_game_reward  = deque(maxlen=100)
        self.sess = tf.compat.v1.Session()
        
        self.mainDQN = dqn.DQN(self.sess, self.INPUT_SIZE, self.OUTPUT_SIZE, name="main")
        self.targetDQN = dqn.DQN(self.sess, self.INPUT_SIZE, self.OUTPUT_SIZE, name="target")
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.copy_ops = self.get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        self.sess.run(self.copy_ops)
        pass


def getApi():
    return AVoidBallAgent.getInstance()
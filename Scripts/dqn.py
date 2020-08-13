import numpy as np
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI
import tensorflow as tf

class DQN:
  def __init__(self, session: tf.compat.v1.Session, input_size: int, output_size: int, name: str="main") -> None:
    self.session = session
    self.input_size = input_size
    self.output_size = output_size
    self.net_name = name
    self._build_network()
  
  def _build_network(self, h_size=16, l_rate=0.001) -> None:
    with tf.compat.v1.variable_scope(self.net_name):
      self._X = tf.compat.v1.placeholder(tf.float32, [None, self.input_size], name="input_x")
      net = self._X

      net = tf.compat.v1.layers.dense(net, h_size, activation=tf.nn.relu)
      net = tf.compat.v1.layers.dense(net, self.output_size)
      self._Qpred = net

      self._Y = tf.compat.v1.placeholder(tf.float32, shape=[None, self.output_size])
      self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

      optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate)
      self._train = optimizer.minimize(self._loss)

  def predict(self, state: np.ndarray) -> np.ndarray:
    x = np.reshape(state, [-1, self.input_size])
    return self.session.run(self._Qpred, feed_dict={self._X: x})

  def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
    feed = {
        self._X : x_stack,
        self._Y : y_stack
    }
    return self.session.run([self._loss, self._train], feed)

def getApi():
  return DQN.getInstance()
import shutil
import os
import gym
import numpy as np
import tensorflow as tf
import threading
import matplotlib.pyplot as plt

env_name = 'Pendulum-v0'
test_env = gym.make(env_name)
n_states = test_env.observation_space.shape[0]
n_actions = test_env.action_space.shape[0]
action_bound = [test_env.action_space.low, test_env.action_space.high]

max_episode = 2000
episode_counter = 0
episode_steps = 200
global_scope = 'GlobalNetwork'

a_hidden = 256
c_hidden = 128

beta = 0.01
gamma = 0.9

a_lr = 0.0001
c_lr = 0.001
update_interval = 10
n_workers = os.cpu_count()
global_running_rewards = []
log_dir = './logs'


class ActorCritic:
    def __init__(self, scope):
        self.scope = scope
        self.a_opt = tf.train.RMSPropOptimizer(a_lr)
        self.c_opt = tf.train.RMSPropOptimizer(c_lr)

        if self.scope == global_scope:
            with tf.variable_scope(self.scope):
                self.state_p = tf.placeholder(dtype=tf.float32, shape=[None, n_states], name='state_p')
                self.a_params, self.c_params = self.build_model()[-2:]
        else:
            with tf.variable_scope(self.scope):
                self.state_p = tf.placeholder(dtype=tf.float32, shape=[None, n_states], name='state_p')
                self.v_target_p = tf.placeholder(dtype=tf.float32,
                                                 shape=[None, 1], name='v_target_p')
                self.actions_p = tf.placeholder(dtype=tf.float32,
                                                shape=[None, n_actions],
                                                name='actions_p')
                self.mu, self.sigma, self.v, self.a_params, self.c_params = self.build_model()

                with tf.name_scope('wrap_actor'):
                    self.mu *= action_bound[1]
                    self.sigma += 1e-4

                a_dist = tf.distributions.Normal(self.mu, self.sigma, name='a_dist')
                td = tf.subtract(self.v_target_p, self.v, name='td_error')

                with tf.name_scope('c_loss'):
                    c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    entropy = a_dist.entropy()
                    exp_v = a_dist.log_prob(self.actions_p) * tf.stop_gradient(td)
                    a_loss = - tf.reduce_mean(beta * entropy + exp_v)

                with tf.name_scope('grads'):
                    a_grads = tf.gradients(a_loss, self.a_params)
                    c_grads = tf.gradients(c_loss, self.c_params)

                with tf.name_scope('choose_action'):
                    self.action = tf.clip_by_value(tf.squeeze(a_dist.sample(1), axis=0),
                                                   action_bound[0],
                                                   action_bound[1])
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_ac.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_ac.c_params)]
                with tf.name_scope('push'):
                    self.push_a_params_op = self.a_opt.apply_gradients(zip(a_grads, global_ac.a_params))
                    self.push_c_params_op = self.c_opt.apply_gradients(zip(c_grads, global_ac.c_params))

    def push_global(self, states, actions, v_target):
        sess.run([self.push_a_params_op, self.push_c_params_op],
                 feed_dict={self.state_p: states,
                            self.actions_p: actions,
                            self.v_target_p: v_target})

    def pull_global(self):
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def build_model(self):
        with tf.variable_scope('actor'):
            a_l = tf.layers.dense(self.state_p, a_hidden, tf.nn.relu,
                                  kernel_initializer=tf.initializers.he_normal(), name='a_l')
            mu = tf.layers.dense(a_l, n_actions, tf.nn.tanh,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='mu')
            sigma = tf.layers.dense(a_l, n_actions, tf.nn.softplus,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='sigma')
        with tf.variable_scope('critic'):
            c_l = tf.layers.dense(self.state_p, c_hidden, tf.nn.relu,
                                  kernel_initializer=tf.initializers.he_normal(), name='c_l')
            v = tf.layers.dense(c_l, 1, None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='v')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/critic')

        return mu, sigma, v, a_params, c_params

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        action = sess.run(self.action, feed_dict={self.state_p: state})[0]
        return action

    def get_value(self, state):
        state = np.expand_dims(state, axis=0)
        value = sess.run(self.v, feed_dict={self.state_p: state})[0, 0]
        return value


class Worker:
    def __init__(self, name):
        self.name = name
        self.env = gym.make(env_name)
        self.ac = ActorCritic(self.name)

    def work(self):
        global episode_counter, global_running_rewards
        buffer_s, buffer_r, buffer_a = [], [], []
        while not coord.should_stop() and episode_counter < max_episode:
            state = self.env.reset()
            ep_r = 0
            for step in range(1, episode_steps + 1):
                # if self.name == 'W_0':
                #     self.env.render()
                action = self.ac.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                done = True if done or step == episode_steps else False

                buffer_a.append(action)
                buffer_s.append(state)
                buffer_r.append((reward + 8) / 8)

                ep_r += reward

                if done or step % update_interval == 0:
                    if done:
                        v_target = 0
                    else:
                        v_target = self.ac.get_value(next_state)
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_target = r + gamma * v_target
                        buffer_v_target.append(v_target)
                    buffer_v_target.reverse()

                    buffer_a = np.vstack(buffer_a)
                    buffer_s = np.vstack(buffer_s)
                    buffer_v_target = np.vstack(buffer_v_target)

                    self.ac.push_global(buffer_s, buffer_a, buffer_v_target)
                    self.ac.pull_global()
                    buffer_s, buffer_r, buffer_a = [], [], []
                if done:
                    if len(global_running_rewards) == 0:  # record running episode reward
                        global_running_rewards.append(ep_r)
                    else:
                        global_running_rewards.append(0.9 * global_running_rewards[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", episode_counter,
                        "| Ep_r: %i" % global_running_rewards[-1],
                    )
                    break
                state = next_state

            episode_counter += 1


if __name__ == '__main__':
    print('n_states: ', n_states)
    print('n_actions: ', n_actions)
    print('action_bounds: ', action_bound)
    print('cpu_count: ', n_workers)

    train = False
    if train:
        with tf.device('/cpu:0'):
            global_ac = ActorCritic(global_scope)
            workers = []
            for i in range(n_workers):
                workers.append(Worker(f'W_{i}'))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        threads = []
        for worker in workers:
            thread = threading.Thread(target=worker.work)
            thread.start()
            threads.append(thread)

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tf.summary.FileWriter(log_dir, sess.graph)

        coord.join(threads)
        saver = tf.train.Saver()
        saver.save(sess, 'model_test_now/pendulum.ckpt')
        plt.figure()
        plt.plot(np.arange(len(global_running_rewards)), global_running_rewards)
        plt.xlabel('step')
        plt.ylabel('Total moving reward')
        plt.savefig('rewards_1.png')
        plt.show()
    else:
        sess = tf.Session()
        with tf.device('/cpu:0'):
            global_ac = ActorCritic(global_scope)
            agent = Worker("W_2")
        saver = tf.train.Saver()
        saver.restore(sess, 'model_test_now/pendulum.ckpt')

        state = agent.env.reset()
        for _ in range(5000):
            agent.env.render()
            action = agent.ac.choose_action(state)
            next_state, reward, done, info = agent.env.step(action)
            state = next_state
            if done:
                state = agent.env.reset()
                print('env has been reset')

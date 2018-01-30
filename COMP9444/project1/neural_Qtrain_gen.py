import sys
import gym
import tensorflow as tf
import numpy as np
import random
import datetime
import pandas as pd


# By z5034054, Boshen Hu & z4002386, Xi Tan, Group 95.
#
# Some statements:
# 1. A pandas data frame has been added in order to record the last 100 results,
#    and the ave reward of the last 100 is also computed at the end of the operation.
# 2. Using the same model in all of the three games, while
#    some important hyper parameters have been tuned separately
#    in terms of each agme. (determined in "setup()")
# 3. Only add functions and some lines, never changed the API signatures.
# 4. In the game "MountainCar-v0" and "Pendulum-v0", a double DQN has been implemented.been
# 5. The model has not reached its best performance due to the limitation of 15 mins processing time.
#    If without considering the time limit, the performance could be near the best
#    at around NUM_EPISODES = 800 in all three games.
# What is more,
# With regards to the 15 mins limites of the processing time,
# it could be vague beacuse of the difference of all computers in computational performance.
# Therefore, to be conservative, we limit our script to make sure every problem can get a
# result in 15 mins in a CSE lab computer, which also limits the perfoamce
# of our code.


"""
Hyper Parameters
"""
GAMMA = 0.97  # discount factor for target Q
INITIAL_EPSILON = 0.6  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
EPSILON_DECAY_STEPS = 100
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 128  # size of minibatch
TEST_FREQUENCY = 10  # How many episodes to run before visualizing test accuracy
SAVE_FREQUENCY = 1000  # How many episodes to run before saving model (unused)
NUM_EPISODES = 1000  # Episode limitation
EP_MAX_STEPS = 200  # Step limitation in an episode
# The number of test iters (with epsilon set to 0) to run every
# TEST_FREQUENCY episodes
NUM_TEST_EPS = 4
HIDDEN_NODES = 120
HIDDEN_NODES2 = 120
Learning_rate = 0.13
WHICH_OPITIMIZER = 'AdamOptimizer'
DOUBLE_Q = True


# record last 100 results
def last100result(last100):
    print('Last 100 episodes reward: ')
    print(pd.DataFrame(last100[:, 1], index=last100[
          :, 0].astype('int'), columns=['ep reward']))
    print('\nAve reward of the last 100 episodes: ', np.mean(last100[:, 1]))


def init(env, env_name):
    """
    Initialise any globals, e.g. the replay_buffer, epsilon, etc.
    return:
        state_dim: The length of the state vector for the env
        action_dim: The length of the action space, i.e. the number of actions

    NB: for discrete action envs such env_nameas the cartpole and mountain car, this
    function can be left unchanged.

    Hints for envs with continuous action spaces, e.g. "Pendulum-v0"
    1) you'll need to modify this function to discretise the action space and
    create a global dictionary mapping from action index to action (which you
    can use in `get_env_action()`)
    2) for Pendulum-v0 `env.action_space.low[0]` and `env.action_space.high[0]`
    are the limits of the action space.
    3) setting a global flag iscontinuous which you can use in `get_env_action()`
    might help in using the same code for discrete and (discretised) continuous
    action spaces
    """
    global replay_buffer, epsilon
    replay_buffer = []
    epsilon = INITIAL_EPSILON

    state_dim = env.observation_space.shape[0]

    if not iscontinuous:
        action_dim = env.action_space.n
    else:
        gap = 10
        action_dim = int(
            ((env.action_space.high[0] - env.action_space.low[0]) * gap) + 1)
        global action_dictionary

        action_dictionary = {}
        for i in range(action_dim):
            action_dictionary[i] = round(
                float(i / gap + env.action_space.low[0]), 1)

    return state_dim, action_dim


def get_network(state_dim, action_dim, hidden_nodes=HIDDEN_NODES):
    """Define the neural network used to approximate the q-function

    The suggested structure is to have each output node represent a Q value for
    one action. e.g. for cartpole there will be two output nodes.

    Hints:
    1) Given how q-values are used within RL, is it necessary to have output
    activation functions?
    2) You will set `target_in` in `get_train_batch` further down. Probably best
    to implement that before implementing the loss (there are further hints there)
    """
    hidden_nodes = HIDDEN_NODES
    action_in = tf.placeholder("float", [None, action_dim])  # one hot

    # q value for the target network for the state, action taken
    target_in = tf.placeholder("float", [None])

    _, _, _, _, _, _, q_values, state_in = build_network(
        state_dim, action_dim, hidden_nodes)

    q_selected_action = \
        tf.reduce_sum(tf.multiply(q_values, action_in),
                      reduction_indices=1)  # [?, 1] one action per sample

    # TO IMPLEMENT: loss function
    # should only be one line, if target_in is implemented correctly
    loss = tf.reduce_mean(tf.square(tf.subtract(target_in, q_selected_action)))
    gs = tf.Variable(0, dtype=tf.int64)

    if WHICH_OPITIMIZER == 'AdagradDAOptimizer':
        optimise_step = tf.train.AdagradDAOptimizer(
            Learning_rate, global_step=gs).minimize(loss)
    elif WHICH_OPITIMIZER == 'AdamOptimizer':
        optimise_step = tf.train.AdamOptimizer(Learning_rate).minimize(loss)
    elif WHICH_OPITIMIZER == 'RMSPropOptimizer':
        optimise_step = tf.train.RMSPropOptimizer(Learning_rate).minimize(loss)

    train_loss_summary_op = tf.summary.scalar("TrainingLoss", loss)
    return state_in, action_in, target_in, q_values, q_selected_action, \
        loss, optimise_step, train_loss_summary_op

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# New function, the constructure of the network


def build_network(state_dim, action_dim, hidden_nodes=HIDDEN_NODES):

    hidden_nodes = HIDDEN_NODES

    state_in = tf.placeholder("float", [None, state_dim])

    def weights(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # fisrt layer
    w1 = weights([state_dim, hidden_nodes])
    b1 = bias([hidden_nodes])
    logits1 = tf.matmul(state_in, w1) + b1  # [?, 4] mul [4, 16] is [?, 16]
    logits1_relu = tf.nn.tanh(logits1)  # [?, 16]

    # second layer
    hidden2 = HIDDEN_NODES2
    w2 = weights([hidden_nodes, hidden2])  # [?, 16] mul [16, 32] is [?, 32]
    b2 = bias([hidden2])
    logits2 = tf.matmul(logits1_relu, w2) + b2
    logits2_relu = tf.nn.relu(logits2)

    # full-connected layer
    w = weights([hidden2, action_dim])  # [?, 32]
    b = bias([action_dim])
    q_values = tf.matmul(logits2_relu, w) + b  # [?, action_dim]

    return w1, b1, w2, b2, w, b, q_values, state_in
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def init_session():
    global session, writer
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    # Setup Logging
    logdir = "tensorboard/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, session.graph)


def get_action(state, state_in, q_values, epsilon, test_mode, action_dim):
    Q_estimates = q_values.eval(feed_dict={state_in: [state]})[0]
    epsilon_to_use = 0.0 if test_mode else epsilon
    if random.random() < epsilon_to_use:
        action = random.randint(0, action_dim - 1)
    else:
        action = np.argmax(Q_estimates)
    return action


def get_env_action(action):
    """
    Modify for continous action spaces that you have discretised, see hints in
    `init()`
    """
    if not iscontinuous:
        return action
    else:
        lis = [action_dictionary[action]]
        return lis


def update_replay_buffer(replay_buffer, state, action, reward, next_state, done,
                         action_dim):
    """
    Update the replay buffer with provided input in the form:
    (state, one_hot_action, reward, next_state, done)

    Hint: the minibatch passed to do_train_step is one entry (randomly sampled)
    from the replay_buffer
    """
    # TO IMPLEMENT: append to the replay_buffer
    # ensure the action is encoded one hot

    # Action is a number waiting for one hot.
    one_hot_action = np.zeros((action_dim, ))
    one_hot_action[action] = 1
    # append to buffer
    replay_buffer.append((state, one_hot_action, reward, next_state, done))
    # Ensure replay_buffer doesn't grow larger than REPLAY_SIZE
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer.pop(0)
    return None


def do_train_step(replay_buffer, state_in, action_in, target_in,
                  q_values, q_selected_action, loss, optimise_step,
                  train_loss_summary_op, batch_presentations_count):

    target_batch, state_batch, action_batch = \
        get_train_batch(q_values, state_in, replay_buffer)

    global weights_list
    summary, _, weights_list = session.run([train_loss_summary_op, optimise_step, tf.trainable_variables()], feed_dict={
        target_in: target_batch,  # [None]
        state_in: state_batch,  # [None, state_dim]
        action_in: action_batch  # [None, action_dim]
    })
    writer.add_summary(summary, batch_presentations_count)


def get_train_batch(q_values, state_in, replay_buffer):
    """
    Generate Batch samples for training by sampling the replay buffer"
    Batches values are suggested to be the following;
        state_batch: Batch of state values
        action_batch: Batch of action values
        target_batch: Target batch for (s,a) pair i.e. one application
            of the bellman update rule.

    return:
        target_batch, state_batch, action_batch

    Hints:
    1) To calculate the target batch values, you will need to use the
    q_values for the next_state for each entry in the batch.
    2) The target value, combined with your loss defined in `get_network()` should
    reflect the equation in the middle of slide 12 of Deep RL 1 Lecture
    notes here: https://webcms3.cse.unsw.edu.au/COMP9444/17s2/resources/12494
    """
    minibatch = random.sample(replay_buffer, BATCH_SIZE)

    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    target_batch = []
    Q_value_batch = q_values.eval(feed_dict={
        state_in: next_state_batch
    })

    if DOUBLE_Q:
        action_Q_value_batch_one_hot = np.zeros((BATCH_SIZE, action_dim))
        Q_value_batch_max_index = np.argmax(Q_value_batch, axis=1)
        # print(action_Q_value_batch_one_hot)
        target_q_batch = q_values2.eval(feed_dict={
            state_in2: next_state_batch
        })

    for i in range(0, BATCH_SIZE):
        sample_is_done = minibatch[i][4]
        if sample_is_done:
            target_batch.append(reward_batch[i])
        else:
            # TO IMPLEMENT: set the target_val to the correct Q value update
            if DOUBLE_Q:
                action_Q_value_batch_one_hot[i, Q_value_batch_max_index[i]] = 1
                QQ = np.sum(np.multiply(target_q_batch[
                            i], action_Q_value_batch_one_hot[i]))
                target_val = reward_batch[i] + GAMMA * QQ
            else:
                target_val = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])
            target_batch.append(target_val)
    # print(action_Q_value_batch_one_hot)
    return target_batch, state_batch, action_batch


def qtrain(env, state_dim, action_dim,
           state_in, action_in, target_in, q_values, q_selected_action,
           loss, optimise_step, train_loss_summary_op,
           num_episodes=NUM_EPISODES, ep_max_steps=EP_MAX_STEPS,
           test_frequency=TEST_FREQUENCY, num_test_eps=NUM_TEST_EPS,
           final_epsilon=FINAL_EPSILON, epsilon_decay_steps=EPSILON_DECAY_STEPS,
           force_test_mode=False, render=False):
    global epsilon
    num_episodes = NUM_EPISODES
    ep_max_steps = EP_MAX_STEPS
    # print(ep_max_steps)

    # Record the number of times we do a training batch, take a step, and
    # the total_reward across all eps
    batch_presentations_count = total_steps = total_reward = 0
    last100 = []

    for episode in range(num_episodes):
        # initialize task
        state = env.reset()
        # if episode > 400:
        #     render = True
        if render:
            env.render()

        # Update epsilon once per episode - exp decaying
        epsilon -= (epsilon - final_epsilon) / epsilon_decay_steps

        # in test mode we set epsilon to 0
        test_mode = force_test_mode or \
            ((episode % test_frequency) < num_test_eps and
             episode > num_test_eps
             )
        if test_mode:
            print("Test mode (epsilon set to 0.0)")

        ep_reward = 0
        for step in range(ep_max_steps):
            total_steps += 1

            # get an action and take a step in the environment
            action = get_action(state, state_in, q_values, epsilon, test_mode,
                                action_dim)
            env_action = get_env_action(action)
            next_state, reward, done, _ = env.step(env_action)

            ep_reward += reward

            # display the updated environment
            if render:
                env.render()  # comment this line to possibly reduce training time

            # add the s,a,r,s' samples to the replay_buffer
            update_replay_buffer(replay_buffer, state, action, reward,
                                 next_state, done, action_dim)
            state = next_state

            # perform a training step if the replay_buffer has a batch worth of
            # samples

            if (len(replay_buffer) > BATCH_SIZE):
                do_train_step(replay_buffer, state_in, action_in, target_in,
                              q_values, q_selected_action, loss, optimise_step,
                              train_loss_summary_op, batch_presentations_count)
                batch_presentations_count += 1

            if done:
                break
        if DOUBLE_Q:
            if episode % target_network_update_gap == 0 and (len(replay_buffer) > BATCH_SIZE):
                assign_list = [tf.assign(w1t, weights_list[0]), tf.assign(b1t, weights_list[1]),
                               tf.assign(w2t, weights_list[2]), tf.assign(
                                   b2t, weights_list[3]),
                               tf.assign(wt, weights_list[4]), tf.assign(bt, weights_list[5])]
                session.run(assign_list)
                # print('good')
        total_reward += ep_reward
        test_or_train = "test" if test_mode else "train"
        print("end {0} episode {1}, ep reward: {2}, ave reward: {3}, \
            Batch presentations: {4}, epsilon: {5}".format(
            test_or_train, episode, ep_reward, total_reward / (episode + 1),
            batch_presentations_count, epsilon
        ))

        # added to record last 100 results
        if not (episode < (num_episodes - 100)):
            last100.append([episode, ep_reward])
    last100 = np.array(last100)
    last100result(last100)


def setup():
    global DOUBLE_Q
    global GAMMA
    global NUM_EPISODES
    global HIDDEN_NODES
    global HIDDEN_NODES2
    global Learning_rate
    global WHICH_OPITIMIZER
    global EP_MAX_STEPS
    global target_network_update_gap

    global iscontinuous

    # default_env_name = 'CartPole-v0'
    default_env_name = 'MountainCar-v0'
    # default_env_name = 'Pendulum-v0'

    # if env_name provided as cmd line arg, then use that
    env_name = sys.argv[1] if len(sys.argv) > 1 else default_env_name

    if env_name == 'CartPole-v0':
        """
        Hyper Parameters
        """
        iscontinuous = False
        GAMMA = 0.90  # discount factor for target Q
        NUM_EPISODES = 600
        HIDDEN_NODES = 40
        HIDDEN_NODES2 = 20
        Learning_rate = 0.1
        WHICH_OPITIMIZER = 'AdagradDAOptimizer'
        DOUBLE_Q = True
        target_network_update_gap = 2

    elif env_name == 'MountainCar-v0':
        iscontinuous = False
        GAMMA = 0.99
        NUM_EPISODES = 230
        HIDDEN_NODES = 120
        HIDDEN_NODES2 = 120
        Learning_rate = 0.009
        WHICH_OPITIMIZER = 'AdamOptimizer'
        DOUBLE_Q = True
        target_network_update_gap = 2

    elif env_name == 'Pendulum-v0':

        iscontinuous = True
        GAMMA = 0.99
        NUM_EPISODES = 230
        HIDDEN_NODES = 80
        HIDDEN_NODES2 = 80
        Learning_rate = 0.01
        WHICH_OPITIMIZER = 'AdamOptimizer'
        DOUBLE_Q = True
        EP_MAX_STEPS = 99999999
        target_network_update_gap = 2

    env = gym.make(env_name)
    global action_dim
    state_dim, action_dim = init(env, env_name)
    network_vars = get_network(state_dim, action_dim)
    # ++++++++++++++++++++++++++
    if DOUBLE_Q:
        global w1t, b1t, w2t, b2t, wt, bt, q_values2, state_in2
        w1t, b1t, w2t, b2t, wt, bt, q_values2, state_in2 = build_network(
            state_dim, action_dim)
    # ++++++++++++++++++++++++++
    init_session()
    return env, state_dim, action_dim, network_vars


def main():
    env, state_dim, action_dim, network_vars = setup()
    # print(GAMMA, NUM_EPISODES, HIDDEN_NODES, HIDDEN_NODES2, Learning_rate, WHICH_OPITIMIZER, DOUBLE_Q)
    qtrain(env, state_dim, action_dim, *network_vars, render=False)


if __name__ == "__main__":
    main()


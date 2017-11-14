import numpy as np
import tensorflow as tf
import logz
import os
import time
import inspect
from MASimulator import MASimulator as Simulator
import random
#============================================================================================#
# Utilities
#============================================================================================#

def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None
        ):
    Layer = {}
    with tf.variable_scope(scope):
        for i in range(n_layers):
            if i == 0:
                Layer[i] = tf.layers.dense(inputs = input_placeholder, 
                                        units = size,
                                        activation = activation,
                                        use_bias = True)
            else:
                Layer[i] = tf.layers.dense(inputs = Layer[i-1], 
                                        units = size,
                                        activation = activation,
                                        use_bias = True)
            
        Layer['logits'] = tf.layers.dense(inputs = Layer[n_layers-1], 
                                units = output_size,
                                activation = output_activation,
                                use_bias = True)
    return Layer['logits']

def pathlength(path):
    return len(path["reward"])

#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_MAPG(exp_name='',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             learning_rate=5e-3, 
             logdir=None, 
             normalize_advantages=True,
             seed=101,
             # network arguments
             n_layers=1,
             size=32
             ):
    #========================================================================================#
    # Logfile setup
    #========================================================================================#
    start = time.time()
    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_MAPG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)
    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    #========================================================================================#
    # Env setup
    #========================================================================================#
    env = Simulator(seed = 101,
                    N_prod = 2,
                    Tstamp = 20,
                    price = np.array([[1.5, 1.5]]),
                    costQ = np.array([[0.1, 0.1]]),
                    costInv = np.array([[0.2, 0.2]]),
                    costLastInv = np.array([[1, 1]]),
                    costBack = np.array([[0.5, 0.5]]) )
    # Observation and action sizes
    ob_dim = env.obs_dim()
    ac_dim = env.act_dim()
    nAgent = env.agent_dim()
    print('observation dimension is: ', ob_dim)
    print('action dimension is: ', ac_dim)
    #========================================================================================#
    # PG Networks
    #========================================================================================#
    

    def PGNet(sy_ob_no, sy_ac_na, sy_adv_n, agent_id):

        sy_mean = build_mlp(input_placeholder = sy_ob_no, 
                            output_size = ac_dim,
                            scope = str(seed) + 'MA_' + str(agent_id), 
                            n_layers = n_layers, 
                            size = size)

        sy_logstd = tf.Variable(tf.truncated_normal(shape = [1, ac_dim], stddev = 0.1), name = 'var_std' + str(agent_id))
        sy_sampled_ac = sy_mean + tf.multiply(tf.random_normal(shape = tf.shape(sy_mean)), tf.exp(sy_logstd))
        MVN_dist = tf.contrib.distributions.MultivariateNormalDiag(sy_mean, tf.exp(sy_logstd))
        sy_logprob_n = MVN_dist.log_prob(sy_ac_na)

        # Loss function for PG network
        loss = -tf.reduce_mean(tf.multiply(sy_logprob_n, sy_adv_n)) # Loss function that we'll differentiate to get the policy gradient.
        update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return sy_sampled_ac, loss, update_op

    #========================================================================================#
    # Critic network
    #========================================================================================#
    
    def CriticNet(sy_ob_critic, baseline_target, agent_id):
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_critic, 
                                output_size = 1, 
                                scope = str(seed) + "critic_" + str(agent_id),
                                n_layers = n_layers,
                                size = size))
    
        
        baseline_loss = tf.nn.l2_loss(baseline_target - baseline_prediction)
        baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(baseline_loss)
        return baseline_prediction, baseline_loss, baseline_update_op
    #========================================================================================#
    # Add networks in a loop
    #========================================================================================#
    for agent in range(nAgent):
        exec("sy_ob_no_%s = tf.placeholder(shape=[None, ob_dim], name='ob' + str(agent), dtype=tf.float32)"%(agent))
        exec("sy_ac_na_%s = tf.placeholder(shape=[None, ac_dim], name='ac' + str(agent), dtype=tf.float32)"%(agent))
        exec("sy_adv_n_%s = tf.placeholder(shape = [None], name = 'adv' + str(agent), dtype = tf.float32)"%(agent))
        exec("sy_ob_critic_%s = tf.placeholder(shape=[None, ob_dim + ac_dim * nAgent], name='critic_ob' + str(agent), dtype=tf.float32)"%(agent))
        exec("baseline_target_%s = tf.placeholder(shape = [None], name = 'baseline_target_qn' + str(agent), dtype = tf.float32)"%(agent))

        exec("sy_sampled_ac_%s, loss_%s, update_op_%s = PGNet(sy_ob_no_%s, sy_ac_na_%s, sy_adv_n_%s, agent)"%(agent, agent, agent, agent, agent, agent))
        exec("baseline_prediction_%s, baseline_loss_%s, baseline_update_op_%s = CriticNet(sy_ob_critic_%s, baseline_target_%s, agent)"%(agent, agent, agent, agent, agent))
    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#
    num_gpu = 0
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, device_count = {'GPU': num_gpu}) 
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    total_numpaths = 0
    
    for itr in range(n_iter):
        #========================#
        # Sampling
        #========================#
        randk = 0 + itr * seed
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        num_path = 0
        paths = []
        while True:
            demand = env.demandGenerator(mu = np.array([5, 5]), cov = np.diag(np.array([0.25, 0.25])), seed = randk)
            # Could be optimized by generating a batch demand vector. Fix this later!
            randk += 1
            ob = env.randomInitialStateGenerator()
            obs, acs, rewards = [], [], []
            last = False
            steps = 0

            while steps < env.Tstamp:
                if steps == env.Tstamp - 1:
                    last = True
                obs.append(ob.flatten())
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob})
                # ac = ac[0]
                acs.append(ac.flatten())
                ob, rew = env.step(ac, ob, demand[steps, :], last)
                rewards.append(rew)
                steps += 1

            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs)}
            paths.append(path)
            num_path += 1
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_numpaths += num_path
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        # print(ob_no.shape)
        # print(ac_na.shape)
        # print(path['reward'].shape)

        #========================#
        # Compute Q value
        #========================#
        q_n = np.concatenate([[np.npv((1/gamma - 1), path["reward"][i:]) for i in range(len(path["reward"]))] for path in paths])
        
        #========================#
        # Compute Baselines
        #========================#

        q_n_mean = q_n.mean()
        q_n_std = q_n.std()
        q_n = (q_n - q_n_mean)/q_n_std
        b_n = baseline_prediction
        adv_n_baseline = q_n - b_n

        #====================================#
        # Optimizing Neural Network Baseline
        #====================================#
        _, adv_n = sess.run([baseline_update_op, adv_n_baseline], feed_dict={baseline_target: q_n, sy_ob_critic: TODO})
        adv_n = adv_n * q_n_std + q_n_mean

        #====================================================================================#
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            adv_n = (adv_n - adv_n.mean())/adv_n.std()

        #====================================================================================#
        # Performing the Policy Update
        #====================================================================================#
        _, train_loss= sess.run([update_op, loss], feed_dict={sy_adv_n: adv_n, sy_ac_na: ac_na, sy_ob_no: ob_no})
        print("PG Network training loss: %.5f"%train_loss)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("NumPathsThisBatch", num_path)
        logz.log_tabular("NumPathsSoFar", total_numpaths)
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='SingleAgent')
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--n_iter', '-n', type=int, default=500)
    parser.add_argument('--batch_size', '-b', type=int, default=1280)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        train_MAPG(
            exp_name=args.exp_name,   
            n_iter=args.n_iter,
            gamma=args.discount,
            min_timesteps_per_batch=args.batch_size,
            learning_rate=args.learning_rate,
            reward_to_go=args.reward_to_go,
            logdir=os.path.join(logdir,'%d'%seed),
            normalize_advantages=not(args.dont_normalize_advantages),
            seed=seed,
            n_layers=args.n_layers,
            size=args.size
            )
        

if __name__ == "__main__":
    main()

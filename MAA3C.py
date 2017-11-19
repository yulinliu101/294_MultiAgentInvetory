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
        output_activation=None,
        scale = None
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
        if scale is not None:
            Layer['logits'] = tf.multiply(Layer['logits'], scale)
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
    nAgent = 2 # hard coded!
    env1 = Simulator(seed = 101,
                    N_agent = nAgent,
                    N_prod = 3,
                 Tstamp = 50,
                 costQ = np.array([[0.3, 0.3, 0.3]]),
                 costInv = np.array([[0.2, 0.2, 0.2]]),
                 costLastInv = np.array([[2, 2, 2]]),
                 costBack = np.array([[0.75, 0.75, 0.75]])  )

    env2 = Simulator(seed = 202,
                    N_agent = nAgent,
                    N_prod = 3,
                 Tstamp = 50,
                 costQ = np.array([[0.3, 0.3, 0.3]]),
                 costInv = np.array([[0.2, 0.2, 0.2]]),
                 costLastInv = np.array([[2, 2, 2]]),
                 costBack = np.array([[0.75, 0.75, 0.75]]) )
    # Observation and action sizes
    ob_dim = env1.obs_dim()
    ac_dim = env1.act_dim()
    
    print('observation dimension is: ', ob_dim)
    print('action dimension is: ', ac_dim)
    #========================================================================================#
    # PG Networks
    #========================================================================================#
    

    def PGNet(sy_ob_no, sy_ac_na, sy_adv_n, agent_id):

        sy_mean = build_mlp(input_placeholder = sy_ob_no, 
                            output_size = ac_dim[0] * ac_dim[1],
                            scope = str(seed) + 'MA_' + str(agent_id), 
                            n_layers = n_layers, 
                            output_activation = tf.sigmoid,
                            size = size,
                            scale = 10.)
        

        sy_logstd = tf.Variable(tf.truncated_normal(shape = [1, ac_dim[0] * ac_dim[1]], stddev = 0.1), name = 'var_std' + str(agent_id))
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
    
    sy_ob_no_1 = tf.placeholder(shape=[None, ob_dim[0]], name='ob' + str(1), dtype=tf.float32)
    sy_ac_na_1 = tf.placeholder(shape=[None, ac_dim[0] * ac_dim[1]], name='ac' + str(1), dtype=tf.float32)
    sy_adv_n_1 = tf.placeholder(shape = [None], name = 'adv' + str(1), dtype = tf.float32)
    sy_ob_critic_1 = tf.placeholder(shape=[None, ob_dim[0] + ac_dim[0] * ac_dim[1] * nAgent], name='critic_ob' + str(1), dtype=tf.float32)
    baseline_target_1 = tf.placeholder(shape = [None], name = 'baseline_target_qn' + str(1), dtype = tf.float32)

    sy_sampled_ac_1, loss_1, update_op_1 = PGNet(sy_ob_no_1, sy_ac_na_1, sy_adv_n_1, 1)
    baseline_prediction_1, baseline_loss_1, baseline_update_op_1 = CriticNet(sy_ob_critic_1, baseline_target_1, 1)

    sy_ob_no_2 = tf.placeholder(shape=[None, ob_dim[0]], name='ob' + str(2), dtype=tf.float32)
    sy_ac_na_2 = tf.placeholder(shape=[None, ac_dim[0] * ac_dim[1]], name='ac' + str(2), dtype=tf.float32)
    sy_adv_n_2 = tf.placeholder(shape = [None], name = 'adv' + str(2), dtype = tf.float32)
    sy_ob_critic_2 = tf.placeholder(shape=[None, ob_dim[0] + ac_dim[0] * ac_dim[1] * nAgent], name='critic_ob' + str(2), dtype=tf.float32)
    baseline_target_2 = tf.placeholder(shape = [None], name = 'baseline_target_qn' + str(2), dtype = tf.float32)

    sy_sampled_ac_2, loss_2, update_op_2 = PGNet(sy_ob_no_2, sy_ac_na_2, sy_adv_n_2, 2)
    baseline_prediction_2, baseline_loss_2, baseline_update_op_2 = CriticNet(sy_ob_critic_2, baseline_target_2, 2)

        # exec("sy_sampled_ac_%s, loss_%s, update_op_%s = PGNet(sy_ob_no_%s, sy_ac_na_%s, sy_adv_n_%s, agent)"%(agent, agent, agent, agent, agent, agent))
        # exec("baseline_prediction_%s, baseline_loss_%s, baseline_update_op_%s = CriticNet(sy_ob_critic_%s, baseline_target_%s, agent)"%(agent, agent, agent, agent, agent))
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
        randk1 = 0 + itr * seed
        randk2 = 12306 + itr * seed
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        num_path = 0
        paths1 = []
        paths2 = []
        while True:
            steps = 0
            last = False
            
            ob1 = env1.randomInitialStateGenerator()
            obs1, acs1, rewards1, criticObs1 = [], [], [], []
            
            ob2 = env2.randomInitialStateGenerator()
            obs2, acs2, rewards2, criticObs2 = [], [], [], []

            while steps < env1.Tstamp:
                if steps == env1.Tstamp - 1:
                    last = True

                obs1.append(ob1.flatten())
                obs2.append(ob2.flatten())

                ac1 = sess.run(sy_sampled_ac_1, feed_dict={sy_ob_no_1 : ob1})
                ac2 = sess.run(sy_sampled_ac_2, feed_dict={sy_ob_no_2 : ob2})
                acs1.append(ac1.flatten())
                acs2.append(ac2.flatten())

                criticObs1.append(np.append(np.append(ob1.flatten(), ac1.flatten()), ac2.flatten()).flatten())
                criticObs2.append(np.append(np.append(ob2.flatten(), ac2.flatten()), ac1.flatten()).flatten())

                actList = [ac1.reshape(-1, 2), ac2.reshape(-1, 2)]

                demand = env1.demandGenerator_p(actList,
                                                 M = np.array([10, 10, 10]).reshape(-1,1),
                                                 V = np.array([5,5,5]).reshape(-1,1),
                                                 sens = np.array([1.5, 1.5, 1.5]).reshape(-1,1),
                                                 cov = np.diag(np.array([0.1, 0.1, 0.1])),
                                                 seed = randk1)
                demand1 = demand[:,0]
                demand2 = demand[:,1]

                # demand2 = env2.demandGenerator_p(actList,
                #                                  M = np.array([3, 3, 3]).reshape(-1,1),
                #                                  V = np.array([5,5,5]).reshape(-1,1),
                #                                  sens = np.array([1, 1, 1]).reshape(-1,1),
                #                                  cov = np.diag(np.array([0.25, 0.25, 0.25])),
                #                                  seed = randk2)

                ob1, rew1 = env1.step(actList[0], ob1.flatten(), demand1, last)
                ob2, rew2 = env2.step(actList[1], ob2.flatten(), demand2, last)

                randk1 += 1
                randk2 += 1                
                
                rewards1.append(rew1)
                rewards2.append(rew2)
                steps += 1

            path1 = {"observation" : np.array(obs1), 
                    "reward" : np.array(rewards1), 
                    "action" : np.array(acs1),
                    "criticObservation": np.array(criticObs1)}

            path2 = {"observation" : np.array(obs2), 
                    "reward" : np.array(rewards2), 
                    "action" : np.array(acs2),
                    "criticObservation": np.array(criticObs2)}

            paths1.append(path1)
            paths2.append(path2)
            num_path += 1
            timesteps_this_batch += pathlength(path1)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_numpaths += num_path
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no1 = np.concatenate([path["observation"] for path in paths1])
        ac_na1 = np.concatenate([path["action"] for path in paths1])
        critic_ob_no1 = np.concatenate([path["criticObservation"] for path in paths1])

        ob_no2 = np.concatenate([path["observation"] for path in paths2])
        ac_na2 = np.concatenate([path["action"] for path in paths2])
        critic_ob_no2 = np.concatenate([path["criticObservation"] for path in paths2])
        # print(ob_no.shape)
        # print(ac_na.shape)
        # print(path['reward'].shape)

        #========================#
        # Compute Q value
        #========================#
        q_n1 = np.concatenate([[np.npv((1/gamma - 1), path["reward"][i:]) for i in range(len(path["reward"]))] for path in paths1])
        q_n2 = np.concatenate([[np.npv((1/gamma - 1), path["reward"][i:]) for i in range(len(path["reward"]))] for path in paths2])
        
        #========================#
        # Compute Baselines
        #========================#


        q_n_mean1 = q_n1.mean()
        q_n_std1 = q_n1.std()
        q_n1 = (q_n1 - q_n_mean1)/q_n_std1
        b_n1 = baseline_prediction_1
        adv_n_baseline1 = q_n1 - b_n1

        q_n_mean2 = q_n2.mean()
        q_n_std2 = q_n2.std()
        q_n2 = (q_n2 - q_n_mean2)/q_n_std2
        b_n2 = baseline_prediction_2
        adv_n_baseline2 = q_n2 - b_n2

        # if bootstrap:
        #     last_critic_ob_no1 = np.concatenate([path["criticObservation"] for path in paths1])
        #     lastFit1 = sess.run(baseline_prediction_1, 
        #                         feed_dict = {sy_ob_critic_1: critic_ob_no1[]})

        #====================================#
        # Optimizing Neural Network Baseline
        #====================================#
        _, adv_n1 = sess.run([baseline_update_op_1, adv_n_baseline1], feed_dict={baseline_target_1: q_n1, sy_ob_critic_1: critic_ob_no1})
        adv_n1 = adv_n1 * q_n_std1 + q_n_mean1

        _, adv_n2 = sess.run([baseline_update_op_2, adv_n_baseline2], feed_dict={baseline_target_2: q_n2, sy_ob_critic_2: critic_ob_no2})
        adv_n2 = adv_n2 * q_n_std2 + q_n_mean2

        #====================================================================================#
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            adv_n1 = (adv_n1 - adv_n1.mean())/adv_n1.std()
            adv_n2 = (adv_n2 - adv_n2.mean())/adv_n2.std()

        #====================================================================================#
        # Performing the Policy Update
        #====================================================================================#
        _, train_loss1 = sess.run([update_op_1, loss_1], feed_dict={sy_adv_n_1: adv_n1, sy_ac_na_1: ac_na1, sy_ob_no_1: ob_no1})
        _, train_loss2 = sess.run([update_op_2, loss_2], feed_dict={sy_adv_n_2: adv_n2, sy_ac_na_2: ac_na2, sy_ob_no_2: ob_no2})
        print("PG Network 1 training loss: %.5f"%train_loss1)
        print("PG Network 2 training loss: %.5f"%train_loss2)

        # Log diagnostics
        returns1 = np.array([path["reward"].sum() for path in paths1])
        returns2 = np.array([path["reward"].sum() for path in paths2])
        totalReturn = returns1 + returns2

        ep_lengths = [pathlength(path) for path in paths1]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        
        logz.log_tabular("AverageReturn1", np.mean(returns1))
        logz.log_tabular("StdReturn1", np.std(returns1))
        logz.log_tabular("MaxReturn1", np.max(returns1))
        logz.log_tabular("MinReturn1", np.min(returns1))
        logz.log_tabular("AverageReturn2", np.mean(returns2))
        logz.log_tabular("StdReturn2", np.std(returns2))
        logz.log_tabular("MaxReturn2", np.max(returns2))
        logz.log_tabular("MinReturn2", np.min(returns2))

        logz.log_tabular("AverageTotalReturn", np.mean(totalReturn))
        logz.log_tabular("StdReturn", np.std(totalReturn))
        logz.log_tabular("MaxReturn", np.max(totalReturn))
        logz.log_tabular("MinReturn", np.min(totalReturn))

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
    parser.add_argument('--exp_name', type=str, default='MultiAgentWTime')
    parser.add_argument('--discount', type=float, default=0.75)
    parser.add_argument('--n_iter', '-n', type=int, default=2000)
    parser.add_argument('--batch_size', '-b', type=int, default=2560)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
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
            logdir=os.path.join(logdir,'%d'%seed),
            normalize_advantages=not(args.dont_normalize_advantages),
            seed=seed,
            n_layers=args.n_layers,
            size=args.size
            )
        

if __name__ == "__main__":
    main()

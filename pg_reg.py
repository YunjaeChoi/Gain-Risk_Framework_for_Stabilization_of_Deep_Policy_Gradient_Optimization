import numpy as np
import tensorflow as tf
import gym
import pybullet_envs
import time
from utils.tf_utils import *
from utils.math_utils import *
from utils.logx import EpochLogger
import model
from lr_scheduler import LinearLR
from replay_buffer import AdvantageBuffer as Buffer

def pg_reg(env_fn, actor_critic=model.mlp_actor_critic, seed=0,
        epochs=50, steps_per_epoch=5000, max_ep_len=1000,
        lam=0.97, gamma=0.99,
        alpha_ratio=5.0, alpha_risk=5.0,
        beta_ratio=0.2, beta_risk=1.0,
        initial_pi_lr=5e-5, min_pi_lr=5e-6,
        linear_dec_pi_lr=1e-7, train_pi_iters=100,
        max_grad_norm=0.5,
        vf_lr=1e-3, train_v_iters=100,
        logger_kwargs=dict(), save_freq=10):

    logger = EpochLogger(**logger_kwargs)
    local_vars_dict = locals().copy()
    local_vars_dict.pop('logger', None)
    local_vars_dict.pop('logger_kwargs', None)
    logger.save_config(local_vars_dict)

    #seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Inputs to computation graph
    x_ph, a_dn_ph, next_x_ph = placeholders_from_spaces(
        env.observation_space, env.action_space, env.observation_space)
    adv_ph, logp_old_ph, v_ph = placeholders(None, None, None)
    ret_ph, rew_ph = placeholders(None, None)
    act_mu_dn_ph, act_std_dn_ph = placeholders_from_spaces(env.action_space, env.action_space)

    # Main outputs from computation graph
    pi, pi_dn, logp_pi, logp, act_mu_dn, act_std_dn, v, next_v = actor_critic(x_ph, a_dn_ph, next_x_ph)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    vf_input_phs = [x_ph, ret_ph]
    obs_phs = [x_ph, next_x_ph]
    pi_input_phs = [x_ph, a_dn_ph, act_mu_dn_ph, act_std_dn_ph, logp_old_ph, adv_ph]

    get_action_ops = [pi, pi_dn, logp_pi, act_mu_dn, act_std_dn]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch)
    buf = Buffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d \n'%var_counts)

    # objectives
    logp_diff = logp - logp_old_ph
    adv_eta = logp_diff * adv_ph
    mean_adv_eta = tf.reduce_mean(adv_eta)
    action_gain = tf.math.maximum(adv_eta, 0.0)
    action_risk = tf.math.minimum(adv_eta, 0.0)
    abs_action_risk = tf.math.abs(action_risk)

    abs_logp_diff = tf.math.abs(logp_diff)

    softmax_abs_logp_diff = tf.nn.softmax(alpha_ratio * abs_logp_diff)
    smooth_max_abs_logp_diff = tf.math.reduce_sum(softmax_abs_logp_diff * abs_logp_diff)
    smooth_max_abs_logp_reg = smooth_max_abs_logp_diff**2

    softmax_abs_action_risk = tf.nn.softmax(alpha_risk * abs_action_risk)
    smooth_min_abs_action_risk = tf.math.reduce_sum(softmax_abs_action_risk * action_risk)
    smooth_min_abs_action_risk_reg = smooth_min_abs_action_risk**2

    regularization = beta_ratio * smooth_max_abs_logp_reg + beta_risk * smooth_min_abs_action_risk_reg

    pi_loss = -mean_adv_eta + regularization

    v_loss = tf.reduce_mean((ret_ph - v)**2)

    #Info
    sample_kl = tf.reduce_mean(logp_old_ph - logp)
    sample_ent = tf.reduce_mean(-logp)

    dn_ent = diag_normal_entropy(act_mu_dn, act_std_dn)
    mean_dn_ent = tf.reduce_mean(dn_ent)
    dn_ent_old = diag_normal_entropy(act_mu_dn_ph, act_std_dn_ph)
    dn_kl = diag_normal_kl_divergence(act_mu_dn_ph, act_std_dn_ph, act_mu_dn, act_std_dn)
    mean_dn_kl = tf.reduce_mean(dn_kl)

    #Info-others
    action_risk_bool = adv_eta < 0.0
    action_risk_ratio = tf.reduce_mean(tf.cast(action_risk_bool, action_gain.dtype))

    sum_action_gain = tf.math.reduce_sum(action_gain)
    sum_action_risk = tf.math.reduce_sum(action_risk)
    abs_sum_action_risk = tf.math.abs(sum_action_risk)
    risk_gain_ratio = sum_action_gain / (abs_sum_action_risk + 1e-8)

    mean_abs_logp_diff = tf.math.reduce_mean(abs_logp_diff)
    max_abs_logp_diff = tf.math.reduce_max(abs_logp_diff)

    max_action_gain = tf.math.reduce_max(action_gain)
    min_action_risk = tf.math.reduce_min(action_risk)

    #pi lr scheduler
    pi_lr = LinearLR(
        initial_pi_lr, min_pi_lr, linear_dec_pi_lr
    )

    # Optimizers
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr.lr_ph)
    grads_and_var = pi_optimizer.compute_gradients(pi_loss, get_vars(scope='pi'))
    grads_and_var = clip_grads_by_global_norm(grads_and_var, max_grad_norm)
    train_pi = pi_optimizer.apply_gradients(grads_and_var)

    vf_optimizer = tf.train.AdamOptimizer(learning_rate=vf_lr)
    grads_and_var = vf_optimizer.compute_gradients(v_loss, get_vars(scope='v'))
    grads_and_var = clip_grads_by_global_norm(grads_and_var, max_grad_norm)
    train_v = vf_optimizer.apply_gradients(grads_and_var)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    #sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(
        sess,
        inputs={'x': x_ph},
        outputs={
            'pi': pi,
            'v': v
        }
    )


    def update(epoch, logger):
        vf_inputs_data = buf.get_vf_update_inputs()
        vf_inputs = {k:v for k,v in zip(vf_input_phs, vf_inputs_data)}
        v_old = sess.run(v, feed_dict=vf_inputs)
        v_l_old = sess.run(v_loss, feed_dict=vf_inputs)
        for i in range(train_v_iters):
            sess.run(train_v, feed_dict=vf_inputs)

        v_l_new = sess.run(v_loss, feed_dict=vf_inputs)

        obs_data = buf.get_observations()
        obs_dict = {k:v for k,v in zip(obs_phs, obs_data)}
        v_new, next_v_new = sess.run([v, next_v], feed_dict=obs_dict)
        buf.store_vf_outputs(v_new, next_v_new)
        logger.store(VVals=v_new)

        pi_inputs_data = buf.get_pi_update_inputs()
        pi_inputs = {k:v for k,v in zip(pi_input_phs, pi_inputs_data)}

        sent, dent_old, pi_l_old = sess.run([
            sample_ent, mean_dn_ent, pi_loss], feed_dict=pi_inputs)
        logger.store(SampleEntropy=sent,
                     DiagNormalEntropy=dent_old,
                     LossPi=pi_l_old,
                     LossV=v_l_old,
                     DeltaLossV=(v_l_new - v_l_old))

        if epoch == 0:
            pi_lr.reset()
        else:
            pi_lr.update()
        pi_inputs[pi_lr.lr_ph] = pi_lr.lr
        logger.store(PiLr=pi_lr.lr)

        for i in range(1, train_pi_iters+1):
            sess.run(train_pi, feed_dict=pi_inputs)

        maxag, minar, meanalpd, maxalpd, rgr, arr = sess.run([
            max_action_gain, min_action_risk,
            mean_abs_logp_diff, max_abs_logp_diff,
            risk_gain_ratio, action_risk_ratio], feed_dict=pi_inputs)

        logger.store(MaxActionGain=maxag, MinActionRisk=minar,
                     MeanAbsLogpDiff=meanalpd, MaxAbsLogpDiff=maxalpd,
                     RiskGainRatio=rgr, ActionRiskRatio=arr)

        dent_new, pi_l_new, skl, dkl = sess.run([
            mean_dn_ent, pi_loss, sample_kl, mean_dn_kl], feed_dict=pi_inputs)
        logger.store(StopIter=i, DiagNormalKL=dkl, SampleKL=skl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaDiagNormalEntropy=(dent_new - dent_old))

        #opt results log
        #adv, ae, logpd = sess.run([adv_ph, adv_eta, logp_diff], feed_dict=pi_inputs)
        #opt_results_logger.store(Adv=adv, AdvEta=ae, LogpDiff=logpd)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, a_dn, logp_t, a_mu_dn, a_std_dn = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})

            next_o, r, d, _ = env.step(a[0])

            buf.store(o, a, a_dn, a_mu_dn, a_std_dn, next_o, logp_t, r, d)

            o = next_o
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                # if trajectory didn't reach terminal state, bootstrap value target
                if d:
                    last_val = 0.0
                else:
                    last_val = sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val=last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    #print('traj finished')
                else:
                    logger.log('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len))
                    # no value function update for cut off trajectory
                    buf.store_cut_off_ep_len(ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        update(epoch, logger)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('MaxActionGain', average_only=True)
        logger.log_tabular('MinActionRisk', average_only=True)
        logger.log_tabular('MeanAbsLogpDiff', average_only=True)
        logger.log_tabular('MaxAbsLogpDiff', average_only=True)
        logger.log_tabular('RiskGainRatio', average_only=True)
        logger.log_tabular('ActionRiskRatio', average_only=True)
        logger.log_tabular('PiLr', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('DiagNormalEntropy', average_only=True)
        logger.log_tabular('DeltaDiagNormalEntropy', average_only=True)
        logger.log_tabular('DiagNormalKL', average_only=True)
        logger.log_tabular('SampleEntropy', average_only=True)
        logger.log_tabular('SampleKL', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

        #Save
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_model()
            #opt_results_logger.save_csv('opt_results.csv')

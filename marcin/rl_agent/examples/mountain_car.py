import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import gym

import rl_agent as rl

import tensorflow as tf


def on_step_end(agent, reward, observation, done, action, extra_params):
    plotter = extra_params

    if agent.total_step % 1000 == 0:
        print()
        print('total_step', agent.total_step,
            'e_rand', agent._epsilon_random)
        print('EPISODE', agent.completed_episodes, agent.get_avg_reward(50))

    if plotter is not None:
        plotter.process(agent.logger, agent.total_step)
        if agent.total_step >= agent.nb_rand_steps:
            res = plotter.conditional_plot(agent.logger, agent.total_step)
            if res:
                plt.pause(0.001)
                pass

    if done:
        print('espiode finished after iteration', agent.step)





def main():
    
    args = rl.util.parse_common_args()
    rl.util.try_freeze_random_seeds(args.seed, args.reproducible)
    logger = None
    if args.logfile is not None or args.plot:
        logger = rl.util.Logger()

    
    #
    #   Init plotter
    #
    plotter = None
    if args.plot:
        fig = plt.figure()
        plotter = rl.util.Plotter(
            realtime_plotting=True, plot_every=1000, disp_len=1000,
            ax_qmax_wf=fig.add_subplot(2,4,1, projection='3d'),
            ax_qmax_im=fig.add_subplot(2,4,2),
            ax_policy=fig.add_subplot(2,4,3),
            ax_trajectory=fig.add_subplot(2,4,4),
            ax_stats=None,
            ax_memory=None,
            ax_q_series=None,
            ax_reward=fig.add_subplot(2,1,2)  )
        


    env = gym.make('MountainCar-v0').env
    env.seed(args.seed)

    q_model = tf.keras.models.Sequential()
    q_model.add(tf.keras.layers.Dense(units=256, activation='relu', input_dim=2))
    q_model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    q_model.add(tf.keras.layers.Dense(units=3, activation='linear'))
    q_model.compile(loss='mse', 
        optimizer=tf.keras.optimizers.RMSprop(lr=0.00025))

    agent = rl.Agent(
        state_space=env.observation_space,
        action_space=env.action_space,
        discount=0.99,
        expl_start=False,
        nb_rand_steps=0,
        e_rand_start=1.0,
        e_rand_target=0.1,
        e_rand_decay=1/10000,
        mem_size_max=10000,
        mem_batch_size=64,
        mem_enable_pmr=False,
        q_fun_approx=rl.TilesApproximator(
            step_size=0.3,
            num_tillings=8,
            init_val=0),

        logger=logger)

    agent.register_callback('on_step_end', on_step_end, extra_params=plotter)


    #
    #   Run application
    #
    try:
        rl.train_agent(env=env, agent=agent, 
            total_steps=10000, target_avg_reward=-200)
    finally:
        if args.logfile is not None:
            logger.save(args.logfile)
            print('Log saved')

    fp, ws, st, act, rew, done = agent.get_fingerprint()
    print('FINGERPRINT:', fp)
    print('  wegight sum:', ws)
    print('  st, act, rew, done:', st, act, rew, done)
    
    if plotter is not None:
        plt.show()


if __name__ == '__main__':
    main()
    

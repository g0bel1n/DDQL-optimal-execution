Quickstart Guide
==================

Training

.. code-block:: python

   from ddql_optimal_execution import DDQL, MarketEnvironnement, Trainer


   #Create environnement
   env = MarketEnvironnement(initial_inventory=500, multi_episodes=True, QV=True, Volume=True, data_path='../data/train')
   #Create agent
   agent = DDQL(state_size=env.state_size, initial_budget=env.initial_inventory, horizon=env.horizon)

   #Create experience replay for storing experience
   trainer = Trainer(agent, env, capacity=10000)

   trainer.fill_exp_replay(max_steps=10000)

   trainer.pretrain(max_steps=100, batch_size=128)

   trainer.train(max_steps=1000, batch_size=128)


Testing vs TWAP

.. code-block:: python

   from ddql_optimal_execution import TWAP

   #Create environnement
   env = MarketEnvironnement(initial_inventory=500, multi_episodes=True, QV=True, Volume=True, data_path='../data/test')

   #Create agent
   agent = DDQL(state_size=env.state_size, initial_budget=env.initial_inventory, horizon=env.horizon)
   twap = TWAP(initial_inventory=env.initial_inventory, horizon=env.horizon)

   pnl_twap = []
   pnl_ddql = []

   n_episodes = min(len(test_env.historical_data_series), 100)

   random_ep = np.random.choice(np.arange(n_episodes), size=n_episodes, replace=True)

   for ep in random_ep:
      test_env.swap_episode(ep)
      _pnl_twap = [0]
      while not test_env.done:
         current_state = test_env.state.copy()
         action = twap(current_state)
         _ = test_env.step(action)
         
      pnl_twap.append(test_env.pnl_for_episode + [test_env.state['Price']*test_env.state['inventory'] - test_env.quadratic_penalty_coefficient*(test_env.state['inventory']/test_env.initial_inventory)**2 / test_env.horizon])

      test_env.reset()
      
      _pnl_ddql = [0]
      while not test_env.done:
         current_state = test_env.state.copy()
         action = trainer.agent(current_state)
         _ = test_env.step(action)
      pnl_ddql.append(test_env.pnl_for_episode + [test_env.state['Price']*test_env.state['inventory'] - test_env.quadratic_penalty_coefficient*(test_env.state['inventory']/test_env.initial_inventory)**2 / test_env.horizon])pnl_twap = []


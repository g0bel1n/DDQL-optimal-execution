Quickstart Guide
==================

.. code-block:: python

    from ddql_optimal_execution import DDQL, MarketEnvironnement, ExperienceReplay, TWAP

   #Create environnement
   env = MarketEnvironnement(initial_inventory=500, multi_episodes=True)

   #Create agent
   agent = DDQL(state_size=env.state_size, initial_budget=env.initial_inventory, horizon=env.horizon)

   #Create experience replay for storing experience
   exp_replay = ExperienceReplay(state_size=env.state_size, capacity=1000)

   #Train agent on every detected episode
   for episode in range(len(env.historical_data_series)):
      env.swap_episode(episode)
      while not env.done:
         current_state = env.get_state()
         action = agent(current_state)
         reward = env.step(action)

         distance2horizon = env.horizon - env.state["period"]
         exp_replay.push(current_state, action, reward, env.state.copy(), distance2horizon if distance2horizon <=1 else 2)
      
      agent.learn(exp_replay.sample(128)) #Learn from experience replay
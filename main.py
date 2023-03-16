from world import *

simul = world(world_population=1000, input_l=3, invest_disc=10, sales_disc=10,
              simulation_len=500, N_episodes=3, start_money_loc=100)

simul.run_world()
simul.show_data()
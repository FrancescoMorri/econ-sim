from world import *

pop = 1000
sim = 300
eps = 100

simul = world(world_population=pop, input_l=3, invest_disc=10, sales_disc=10,
              simulation_len=sim, N_episodes=eps, start_money_loc=100)

simul.run_world(f"pop{pop}-len{sim}-eps{eps}")
simul.save_data(f"data/pop{pop}-len{sim}-eps{eps}")
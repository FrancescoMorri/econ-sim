from agents import jhonnies
import numpy as np
import pickle as pk

class world():

    def __init__(self, world_population:int, input_l:int, invest_disc:int, sales_disc:int,
                 simulation_len:int, N_episodes:int, start_money_loc=100, start_money_scale=10,
                 cost_loc=60, epsilon=1, beta=10, delta=20, income_scaling=5, noise_scale=1.5,
                 top_percent = 0.15) -> None:
        '''
        Stuff to do:
        - initialize the population with:
            - initial money
            - cost of living
            - discretization of input and output
        - initialize parameters for price update and noise signal
        '''
        # PEOPLE INIT
        self.peoples = None
        self.world_population = world_population
        for _ in range(world_population):
            money = int(np.random.normal(loc=start_money_loc, scale=start_money_scale))
            # making sure that the cost of living is smaller than the initial money they have
            cost_of_living = 0
            #while cost_of_living > money:
            #    cost_of_living = int(money - np.random.normal(loc=(start_money_loc-cost_loc), scale=start_money_scale))
            if self.peoples is None:
                self.peoples = np.array(jhonnies(li=input_l, loi=invest_disc, loa=sales_disc, M=money, C=cost_of_living))
            self.peoples = np.append(self.peoples, jhonnies(li=input_l, loi=invest_disc, loa=sales_disc, M=money, C=cost_of_living))
        
        # PARS INIT
        self.input_len = input_l
        self.invest_disc = invest_disc
        self.sales_disc = sales_disc
        self.epsilon = epsilon
        self.beta = beta
        self.delta = delta
        self.start_price = start_money_loc//10
        self.asset_price = start_money_loc//10
        self.income_scale = income_scaling
        self.sold = 0
        self.bought = 0
        self.previous_noise = 0
        self.noise_scale = noise_scale
        self.percent = top_percent
        self.simul_len = simulation_len
        self.eps = N_episodes
        self.removed_people = 0

        # SAVING PARS
        self.noises = np.zeros((N_episodes, simulation_len))
        self.prices = np.zeros((N_episodes, simulation_len))
        self.general_liquidity = np.zeros((N_episodes, simulation_len))
        self.general_assets = np.zeros((N_episodes, simulation_len))
        self.avg_buying = np.zeros((N_episodes, simulation_len))
        self.avg_selling = np.zeros((N_episodes, simulation_len))
        self.avg_age = []
        self.hist_removed_people = []
    
    def update_prices(self):
        '''
        Update the price following:
        - if `sold > bought+delta` -> `price -= epsilon`
        - if `bought > sold+delta` -> `price += epsilon`
        - else -> `price +/-= epsilon/beta` with 50/50 chance
        '''
        if self.sold > self.bought + self.delta:
            self.asset_price -= self.epsilon
        elif self.sold < self.bought - self.delta:
            self.asset_price += self.epsilon
        else:
            if np.random.rand() < 0.5:
                self.asset_price += self.epsilon/self.beta
            else:
                self.asset_price -= self.epsilon/self.beta
        
        if self.asset_price <= 0:
            self.asset_price = 1
    

    def agents_step(self, x, t, e):
        '''
        Get actions and update liquidity/assets
        '''
        # FOR NOW NO INCOME, NO COST
        #for p in self.peoples:
        #    p.liquidity += int((p.cost_of_living//self.income_scale)*(np.sin(t)+1))
        #    if p.liquidity < p.cost_of_living:
        #        print("Lost someone to poor to pay :(")
        #        self.peoples.remove(p)
        self.bought = 0
        self.sold = 0
        for p in self.peoples:
            # if the agent has no money and no assets he's out
            if p.liquidity == 0 and p.assets == 0:
                #print("Very poor guy removed")
                self.removed_people += 1
                self.peoples = self.peoples[self.peoples != p]
                continue
            # otherwise it can buy and/or sell things
            inv, sale = p.action(x)
            self.avg_buying[e,t] += inv
            self.avg_selling[e,t] += sale
            # possibly buy stocks at current price and decrease its liquidity
            stocks = inv//self.asset_price
            p.assets += stocks
            p.liquidity -= inv
            self.bought += stocks
            # and sell assets at current price, decreasing its assets and increasing its liquidity
            p.liquidity += self.asset_price*sale
            p.assets -= sale
            self.sold += sale
        
        self.avg_buying[e,t] /= len(self.peoples)
        self.avg_selling[e,t] /= len(self.peoples)
    
    def get_input(self, t, e):
        x = np.zeros(self.input_len)
        x[0] = self.asset_price
        x[1] = self.previous_noise + (np.random.rand()*2 - 1)*self.noise_scale
        self.previous_noise = x[1]
        x[2] = np.mean([p.liquidity for p in self.peoples])
        self.noises[e,t] = x[1]
        self.prices[e,t] = x[0]
        self.general_liquidity[e,t] = x[2]
        self.general_assets[e,t] = np.mean([p.assets for p in self.peoples])

        return x
        
    def evolve(self):
        '''
        For now is extremely non-optimized
        '''
        combined_wealth = np.argsort([p.liquidity+p.assets*self.asset_price for p in self.peoples])
        liquidity = np.argsort([p.liquidity for p in self.peoples])
        assets = np.argsort([p.assets for p in self.peoples])
        for p in self.peoples:
            p.age += 1
        

        keep = int(len(self.peoples)*self.percent)
        
        best_combined = self.peoples[combined_wealth[:keep]]
        best_liquidity = self.peoples[liquidity[:keep]]
        best_assets = self.peoples[assets[:keep]]


        evol_choices = [0,1,2]

        new_people = np.concatenate((best_combined, best_liquidity, best_assets))

        for _ in range(self.world_population-len(new_people)):
            # choose random parents from the three groups
            parents = np.random.choice(evol_choices, size=2, replace=True)
            if parents[0] == 0:
                parent1 = best_combined[np.random.randint(0,high=len(best_combined))]
            elif parents[0] == 1:
                parent1 = best_liquidity[np.random.randint(0,high=len(best_liquidity))]
            elif parents[0] == 2:
                parent1 = best_assets[np.random.randint(0,high=len(best_assets))]
            
            if parents[1] == 0:
                parent2 = best_combined[np.random.randint(0,high=len(best_combined))]
            elif parents[1] == 1:
                parent2 = best_liquidity[np.random.randint(0,high=len(best_liquidity))]
            elif parents[1] == 2:
                parent2 = best_assets[np.random.randint(0,high=len(best_assets))]
            
            newWI = np.zeros(parent1.WI.shape)
            newWA = np.zeros(parent1.WA.shape)
            
            for i in range(parent1.WI.shape[0]):
                for j in range(parent1.WI.shape[1]):
                    r = np.random.rand()
                    mutation = np.random.rand()
                    if r < 0.5:
                        if mutation < 0.01:
                            newWI[i,j] = parent1.WI[i,j] + (np.random.rand()*2 - 1)/20
                        else:
                            newWI[i,j] = parent1.WI[i,j]
                    else:
                        if mutation < 0.01:
                            newWI[i,j] = parent2.WI[i,j] + (np.random.rand()*2 - 1)/20
                        else:
                            newWI[i,j] = parent2.WI[i,j]
            
            for i in range(parent1.WA.shape[0]):
                for j in range(parent1.WA.shape[1]):
                    r = np.random.rand()
                    mutation = np.random.rand()
                    if r < 0.5:
                        if mutation < 0.01:
                            newWA[i,j] = parent1.WA[i,j] + (np.random.rand()*2 - 1)/20
                        else:
                            newWA[i,j] = parent1.WA[i,j]
                    else:
                        if mutation < 0.01:
                            newWA[i,j] = parent2.WA[i,j] + (np.random.rand()*2 - 1)/20
                        else:
                            newWA[i,j] = parent2.WA[i,j]
            
            new_jhon = jhonnies(li=self.input_len, loi=self.invest_disc, loa=self.sales_disc,
                                 M=int((parent1.start_liquidity+parent2.start_liquidity)/2), C=0)
            new_jhon.set_matrix(newWI, newWA)
            new_people = np.append(new_people, new_jhon)

        self.peoples = new_people
        if len(self.peoples) != self.world_population:
            print(f"The new people are a different number from the start, something wrong: new={len(self.peoples)}-->stating={self.world_population}")
                

    def reset_world(self):
        for p in self.peoples:
            p.reset_attr()
        
        self.avg_age.append(np.mean([p.age for p in self.peoples]))
        self.hist_removed_people.append(self.removed_people)
        
        self.asset_price = self.start_price
        self.previous_noise = 0
        self.removed_people = 0
                    
    
    def save_best_ag(self, eps, PATH):

        combined_wealth = np.argsort([p.liquidity+p.assets*self.asset_price for p in self.peoples])
        top = self.peoples[combined_wealth[0]]
        with open(PATH+f"bestAG-eps{eps}", 'wb') as file:
            pk.dump(top, file, pk.HIGHEST_PROTOCOL)

    def run_world(self, PATH):

        for e in range(self.eps):
            print(f"Episode {e}")
            for t in range(self.simul_len):
                self.agents_step(self.get_input(t=t, e=e), t=t, e=e)
                self.update_prices()
            
            if (e+1)%10 == 0:
                print(f"Removed {self.removed_people} people")
                print("Saving best agent (assets+liquidity)...")
                self.save_best_ag(e, "data/agents/"+PATH)
            self.evolve()
            self.reset_world()
    

    def save_data(self, PATH):
        np.save(PATH+"noises",self.noises)
        np.save(PATH+"prices",self.prices)
        np.save(PATH+"liquidity",self.general_liquidity)
        np.save(PATH+"assets",self.general_assets)
        np.save(PATH+"buy",self.avg_buying)
        np.save(PATH+"sell",self.avg_selling)
        np.save(PATH+"age",self.avg_age)
        np.save(PATH+"removed",self.hist_removed_people)

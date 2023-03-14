from agents import jhonnies
import numpy as np

class world():

    def __init__(self, world_population:int, input_l:int, invest_disc:int, sales_disc:int,
                 start_money_loc=100, start_money_scale=10, cost_loc=60, epsilon=1, beta=10,
                 delta=20, income_scaling=5, noise_scale=1.5) -> None:
        '''
        Stuff to do:
        - initialize the population with:
            - initial money
            - cost of living
            - discretization of input and output
        - initialize parameters for price update and noise signal
        '''
        # PEOPLE INIT
        self.peoples = []
        for _ in range(world_population):
            money = int(np.random.normal(loc=start_money_loc, scale=start_money_scale))
            # making sure that the cost of living is smaller than the initial money they have
            cost_of_living = 0
            #while cost_of_living > money:
            #    cost_of_living = int(money - np.random.normal(loc=(start_money_loc-cost_loc), scale=start_money_scale))
            
            self.peoples.append(jhonnies(li=input_l, loi=invest_disc, loa=sales_disc, M=money, C=cost_of_living))
        
        # PARS INIT
        self.input_len = input_l
        self.epsilon = epsilon
        self.beta = beta
        self.delta = delta
        self.asset_price = start_money_loc//10
        self.income_scale = income_scaling
        self.sold = 0
        self.bought = 0
        self.previous_noise = 0
        self.noise_scale = noise_scale
    
    def update_prices(self):
        '''
        Update the price following:
        - if `sold > bought+delta` -> `price -= epsilon`
        - if `bought > sold+delta` -> `price += epsilon`
        - else -> `price +/-= epsilon/beta` with 50/50 chance
        '''
        if self.sold > self.bought + self.delta:
            self.asset_price += self.epsilon
        elif self.sold < self.bought - self.delta:
            self.asset_price -= self.epsilon
        else:
            if np.random.rand() < 0.5:
                self.asset_price += self.epsilon/self.beta
            else:
                self.asset_price -= self.epsilon/self.beta
    

    def agents_step(self, x):
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
                print("Very poor guy removed")
                self.peoples.remove(p)
                continue
            # otherwise it can buy and/or sell things
            inv, sale = p.action(x)
            # possibly buy stocks at current price and decrease its liquidity
            stocks = inv//self.asset_price
            p.assets += stocks
            p.liquidity -= inv
            self.bought += stocks
            # and sell assets at current price, decreasing its assets and increasing its liquidity
            p.liquidity += self.asset_price*sale
            p.assets -= sale
            self.sold += sale
    
    def get_input(self):
        x = np.zeros(self.input_len)
        x[0] = self.asset_price
        x[1] = self.previous_noise + (np.random.rand()*2 - 1)*self.noise_scale
        self.previous_noise = x[1]
        x[2] = np.mean([p.liquidity for p in self.people])

        return x
        
    def evolve(self):
        '''
        Crucial point of everything
        '''
        pass
    
    def run_world(self):
        self.agents_step(self.get_input())
        self.update_prices()
        self.evolve()
import numpy as np


class jhonnies():

    def __init__(self, li, loi, loa, M, C) -> None:
        '''
        Input:
        - li = length of expected input
        - loi = lenght of output for investment
        - loa = lenght of output for selling assets
        - M = starting liquidity
        - C = cost of surviving
        
        Here we intialize the weight matrices of this agent, its activation function,
        decide its income
        '''

        self.in_len = li
        self.outI_len = loi
        self.invest_range = np.linspace(0,1,num=self.outI_len)
        self.outA_len = loa
        self.sales_range = np.linspace(0,1,num=self.outA_len)

        self.WI = np.random.normal(size=(self.in_len, self.outI_len))
        self.WA = np.random.normal(size=(self.in_len, self.outA_len))

        self.liquidity = M
        self.cost_of_living = C
        self.assets = 0
    

    def action(self, x):
        '''
        Input:
        - x = input vector

        Output:
        - invest_decision = % of self.liquidity to invest
        - sale_decision = % of self.assets to sell
        '''
        if len(x) != self.in_len:
            print(f"Wrong input: len(x)={len(x)}, expected {self.in_len}")
            raise RuntimeError

        outI = x @ self.WI
        outA = 0
        if self.assets > 0:
            outA = x @ self.WA
            outA = outA/np.linalg.norm(outA)
        outI = outI/np.linalg.norm(outI)

        invest_decision = self.invest_range[np.unravel_index(np.argmax(outI, axis=None), outI.shape)]
        sale_decision = self.sales_range[np.unravel_index(np.argmax(outA, axis=None), outA.shape)]

        return invest_decision, sale_decision
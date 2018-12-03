import numpy as np
from scipy.optimize import minimize_scalar

class Seller:
    def __init__(self, a, b, E_sold, Emax):

        self.a = a
        self.b = b
        self.E_sold = 0
        self.Emax = Emax
        self.utility = 0

    def compute_utility(self, E_sold, w):
        return self.a*np.log(1+self.b*(self.Emax - E_sold)) + w - self.a*np.log(1+self.b*(self.Emax))

    def set_utility(self, E_sold, w):
        self.utility = self.compute_utility(E_sold, w)

    def max_utility_arg(self, unit_price):
        E = self.Emax - self.a/unit_price+1/self.b
        if E >= self.Emax :
            E = self.Emax
        if E <= 0:
            E = 0
        return E

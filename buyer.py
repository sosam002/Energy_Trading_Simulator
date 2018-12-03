import numpy as np
import pdb

class Buyer:
    def __init__(self, a,b,Eo,Emax):

        self.a = a
        self.b = b
        self.Eo = Eo
        self.Emax = Emax
        self.E = 0
        self.w = 0
        self.utility = 0

#################### only useed in manager.get_mu() ###########################
    def compute_E(self, E_total, mu):
        # pdb.set_trace()
        E = E_total*(-self.Eo*self.b*mu + self.a*self.b -mu)/(self.b*(E_total*mu + self.a))
        return max(E, 0)

        # return E_total*(-self.Eo*self.b*mu + self.a*self.b -mu)/(self.b*(E_total*mu + self.a))

    def set_E(self, E_total, mu):
        self.E = self.compute_E(E_total, mu)
        return  self.E

    def compute_w(self, unit_price):
        return self.E*unit_price

    def set_w(self, unit_price):
        self.w = self.compute_w(unit_price)
        return self.w
#################### only useed in manager.get_mu() ###########################

    def compute_utility(self, E, w):
        return self.a*np.log(1+self.b*(self.Eo+E)) - w - self.a*np.log(1+self.b*(self.Eo))

    def set_utility(self, E, w):
        self.utility = self.compute_utility(E, w)
        return self.utility

    def compute_diff_u(self):
        return self.a*self.b/(1+self.b*(self.Eo+self.E))

    def buy_max_utility(self, unit_price, set = False):
        E = self.a/unit_price - 1/self.b - self.Eo
        if E < 0:
            E = 0
        # elif E > self.Emax:
        #     E = self.Emax
        else:
            pass
        if set:
            self.E = E
        return E

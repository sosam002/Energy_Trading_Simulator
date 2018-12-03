import time
import pdb
from scipy.optimize import minimize
import numpy as np
# from cvxopt import matrix, log, div, spdiag, solvers

class Manager:
    def __init__(self, E_total, users):
        self.E_total = E_total
        self.users = users
        self.mu = 0
        self.unit_price = 0
        self.num_users = len(users)

    def initialize(self, E_total):
        self.E_total = E_total
        self.mu = 0
        self.unit_price = 0

    # KKT optimization self written codes
    # return value mu is the unit price.
    def get_mu(self):
        self.mu = self.unit_price
        start_time = time.time()
        dmu = 0.001
        lr = 0.0001
        while True:
            sum = 0
            dsum = 0
            for user in self.users:
                sum += user.set_E(self.E_total, self.mu)
                dsum += (user.set_E(self.E_total, self.mu + dmu) - user.set_E(self.E_total, self.mu - dmu))
            dsum = dsum / 2 / dmu
            diff = self.E_total - sum
            if diff < 0.00001 and diff > -0.00001:
                for user in self.users:
                    user.set_E(self.E_total, self.mu)
                # self.get_unit_price()
                self.unit_price = self.mu
                for user in self.users:
                    user.set_w(self.unit_price)
                break
            else:
                if dsum==0:
                    self.mu += 0.1
                else:
                    self.mu += diff / dsum *lr
        print("sum : {}, E : {}, mu : {}, unit_price : {}, execution time: {:.6f} seconds".format(sum, self.E_total, self.mu, self.unit_price, time.time()-start_time))
        return self.mu

    def KKT(self, get_mu = False):
        # minimize input variable
        E = []
        for k in range(self.num_users):
            E.append(self.users[k].E)

        # objective function sum(u_hat(E))
        def objective(E):
            sum = 0
            for k in range(self.num_users):
                star = self.users[k].b*(self.users[k].Eo+E[k])+1
                sum += self.users[k].a*(((self.E_total+self.users[k].Eo)*self.users[k].b+1)*np.log(star)-E[k]*self.users[k].b-((self.users[k].Eo+self.E_total)*self.users[k].b+1)*np.log(self.users[k].Eo*self.users[k].b+1))/(self.E_total*self.users[k].b)
            return -sum

        # constraints
        def constraint(E):
            return self.E_total - sum(E)
        const = {'type':'ineq', 'fun':constraint}

        def const_function(i):
            return lambda E: E[i]
        constraints = [const_function(i) for i in range(self.num_users)]
        consts = []
        for i in range(self.num_users):
            consts.append({'type':'ineq', 'fun':constraints[i]})

        cons = ([const]+consts)

        # bounds
        b = (0.0, self.E_total)
        bnds = [b]*self.num_users

        # print(objective(E))
        # get solution
        solution = minimize(objective, E, bounds = tuple(bnds), constraints=cons)#, tol =1e-6 )
        E = solution.x

        # save users.E values
        for k in range(self.num_users):
            self.users[k].E = E[k]
            # print("user {}'s a : {}, E : {}".format(k, self.users[k].a, self.users[k].E))

        # solve unit price
        self.get_unit_price()
        # print("unit_price = ", self.unit_price)
        if get_mu == True:
            self.get_mu()
        # solve users.w and users.utility values with unit price
        for user in self.users:
            user.w = user.E*self.unit_price
            user.set_utility(user.E, user.w)
            # user.utility = user.a*np.log(1+user.b*(user.Eo+user.E)) - user.w - user.a*np.log(1+user.b*(user.Eo))
            # print("user ?'s a : {}, E : {}, w : {}, utility: {}, user's ab/(1+bEo) = {}".format(user.a, user.E, user.w, user.utility, user.a*user.b/(1+user.b*user.Eo)))
        return E

    # price is determined by the seller, and the seller also determines E[k].
    def KKT_priced(self, E_total, unit_price = 0):
        self.E_total = E_total
        if unit_price == 0:
            unit_price = self.unit_price

        # minimize input variable
        E = []
        for k in range(self.num_users):
            E.append(self.users[k].E)

        # objective function sum(u_hat(E))
        def objective(E):
            sum = 0
            for k in range(self.num_users):
                a = self.users[k].a
                b = self.users[k].b
                Eo = self.users[k].Eo
                sum += a*np.log(1+b*(Eo+E[k])) - a*np.log(1+b*(Eo)) - unit_price*E[k]
            return -sum

        # constraints
        def constraint(E):
            return self.E_total - sum(E)
        const = {'type':'ineq', 'fun':constraint}

        def const_function(i):
            return lambda E: E[i]
        constraints = [const_function(i) for i in range(self.num_users)]

        consts = []
        for i in range(self.num_users):
            consts.append({'type':'ineq', 'fun':constraints[i]})

        cons = ([const]+consts)

        # bounds
        b = (0.0, self.E_total)
        bnds = [b]*self.num_users

        # print(objective(E))
        # get solution
        solution = minimize(objective, E, bounds = tuple(bnds), constraints=cons)
        E = solution.x

        # save users.E values
        for k in range(self.num_users):
            self.users[k].E = E[k]
            # print("user {}'s a : {}, E : {}".format(k, self.users[k].a, self.users[k].E))

        # solve users.w and users.utility values with unit price
        for user in self.users:
            user.w = user.E*unit_price
            user.set_utility(user.E, user.w)
            # user.utility = user.a*np.log(1+user.b*(user.Eo+user.E)) - user.w - user.a*np.log(1+user.b*(user.Eo))
            # print("user ?'s a : {}, E : {}, w : {}, utility: {}, user's ab/(1+bEo) = {}".format(user.a, user.E, user.w, user.utility, user.a*user.b/(1+user.b*user.Eo)))
        return E

    # price is determined by the seller, and the buyers determine energy
    def max_users_utility(self, unit_price = 0):
        if unit_price == 0:
            unit_price = self.unit_price
        sum=0
        for i in range(self.num_users):
            sum += self.users[i].buy_max_utility(unit_price)
        return sum

    # Thm4 from Yudong Yang et.al in PE.
    # errata of the paper : summation index set is not N, but indeces with d_i>0
    def get_unit_price(self):
        self.unit_price = 0
        sum = 0
        for user in self.users:
            if user.E > 0.00001:
                sum += 1
                self.unit_price += user.compute_diff_u()*(self.E_total-user.E)
        # try:
        self.unit_price = self.unit_price/(sum*self.E_total)
        # except:
        #     pdb.set_trace()
        # print("for unit price, E>0 number : {}".format(sum))
        return self.unit_price

    # verification of NE.
    # hold w of the other players and set users[index].w = w
    def user_variation(self, index, w):
        new_sum_w=0
        for user in self.users:
            new_sum_w += user.w
        new_sum_w = new_sum_w - self.users[index].w + w
        new_E_of_index = w*self.E_total/new_sum_w
        # new_E_of_index = w/self.unit_price

        utility = self.users[index].compute_utility(new_E_of_index, w)
        # print(self.unit_price, new_sum_w/self.E_total, new_E_of_index, utility, w)
        # print(new_sum_w, w, new_E_of_index, utility)

        return [new_E_of_index, utility]

    def trading_user_count(self):
        sum=0
        for user in self.users:
            if user.E>0.001:
                sum +=1
        return sum

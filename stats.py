import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from buyer import Buyer
from seller import Seller
from manager import Manager
import pdb
import logging

min_unit_price = 0.2
max_unit_price = 3.0
num_users, a, b, Emax = 5, 100, 20, 100

userinfo = []
uservar = []
stats = []
x,y = [],[]

userinfo = []
bps_buyer = []
bps_efficiency = []
sps_buyer = []
bps_seller = []
sps_seller = []
buyer_delta = []
seller_delta = []

for iter in range(1000):

    users = []
    aes, bs, Eos, Emaxs = [],[],[],[]

    # random한 a, b, Eo, Emax를 num_users만큼 생산
    for i in range(num_users):
        aes.append(a*np.random.random())
        bs.append(b*np.random.random())
        tempmax = Emax*np.random.random() + Emax
        Emaxs.append(tempmax)
        Eos.append(tempmax*0.2*np.random.random())

    # buyer가 energy거래에 참여하는 unit_price 기준이 되는 ab/(b*Eo+1) 계산 및 정렬
    ab_over_bEos = [aes[i]*bs[i]/(1+bs[i]*Eos[i]) for i in range(len(aes))]
    sorted_idx = sorted(range(num_users), key = lambda i: aes[i]*bs[i]/(1+bs[i]*Eos[i]))

    aes = [aes[sorted_idx[i]] for i in range(num_users)]
    bs = [bs[sorted_idx[i]] for i in range(num_users)]
    Eos = [Eos[sorted_idx[i]] for i in range(num_users)]
    Emaxs = [Emaxs[sorted_idx[i]] for i in range(num_users)]
    ab_over_bEos.sort()

    # ab_over_bEos 순으로 정렬된 buyer 생성
    for i in range(num_users):
        users.append(Buyer(aes[i], bs[i], Eos[i], Emaxs[i]))

    # seller 생성
    a_seller = a*np.random.random()
    b_seller = b*np.random.random()
    Emax_seller = Emax*np.random.random() + Emax
    E_sold = Emax_seller # 맨 처음으로는 가지고 있는 모든 에너지를 판다.
    seller = Seller(a_seller, b_seller, E_sold, Emax_seller)

    # 시장거래를 위한 manager 생성
    manager = Manager(E_sold, users)

    # 파는 E (E_sold)에 따른 유저별 w,E,utility를 임시 저장하기 위한 배열 생성
    ws = [None]*num_users
    Es = [None]*num_users
    us = [None]*num_users
    final_E = E_sold

    # E_sold를 -1씩 감소시키면서 utility_seller를 관찰
    while True:
        manager.initialize(E_sold)
        E = manager.KKT()
        unit_price = manager.unit_price
        nd_user = manager.users[num_users-2]
        if E_sold <= 1 or nd_user.a*nd_user.b/(1+nd_user.b*nd_user.Eo) < unit_price or unit_price > max_unit_price:
            # final_E = Emax_seller
            break

        if unit_price < min_unit_price:
            pass
        else:
            for i in range(num_users):
                ws[i] = manager.users[i].w
                Es[i] = manager.users[i].E
                us[i] = manager.users[i].set_utility(Es[i], ws[i])
                # print("user {}'s E_i : {}, w_i : {}, utility : {}".format(i, Es[i], ws[i], us[i]))
            new_utility = seller.compute_utility(E_sold, sum(ws))
            # xs.append(E_sold)
            # utility_seller.append(new_utility)
            # utility_buyers.append(sum(us)/num_users)
            # unit_prices.append(unit_price)

            if seller.utility < new_utility:
                seller.utility = new_utility
                final_E = E_sold
        E_sold -= 1

    bps_seller_utility=seller.utility

    ws_final = []
    Es_final = []
    us_final = []
    manager.initialize(final_E)
    manager.KKT()
    for i in range(len(users)):
        ws_final.append(users[i].w)
        Es_final.append(users[i].E)
        us_final.append(users[i].set_utility(Es_final[-1], ws_final[-1]))

    ws_priced = []
    Es_priced = []
    us_priced = []
    manager.KKT_priced(final_E)
    for i in range(len(users)):
        ws_priced.append(users[i].w)
        Es_priced.append(users[i].E)
        us_priced.append(users[i].set_utility(Es_priced[-1], ws_priced[-1]))




    utility_seller2 = []
    # initialize final_E. final_E는 seller의 utility가 최대가 되는 E_sold가 될 예정
    E_sold = Emax_seller
    final_E = E_sold
    unit_price = min_unit_price
    # 파는 E (E_sold)에 따른 seller의 utility를 저장할 배열 생성 (figure 4 그래프)
    market_clear_Es = []
    unit_prices2 = []
    flag = 0
    new_max_unit_price = min_unit_price

    # unit_price에 따른 market_clear E_sold (max부터 min까지)
    while True:
        E= manager.max_users_utility(unit_price)
        market_clear_Es.append(E)
        unit_prices2.append(unit_price)

        # unit_price를 올리다가, 어느 시점에 seller가 팔 수 있는 E보다 market clearing E가 더 작다면,
        # 그 직전의 unit_price를 저장함 (E>E_sold였던 가장 큰 unit_price)
        # 즉, 팔고자 하는 모든 E를 다 팔 수 있는 가장 비싼 unit_price
        if flag == 0 and E <= E_sold:
            new_max_unit_price = unit_price
            flag = 1

        if unit_price >= max_unit_price:
            break
        unit_price += 0.001

    # market clearing E가 E_sold 이하로 들어오는 범위에서, (price, clearing E) tuple을 이용하여 seller의 utility를 측정
    unit_price = new_max_unit_price
    idx = unit_prices2.index(unit_price)

    utility_seller2 = []
    utility_buyers2 = []
    best_E_solds, best_utility_seller2 = [], []
    real_E_solds, real_utility_seller2 = [], []
    flag = 0
    impossible = []
    for i in range(idx, len(unit_prices2)):
        E_sold = market_clear_Es[i]
        new_utility = seller.compute_utility(E_sold, E_sold*unit_prices2[i])
        best_E_sold = seller.max_utility_arg(unit_prices2[i])
        best_utility = seller.compute_utility(best_E_sold, best_E_sold*unit_prices2[i])
        if best_E_sold < E_sold:
            real_utility = best_utility
            real_E_solds.append(best_E_sold)
        else:
            real_utility = new_utility
            real_E_solds.append(E_sold)
        best_E_solds.append(best_E_sold)
        if flag == 0 and best_E_sold >= E_sold:
            impossible = [unit_prices2[i], E_sold]
            flag = 1
        utility_seller2.append(new_utility)
        real_utility_seller2.append(real_utility)
        best_utility_seller2.append(best_utility)

        ws_priced = []
        Es_priced = []
        us_priced = []
        manager.KKT_priced(E_sold, unit_price)
        for i in range(len(users)):
            ws_priced.append(users[i].w)
            Es_priced.append(users[i].E)
            us_priced.append(users[i].set_utility(Es_priced[-1], ws_priced[-1]))
        utility_buyers2.append(sum(us_priced)/num_users)

        E_sold -= 1

    seller_max_at = seller.max_utility_arg(unit_price)
    if seller_max_at <=0:
        seller_max_at = 0
    if seller_max_at >= Emax_seller:
        seller_max_at = Emax_seller

    arg = np.argmax(real_utility_seller2)
    xmax, ymax, price= real_E_solds[arg], np.max(real_utility_seller2), unit_prices2[idx+arg]

    ws_sps_priced = []
    Es_sps_priced = []
    us_sps_priced = []
    manager.KKT_priced(xmax, price)
    for i in range(len(users)):
        ws_sps_priced.append(users[i].w)
        Es_sps_priced.append(users[i].E)
        us_sps_priced.append(users[i].set_utility(Es_sps_priced[-1], ws_sps_priced[-1]))
    new_utility = seller.compute_utility(xmax, xmax*price)
    # print("seller's E_i : {}, w_i : {}, utility : {}".format(xmax, sum(ws_sps_priced), new_utility))
    # print("buyers' total utility : {}".format(sum(us_sps_priced)))


    userinfo.append(np.array(ab_over_bEos))
    bps_buyer.append(sum(us_final))
    bps_efficiency.append(sum(us_priced))
    sps_buyer.append(sum(us_sps_priced))
    buyer_delta.append(sum(us_final)-sum(us_sps_priced))
    bps_seller.append(bps_seller_utility)
    sps_seller.append(new_utility)
    seller_delta.append(bps_seller_utility-new_utility)
    # x.append(np.std(ab_over_bEos))
    # y.append(1-sum(us_final)/sum(us_priced))

userinfo = np.array(userinfo)
stats = np.array([bps_buyer, bps_efficiency, sps_buyer])
seller_stats = np.array([bps_seller, sps_seller])
delta_stats = np.array([buyer_delta, seller_delta])

np.save("userinfo3", userinfo)
np.save("stats3", stats)
np.save("seller_stats3", seller_stats)
np.save("delta_stats3", delta_stats)

plt.scatter(buyer_delta, seller_delta)
# plt.scatter(sps_buyer, sps_seller, label = "SPS utility")
plt.xlabel("Total utility changes of buyers (SPS to BPS)")
plt.ylabel("Utility changes of seller (SPS to BPS)")
plt.show()

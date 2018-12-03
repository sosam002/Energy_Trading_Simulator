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
users = []
aes = [13.212381088639901, 71.79824982511163, 72.56821931434466, 71.91058982800065, 85.07600737444164]
bs = [18.689563724408266, 11.678934511872836, 1.424129347448777, 13.613529960278933, 0.908226884551544]
Eos = [12.06509353922039, 16.296671099614052, 14.699426406864657, 1.9207769636975447, 0.8275107140100313]
Emaxs = [156.8798473493223, 156.8798473493223,156.8798473493223,156.8798473493223, 156.8798473493223]

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
    print(aes[i], bs[i], Eos[i]) #, Emaxs[i])

# seller 생성
a_seller = 89.89911431262864
b_seller = 12.17899767085905
Emax_seller = 156.8798473493223
E_sold = Emax_seller # 맨 처음으로는 가지고 있는 모든 에너지를 판다.
seller = Seller(a_seller, b_seller, E_sold, Emax_seller)
print(a_seller, b_seller, E_sold, Emax_seller)

# 시장거래를 위한 manager 생성
manager = Manager(E_sold, users)

# 막대그래프 plot과 labeling을 위한 함수
bar_idx = np.arange(num_users)
bar_width = 0.15
opacity = 1
def autolabel(rects, text=True, place = 'top'):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if text == True and height > 0.0001:
            if place == 'top':
                bar_graph.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '{:.3f}'.format(height),
                    ha='center', va='bottom')
            elif place == 'center':
                bar_graph.text(rect.get_x() + rect.get_width()/2., 0.5*height,
                    '{:.3f}'.format(height),
                    ha='center', va='bottom')
markers = ['o','v','^','s','D']

# 파는 E (E_sold)에 따른 유저별 w,E,utility를 임시 저장하기 위한 배열 생성
ws = [None]*num_users
Es = [None]*num_users
us = [None]*num_users

# 파는 E (E_sold)에 따른 seller의 utility를 저장할 배열 생성 (figure 0 마지막 그래프)
utility_seller = []
utility_buyers = []
unit_prices = []
trading_user_count = []
xs = []
# initialize final_E. final_E는 seller의 utility가 최대가 되는 E_sold가 될 예정
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
        xs.append(E_sold)
        utility_seller.append(new_utility)
        utility_buyers.append(sum(us)/num_users)
        unit_prices.append(unit_price)
        trading_user_count.append(manager.trading_user_count())

        if seller.utility < new_utility:
            seller.utility = new_utility
            final_E = E_sold
    E_sold -= 1

print("final_E = {} Emax_seller = {}".format(final_E, Emax_seller))
print("Final graph loading...")

# seller의 utiltiy 최대로 하는 E_sold일 때 (final_E일 때)
# NE가 구해졌을 때 buyer별 solution w, E, utility를 막대그래프로 그린다.
# E_final이 시장에 거래될 때 NE를 풀기 위한 KKT
ws_final = []
Es_final = []
us_final = []
manager.initialize(final_E)
manager.KKT(True)
for i in range(len(users)):
    ws_final.append(users[i].w)
    Es_final.append(users[i].E)
    us_final.append(users[i].set_utility(Es_final[-1], ws_final[-1]))
print("seller's E_i : {}, w_i : {}, utility : {}".format(final_E, sum(ws_final), new_utility))
print("buyers' total utility : {}".format(sum(us_final)))


# figure 0 생성
fig = plt.figure(0)
################### figure 0의 막대 그래프 : NE solution과 utility ##############
# bar_graph = fig.add_subplot(2, 2, 1)
bar_graph = fig.add_subplot(111)
rects2 = bar_graph.bar(bar_idx, ab_over_bEos, bar_width,
                        label ='$r_k$',
                        alpha = opacity)
rects4 = bar_graph.bar(bar_idx+2*bar_width, ws_final, bar_width,
                        label ='$r_k$',
                        alpha = opacity)
rects5 = bar_graph.bar(bar_idx+3*bar_width, Es_final, bar_width,
                        label ='$E_k$',
                        alpha = opacity)
rects6 = bar_graph.bar(bar_idx+4*bar_width, us_final, bar_width,
                        label ='$Utility_k$',
                        alpha = opacity)
autolabel(rects2)
autolabel(rects4)
autolabel(rects5, place = 'center')
autolabel(rects6)
# plt.title('distributed energy $E_i$ and payments $w_i$ when $E_s=$ {:.3f}'.format(final_E))
plt.xlabel("Buyers \n total utility : {:.3f}, unit price : {:.3f}, seller's utility : {:.3f}, trading energy : {:.3f}".format(sum(us_final), manager.unit_price, new_utility, final_E))
plt.ylabel('$r_k$, $w_k$, $E_k$, $Utility_k$')
plt.tight_layout()
plt.legend()
################ figure 0의 막대 그래프 : NE solution과 utility 끝 ##############

fig = plt.figure(1)
################### figure 0의 NE varification 그래프 ###########################
# utility_graph = fig.add_subplot(2, 2, 3)
utility_graph = fig.add_subplot(111)
xs_var = np.arange(0,200, 0.001)
ys = []
zs = []
maxes = []

for i in range(num_users):
    ys.append([manager.user_variation(i, w)[1] for w in xs_var])
    zs.append([manager.user_variation(i, w)[0] for w in xs_var])
    xmax, ymax, zmax = xs_var[np.argmax(ys[i])], np.max(ys[i]), zs[i][np.argmax(ys[i])]
    maxes.append(xmax)
    print(xmax, ymax, zmax)
    user = manager.users[i]
    utility_graph.plot(xs_var, ys[i], label='$r_k$ {:3f}'.format(user.a*user.b/(1+user.b*user.Eo)), marker=markers[i], markerfacecolor = 'none', markevery=10000, linewidth=1)
    utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                    xytext=(-20,-50+10*i), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->")
                    )
    # utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-25+5*i)
    #         )
# fig.subplots_adjust(hspace = 0.5)
# plt.title('Utilities vary according to $w_i$s')

plt.xlabel("Payment strategy $w_k$ of each buyer \n total utility : {:.3f}, unit price : {:.3f}, seller's utility : {:.3f}, trading energy : {:.3f}".format(sum(us_final), manager.unit_price, new_utility, final_E))
plt.ylabel('Utility $u_k(w_k,w_{-k}^*)$ of each buyer')
plt.tight_layout()
plt.legend()
################### figure 0의 NE varification 그래프 끝 ########################


# Johari 결과랑 효율성 비교를 위한 KKT
# E_final만큼 거래될 때, buyer utility의 sum을 최대화하는 거래가 이루어지면,
# NE에 의해 거래가 이루어질 때랑 공익이 얼마나 차이나는지를 보여주기 위한 그래프
ws_priced = []
Es_priced = []
us_priced = []
manager.KKT_priced(final_E)
for i in range(len(users)):
    ws_priced.append(users[i].w)
    Es_priced.append(users[i].E)
    us_priced.append(users[i].set_utility(Es_priced[-1], ws_priced[-1]))
print("seller's E_i : {}, w_i : {}, utility : {}".format(final_E, sum(ws_priced), new_utility))
print("buyers' total utility : {}".format(sum(us_priced)))

fig = plt.figure(2)
################ figure 0의 total utility 최대화 solution과 utiltiy##############
# bar_graph = fig.add_subplot(2,2,2)
bar_graph = fig.add_subplot(111)
rects2 = bar_graph.bar(bar_idx, ab_over_bEos, bar_width,
                        label ='$r_k$',
                        alpha = opacity)
rects4 = bar_graph.bar(bar_idx+2*bar_width, ws_priced, bar_width,
                        label ='$w_k$',
                        alpha = opacity)
rects5 = bar_graph.bar(bar_idx+3*bar_width, Es_priced, bar_width,
                        label ='$E_k$',
                        alpha = opacity)
rects6 = bar_graph.bar(bar_idx+4*bar_width, us_priced, bar_width,
                        label ='$Utility_k$',
                        alpha = opacity)
autolabel(rects2)
autolabel(rects4)
autolabel(rects5, place = 'center')
autolabel(rects6)

# plt.title('distributed energy $E_i$ and payments $w_i$ when $E_s=$ {:.3f}'.format(final_E))

plt.xlabel("Buyers \ntotal utility : {:.3f}, unit price : {:.3f}, seller's utility : {:.3f}, trading energy : {:.3f}".format(sum(us_priced), manager.unit_price, new_utility, final_E))
plt.ylabel('$r_k$, $w_k$, $E_k$, $Utility_k$')
plt.tight_layout()
plt.legend()
############ figure 0의 total utility 최대화 solution과 utiltiy 끝 ##############


# # figure1 생성
# fig = plt.figure(1)
fig = plt.figure(3)
############## figure 0. seller의 utility 변화 곡선 #############################
# seller_utility_graph = fig.add_subplot(2, 2, 4)
utility_graph = fig.add_subplot(111)
unit_price_graph = utility_graph.twinx()
# xs.reverse()
# utility_seller.reverse()
# utility_buyers.reverse()
# unit_prices.reverse()

xmax, ymax = xs[np.argmax(utility_seller)], np.max(utility_seller)
# xmax2, ymax2 = xs[np.argmax(utility_buyers)], np.max(utility_buyers)
print(xmax, ymax)
l1 = utility_graph.plot(xs, utility_seller, marker='o', markerfacecolor = 'none', markevery=5, linewidth = 1)#, label='$ab/(1+bE_m)$ {:3f}'.format(a_seller*b_seller/(1+b_seller*(Emax_seller))))# + str(n))
l2 = unit_price_graph.plot(xs, unit_prices, color='r', marker='v', markerfacecolor = 'none', markevery=5, linewidth=1)
l3 = utility_graph.plot(xs, utility_buyers, marker='^',  markerfacecolor = 'none', markevery=5, linewidth = 1)
# utility_graph.legend(loc=2)
# bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
# arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
# kw = dict(xycoords='data',textcoords="axes fraction",
#           arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                xytext=(-50,-50), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
# utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-5)
#         # arrowprops=dict(arrowstyle = "->", facecolor='black', shrink=0.05),
#         )

    # plt.legend()
# fig.subplots_adjust(hspace = 0.5)
# plt.title('Utilities vary according to $E_{seller}$')

utility_graph.set_xlabel('Trading energy $E_{s}$')
utility_graph.set_ylabel("Utility")
unit_price_graph.set_ylabel('Unit price')
plt.legend(l1+l2+l3, ["seller's utility", 'unit price', "buyer's mean utility"], loc='lower left')
plt.tight_layout()
############## figure 0. seller의 utility 변화 곡선 끝 ##########################





fig = plt.figure(13)
############## figure 0. seller의 utility 변화 곡선 #############################
# seller_utility_graph = fig.add_subplot(2, 2, 4)
utility_graph = fig.add_subplot(111)
# xs.reverse()
# utility_seller.reverse()
# utility_buyers.reverse()
# unit_prices.reverse()

xmax, ymax = xs[np.argmax(utility_seller)], np.max(utility_seller)
# xmax2, ymax2 = xs[np.argmax(utility_buyers)], np.max(utility_buyers)
print(xmax, ymax)
l1 = utility_graph.plot(xs, utility_seller)#, label='$ab/(1+bE_m)$ {:3f}'.format(a_seller*b_seller/(1+b_seller*(Emax_seller))))# + str(n))
l3 = utility_graph.plot(xs, utility_buyers)
# utility_graph.legend(loc=2)
# bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
# arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
# kw = dict(xycoords='data',textcoords="axes fraction",
#           arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                xytext=(-50,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
# utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-5)
#         # arrowprops=dict(arrowstyle = "->", facecolor='black', shrink=0.05),
#         )

    # plt.legend()
# fig.subplots_adjust(hspace = 0.5)
# plt.title('Utilities vary according to $E_{seller}$')

utility_graph.set_xlabel('Trading energy $E_{s}$ \n under BPS')
utility_graph.set_ylabel("Utility")
plt.legend(l1+l3, ["seller's utility", "buyer's mean utility"], loc='lower left')
plt.tight_layout()
############## figure 0. seller의 utility 변화 곡선 끝 ##########################


fig = plt.figure(14)
############## figure 0. seller의 utility 변화 곡선 #############################
# seller_utility_graph = fig.add_subplot(2, 2, 4)
utility_graph = fig.add_subplot(111)
# xs.reverse()
# utility_seller.reverse()
# utility_buyers.reverse()
# unit_prices.reverse()

xmax, ymax = unit_prices[np.argmax(utility_seller)], np.max(utility_seller)
# xmax2, ymax2 = xs[np.argmax(utility_buyers)], np.max(utility_buyers)
print(xmax, ymax)
l1 = utility_graph.plot(unit_prices, utility_seller)#, label='$ab/(1+bE_m)$ {:3f}'.format(a_seller*b_seller/(1+b_seller*(Emax_seller))))# + str(n))
l3 = utility_graph.plot(unit_prices, utility_buyers)
# utility_graph.legend(loc=2)
# bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
# arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
# kw = dict(xycoords='data',textcoords="axes fraction",
#           arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                xytext=(-50,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
# utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-5)
#         # arrowprops=dict(arrowstyle = "->", facecolor='black', shrink=0.05),
#         )

    # plt.legend()
# fig.subplots_adjust(hspace = 0.5)
# plt.title('Utilities vary according to $E_{seller}$')

utility_graph.set_xlabel('Unit price')
utility_graph.set_ylabel("Utility")
plt.legend(l1+l3, ['unit price', "buyer's mean utility"], loc='lower left')
plt.tight_layout()
############## figure 0. seller의 utility 변화 곡선 끝 ##########################

fig = plt.figure(15)
################### figure 3. seller의 utility 변화 곡선 ########################
seller_utility_graph = fig.gca(projection='3d')

# xs.reverse()
# best_E_sold.reverse()
# utility_seller.reverse()
# best_utility_seller.reverse()

arg = np.argmax(utility_seller)
xmax, ymax, price = xs[arg], np.max(utility_seller), unit_prices[arg]
seller_utility_graph.plot(unit_prices, xs, utility_seller, color = 'blue', label="seller's utility")
seller_utility_graph.text(price, xmax, ymax, 'max ({:.3f}, {:.3f}, {:.3f})'.format(price, xmax, ymax))

arg = np.argmax(utility_buyers)
xmax, ymax, price = xs[arg], np.max(utility_buyers), unit_prices[arg]
seller_utility_graph.plot(unit_prices, xs, utility_buyers, color = 'orange', label="buyers's mean utility")
# seller_utility_graph.annotate('max ({:.3f}, {:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-20))

# plt.title('Utility of seller when $E_{max}$='+'{:.3f}'.format(Emax_seller))
seller_utility_graph.set_xlabel('unit price') #+' max at {:.3f} theoretically'.format(seller_max_at))
seller_utility_graph.set_ylabel('trading energy $E_{s}$')
seller_utility_graph.set_zlabel("utility")
seller_utility_graph.legend(loc = 'upper left')
plt.tight_layout()
############## figure 3. seller의 utility 변화 곡선 끝 ##########################






############################## figure 0 끝 #####################################
# # E_sold 고정에서 seller의 utiltiy 최대화하는 unit_price 찾기
# price_min = ab_over_bEos[0]
# price_max = ab_over_bEos[-1]
# unit_price = price_min



##### NE 끝. NE와 공익최대의 solution을 비교하여 효율이 얼마나 떨어지는가 관찰 시작 ########

# unit_price를 가장 비싸게 놓고 돌려야 하나? 일단 E_final의 NE에서 구해진 unit_price로 돌렸음.
unit_price = manager.unit_price
# unit_price가 고정되어 있을 떄, seller가 공리주의를 추구하는 경우에 utility 최대로 되는 점은 언제인가.

# 파는 E (E_sold)에 따른 seller의 utility를 저장할 배열 생성 (figure 4 그래프)
utility_seller2 = []
xs2 = []
# initialize final_E. final_E는 seller의 utility가 최대가 되는 E_sold가 될 예정
E_sold = Emax_seller
final_E = E_sold
# E_sold를 -1씩 감소시키면서 utility_seller를 관찰
while True:
    manager.KKT_priced(E_sold)
    if E_sold <= 1:
        break
    for i in range(num_users):
        ws[i] = manager.users[i].w
        Es[i] = manager.users[i].E
        us[i] = manager.users[i].set_utility(Es[i], ws[i])
        # print("user {}'s E_i : {}, w_i : {}, utility : {}".format(i, Es[i], ws[i], us[i]))
    new_utility = seller.compute_utility(E_sold, sum(ws))
    xs2.append(E_sold)
    utility_seller2.append(new_utility)

    if seller.utility < new_utility:
        seller.utility = new_utility
        final_E = E_sold

    # print("final_E = {} Emax_seller = {}".format(final_E, Emax_seller))
    E_sold -= 1

print("final_E = {} Emax_seller = {}".format(final_E, Emax_seller))
print("Final graph loading...")


# 가장 싼 unit_price 부터 market clearing E 측정
unit_price = min_unit_price
# 파는 E (E_sold)에 따른 seller의 utility를 저장할 배열 생성 (figure 4 그래프)
market_clear_Es = []
unit_prices2 = []
# initialize final_E. final_E는 seller의 utility가 최대가 되는 E_sold가 될 예정
E_sold = Emax_seller
final_E = E_sold
flag = 0
new_max_unit_price = 0

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
trading_user_count2 = []
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
    trading_user_count2.append(manager.trading_user_count())

    E_sold -= 1

seller_max_at = seller.max_utility_arg(unit_price)
if seller_max_at <=0:
    seller_max_at = 0
if seller_max_at >= Emax_seller:
    seller_max_at = Emax_seller

# figure 2 생성
fig = plt.figure(4)
################### figure 2. unit_price에 따른 market clear E #################
# seller_utility_graph = fig.add_subplot(3, 1, 1)
seller_utility_graph = fig.add_subplot(111)


seller_utility_graph.plot(unit_prices2, market_clear_Es, color = 'blue', marker=markers[0], markerfacecolor = 'none', markevery=100, linewidth=1, label = 'AD curve')
seller_utility_graph.plot(unit_prices2[idx:], best_E_solds, color = 'green',marker=markers[1], markerfacecolor = 'none', markevery=100, linewidth=1, label =  "$(p_{trd}, E_s^{sp}(p_{trd}))$")
seller_utility_graph.plot(unit_prices2[idx:], real_E_solds, color = 'red', marker=markers[2], markerfacecolor = 'none', markevery=100, linewidth=1, linestyle = 'dashed', label = "$(p_{trd}, min(E_s^{sp}(p_{trd}), E_{AD}(p_{trd})))$")
seller_utility_graph.scatter(impossible[0], impossible[1])
seller_utility_graph.annotate('cross point:({:.3f}, {:.3f})'.format(impossible[0], impossible[1]), xy=(impossible[0], impossible[1]),  xycoords='data',
                xytext=(-50,20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )

# fig.subplots_adjust(hspace = 0.5)
# plt.title('Selling energy vs unit price')# : market clearing E_total when unit price is given')
plt.xlabel('Unit price')
plt.ylabel('Trading energy $E_{s}$')
plt.legend(loc = 'upper right')
plt.tight_layout()
################ figure 2. unit_price에 따른 market clear E 끝 #################

fig = plt.figure(5)
################### figure 2. seller의 utility 변화 곡선 ########################
# seller_utility_graph = fig.add_subplot(3, 1, 2)
utility_graph = fig.add_subplot(111)

# xs.reverse()
# best_E_sold.reverse()
# utility_seller.reverse()
# best_utility_seller.reverse()

arg = np.argmax(utility_seller2)
xmax, ymax, price = market_clear_Es[idx+arg], np.max(utility_seller2), unit_prices2[idx+arg]
utility_graph.plot(market_clear_Es[idx:], utility_seller2, color = 'blue', label="seller's utility on AD curve", linewidth=1, marker = markers[0], markerfacecolor='none', markevery = 100)
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                xytext=(0,10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
arg = np.argmax(best_utility_seller2)
xmax, ymax, price= best_E_solds[arg], np.max(best_utility_seller2), unit_prices2[idx+arg]
utility_graph.plot(best_E_solds, best_utility_seller2, color = 'green', label="seller's best utility regardless of AD", linewidth=1, marker = markers[1], markerfacecolor='none', markevery = 100)
# seller_utility_graph.annotate('max ({:.3f}, {:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-20))

arg = np.argmax(real_utility_seller2)
xmax, ymax, price= real_E_solds[arg], np.max(real_utility_seller2), unit_prices2[idx+arg]
utility_graph.plot(real_E_solds, real_utility_seller2, color = 'red', linestyle = 'dashed', label="seller's best utility under AD constraint", linewidth=1, marker = markers[2], markerfacecolor='none', markevery = 100)
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                xytext=(0,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
arg = np.argmax(utility_buyers2)
xmax, ymax, price= real_E_solds[arg], np.max(utility_buyers2), unit_prices2[idx+arg]
utility_graph.plot(real_E_solds, utility_buyers2, color = 'orange', label="buyers's best mean utility under AD constraint",linewidth=1, marker = markers[3], markerfacecolor='none', markevery = 100)
# seller_utility_graph.annotate('max ({:.3f}, {:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-20))

# fig.subplots_adjust(hspace = 0.5)
# plt.title('Utility varies according trading energy $E_{seller}$')
plt.xlabel('Trading energy $E_{s}$')#+' max at {:.3f} theoretically'.format(seller_max_at))
plt.ylabel("Utility")
plt.legend(loc = 'lower center')
plt.tight_layout()
############## figure 2. seller의 utility 변화 곡선 끝 ##########################

fig = plt.figure(6)
################### figure 2. seller의 utility 변화 곡선 ########################
# seller_utility_graph = fig.add_subplot(3, 1, 3)
utility_graph = fig.add_subplot(111)

arg = np.argmax(utility_seller2)
xmax, ymax = unit_prices2[idx+arg], np.max(utility_seller2)
utility_graph.plot(unit_prices2[idx:], utility_seller2, color = 'blue', label="seller's utility on AD curve", linewidth=1, marker = markers[0], markerfacecolor='none', markevery = 100)
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                xytext=(-70,10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
arg = np.argmax(best_utility_seller2)
xmax, ymax=unit_prices2[idx+arg], np.max(best_utility_seller2)
utility_graph.plot(unit_prices2[idx:], best_utility_seller2, color = 'green', label="seller's best utility regardless of AD", linewidth=1, marker = markers[1], markerfacecolor='none', markevery = 100)
# seller_utility_graph.annotate('max ({:.3f}, {:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-20))

arg = np.argmax(real_utility_seller2)
xmax, ymax= unit_prices2[idx+arg], np.max(real_utility_seller2)
utility_graph.plot(unit_prices2[idx:], real_utility_seller2, color = 'red', linestyle = 'dashed', label="seller's best utility under AD constraint", linewidth=1, marker = markers[2], markerfacecolor='none', markevery = 100)
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                xytext=(-70,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
arg = np.argmax(utility_buyers2)
xmax, ymax = unit_prices2[idx+arg], np.max(utility_buyers2)
utility_graph.plot(unit_prices2[idx:], utility_buyers2, color = 'orange', label="buyers's best mean utility under AD constraint", linewidth=1, marker = markers[3], markerfacecolor='none', markevery = 100)
# seller_utility_graph.annotate('max ({:.3f}, {:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-20))

# fig.subplots_adjust(hspace = 0.5)
# plt.title('Utility varies according unit price')
plt.xlabel('Unit price')#+' max at {:.3f} theoretically'.format(seller_max_at))
plt.ylabel("Utility")
plt.legend(loc = 'lower center')
plt.tight_layout()
############## figure 2. seller의 utility 변화 곡선 끝 ##########################
############## figure 2 끝 ############################## figure 2끝 ################

fig = plt.figure(7)
################### figure 3. seller의 utility 변화 곡선 ########################
utility_graph = fig.gca(projection='3d')

# xs.reverse()
# best_E_sold.reverse()
# utility_seller.reverse()
# best_utility_seller.reverse()

arg = np.argmax(utility_seller2)
xmax, ymax, price = market_clear_Es[idx+arg], np.max(utility_seller2), unit_prices2[idx+arg]
utility_graph.plot(unit_prices2[idx:], market_clear_Es[idx:], utility_seller2, color = 'blue', markerfacecolor = 'none', markevery=100, marker = markers[0], linewidth=1, label="seller's utility on AD curve")
utility_graph.text(price, xmax, ymax, 'max ({:.3f}, {:.3f}, {:.3f})'.format(price, xmax, ymax))

arg = np.argmax(best_utility_seller2)
xmax, ymax, price= best_E_solds[arg], np.max(best_utility_seller2), unit_prices2[idx+arg]
utility_graph.plot(unit_prices2[idx:], best_E_solds, best_utility_seller2, color = 'green', markerfacecolor = 'none', markevery=100, marker = markers[1], linewidth=1, label="seller's best utility regardless of AD")
# seller_utility_graph.text(price, xmax, ymax, 'max ({:.3f}, {:.3f}, {:.3f})'.format(price, xmax, ymax))

arg = np.argmax(utility_buyers2)
xmax, ymax, price = real_E_solds[arg], np.max(utility_buyers2), unit_prices2[idx+arg]
utility_graph.plot(unit_prices2[idx:], real_E_solds, utility_buyers2, color = 'orange',  markerfacecolor = 'none', markevery=100, marker = markers[3], linewidth=1, label="buyers's best mean utility under AD constraint")
# seller_utility_graph.annotate('max ({:.3f}, {:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-20))

arg = np.argmax(real_utility_seller2)
xmax, ymax, price= real_E_solds[arg], np.max(real_utility_seller2), unit_prices2[idx+arg]
utility_graph.plot(unit_prices2[idx:], real_E_solds, real_utility_seller2, color = 'red',  markerfacecolor = 'none', markevery=100, marker = markers[2], linewidth=1, linestyle = 'dashed', label="seller's best utility under AD constraint")
utility_graph.text(price, xmax, ymax, 'max ({:.3f}, {:.3f}, {:.3f})'.format(price, xmax, ymax))

# plt.title('Utility of seller when $E_{max}$='+'{:.3f}'.format(Emax_seller))
utility_graph.set_xlabel('Unit price') #+' max at {:.3f} theoretically'.format(seller_max_at))
utility_graph.set_ylabel('Trading energy $E_{s}$')
utility_graph.set_zlabel("Utility")
utility_graph.legend(loc = 'upper left')
plt.tight_layout()
############## figure 3. seller의 utility 변화 곡선 끝 ##########################
############## figure 3 끝 ############################## figure 3끝 ################.03



################### figure 0의 NE varification 그래프 끝 ########################

# Johari 결과랑 효율성 비교를 위한 KKT
# E_final만큼 거래될 때, buyer utility의 sum을 최대화하는 거래가 이루어지면,
# NE에 의해 거래가 이루어질 때랑 공익이 얼마나 차이나는지를 보여주기 위한 그래프
ws_priced = []
Es_priced = []
us_priced = []
manager.KKT_priced(xmax, price)
for i in range(len(users)):
    ws_priced.append(users[i].w)
    Es_priced.append(users[i].E)
    us_priced.append(users[i].set_utility(Es_priced[-1], ws_priced[-1]))
new_utility = seller.compute_utility(xmax, xmax*price)
print("seller's E_i : {}, w_i : {}, utility : {}".format(xmax, sum(ws_priced), new_utility))
print("buyers' total utility : {}".format(sum(us_priced)))

fig = plt.figure(8)
################ figure 0의 total utility 최대화 solution과 utiltiy##############
# bar_graph = fig.add_subplot(3,1,1)
bar_graph = fig.add_subplot(111)
rects2 = bar_graph.bar(bar_idx, ab_over_bEos, bar_width,
                        label ='$r_k$',
                        alpha = opacity)
rects4 = bar_graph.bar(bar_idx+2*bar_width, ws_priced, bar_width,
                        label ='$w_k$',
                        alpha = opacity)
rects5 = bar_graph.bar(bar_idx+3*bar_width, Es_priced, bar_width,
                        label ='$E_k$',
                        alpha = opacity)
rects6 = bar_graph.bar(bar_idx+4*bar_width, us_priced, bar_width,
                        label ='$Utility_k$',
                        alpha = opacity)
autolabel(rects2)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6, place='center')

# plt.title('distributed energy $E_i$ and payments $w_i$ when $E_s=$ {:.3f}'.format(xmax))

plt.xlabel("Buyers \ntotal utility : {:.3f}, unit price : {:.3f}, seller's utility : {:.3f}, trading energy : {:.3f}".format(sum(us_priced), price, new_utility, xmax))
plt.ylabel('$r_k$, $w_k$, $E_k$, $Utility_k$')
# plt.subplots_adjust(top=0.96,
# bottom=0.062,
# left=0.096,
# right=0.976,
# hspace=0.211,
# wspace=0.2)
plt.legend()
plt.tight_layout()

############ figure 0의 total utility 최대화 solution과 utiltiy 끝 ##############
ws_final = []
Es_final = []
us_final = []
manager.initialize(xmax)
for i in range(len(users)):
    Ei = users[i].buy_max_utility(price)
    Es_final.append(Ei)
    ws_final.append(Ei*price)
    us_final.append(users[i].set_utility(Ei, Ei*price))
new_utility = seller.compute_utility(xmax, xmax*price)
print("seller's E_i : {}, w_i : {}, utility : {} {}".format(xmax, sum(ws_final), ymax, new_utility))
print("buyers' total utility : {}, total E_sum={}".format(sum(us_final), sum(Es_final)))


fig = plt.figure(9)
################### figure 0의 막대 그래프 : NE solution과 utility ##############
# bar_graph = fig.add_subplot(3,1,2)
bar_graph = fig.add_subplot(111)
rects2 = bar_graph.bar(bar_idx, ab_over_bEos, bar_width,
                        label ='$r_k$',
                        alpha = opacity)
rects4 = bar_graph.bar(bar_idx+2*bar_width, ws_final, bar_width,
                        label ='$w_k$',
                        alpha = opacity)
rects5 = bar_graph.bar(bar_idx+3*bar_width, Es_final, bar_width,
                        label ='$E_k$',
                        alpha = opacity)
rects6 = bar_graph.bar(bar_idx+4*bar_width, us_final, bar_width,
                        label ='$Utility_k$',
                        alpha = opacity)
autolabel(rects2)
autolabel(rects4)
autolabel(rects5, place = 'center')
autolabel(rects6)
# plt.title('distributed energy $E_i$ and payments $w_i$ when $E_s=$ {:.3f}'.format(xmax))
plt.xlabel("Buyers \ntotal utility : {:.3f}, unit price : {:.3f}, seller's utility : {:.3f}, trading energy : {:.3f}".format(sum(us_final), price, new_utility, xmax))
plt.ylabel('$r_k$, $w_k$, $E_k$, $Utility_k$')
plt.tight_layout()
# plt.subplots_adjust(top=0.96,
# bottom=0.062,
# left=0.096,
# right=0.976,
# hspace=0.211,
# wspace=0.2)
plt.legend()
################ figure 0의 막대 그래프 : NE solution과 utility 끝 ##############

ws_final = []
Es_final = []
us_final = []
manager.initialize(xmax)
manager.KKT(True)
for i in range(len(users)):
    ws_final.append(users[i].w)
    Es_final.append(users[i].E)
    us_final.append(users[i].set_utility(Es_final[-1], ws_final[-1]))
new_utility = seller.compute_utility(xmax, sum(ws_final))
print("seller's E_i : {}, w_i : {}, utility : {}".format(xmax, sum(ws_final), new_utility))
print("buyers' total utility : {}".format(sum(us_final)))


fig = plt.figure(10)
################### figure 0의 막대 그래프 : NE solution과 utility ##############
# bar_graph = fig.add_subplot(3,1,3)
bar_graph = fig.add_subplot(111)
rects2 = bar_graph.bar(bar_idx, ab_over_bEos, bar_width,
                        label ='$r_k$',
                        alpha = opacity)
rects4 = bar_graph.bar(bar_idx+2*bar_width, ws_final, bar_width,
                        label ='$w_k$',
                        alpha = opacity)
rects5 = bar_graph.bar(bar_idx+3*bar_width, Es_final, bar_width,
                        label ='$E_k$',
                        alpha = opacity)
rects6 = bar_graph.bar(bar_idx+4*bar_width, us_final, bar_width,
                        label ='$Utility_k$',
                        alpha = opacity)
autolabel(rects2)
autolabel(rects4)
autolabel(rects5, place = 'center')
autolabel(rects6)
# plt.title('distributed energy $E_i$ and payments $w_i$ when $E_s=$ {:.3f}'.format(xmax))
plt.xlabel("Buyers \n total utility : {:.3f}, unit price : {:.3f}, seller's utility : {:.3f}, trading energy : {:.3f}".format(sum(us_final), manager.unit_price, new_utility, xmax))
plt.ylabel('$r_k$, $w_k$, $E_k$, $Utility_k$')
plt.tight_layout()
# plt.subplots_adjust(top=0.96,
# bottom=0.062,
# left=0.096,
# right=0.976,
# hspace=0.211,
# wspace=0.2)
plt.legend()
################ figure 0의 막대 그래프 : NE solution과 utility 끝 ##############

fig = plt.figure(11)
############## figure 0. seller의 utility 변화 곡선 #############################
# seller_utility_graph = fig.add_subplot(2, 2, 4)
utility_graph = fig.add_subplot(211)
user_count_graph = fig.add_subplot(212)
# xs.reverse()
# utility_seller.reverse()
# utility_buyers.reverse()
# unit_prices.reverse()

xmax, ymax = xs[np.argmax(utility_seller)], np.max(utility_seller)
xmax2, ymax2 = xs[np.argmax(utility_buyers)], np.max(utility_buyers)
ymax_ = utility_buyers[np.argmax(utility_seller)]
xmax3, ymax3 = real_E_solds[np.argmax(utility_seller2)], np.max(utility_seller2)
xmax4, ymax4 = real_E_solds[np.argmax(utility_buyers2)], np.max(utility_buyers2)
ymax_4 = utility_buyers2[np.argmax(utility_seller2)]
print(xmax, ymax)
l1 = utility_graph.plot(xs, utility_seller)
l2 = utility_graph.plot(xs, utility_buyers)
l3 = utility_graph.plot(real_E_solds, real_utility_seller2)
l4 = utility_graph.plot(real_E_solds, utility_buyers2)
l5 = user_count_graph.plot(xs, trading_user_count, linestyle='dashed', color ='purple')
l6 = user_count_graph.plot(real_E_solds, trading_user_count2, linestyle='dashed', color = 'yellow')


utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                xytext=(-50,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
# utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax2, ymax2), xy=(xmax2, ymax2),  xycoords='data',
#                 xytext=(-50,-20), textcoords='offset points',
#                 arrowprops=dict(arrowstyle="->")
#                 )
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax3, ymax3), xy=(xmax3, ymax3),  xycoords='data',
                xytext=(-50,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
# utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax4, ymax4), xy=(xmax4, ymax4),  xycoords='data',
#                 xytext=(-50,-20), textcoords='offset points',
#                 arrowprops=dict(arrowstyle="->")
#                 )
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax_), xy=(xmax, ymax_),  xycoords='data',
                xytext=(-50,-50), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax3, ymax_4), xy=(xmax3, ymax_4),  xycoords='data',
                xytext=(50,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
# utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-5)
#         # arrowprops=dict(arrowstyle = "->", facecolor='black', shrink=0.05),
#         )

    # plt.legend()
# fig.subplots_adjust(hspace = 0.5)
# plt.title('Utilities vary according to $E_{seller}$')

utility_graph.set_xlabel('trading energy $E_{s}$')
utility_graph.set_ylabel("utility")
user_count_graph.set_xlabel("trading energy $E_{s}$")
user_count_graph.set_ylabel("number of trading users")
user_count_graph.set_ylim((0, num_users+0.5))
plt.legend(l1+l2+l3+l4+l5+l6, ["seller's utility 1", "buyer's mean utility1", "seller's utility 2", "buyer's mean utility2", "#of trading users1", "#of trading users2"], loc='lower left')
plt.tight_layout()
############## figure 0. seller의 utility 변화 곡선 끝 ##########################


fig = plt.figure(12)
############## figure 0. seller의 utility 변화 곡선 #############################
# seller_utility_graph = fig.add_subplot(2, 2, 4)
unit_price_graph = fig.add_subplot(111)
# xs.reverse()
# utility_seller.reverse()
# utility_buyers.reverse()
# unit_prices.reverse()

l1 = unit_price_graph.plot(unit_prices, xs, linewidth=1, marker = markers[0], markerfacecolor='none', markevery = 4)
l2 = unit_price_graph.plot(unit_prices2[idx:], real_E_solds,linewidth=1, marker = markers[1], markerfacecolor='none', markevery = 50)

unit_price_graph.set_xlabel("Unit price")
unit_price_graph.set_ylabel('Trading energy $E_{s}$')
plt.legend(l1+l2, ["BPS", "SPS"], loc='lower left')
plt.tight_layout()
############## figure 0. seller의 utility 변화 곡선 끝 ##########################



fig = plt.figure(16)
############## figure 0. seller의 utility 변화 곡선 #############################
# seller_utility_graph = fig.add_subplot(2, 2, 4)
utility_graph = fig.add_subplot(211)
user_count_graph = fig.add_subplot(212)
# xs.reverse()
# utility_seller.reverse()
# utility_buyers.reverse()
# unit_prices.reverse()

xmax, ymax = unit_prices[np.argmax(utility_seller)], np.max(utility_seller)
xmax2, ymax2 = unit_prices[np.argmax(utility_buyers)], np.max(utility_buyers)
ymax_ = utility_buyers[np.argmax(utility_seller)]
xmax3, ymax3 = unit_prices2[idx+np.argmax(utility_seller2)], np.max(utility_seller2)
xmax4, ymax4 = unit_prices2[idx+np.argmax(utility_buyers2)], np.max(utility_buyers2)
ymax_4 = utility_buyers2[np.argmax(utility_seller2)]
print(xmax, ymax)
l1 = utility_graph.plot(unit_prices, utility_seller)
l2 = utility_graph.plot(unit_prices, utility_buyers)
l3 = utility_graph.plot(unit_prices2[idx:], real_utility_seller2)
l4 = utility_graph.plot(unit_prices2[idx:], utility_buyers2)
l5 = user_count_graph.plot(unit_prices, trading_user_count, linestyle='dashed', color='purple')
l6 = user_count_graph.plot(unit_prices2[idx:], trading_user_count2, linestyle='dashed', color='yellow')

utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax),  xycoords='data',
                xytext=(-50,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
# utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax2, ymax2), xy=(xmax2, ymax2),  xycoords='data',
#                 xytext=(-50,-20), textcoords='offset points',
#                 arrowprops=dict(arrowstyle="->")
#                 )
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax3, ymax3), xy=(xmax3, ymax3),  xycoords='data',
                xytext=(-50,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
# utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax4, ymax4), xy=(xmax4, ymax4),  xycoords='data',
#                 xytext=(-50,-20), textcoords='offset points',
#                 arrowprops=dict(arrowstyle="->")
#                 )
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax_), xy=(xmax, ymax_),  xycoords='data',
                xytext=(-50,-50), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax3, ymax_4), xy=(xmax3, ymax_4),  xycoords='data',
                xytext=(50,-20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
# utility_graph.annotate('max ({:.3f},{:.3f})'.format(xmax, ymax), xy=(xmax, ymax), xytext=(xmax, ymax-5)
#         # arrowprops=dict(arrowstyle = "->", facecolor='black', shrink=0.05),
#         )

    # plt.legend()
# fig.subplots_adjust(hspace = 0.5)
# plt.title('Utilities vary according to $E_{seller}$')

utility_graph.set_xlabel('unit price')
utility_graph.set_ylabel("utility")
user_count_graph.set_xlabel("unit price")
user_count_graph.set_ylabel("number of trading users")
user_count_graph.set_ylim((0,num_users+0.5))
plt.legend(l1+l2+l3+l4+l5+l6, ["seller's utility 1", "buyer's mean utility1", "seller's utility 2", "buyer's mean utility2", "#of trading users1", "#of trading users2"], loc='lower left')
plt.tight_layout()
############## figure 0. seller의 utility 변화 곡선 끝 ##########################





plt.show()

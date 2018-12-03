import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from buyer import Buyer
from seller import Seller
from manager import Manager
import pdb
import logging
import seaborn as sns
from scipy.stats import lognorm
import pylab as pl

# userinfo = np.load("userinfo.npy")
buyer_stats = np.load("./stats3.npy")
seller_stats = np.load("./seller_stats3.npy")
delta_stats = np.load("./delta_stats3.npy")
userinfo = np.load("./userinfo3.npy")

i=0
for bdelta in delta_stats[0]:
    if bdelta<0:
        print("{}, ".format(i))
        print(userinfo[i])
    i = i+1

plt.figure(0)
s1=plt.scatter(buyer_stats[2], seller_stats[1], label = "SPS utility", alpha='0.7', marker=MarkerStyle(marker='o',fillstyle='top'))
s2=plt.scatter(buyer_stats[0], seller_stats[0], label = "BPS utility", alpha='0.7', marker=MarkerStyle(marker='v', fillstyle='bottom'))
# plt.xticks([0, 100, 200, 300, 400, 500, 600, 700])
# plt.yticks([0, 100, 200, 300])#, 400, 500, 600, 700])
plt.axis('equal')
plt.xlabel("Total utility of buyers")
plt.ylabel("Utility of seller")
plt.tight_layout()
plt.legend()

plt.figure(1)
plt.scatter(delta_stats[0], delta_stats[1], alpha='0.7')
# plt.yticks([-90, -80, -70, -60, -50, -40, -30, -20, -10, 0])#, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.xticks([-40, -20, 0, 20, 40, 60, 80, 100, 120, 140])
plt.axis('equal')
plt.xlabel("Total utility changes of buyers (SPS to BPS)")
plt.ylabel("Utility changes of seller (SPS to BPS)")
# plt.plot(x, pdf, linewidth=2, color='r')
plt.tight_layout()
plt.show()

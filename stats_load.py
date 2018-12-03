import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from buyer import Buyer
from seller import Seller
from manager import Manager
import pdb
import logging
import seaborn as sns
from scipy.stats import lognorm
import pylab as pl

# userinfo = np.load("userinfo.npy")
stats = np.load("./stats.npy")

y = []
for stat in stats:
    y.append(1-stat[0]/stat[1])

mu = np.mean(y)
sigma = np.std(y)

print(mu, sigma)

ax = plt.subplot(111)
n, bins, patches = ax.hist(y, bins=100, normed=True, align='mid')
ax.set_xscale("linear")

shape, loc, scale = lognorm.fit(y)
x = np.linspace(min(bins),max(bins), 10000)
pdf = lognorm.pdf(x, shape, loc, scale)
# pdb.set_trace()
# dist = lognorm(s=sigma)

ax.plot(x, pdf, 'r')
ax.set_xlabel("inefficiency")
ax.set_ylabel("frequency and fitted log normal pdf")
# plt.plot(x, pdf, linewidth=2, color='r')

print("mu:{}, sigma:{}, shape:{}, loc:{}, scale:{}".format(mu, sigma, shape, loc, scale))
plt.show()

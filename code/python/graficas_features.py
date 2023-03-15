import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys


cap = str(sys.argv[1])
lineas = sys.argv[2].strip().split()[0]

#featuresOriginales = cap + "_features.csv"
#features = "features_normalized/" + cap + "_features_normalized.csv"
features = cap + "_features_normalized.csv"

datasetOriginales = pd.read_csv(str(features))
dataset = pd.read_csv(str(features))

firstDigit = lineas[0]
nroCifras = len(lineas)
next = 0
#if ((int(firstDigit)+1)*(10**(nroCifras-1)) >= int(firstDigit)*(10**(nroCifras-1))+int(int(firstDigit)*(10**(nroCifras-1))/2)):
if ((int(firstDigit)*(10**(nroCifras-1))+(10**(nroCifras-1))/2) <= int(lineas)): #int(firstDigit)*(10**(nroCifras-1))):
    next = (int(firstDigit)+1)*(10**(nroCifras-1))
else: 
    next = int(firstDigit)*(10**(nroCifras-1)) + int((10**(nroCifras-1))/2)

alto = next + int("30"+"0"*(nroCifras-3))



fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, sharex=False, sharey=True,  figsize=(10,10))
_=plt.rc('xtick',labelsize=10)
_=plt.rc('ytick',labelsize=10)
num_bins=5

freq1, bins1, patches1 = ax1.hist(dataset.iloc[:,1], num_bins, facecolor='tab:blue', edgecolor="black",)
_=ax1.ticklabel_format(style='sci',scilimits=(-5,5))
_=ax1.set_xlabel('ID',fontsize=14)
_=ax1.set_ylim(0, alto)
bin_centers1 = np.diff(bins1)*0.5 + bins1[:-1]
for n, (fr, x, patch) in enumerate(zip(freq1, bin_centers1, patches1)):
    height = int(freq1[n])
    _=ax1.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2), textcoords = "offset points", ha = 'center', va = 'bottom')

freq2, bins2, patches2 = ax2.hist(dataset.iloc[:,2], num_bins, facecolor='tab:orange', edgecolor="black")
_=ax2.ticklabel_format(style='sci',scilimits=(-5,5))
_=ax2.set_xlabel('OD',fontsize=14)
_=ax2.set_ylim(0, alto)
bin_centers2 = np.diff(bins2)*0.5 + bins2[:-1]
for n, (fr, x, patch) in enumerate(zip(freq2, bin_centers2, patches2)):
    height = int(freq2[n])
    _=ax2.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2), textcoords = "offset points", ha = 'center', va = 'bottom')

freq3, bins3, patches3 = ax3.hist(dataset.iloc[:,3], num_bins, facecolor='tab:green', edgecolor="black")
_=ax3.ticklabel_format(style='sci',scilimits=(-5,5))
_=ax3.set_xlabel('IDW',fontsize=14)
_=ax3.set_ylabel('Frequency',fontsize=16)
_=ax3.set_ylim(0, alto)
bin_centers3 = np.diff(bins3)*0.5 + bins3[:-1]
for n, (fr, x, patch) in enumerate(zip(freq3, bin_centers3, patches3)):
    height = int(freq3[n])
    _=ax3.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2), textcoords = "offset points", ha = 'center', va = 'bottom')

freq4, bins4, patches4 = ax4.hist(dataset.iloc[:,4], num_bins, facecolor='tab:red', edgecolor="black")
_=ax4.ticklabel_format(style='sci',scilimits=(-5,5))
_=ax4.set_xlabel('ODW',fontsize=14)
_=ax4.set_ylim(0, alto)
bin_centers4 = np.diff(bins4)*0.5 + bins4[:-1]
for n, (fr, x, patch) in enumerate(zip(freq4, bin_centers4, patches4)):
    height = int(freq4[n])
    _=ax4.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2), textcoords = "offset points", ha = 'center', va = 'bottom')

freq5, bins5, patches5 = ax5.hist(dataset.iloc[:,5], num_bins, facecolor='tab:purple', edgecolor="black")
_=ax5.ticklabel_format(style='sci',scilimits=(-5,5))
_=ax5.set_xlabel('BC',fontsize=14)
_=ax5.set_ylim(0, alto)
bin_centers5 = np.diff(bins5)*0.5 + bins5[:-1]
for n, (fr, x, patch) in enumerate(zip(freq5, bin_centers5, patches5)):
    height = int(freq5[n])
    _=ax5.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2), textcoords = "offset points", ha = 'center', va = 'bottom')

freq6, bins6, patches6 = ax6.hist(dataset.iloc[:,6], num_bins, facecolor='tab:brown', edgecolor="black")
_=ax6.ticklabel_format(style='sci',scilimits=(-5,5))
_=ax6.set_xlabel('LCC',fontsize=14)
_=ax6.set_ylim(0, alto)
bin_centers6 = np.diff(bins6)*0.5 + bins6[:-1]
for n, (fr, x, patch) in enumerate(zip(freq6, bin_centers6, patches6)):
    height = int(freq6[n])
    _=ax6.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2), textcoords = "offset points", ha = 'center', va = 'bottom')

freq7, bins7, patches7 = ax7.hist(dataset.iloc[:,7], num_bins, facecolor='tab:pink', edgecolor="black")
_=ax7.ticklabel_format(style='sci',scilimits=(-5,5))
_=ax7.set_xlabel('AC',fontsize=14)
_=ax7.set_ylim(0, alto)
bin_centers7 = np.diff(bins7)*0.5 + bins7[:-1]
for n, (fr, x, patch) in enumerate(zip(freq7, bin_centers7, patches7)):
    height = int(freq7[n])
    _=ax7.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2), textcoords = "offset points", ha = 'center', va = 'bottom')

_=ax8.axis('off')

_=fig.suptitle(str(cap),fontsize=16, y=0.995)
_=fig.tight_layout()
_=plt.subplot_tool()

#plt.savefig(str(cap)+'.pdf', dpi=300)
plt.savefig(str(cap)+'.png', dpi=300)

################################################################################################

# PRIMER HISTOGRAMA

"""
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


dataset=pd.read_csv("capture20110816-2_features_normalized.csv")

num_bins=5
freq, bins, patches = plt.hist(dataset.iloc[:,1], num_bins, facecolor='blue', edgecolor="black") #, alpha=0.5)
_=plt.xlabel("ID")
_=plt.ylabel("Frequency")
_=plt.title("capture20110818 completo")

bin_centers = np.diff(bins)*0.5 + bins[:-1]
for n, (fr, x, patch) in enumerate(zip(freq, bin_centers, patches)):
    height = int(freq[n])
    _=plt.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2), textcoords = "offset points", ha = 'center', va = 'bottom')

plt.show()


dataset[dataset["ID"]>=20845]
# fila "3": 147.32.84.229

num_bins=5
freq, bins, patches = plt.hist(dataset.drop(3).iloc[:,1], num_bins, facecolor='blue', edgecolor="black") #, alpha=0.5)
_=plt.xlabel("ID")
_=plt.ylabel("Frequency")
_=plt.title("capture20110816-2 sin nodo 147.32.84.229")

bin_centers = np.diff(bins)*0.5 + bins[:-1]
for n, (fr, x, patch) in enumerate(zip(freq, bin_centers, patches)):
    height = int(freq[n])
    _=plt.annotate("{}".format(height), xy = (x, height), xytext = (0,0.2), textcoords = "offset points", ha = 'center', va = 'bottom')

plt.show()
"""

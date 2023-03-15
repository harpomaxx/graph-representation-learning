import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import rc


df = pd.read_csv("capture20110810_pkts_bytes_noZeros.csv")

#dfdrop_Src = df.drop(index=[1125337,1373835]).copy()
df_filter_Src = df.loc[( (df["SrcPkts"] <= 16) & (df["SrcBytes"] <= 1000) )]
df_filter_Dst = df.loc[( (df["DstPkts"] <= 16) & (df["DstBytes"] <= 1000) )]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False,  figsize=(10,5))
_=plt.rc('xtick',labelsize=12)
_=plt.rc('ytick',labelsize=12)

#_=ax1.scatter(dfdrop_Src["SrcPkts"], dfdrop_Src["SrcBytes"])
_=ax1.scatter(df["SrcPkts"], df["SrcBytes"])
_=ax1.ticklabel_format(style='sci',scilimits=(0,0))
_=ax1.set_title("Todas las observaciones", fontsize=16)

_=ax2.scatter(df_filter_Src["SrcPkts"], df_filter_Src["SrcBytes"])
_=ax2.set_xticks(np.arange(0,17,2))
_=ax2.set_xlim(0, 17)
_=ax2.set_title("SrcPkts $\leq 16$ \n SrcBytes $\leq 1000$", fontsize=16)

# add a big axis, hide frame
_=fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
_=plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
_=plt.xlabel("SrcPkts", labelpad=15.0, fontsize=14)
_=plt.ylabel("SrcBytes", labelpad=5.0, fontsize=14)

#plt.subplot_tool()
#plt.show()

plt.subplots_adjust(bottom=0.154) #, wspace=0.5, hspace=0.4, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.5)

#plt.savefig("src_correlation.pdf", dpi=300)
plt.savefig("src.png", dpi=300)

#########################################

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False,  figsize=(10,5))
_=plt.rc('xtick',labelsize=12)
_=plt.rc('ytick',labelsize=12)

_=ax1.scatter(df["DstPkts"], df["DstBytes"])
_=ax1.ticklabel_format(style='sci',scilimits=(0,0))
_=ax1.set_title("Todas las observaciones", fontsize=16)

_=ax2.scatter(df_filter_Dst["DstPkts"], df_filter_Dst["DstBytes"])
_=ax2.set_xticks(np.arange(0,17,2))
_=ax2.set_xlim(0, 17)
_=ax2.set_title("DstPkts $\leq 16$ \n DstBytes $\leq 1000$", fontsize=16)

# add a big axis, hide frame
_=fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
_=plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
_=plt.xlabel("DstPkts", labelpad=16.0, fontsize=14)
_=plt.ylabel("DstBytes", labelpad=10.0, fontsize=14)

#plt.subplot_tool()
#plt.show()

plt.subplots_adjust(bottom=0.143,left=0.095,right=0.963)
#plt.savefig("dst_correlation.pdf", dpi=300)
plt.savefig("dst.png", dpi=300)

####################################################

fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False,  figsize=(15,10))
_=plt.rc('xtick',labelsize=12)
_=plt.rc('ytick',labelsize=12)

_=ax1.scatter(df["SrcPkts"], df["SrcBytes"])
_=ax1.ticklabel_format(style='sci',scilimits=(0,0))
_=ax1.set_title("$\mathbf{Src}$ \n\n Todas las observaciones", fontsize=15)

_=ax2.scatter(df["DstPkts"], df["DstBytes"])
_=ax2.ticklabel_format(style='sci',scilimits=(0,0))
_=ax2.set_title("$\mathbf{Dst}$ \n\n Todas las observaciones", fontsize=15)

_=fig.add_subplot(121, frameon=False)
_=plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
_=plt.xlabel("SrcPkts", labelpad=18.0, fontsize=14)
_=plt.ylabel("SrcBytes", labelpad=9.0, fontsize=14)

_=ax3.scatter(df_filter_Src["SrcPkts"], df_filter_Src["SrcBytes"])
_=ax3.set_xticks(np.arange(0,17,2))
_=ax3.set_xlim(0, 17)
_=ax3.set_title("SrcPkts$\leq16\quad$ & $\quad$SrcBytes$\leq1000$", fontsize=15)

_=ax4.scatter(df_filter_Dst["DstPkts"], df_filter_Dst["DstBytes"])
_=ax4.set_xticks(np.arange(0,17,2))
_=ax4.set_xlim(0, 17)
_=ax4.set_title("DstSrc$\leq16\quad$ & $\quad$DstBytes$\leq1000$", fontsize=15)

_=fig.add_subplot(122, frameon=False)
_=plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
_=plt.xlabel("DstPkts", labelpad=18.0, fontsize=14)
_=plt.ylabel("DstBytes", labelpad=9.0, fontsize=14)

plt.subplots_adjust(wspace=0.5, hspace=0.4)
#left=0.1,
                    #bottom=0.1, 
                    #right=0.9, 
                    #top=0.9, 
                    #wspace=0.3, 
                    #hspace=0.5)

plt.savefig("pkts_vs_bytes.pdf", dpi=300)


####################################################

fig, (ax1, ax2) = plt.subplots(nrows=, ncols=2, sharex=False, sharey=False,  figsize=(15,10))
_=plt.rc('xtick',labelsize=12)
_=plt.rc('ytick',labelsize=12)

_=ax1.scatter(df["SrcPkts"], df["SrcBytes"])
_=ax1.ticklabel_format(style='sci',scilimits=(0,0))
_=ax1.set_title("$\mathbf{Src}$ \n\n Todas las observaciones", fontsize=15)

_=ax2.scatter(df["DstPkts"], df["DstBytes"])
_=ax2.ticklabel_format(style='sci',scilimits=(0,0))
_=ax2.set_title("$\mathbf{Dst}$ \n\n Todas las observaciones", fontsize=15)

_=fig.add_subplot(121, frameon=False)
_=plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
_=plt.xlabel("SrcPkts", labelpad=18.0, fontsize=14)
_=plt.ylabel("SrcBytes", labelpad=9.0, fontsize=14)

_=ax3.scatter(df_filter_Src["SrcPkts"], df_filter_Src["SrcBytes"])
_=ax3.set_xticks(np.arange(0,17,2))
_=ax3.set_xlim(0, 17)
_=ax3.set_title("SrcPkts$\leq16\quad$ & $\quad$SrcBytes$\leq1000$", fontsize=15)

_=ax4.scatter(df_filter_Dst["DstPkts"], df_filter_Dst["DstBytes"])
_=ax4.set_xticks(np.arange(0,17,2))
_=ax4.set_xlim(0, 17)
_=ax4.set_title("DstSrc$\leq16\quad$ & $\quad$DstBytes$\leq1000$", fontsize=15)

_=fig.add_subplot(122, frameon=False)
_=plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
_=plt.xlabel("DstPkts", labelpad=18.0, fontsize=14)
_=plt.ylabel("DstBytes", labelpad=9.0, fontsize=14)

plt.subplots_adjust(wspace=0.5, hspace=0.4)
#left=0.1,
                    #bottom=0.1, 
                    #right=0.9, 
                    #top=0.9, 
                    #wspace=0.3, 
                    #hspace=0.5)

plt.savefig("pkts_vs_bytes.pdf", dpi=300)


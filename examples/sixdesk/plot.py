import sixtracktools

data=sixtracktools.SixBin("singletrackfile.dat")

import matplotlib.pyplot as plt

x=data.part[0,:,2]
px=data.part[0,:,3]
y=data.part[0,:,4]
py=data.part[0,:,5]
sig=data.part[0,:,6]
delta=data.part[0,:,7]

fig,axs=plt.subplots(2,2)
axs[0,0].plot(x,px,'.')
axs[0,0].set_xlabel("x")
axs[0,0].set_ylabel("px")

axs[0,1].plot(y,py,'.')
axs[0,1].set_xlabel("y")
axs[0,1].set_ylabel("py")

axs[1,0].plot(sig,delta,'.')
axs[1,0].set_xlabel("sig")
axs[1,0].set_ylabel("delta")

axs[1,1].plot(delta,x,'.',label='x')
axs[1,1].plot(delta,y,'.',label='y')
axs[1,1].set_xlabel("delta")
axs[1,1].set_ylabel("xy")

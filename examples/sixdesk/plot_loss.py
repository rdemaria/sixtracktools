import sixtracktools


data=sixtracktools.SixBin("stf_loss.dat")

data.plot_lossturns()
data.plot_phasespace(0)
data.plot_phasespace(20)
data.plot_phasespace(50)
data.show()



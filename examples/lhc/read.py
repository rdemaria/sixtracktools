import sixtracktools


six =sixtracktools.SixTrackInput('.')
line,rest=six.expand_struct()

beam=sixtracktools.SixDump3('dump3.dat')
fb=beam.get_full_beam()



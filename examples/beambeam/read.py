import sys
sys.path.append('../../')#containst sixtracktools

import sixtracktools

six =sixtracktools.SixTrackInput('.')
line,rest, iconv =six.expand_struct()



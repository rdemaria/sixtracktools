import sys
sys.path.append('../../')#containst sixtracktools

import sixtracktools

six = sixtracktools.SixInput('.')
line, rest, iconv =six.expand_struct()

# Check for beam beam
for ee in line:
    if ee[0].startswith('bb'):
        print(ee)



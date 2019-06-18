import sixtracktools

six =sixtracktools.SixInput('.')


beam=sixtracktools.SixDump101('res/dump3.dat')
fb=beam.get_minimal_beam()




print("#initial condition particle 1")
for ll in sorted(fb):
    dt=fb[ll]
    if hasattr(dt,'__len__'):
       print(ll,dt[0])
    else:
       print(ll,dt)


print("#initial condition particle 2")

for ll in sorted(fb):
    dt=fb[ll]
    if hasattr(dt,'__len__'):
       print(ll,dt[1])
    else:
       print(ll,dt)





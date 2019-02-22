import numpy
import pylab

# intial parameters
n_iter = 800 #number of timesteps; equivalent to update rate of Navisens app

portion = 0.4
sz_temp = int(portion*n_iter)
sz_temp2 = n_iter - sz_temp

sz = (n_iter,) # size of array
x = 1 # truth value (deck number)
x2 = 2


# to simulate noisy measurements; can rewrite to take in measurements timestep by timestep
z = numpy.random.normal(x,0.1,size=sz_temp) # observations (normal about x, sigma=0.1)
z = numpy.append(z, numpy.random.normal(x2,0.1,size=sz_temp2))

Q = 1e-5 # process variance

# allocate space for arrays
xhat=numpy.zeros(sz)      # a posteri estimate of x
P=numpy.zeros(sz)         # a posteri error estimate
xhatminus=numpy.zeros(sz) # a priori estimate of x
Pminus=numpy.zeros(sz)    # a priori error estimate
K=numpy.zeros(sz)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1,n_iter):
	# time update
	xhatminus[k] = xhat[k-1]
	Pminus[k] = P[k-1]+Q

	# measurement update
	K[k] = Pminus[k]/( Pminus[k]+R )
	xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
	P[k] = (1-K[k])*Pminus[k]

pylab.figure()
pylab.plot(z,'k+',label='noisy measurements')
pylab.plot(xhat,'b-',label='a posteri estimate')
pylab.axhline(x, color='g',label='truth value_deck1')

pylab.axhline(x2, color='g',label='truth value_deck2')

pylab.legend()
pylab.xlabel('Iteration')
pylab.ylabel('Deck number')

"""
pylab.figure()
valid_iter = range(1,n_iter) # Pminus not valid at step 0
pylab.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
pylab.xlabel('Iteration')
pylab.ylabel('$(Voltage)^2$')
pylab.setp(pylab.gca(),'ylim',[0,.01])
"""

pylab.show()
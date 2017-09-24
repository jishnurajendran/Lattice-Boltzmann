"""
Lattice-Boltzmann method for fluid simulation

Simple rectangular barrier 
@author: Jishnu
"""
import numpy, time, matplotlib.pyplot, matplotlib.animation

height = 80							# dimensions of lattice
width = 200
viscosity = 0.02					# viscosity
omega = 1 / (3*viscosity + 0.5)		# parameter for "relaxation"
u0 = 0.1							# initial and in-flow speed
f_n = 4.0/9.0						# lattice-Boltzmann weight factors
o_n   = 1.0/9.0
o_36  = 1.0/36.0
performanceData = True				# True if performance data is needed

# Initialize arrays --steady rightward flow:
n0 = f_n * (numpy.ones((height,width)) - 1.5*u0**2)	# particle densities along 9 directions
nN = o_n * (numpy.ones((height,width)) - 1.5*u0**2)
nS = o_n * (numpy.ones((height,width)) - 1.5*u0**2)
nE = o_n * (numpy.ones((height,width)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nW = o_n * (numpy.ones((height,width)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nNE = o_36 * (numpy.ones((height,width)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nSE = o_36 * (numpy.ones((height,width)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nNW = o_36 * (numpy.ones((height,width)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nSW = o_36 * (numpy.ones((height,width)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW		# macroscopic density
ux = (nE + nNE + nSE - nW - nNW - nSW) / rho				# macroscopic x velocity
uy = (nN + nNE + nNW - nS - nSE - nSW) / rho				# macroscopic y velocity

# Initialize barriers:
barrier = numpy.zeros((height,width), bool)										# True wherever there's a barrier
barrier[(height/2)-8:(height/2)+8, (height/2)-4:(height/2)+4] = True			# simple linear barrier
barrierN = numpy.roll(barrier,  1, axis=0)										# sites just north of barriers
barrierS = numpy.roll(barrier, -1, axis=0)										# sites just south of barriers
barrierE = numpy.roll(barrier,  1, axis=1)										# etc.
barrierW = numpy.roll(barrier, -1, axis=1)
barrierNE = numpy.roll(barrierN,  1, axis=1)
barrierNW = numpy.roll(barrierN, -1, axis=1)
barrierSE = numpy.roll(barrierS,  1, axis=1)
barrierSW = numpy.roll(barrierS, -1, axis=1)

# Move all particles by one step along their directions of motion (pbc):
def stream():
	global nN, nS, nE, nW, nNE, nNW, nSE, nSW
	nN  = numpy.roll(nN,   1, axis=0)	# axis 0 is north-south; + direction is north
	nNE = numpy.roll(nNE,  1, axis=0)
	nNW = numpy.roll(nNW,  1, axis=0)
	nS  = numpy.roll(nS,  -1, axis=0)
	nSE = numpy.roll(nSE, -1, axis=0)
	nSW = numpy.roll(nSW, -1, axis=0)
	nE  = numpy.roll(nE,   1, axis=1)	# axis 1 is east-west; + direction is east
	nNE = numpy.roll(nNE,  1, axis=1)
	nSE = numpy.roll(nSE,  1, axis=1)
	nW  = numpy.roll(nW,  -1, axis=1)
	nNW = numpy.roll(nNW, -1, axis=1)
	nSW = numpy.roll(nSW, -1, axis=1)
	# Using boolean arrays to handle barrier collisions (bounce-back):
	nN[barrierN] = nS[barrier]
	nS[barrierS] = nN[barrier]
	nE[barrierE] = nW[barrier]
	nW[barrierW] = nE[barrier]
	nNE[barrierNE] = nSW[barrier]
	nNW[barrierNW] = nSE[barrier]
	nSE[barrierSE] = nNW[barrier]
	nSW[barrierSW] = nNE[barrier]

# Collide particles within each cell to redistribute velocities (can be optimized a little more):
def collide():
	global rho, ux, uy, n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW
	rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW
	ux = (nE + nNE + nSE - nW - nNW - nSW) / rho
	uy = (nN + nNE + nNW - nS - nSE - nSW) / rho
	ux2 = ux * ux				# pre-compute terms used repeatedly...
	uy2 = uy * uy
	u2 = ux2 + uy2
	omu215 = 1 - 1.5*u2			# "one minus u2 times 1.5"
	uxuy = ux * uy
	n0 = (1-omega)*n0 + omega * f_n * rho * omu215
	nN = (1-omega)*nN + omega * o_n * rho * (omu215 + 3*uy + 4.5*uy2)
	nS = (1-omega)*nS + omega * o_n * rho * (omu215 - 3*uy + 4.5*uy2)
	nE = (1-omega)*nE + omega * o_n * rho * (omu215 + 3*ux + 4.5*ux2)
	nW = (1-omega)*nW + omega * o_n * rho * (omu215 - 3*ux + 4.5*ux2)
	nNE = (1-omega)*nNE + omega * o_36 * rho * (omu215 + 3*(ux+uy) + 4.5*(u2+2*uxuy))
	nNW = (1-omega)*nNW + omega * o_36 * rho * (omu215 + 3*(-ux+uy) + 4.5*(u2-2*uxuy))
	nSE = (1-omega)*nSE + omega * o_36 * rho * (omu215 + 3*(ux-uy) + 4.5*(u2-2*uxuy))
	nSW = (1-omega)*nSW + omega * o_36 * rho * (omu215 + 3*(-ux-uy) + 4.5*(u2+2*uxuy))
	# Force steady rightward flow at ends (no need to set 0, N, and S components):
	nE[:,0] = o_n * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nW[:,0] = o_n * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nNE[:,0] = o_36 * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nSE[:,0] = o_36 * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nNW[:,0] = o_36 * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nSW[:,0] = o_36 * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)

# Compute curl of the macroscopic velocity field:
def curl(ux, uy):
	return numpy.roll(uy,-1,axis=1) - numpy.roll(uy,1,axis=1) - numpy.roll(ux,-1,axis=0) + numpy.roll(ux,1,axis=0)

# graphics and animation...
theFig = matplotlib.pyplot.figure(figsize=(8,3))
fluidImage = matplotlib.pyplot.imshow(curl(ux, uy), origin='lower', norm=matplotlib.pyplot.Normalize(-.1,.1),
									cmap=matplotlib.pyplot.get_cmap('jet'), interpolation='none')
		# for more details on animation look http://www.loria.fr/~rougier/teaching/matplotlib/#colormaps
bImageArray = numpy.zeros((height, width, 4), numpy.uint8)	# an RGBA image
bImageArray[barrier,3] = 255								# set alpha=255 barrier sites only
barrierImage = matplotlib.pyplot.imshow(bImageArray, origin='lower', interpolation='none')

# Function called for each successive animation frame:
startTime = time.clock()
#frameList = open('frameList.txt','w')		# file containing list of images (to make movie)
def nextFrame(arg):							# (arg is the frame number, which we don't need)
	global startTime
	if performanceData and (arg%100 == 0) and (arg > 0):
		endTime = time.clock()
		print "%1.1f" % (100/(endTime-startTime)), 'frames per second'
		startTime = endTime
	#frameName = "frame%04d.png" % arg
	#matplotlib.pyplot.savefig(frameName)
	#frameList.write(frameName + '\n')
	for step in range(15):					# adjust number of steps for smooth animation
		stream()
		collide()
	fluidImage.set_array(curl(ux, uy))
	return (fluidImage, barrierImage)		# return the figure elements to redraw

animate = matplotlib.animation.FuncAnimation(theFig, nextFrame, interval=0.5, blit=True)
matplotlib.pyplot.show()

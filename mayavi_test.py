import numpy
from mayavi import mlab
from matplotlib import pyplot


t = numpy.linspace(0, 4 * numpy.pi, 20)
cos = numpy.cos
sin = numpy.sin

x = sin(2 * t)
y = cos(t)
z = cos(2 * t)
s = 2 + sin(t)



# mlab.options.offscreen = True

mlab.points3d(x, y, z, s, colormap="copper", scale_factor=.25)

#mlab.savefig('mayavi_test.png')
#mlab.show()

plot = mlab.screenshot()
pyplot.figure()
pyplot.imshow(plot)
pyplot.show()

print 'done'

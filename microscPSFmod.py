#!/usr/bin/env python
"""
Generate a PSF using the Gibson and Lanni model.

Note: All distance units are microns.

This is slightly reworked version of the Python code provided by Kyle
Douglass, "Implementing a fast Gibson-Lanni PSF solver in Python".

http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python.html


References:

1. Li et al, "Fast and accurate three-dimensional point spread function computation
   for fluorescence microscopy", JOSA, 2017.

2. Gibson, S. & Lanni, F. "Experimental test of an analytical model of
   aberration in an oil-immersion objective lens used in three-dimensional
   light microscopy", J. Opt. Soc. Am. A 9, 154-166 (1992), [Originally
   published in J. Opt. Soc. Am. A 8, 1601-1613 (1991)].

3. Kirshner et al, "3-D PSF fitting for fluorescence microscopy: implementation
   and localization application", Journal of Microscopy, 2012.

Hazen 04/18
"""
import cmath
import math
import numpy
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.special
import h5py
import time
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt

def calcRv(dxy, xy_size, sampling=4):
	"""
	Calculate rv vector, this is 2x up-sampled.
	"""
	rv_max = math.sqrt(0.5 * xy_size * xy_size) + 1
	return numpy.arange(0, rv_max * dxy, dxy / sampling)

def configure(mp, wvl, num_basis):
	# Scaling factors for the Fourier-Bessel series expansion
	min_wavelength = 0.603 #0.436 # microns
	scaling_factor = mp["NA"] * (3 * numpy.arange(1, num_basis + 1) - 2) * min_wavelength / wvl

	# Not sure this is completely correct for the case where the axial
	# location of the flourophore is 0.0.
	#
	max_rho = min([mp["NA"], mp["ng0"], mp["ng"], mp["ni0"], mp["ni"], mp["ns"]]) / mp["NA"]

	return [scaling_factor, max_rho]

def gLXYZParticleScan(mp, dxy, xy_size, pz, writedisk, sample, num_basis, rho_samples, normalize = True, wvl = 0.6, zd = None, zv = 0.0, px=0, py=0):
	"""
	Calculate 3D G-L PSF. This is models the PSF you would measure by scanning a particle
	through the microscopes focus.

	This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
	is the zeroth dimension of the PSF.

	mp - The microscope parameters dictionary.
	dxy - Step size in the XY plane.
	xy_size - Number of pixels in X/Y.
	pz - A numpy array containing the particle z position above the coverslip (positive values only)
		 in microns.

	normalize - Normalize the PSF to unit height.
	wvl - Light wavelength in microns.
	zd - Actual camera position in microns. If not specified the microscope tube length is used.
	zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
	"""
	# Calculate rv vector, this is 2x up-sampled.
	rv = calcRv(dxy, xy_size)

	# Sampling points in Z used for interpolation later
	pz_ = numpy.arange(0, 1.5, step=0.001)

	# Calculate radial/Z PSF.
	PSF_rz = gLZRParticleScan(mp, rv, pz_, num_basis, rho_samples, normalize = normalize, wvl = wvl, zd = zd, zv = zv)

	# Create XYZ PSF by interpolation.
	return psfRZToPSFXYZ(dxy, xy_size, pz, pz_, rv, PSF_rz, writedisk, sample, px, py).reshape((px.shape[0],-1))

def gLZRScan(mp, pz, rv, zd, zv, num_basis, rho_samples, normalize = True, wvl = 0.6):
	"""
	Calculate radial G-L at specified radius. This function is primarily designed
	for internal use. Note that only one pz, zd and zv should be a numpy array
	with more than one element. You can simulate scanning the focus, the particle
	or the camera but not 2 or 3 of these values at the same time.

	mp - The microscope parameters dictionary.
	pz - A numpy array containing the particle z position above the coverslip (positive values only).
	rv - A numpy array containing the radius values.
	zd - A numpy array containing the actual camera position in microns.
	zv - A numpy array containing the relative z offset value of the coverslip (negative is closer to the objective).

	normalize - Normalize the PSF to unit height.
	wvl - Light wavelength in microns.
	"""
	[scaling_factor, max_rho] = configure(mp, wvl, num_basis)
	rho = numpy.linspace(0.0, max_rho, rho_samples)

	a = mp["NA"] * mp["zd0"] / math.sqrt(mp["M"]*mp["M"] + mp["NA"]*mp["NA"])  # Aperture radius at the back focal plane.
	k = 2.0 * numpy.pi/wvl

	ti = zv.reshape(-1,1) + mp["ti0"]
	pz = pz.reshape(-1,1)
	zd = zd.reshape(-1,1)

	opdt = OPD(mp, rho, ti, pz, wvl, zd)

	# Sample the phase
	#phase = numpy.cos(opdt) + 1j * numpy.sin(opdt)
	phase = numpy.exp(1j * opdt)

	# Define the basis of Bessel functions
	# Shape is (number of basis functions by number of rho samples)
	J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

	# Compute the approximation to the sampled pupil phase by finding the least squares
	# solution to the complex coefficients of the Fourier-Bessel expansion.
	# Shape of C is (number of basis functions by number of z samples).
	# Note the matrix transposes to get the dimensions correct.
	C, residuals, _, _ = numpy.linalg.lstsq(J.T, phase.T, rcond = -1)

	rv = rv*mp["M"]
	b = k * a * rv.reshape(-1, 1)/zd

	# Convenience functions for J0 and J1 Bessel functions
	J0 = lambda x: scipy.special.jv(0, x)
	J1 = lambda x: scipy.special.jv(1, x)

	# See equation 5 in Li, Xue, and Blu
	denom = scaling_factor * scaling_factor - b * b
	R = (scaling_factor * J1(scaling_factor * max_rho) * J0(b * max_rho) * max_rho - b * J0(scaling_factor * max_rho) * J1(b * max_rho) * max_rho)
	R /= denom

	# The transpose places the axial direction along the first dimension of the array, i.e. rows
	# This is only for convenience.
	PSF_rz = (numpy.abs(R.dot(C))**2).T

	# Normalize to the maximum value
	if normalize:
		PSF_rz /= numpy.max(PSF_rz)

	return PSF_rz

def gLZRParticleScan(mp, rv, pz, num_basis, rho_samples, normalize = True, wvl = 0.6, zd = None, zv = 0.0):
	"""
	Calculate radial G-L at specified radius and z values. This is models the PSF
	you would measure by scanning the particle relative to the microscopes focus.

	mp - The microscope parameters dictionary.
	rv - A numpy array containing the radius values.
	pz - A numpy array containing the particle z position above the coverslip (positive values only)
		 in microns.

	normalize - Normalize the PSF to unit height.
	wvl - Light wavelength in microns.
	zd - Actual camera position in microns. If not specified the microscope tube length is used.
	zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
	"""
	if zd is None:
		zd = mp["zd0"]

	zd = numpy.array([zd])
	zv = numpy.array([zv])

	return gLZRScan(mp, pz, rv, zd, zv, num_basis, rho_samples, normalize = normalize, wvl = wvl)

def OPD(mp, rho, ti, pz, wvl, zd):
	"""
	Calculate phase aberration term.

	mp - The microscope parameters dictionary.
	rho - Rho term.
	ti - Coverslip z offset in microns.
	pz - Particle z position above the coverslip in microns.
	wvl - Light wavelength in microns.
	zd - Actual camera position in microns.
	"""
	NA = mp["NA"]
	ns = mp["ns"]
	ng0 = mp["ng0"]
	ng = mp["ng"]
	ni0 = mp["ni0"]
	ni = mp["ni"]
	ti0 = mp["ti0"]
	tg = mp["tg"]
	tg0 = mp["tg0"]
	zd0 = mp["zd0"]

	a = NA * zd0 / mp["M"]  # Aperture radius at the back focal plane.
	k = 2.0 * numpy.pi/wvl  # Wave number of emitted light.

	OPDs = pz * numpy.sqrt(ns * ns - NA * NA * rho * rho) # OPD in the sample.
	OPDi = ti * numpy.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * numpy.sqrt(ni0 * ni0 - NA * NA * rho * rho) # OPD in the immersion medium.
	OPDg = tg * numpy.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * numpy.sqrt(ng0 * ng0 - NA * NA * rho * rho) # OPD in the coverslip.
	OPDt = a * a * (zd0 - zd) * rho * rho / (2.0 * zd0 * zd) # OPD in camera position.

	return k * (OPDs + OPDi + OPDg + OPDt)


def psfRZToPSFXYZ(dxy, xy_size, zv, zv_, rv, PSF_rz, writedisk, sample, px=0, py=0):
	"""
	Use interpolation to create a 3D XYZ PSF from a 2D ZR PSF.
	Modified
	"""
	# Create XY grid of radius values.
	px = px.astype(numpy.float32)
	py = py.astype(numpy.float32)
	PSF_rz = PSF_rz.astype(numpy.float32)
	t0 = time.perf_counter()
	c_xy = float(xy_size) * 0.5
	c_xy = dxy * c_xy
	xy = numpy.mgrid[0:xy_size, 0:xy_size] + 0.5
	xy = dxy * xy
	x = xy[1].astype(numpy.float32)
	y = xy[0].astype(numpy.float32)
	
	if(writedisk != 1):
		
		X = numpy.repeat(x[None, :, :], zv.shape[0], axis=0)
		sub1 = numpy.subtract(X, c_xy)
		del X
		sub2 = numpy.subtract(sub1, px[:,None,None])
		del sub1
		mult1 = numpy.multiply(sub2, sub2)
		del sub2
		Y = numpy.repeat(y[None, :, :], zv.shape[0], axis=0)
		sub3 = numpy.subtract(Y, c_xy)
		del Y
		sub4 = numpy.subtract(sub3, py[:,None,None])
		mult2 = numpy.multiply(sub4, sub4)
		del sub4
		add1 = numpy.add(mult1, mult2)
		sqrt1 = numpy.sqrt(add1)
		del add1
		R_pixel = sqrt1.ravel()
		newDiv1 = numpy.divide(1, (rv[1] - rv[0]))
		#print("partly done with PSF")
		R_pixel2 = numpy.multiply(R_pixel, newDiv1)
		del newDiv1
		del R_pixel
		"""
		# Create XYZ PSF by interpolation.
		PSF_xyz = numpy.zeros((PSF_rz.shape[0], xy_size, xy_size))
		for i in range(PSF_rz.shape[0]):
			psf_rz_interp = scipy.interpolate.interp1d(rv, PSF_rz[i,:], bounds_error=False, fill_value=0.0)
			PSF_xyz[i,:,:] = psf_rz_interp(r_pixel.ravel()).reshape(xy_size, xy_size)
		"""
		repeat1 = numpy.divide(zv, (zv_[1] - zv_[0]))
		repeat2 = numpy.multiply(x.shape[0], x.shape[1])
		Z = numpy.repeat(repeat1, repeat2)
		del repeat1
		del repeat2
		sampling_points = numpy.zeros((2, R_pixel2.shape[0]), numpy.float32)
		sampling_points[0,:] = Z
		del Z
		sampling_points[1,:] = R_pixel2
		del R_pixel2
		PSF_xyz = map_coordinates(PSF_rz, sampling_points)
		return PSF_xyz
	else:
		h5py_file = h5py.File('output/sample_'+str(sample) + '/h5py_file.hdf5', 'w')
		
		#The commented code below refers to the percentage of my RAM that each line seems to use.
		#This was helpful in selecting which variables I need to write to disk using h5py.
		
		X = numpy.repeat(x[None, :, :], zv.shape[0], axis=0)#20
		#total 20
		sub1 = numpy.subtract(X, c_xy)#20
		#total 40
		del X#These variables are quite massive. Deleting them right away is important so I don't consume too much memory
		#total 20
		sub2 = numpy.subtract(sub1, px[:,None,None])#20
		#total 40
		del sub1
		#total 20
		mult1 = numpy.multiply(sub2, sub2)#20
		#total 40
		del sub2
		#total 20
		mult1dset = h5py_file.create_dataset("mult1",  data=mult1)#I write this variable to disk because it needs to be stored while I do several other memory consuming calculations 
		del mult1
		#total 0
		Y = numpy.repeat(y[None, :, :], zv.shape[0], axis=0)#20
		#total 20
		sub3 = numpy.subtract(Y, c_xy)#20
		#total 40
		del Y
		#total 20
		sub4 = numpy.subtract(sub3, py[:,None,None])#20
		#total 40
		del sub3
		#total 20
		mult2 = numpy.multiply(sub4, sub4)#20
		#total 40
		del sub4
		#total 20
		add1 = numpy.add(mult1dset, mult2)#20
		del mult1dset
		#total 40
		del mult2
		#total 20
		sqrt1 = numpy.sqrt(add1)#20
		#total 40
		del add1
		#total 20
		R_pixel = sqrt1.ravel()#0
		#total 20
		del sqrt1
		#total 20   --IDK why, but this is what it says...
		newDiv1 = numpy.divide(1, (rv[1] - rv[0]))#0
		#total 20
		R_pixel2 = numpy.multiply(R_pixel, newDiv1)#20
		#total 40
		del R_pixel
		del newDiv1
		#total 20
		R_pixel2dset = h5py_file.create_dataset("R_pixel2",  data=R_pixel2)#Same reasoning with this one. To conserve memory, I write this var to disk
		del R_pixel2
		#total 0
		"""
		# Create XYZ PSF by interpolation.
		PSF_xyz = numpy.zeros((PSF_rz.shape[0], xy_size, xy_size))
		for i in range(PSF_rz.shape[0]):
			psf_rz_interp = scipy.interpolate.interp1d(rv, PSF_rz[i,:], bounds_error=False, fill_value=0.0)
			PSF_xyz[i,:,:] = psf_rz_interp(r_pixel.ravel()).reshape(xy_size, xy_size)
		"""
		repeat1 = numpy.divide(zv, (zv_[1] - zv_[0]))#0
		repeat2 = numpy.multiply(x.shape[0], x.shape[1])#0
		Z = numpy.repeat(repeat1, repeat2)#40
		del repeat1
		del repeat2
		#total 40
		Zdset = h5py_file.create_dataset("Z",  data=Z)#This huge variable definitely needs to be written to disk
		del Z
		#total 0
		sampling_points = numpy.zeros((2, R_pixel2dset.shape[0]), numpy.float32)#0
		sampling_points[0,:] = Zdset #20
		del Zdset
		#total 20
		sampling_points[1,:] = R_pixel2dset #20
		del R_pixel2dset
		#total 40
		h5py_file.close()
		#total 40
		PSF_xyz = map_coordinates(PSF_rz, sampling_points)#20
		del sampling_points
		#total 60
		return PSF_xyz

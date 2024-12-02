import numpy as np
import scipy.interpolate as si
from numpy.random import uniform
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from numpy import cos,sin
import skimage.measure
import scipy.misc
from PIL import Image
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from numpy import pi, cos, sin, arccos, arange
from scipy.interpolate import CubicSpline
from scipy import interpolate
import os
import microscPSFmod as msPSF
import time
from tifffile import imwrite
from multiprocessing import Pool
import sys
from traceback import format_exception
import linecache
import csv

def set_seed():
	overly_specific_time = time.time() * 100000000000#getting the decimal point super far to the right
	#"{:f}".format     this part makes it so it's not in scientific notation when converted into a string
	#first I strp all the trailing zeros, then I strip the decimal point from the right end of the string, and then I strip any remaining trailing zeros
	useful_digits_from_time_string = "{:f}".format(overly_specific_time).rstrip("0").rstrip(".").rstrip("0")
	rightmost_six_digits_string = useful_digits_from_time_string[-6:]#reduce the number of digits in this number
	#just in case the time is exactly the same during two processes, I need to add it to the unique process id 
	seed = int(rightmost_six_digits_string) + os.getpid()
	np.random.seed(int(seed))

def write_to_attributes(sample, text, frame, override = False):#append some text to a text file. usefull for configuration information, frame duration information, timestamps, length and width information
	if(frame < 900 or override == True):
		fowt="output/sample_"+str(sample)+'/attributes.txt'
		outF = open(fowt, "a")
		outF.write(text)
		outF.close()

def write_to_iqra(sample, text):
	fowt="output/sample_"+str(sample)+'/iqra_record.json'
	outF = open(fowt, "a")
	outF.write(text)
	outF.close()

def save_physics_gt(particlesArray, sample, frame, pixel_size, image_size, max_xy, printGt):#Takes a list of points and generates a black and white ground truth PNG image
	pimage=np.zeros((int(pixel_size*image_size),int(pixel_size*image_size)))
	if(particlesArray != []):
		particlesArray=np.delete(particlesArray,2,1)
		particlesArray*=1000
		particlesArray+=max_xy
		try:
			pimage[particlesArray[:,1].astype(int),particlesArray[:,0].astype(int)]=255
		except:
			return "Fail" #If this fails then it means some particles I am trying to render exist outside our canvas. This should have been fixed beforehand by trimming off those particles prior to sending them here
	if(printGt == 1):
		pimage2=skimage.measure.block_reduce(pimage, (pixel_size,pixel_size), np.max)
		img = Image.fromarray(pimage2)
		img=img.convert('RGB')
		physics_gt_fname='output/sample_'+str(sample)+'/physics_gt/'+str(sample)+'_'+str(frame)+'.png'#File name of ground truth image
		img.save(physics_gt_fname)
	return "success"

def process_matrix_all_z(locations, size_x, step_size_xy, psf_size_x, stage_delta, sampling, mp, 
				writedisk, sample, num_basis, rho_samples, wvl=0.510):#Get output from Point Spread Function in microscPSFmod.py
	psf = msPSF.gLXYZParticleScan(
		mp, step_size_xy, psf_size_x, locations[:,2], writedisk, sample, num_basis, rho_samples, 
		zv=stage_delta, wvl=wvl, normalize=False,
		px=locations[:, 0], py=locations[:, 1])
	psf=np.reshape(psf, (psf.shape[0], -1))
	return psf

def brightness_trace(t_off, t_on, rate, frames):#I didn't edit this function -Aaron
	T = np.zeros((2, frames))
	T[0, :] = np.random.exponential(scale=t_off, size=(1, frames))
	T[1, :] = np.random.exponential(scale=t_on, size=(1, frames))

	B = np.zeros((2, frames))
	B[1, :] = rate * T[1, :]

	T_t = T.ravel(order="F")
	B_t = B.ravel(order="F")

	T_cum = np.cumsum(T_t)
	B_cum = np.cumsum(B_t)

	start = np.random.randint(0, 10)

	br = np.diff(np.interp(np.arange(start, start + frames + 1), T_cum, B_cum))
	return br

def calcBrightness(particlesArray):#I didn't edit this function -Aaron
	# Fluorophore parameters
	t_off = 0 # changed by krishna, needs to be checked, original value 8 -comment from old code
	t_on = 1 # changed by krishna, needs to be checked, original value 2 -comment from old code
	rate = 10
	
	brightness = np.zeros((particlesArray.shape[0], 1))
	for i in range(particlesArray.shape[0]):
		brightness[i, :] = brightness_trace(t_off, t_on, rate, 1)
	return brightness

def particlesData_process_matrix_all_z(particlesArray_ThisEntity, pixels, micrometers_per_pixel, psf_size_x, stage_delta, 
				sampling, mp, writedisk, sample, num_basis, rho_samples): 
	#call process_matrix_all_z function and involve brightness
	image_ThisEntity = np.zeros((1, pixels, pixels))
	if(particlesArray_ThisEntity != []):
		particlesData_ThisEntity = process_matrix_all_z(particlesArray_ThisEntity, pixels, 
						micrometers_per_pixel, psf_size_x, stage_delta, sampling, mp, writedisk, sample, num_basis, rho_samples)
		image=np.array([])
		
		brightness=calcBrightness(particlesArray_ThisEntity)
		brightness = brightness[:, 0]
		image_ThisEntity[0, :, :] = np.reshape(np.sum(particlesData_ThisEntity * brightness[:, None], axis=0), (pixels, pixels))
	
	return image_ThisEntity

def add_entity_images_together(image_array):#after running the PSF function on some subset of points I can add the result to the result of a different subset of points to create the PSF of the entire set of all points
	image = []
	for image_array_element in range(len(image_array)):
		if(image_array_element == 0):#if I are processing the first entity in the list now
			image=image_array[image_array_element]#Then just set the resulting image equal to just the image for this one entity
		else:
			image=image+image_array[image_array_element]#And just add this entity image data on top of the previous ones
	return image

def render(entitiesPts, sample, frame, max_xy, psftogether, printTif, printGt, parallelPSF, writedisk, m_params, num_basis, rho_samples, pixels, nm_per_pixel):
	#gets the list of points that make up the Entities, and uses that to generate the final blurred images and ground truth images
	
	micrometers_per_pixel = 0.001 * nm_per_pixel
	stage_delta = -1  # [um] negative means toward the objective -comment from old code
	sampling = 1
	psf_size_x = sampling * pixels
	
	particlesArray=np.array([])#This is an array of the points of the Entity(s) not seperated by entity, these are just all jumbled together
	image=np.array([])#This will become the blurry .tiff image
	
	timeOnPSF = 0
	
	#The code below sets up the variable particlesArray. Depending on if psftegether is 1 or 0, the entitiesPts variable will be seperated by entity or not. 
	#If psftogether==0, then entities are seperated in entitiesPts variable, so to get particlesArray which should not be seperated, I need to process it differently than psftogether==1
	if(psftogether == 0):
		for entity in range(len(entitiesPts)):
			particles_list = entitiesPts[entity]#grab one entity's points
			particlesArray_ThisEntity = np.array(particles_list)# to numpy
			particlesArray_ThisEntity = particlesArray_ThisEntity.reshape((int(len(particlesArray_ThisEntity)/3),3))#The array of points of this entity
			if(entity == 0):
				particlesArray=particlesArray_ThisEntity
			else:
				particlesArray=np.concatenate((particlesArray, particlesArray_ThisEntity))#I need to add all the Entity's arrays together so I can generate one ground truth
	else:#psftogether==1
		particlesArray = np.array(entitiesPts)
		particlesArray = particlesArray.reshape((int(len(particlesArray)/3),3))#formating
	
	particlesArray_cropped_by_entity_for_psfTogether0 = []#cropped means excluding the points which fall outside the canvas
	particlesArray_cropped = []
	
	#The code below first runs save_physics_gt() and sees if it fails. If it fails, that means some points are falling too close to the edge of the canvas, and I need to crop more points out. So I try running it with a tighter and tighter exclusion zone (out_of_bounds-=0.001) to crop more points out until it runs without failing. 
	#Once save_physics_gt() runs sucessfully, then I use the successful out_of_bounds value to construct a variable particlesArray_cropped_by_entity_for_psfTogether0, if psfTogether==0, then this variable will contain the cropped points for each entity seperated by entity. Seperated meaning there wil be an array element for each entity instead of just cramming all entity points all in the same level of the array
	success = False
	out_of_bounds = max_xy / 1000 #times 2 because origin is in the center so max_xy*2 = width of the canvas
	while(success == False):#try printing ground truth. if it doesn't work, then trim off more particles from the edges until it works
		particlesArray_cropped = []
		for part in particlesArray:
			if(part[0] > -out_of_bounds and part[0] < out_of_bounds and part[1] > -out_of_bounds and part[1] < out_of_bounds):
				#This removes the points that escape outside the edge of the canvas. earlier they were popping up on the other side :/ 
				particlesArray_cropped.append(part)#dumping everything from particlesArray into the cropped variable, but excluding those particles who are out of bounds of the canvas
		particlesArray_cropped = np.array(particlesArray_cropped)#to numpy
		if("success" == save_physics_gt(particlesArray_cropped, sample, frame, nm_per_pixel, pixels, max_xy, printGt)):#print ground truth png
			success = True
			
			particlesArray_cropped_by_entity_for_psfTogether0 = []
			if(psftogether == 0):#create cropped list in special format for seperated by entity for psf
				for entity in range(len(entitiesPts)):
					particlesArray_cropped_by_entity_for_psfTogether0.append([])
					
					particles_list = entitiesPts[entity]#grab one entity's points
					particlesArray_ThisEntity = np.array(particles_list)# to numpy
					particlesArray_ThisEntity = particlesArray_ThisEntity.reshape((int(len(particlesArray_ThisEntity)/3),3))#The array of points of this entity
					
					for part in particlesArray_ThisEntity:
						if(part[0] > -out_of_bounds and part[0] < out_of_bounds and part[1] > -out_of_bounds and part[1] < out_of_bounds):
							if(entity == 0):
								particlesArray_cropped_by_entity_for_psfTogether0[entity]=particlesArray_ThisEntity
							else:
								particlesArray_cropped_by_entity_for_psfTogether0[entity]=np.concatenate((particlesArray, particlesArray_ThisEntity))#I need to add all the Entity's arrays together so I can generate one ground truth
		
		else:#some points fell too close to the edge of the canvas, these need to be cropped and this whole thing should be tried again at the while loop level
			out_of_bounds = out_of_bounds - 0.001#it's important that I don't have any points that are beyond the edges of the canvas or I can't plot the image
			#So I slowly bring in the edges until plotting the image is successfull.
	
	#The code below gets the image variable from the cropped particlesArray array which will later be used to print the tif image
	#The image variable is obtained using the psf function. This function is run one entity at a time if psftogether=0, and then each entity's output is added together to get the full image containing all entities
	#If psftogether==0, then there is also the option to do each entity's psf function in parallel. 
	if(psftogether==0): #PSF One entity at a time, then added together after processing
		image_array = []
		
		if(parallelPSF):#does each entity's PSF in parallel with each other
			argsForParallel = []
			for entity in range(len(entitiesPts)):
				argsForParallel.append([particlesArray_cropped_by_entity_for_psfTogether0[entity], pixels, micrometers_per_pixel, psf_size_x, stage_delta, 
								sampling, m_params, writedisk, sample, num_basis, rho_samples])#these are the arguments that I'll need while parallelly doing the PSF for each entity
			PSFStartTime=int( time.time() )
			if(printTif==1 and len(entitiesPts) > 0):#if I don't need the computation intensive blurry tiff images, I can skip that and just do the ground truth images
				with Pool(len(entitiesPts)) as p:
					image_array = p.starmap(particlesData_process_matrix_all_z, argsForParallel)#parallel printing of entities in one frame. 
			if(len(entitiesPts) == 0):
				image_array.append(np.zeros((1, pixels, pixels)))
			image = add_entity_images_together(image_array)#add all entity's PSFs into one image
			timeOnPSF+=int( time.time() )-PSFStartTime#PSF takes forever so keep track of how long it takes 
			
		else:#sequential PSF // Does each entity's PSF sequentially
			for entity in range(len(entitiesPts)):
				if(printTif==1):#if I don't need the computation intensive blurry tiff images, I can skip that and just do the ground truth images
					PSFStartTime=int( time.time() )
					image_array.append(particlesData_process_matrix_all_z(particlesArray_cropped_by_entity_for_psfTogether0[entity], 
									pixels, micrometers_per_pixel, psf_size_x, stage_delta, sampling, m_params, writedisk, 
									sample, num_basis, rho_samples))#Collect all finished PSFs into an array
					timeOnPSF+=int( time.time() )-PSFStartTime
					print(".")#takes a long time to render on big canvases. If it's taking a long time to see any sign of life, it's nice to get a hint that it's working and hasn't crashed. 
			if(len(entitiesPts) == 0):
				image_array.append(np.zeros((1, pixels, pixels)))
			image = add_entity_images_together(image_array)
		if(printTif==1):#Only do this if I need the tiff blurred images
			imageCopy=image
			image=[]
			image.append(imageCopy)#More Formatting
			image = np.array(image)
	else: # psftogether==1  PSF all entities at once. This is how it was done originaly and is a much more simplified procedure
		image = np.zeros((1, pixels, pixels))
		if(printTif==1 and particlesArray_cropped != []):#Only do this if you need the tiff images and not just ground truth
			brightness=calcBrightness(particlesArray_cropped)
			PSFStartTime=int( time.time() )
			particlesData = process_matrix_all_z(particlesArray_cropped, pixels, micrometers_per_pixel, psf_size_x, stage_delta, 
							sampling, m_params, writedisk, sample, num_basis, rho_samples)#process all entities in the image all at once
			timeOnPSF = int(time.time()) - PSFStartTime
			brightness = brightness[:, 0]
			image[0, :, :] = np.reshape(np.sum(particlesData * brightness[:, None], axis=0), (pixels, pixels))
	
	#The following code generates the tif image from the image variable 
	if(printTif == 1):#Only do this if you need the blurry tiff images
		d=image[0, :, :]
		if(np.max(d) != 0):
			d = d / np.max(d)
		d=d*255
		d=d.astype(np.uint8)
		save_fname='output/sample_'+str(sample)+'/tif/'+str(sample)+'_'+str(frame)+'.tif'
		imwrite(save_fname,d,compress=6)#print the tiff blurry image
	
	return timeOnPSF

def roll( x,y,z, Euler):#I didn't edit this -Aaron 
	# Euler rotation
	rot = np.dot(
		Euler,
		np.array([x.ravel(), y.ravel(), z.ravel()])
	)
	x_rot = rot[0,:].reshape(x.shape)
	y_rot = rot[1,:].reshape(y.shape)
	z_rot = rot[2,:].reshape(z.shape)
	return x_rot, y_rot, z_rot

def arc_length(x, y, z):#I didn't edit this -Aaron
	npts = len(x)
	arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2  + (z[1] - z[0])**2)
	for k in range(1, npts):
		arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2 + (z[k] - z[k-1])**2)
	return arc

def arc_length_2d(x, y):#I didn't edit this -Aaron
	npts = len(x)
	arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2 )
	for k in range(1, npts):
		arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)
	return arc

#remove random photons from the surface of the sphere
def random_selector(data_x,data_y,data_z,percentage):#I didn't edit this -Aaron
	random_list=np.random.randint(0,len(data_x),int(len(data_x)*percentage))
	final_data_x=np.take(data_x, random_list)
	final_data_y=np.take(data_y, random_list)
	final_data_z=np.take(data_z, random_list)
	return final_data_x,final_data_y,final_data_z

def get_mito_center_line_pts(zhigh, zlow, max_xy, max_length, centerLinePtMultiple, sample, num_of_curve_pts=4, curve_pointXs_this_mito=None, curve_pointYs_this_mito=None):
	#Gets a string of points along the mitochondria's central spine line
	#takes the four curve points and draws a smooth curving line between them and uses that to get the center_line_pts
	set_seed()
	zhighTemp=zhigh #the max level that a point should fall under the Z axis
	mitoLength=0#the length of this mito
	needToGenerateRandomCurvePts=False#if this is the first frame, then I need to randomly pick a starting location for this mitochondria before I get the center line points
	if curve_pointXs_this_mito is None:#if this is not the first frame, then there will be a non none value in curve_pointXs
		needToGenerateRandomCurvePts=True#so since there's none, then I need to generate a starting position for this mitochondria's curve_points
	while True:#Try to generate satisfactory curve points, try again until you get a good one
		if(needToGenerateRandomCurvePts):
			#pick a random point to be the center of the box in which I'll generate a mito. important because if I try to generate a mitochondria within the entire max_xy then it will usually be much too long so I need to pick a spot and then generate a reasonably long mitochondria in that spot
			usable_area = max_length * 0.5 #half of the max length. this is half of the width of the square in which I'll pick points for new curve points of this mito. Because the origin is in the center so I go to negative this value too as part of this box 
			if(max_xy <= (max_length / 2)):
				usable_area = max_xy#or full canvas if small canvas
			p=np.random.uniform(-1 * usable_area, usable_area, int(usable_area))#Only choose points within this small box because I'll move the points randomly later to set the mito in a random spot
			np.random.shuffle(p)
			curve_pointXs_this_mito=np.random.choice(p,num_of_curve_pts)#array, each element is the X Y value for one of the four "curve points", 
			curve_pointYs_this_mito=np.random.choice(p,num_of_curve_pts)#the points that define the position of the mitochondria
			curve_pointXs_this_mito.sort()
			
			if(max_xy > (max_length / 2)):
				#pick a value to move the generated mito's curve points so it lies in a random spot
				center_of_mito_gen_box_range = int(max_xy * 2 - max_length - 1)#the center of the box can't be on the edge of the max_xy so I subtract the max_length so that the centerpoint of the generation box will be at least half that length from the edge which is perfect. So the mitochondria has a chance to be on the edge of the canvas but not outside it
				mito_box_center_x = np.random.randint(0, center_of_mito_gen_box_range) - (max_xy-(0.5*max_length))#pick a random point from around the center of the max_xy which I'll use 
				mito_box_center_y = np.random.randint(0, center_of_mito_gen_box_range) - (max_xy-(0.5*max_length))#as the center point for a box in which I'll choose random points
				curve_pointXs_this_mito = curve_pointXs_this_mito + mito_box_center_x#move mito
				curve_pointYs_this_mito = curve_pointYs_this_mito + mito_box_center_y
		tck,u = interpolate.splprep([curve_pointXs_this_mito,curve_pointYs_this_mito], k=2,s=0)
		unew = np.arange(0, 1, 0.01)
		out = interpolate.splev(unew, tck)
		x=out[0]
		y=out[1]
		arc_2d=arc_length_2d(x, y)
		if(arc_2d<(zhigh-zlow)):
			zhighTemp=zlow+int(arc_2d)
		z=np.linspace(zlow,zhighTemp,len(x))
		mitoLength=arc_length(x,y,z)		#I couldn't use the get_length function for this because many of these vars are needed later in this function ......
		if(mitoLength<max_length or needToGenerateRandomCurvePts==False):#If this is true then I have a good set of points and I can exit this while loop
			break
	unew = np.arange(0, 1, (1/arc_2d)*centerLinePtMultiple)
	out = interpolate.splev(unew, tck)
	center_line_Xs=out[0]
	center_line_Ys=out[1]
	center_line_Zs=np.linspace(zlow,zhighTemp,len(center_line_Xs))
	return center_line_Xs, center_line_Ys, center_line_Zs, mitoLength, curve_pointXs_this_mito, curve_pointYs_this_mito

def get_points(r,p1,p2,num_pts,num_of_points_in_circle):#I didn't touch this function. I did delete a semicolon though at the end of a line
	num_pts=np.random.randint(int(num_pts*0.8),num_pts)
	indices = arange(0, num_pts, dtype=float) + 0.5
	phi = arccos(1 - 2*indices/num_pts)
	theta = pi * (1 + 5**0.5) * indices
	xi, yi, zi = r*cos(theta) * sin(phi), r*sin(theta) * sin(phi), r*cos(phi)

	xi+=(p1[0])
	yi+=(p1[1])
	zi+=(p1[2])
	b = np.array([p1[0], p1[1], p1[2]])
	c = np.array([p2[0], p2[1], p2[2]])

	a = np.array([xi, yi,zi])
	ba=np.subtract(a.T,b)
	bc = c-b
	d=np.dot(ba, bc)
	m= np.multiply(np.linalg.norm(ba,axis=-1) , np.linalg.norm(bc))
	cosine_angle = d / m
	angle = np.arccos(cosine_angle)
	theta=np.degrees(angle)
	xii=xi[np.where((theta>88) & (theta<92))]
	yii=yi[np.where((theta>88) & (theta<92))]
	zii=zi[np.where((theta>88) & (theta<92))]
	random_list=np.random.randint(0,len(xii),num_of_points_in_circle)
	final_data_x=np.take(xii, random_list)
	final_data_y=np.take(yii, random_list)
	final_data_z=np.take(zii, random_list)
	return final_data_x,final_data_y,final_data_z

def get_mito_surface_points(center_line_Xs, center_line_Ys, center_line_Zs, dist, width, density, percentage, surfPtsDivisor, max_xy):
	#takes the string of points along the central spine, and generates a set of points sitting on the surface of the body of the mitochondria
	#these are a set distance from the central spine points in a circle. This results in a cylindar shape composed of rings of points all along the mitochondria's body
	
	#The only changes I made here are changing variable names to make it more understandable and deleting unused code
	
	r=int(width/2)
	area=2*np.pi*r*dist
	total_expected_emitters=int((area/1000)*density)
	num_of_points_in_circle=int(math.floor(total_expected_emitters/dist))
	datax=[]
	datay=[]
	dataz=[]
	xx=[]
	yy=[]
	zz=[]
	num_pts=2*np.pi*r*5# num_pts is the number of points I'm gonna put on the surface of the mitochondria. 
	num_pts=int(num_pts/surfPtsDivisor)
	
	out_of_bounds = max_xy * 2
	point_in_bounds_exists = False
	
	for i in range(len(center_line_Xs)-1):#each point in center spine line skeleton
		p1=np.array([center_line_Xs[i],center_line_Ys[i],center_line_Zs[i]])
		p2=np.array([center_line_Xs[i+1],center_line_Ys[i+1],center_line_Zs[i+1]])
		try_to_get_points = 0
		while try_to_get_points < 10:
			try:
				data_x,data_y,data_z=get_points(r,p1,p2,num_pts,num_of_points_in_circle)
				try_to_get_points = 100
			except:
				try_to_get_points = try_to_get_points + 1
		if(try_to_get_points != 100):
			print("\n\nget_points() Failed ten times :(\n\n")
		final_data_x,final_data_y,final_data_z=random_selector(data_x,data_y,data_z,1)
		for k in range(len(final_data_x)):
			datax.append(final_data_x[k])
			datay.append(final_data_y[k])
			dataz.append(final_data_z[k])
			if(final_data_x[k] > -out_of_bounds and final_data_x[k] < out_of_bounds and final_data_y[k] > -out_of_bounds and final_data_y[k] < out_of_bounds):
				point_in_bounds_exists = True
	
	data=False
	if(point_in_bounds_exists == True):
		data = []
		for k in range(len(datax)):
			data.append(datax[k])
			data.append(datay[k])
			data.append(dataz[k])
	return data #this data is the final set of points that make up the mitochondria

def setup_new_coord_csv(filename):
	if(os.path.exists(filename) == False):
		with open(filename, 'w') as csvfile:
			csvwriter = csv.writer(csvfile)
			headers = ["Frame", "Curve_pt_XY_1", "Curve_pt_XY_2", "Curve_pt_XY_3", "Curve_pt_XY_4"]
			for center_line_point_number in range(100):
				headers.append("Center_Line_" + str(center_line_point_number) + "_XYZ")
			csvwriter.writerow(headers)

def setup_new_rohit_csv(filename):
	if(os.path.exists(filename) == False):
		with open(filename, 'w') as csvfile:
			csvwriter = csv.writer(csvfile)
			headers = ["Frame", "Splits", "Merges"]
			for mito_num in range(100):
				headers.append("Mito_" + str(mito_num) + "_Center_XYZ")
			csvwriter.writerow(headers)

def plot(max_xy, mitosPts):
	
	max_xy = max_xy * 0.001
	mitosPts = np.array(mitosPts)
	
	fig2 = plt.figure(1)
	ax = fig2.add_subplot(111, projection='3d')
	
	data3 = mitosPts.reshape((int(len(mitosPts)/3),3))
	cmhot = plt.get_cmap("inferno")
	ff=ax.scatter(data3[:,0],data3[:,1],data3[:,2],c=data3[:,2],cmap=cmhot)
	ax.set_xlim(-max_xy, max_xy)
	ax.set_ylim(-max_xy, max_xy)
	ax.set_zlim(0, 1.4)
	ax.set_xlabel('x (nm)')
	ax.set_ylabel('y  (nm)')
	ax.set_zlabel('z  (nm)')
	ax.auto_scale_xyz([-max_xy, max_xy], [-max_xy, max_xy], [0, 1400])
	plt.show()

def generate_save_mito(sample, frame, num_of_mitos, centermult, surfDivisor, max_xy, psftogether, 
				zlow, zhigh, max_mito_length, density, emmiters_percentage, widths, num_of_curve_pts, 
				recordcoordinates, printGt, printTif, curve_pointXs, curve_pointYs, rohit_behavior, 
				rohit_id_map, recordcoordinates_mito_storage, rohit_mito_storage, frames, splits_amount_this_frame, merges_amount_this_frame, plot_boolean):
	#For each mitochondria, generate new random curve points if it's the first frame, or use the existing curve points, and get all the 
	#surface points that lie all around the body of the mitochondria
	
	set_seed()
	mitoLength=[]#length of each mito
	center_line_Xs=[]#these are all arrays with one element for each mitochondria in the frame, each element is a list defining the XYZ locations of the sets of points in each mitochondria
	center_line_Ys=[]
	center_line_Zs=[]
	surfPts_npArray=[]#to numpy
	firstFrame=False#is this the first frame of the program?
	
	if curve_pointXs is None:
		curve_pointXs=[]
		curve_pointYs=[]
		firstFrame=True
	
	mitosPts = []#set of points of all mitochondria
	mitosPts_separated_by_mito = []#one element for each mito, each element is the set of points making up that mito
	
	for mitoNum in range(num_of_mitos):
		surfPts =[]#an array where each element is an arary of one of the mitochondria's surface points
		center_line_Xs.append([])
		center_line_Ys.append([])
		center_line_Zs.append([])#declaring elements to modify later
		mitoLength.append(0)
		
		if(firstFrame):
			curve_pointXs.append([])
			curve_pointXs[mitoNum]=None#if I pass 'none' to get_mito_center_line_pts(), then it will understand that this is the first frame and it needs to randomly get new curve pts
			curve_pointYs.append([])
			curve_pointYs[mitoNum]=None
		
		#gets center line points of one mito at a time. points lie along a smooth line drawn between the curve points. Curve points are generated in this function if this is the first frame
		center_line_Xs[mitoNum], center_line_Ys[mitoNum], center_line_Zs[mitoNum], mitoLength[mitoNum], curve_pointXs[mitoNum], curve_pointYs[mitoNum]=get_mito_center_line_pts(zhigh,
						zlow, max_xy, max_mito_length, centermult, sample, num_of_curve_pts, curve_pointXs[mitoNum], curve_pointYs[mitoNum])
		
		if(printGt == 1 or printTif == 1):
			#use the center line points to generate a ring of points around those points to construct a cylindar of points curving to meet the curve points these are called surface points
			surfPts = get_mito_surface_points(center_line_Xs[mitoNum], 
					center_line_Ys[mitoNum], center_line_Zs[mitoNum], mitoLength[mitoNum], widths[mitoNum], density, emmiters_percentage, surfDivisor, max_xy)
			
			if(surfPts != False):#if there are points of this mito in the max_xy (Not all are outside the canvas making it totally invisible)
				surfPts_npArray.append(np.array(surfPts))#to numpy
				surfPts_npArray[mitoNum] *= 0.001
				
				if(psftogether==0):#psf function will be run one mito at a time
					mitosPts=[]#So I need to submit a list of the lists of points making up each each mito
				
				#If I need all the points in one big list to run the psf on all mitos at once, then I keep all points saved in mitosPts variable
				for element in surfPts_npArray[mitoNum]:
					mitosPts.append(element)#dump surfPts into mitosPts
				if(psftogether == 0):
					mitosPts_separated_by_mito.append(mitosPts)#if I need pts organized by mito, then save each mito in here
			else:
				surfPts_npArray.append([])
	if(psftogether == 0):
		mitosPts = mitosPts_separated_by_mito
	
	if(recordcoordinates == 1):#This will write the xyz coordinates of all curve points and all center line skeleton points to a csv file every single frame for every single mitochondria 
		recordcoordinates_mito_storage.append([])#[rows][mitos][cells]
		for mito in range(num_of_mitos):
			if(len(recordcoordinates_mito_storage[len(recordcoordinates_mito_storage)-1]) < (mito + 1)):#If the number of mitos is less than the id of the current mito
				recordcoordinates_mito_storage[len(recordcoordinates_mito_storage)-1].append([])#add mito
			cells = [str(frame), str(curve_pointXs[mito][0]) + ", " + str(curve_pointYs[mito][0]), str(curve_pointXs[mito][1]) + ", " + str(curve_pointYs[mito][1]), 
								 str(curve_pointXs[mito][2]) + ", " + str(curve_pointYs[mito][2]), str(curve_pointXs[mito][3]) + ", " + str(curve_pointYs[mito][3])]
			for center_line_point_number in range(len(center_line_Xs[mito])):
				cells.append(str(center_line_Xs[mito][center_line_point_number]) + ", " + str(center_line_Ys[mito][center_line_point_number]) + ", " + str(center_line_Zs[mito][center_line_point_number]))
			recordcoordinates_mito_storage[len(recordcoordinates_mito_storage)-1][mito] = cells
			if(len(recordcoordinates_mito_storage) % 200 == 0 or frame == frames or frame < 100):#append to csv if this is the last frame or if we've saved 200 frames already
				filename = "output/sample_" + str(sample) + "/mito_" + str(mito) + "_coordinates.csv"
				setup_new_coord_csv(filename)
				with open(filename, 'a') as csvfile:
					csvwriter = csv.writer(csvfile)
					for row in recordcoordinates_mito_storage:
						if(mito < len(row)):
							csvwriter.writerow(row[mito])
				if(mito == num_of_mitos - 1):
					recordcoordinates_mito_storage=[]
	
	if(rohit_behavior == 1):#This will write the center points of each mito to a csv file and it will use a custom mitochondria id scheme, different than the rest of the code. 
		rohit_mito_storage.append([])# [rows][cells]
		filename = "output/sample_" + str(sample) + "/rohit_centerpoints.csv"
		setup_new_rohit_csv(filename)
		
		cells = [str(frame), splits_amount_this_frame, merges_amount_this_frame]#["Frame", "Splits", "Merges"]
		for rohit_id in rohit_id_map:
			XYZ = ""
			if(rohit_id != -1):
				center_line_point_number = int(len(center_line_Xs[rohit_id])/2)
				x_coord = center_line_Xs[rohit_id][center_line_point_number]
				y_coord = center_line_Ys[rohit_id][center_line_point_number]
				z_coord = center_line_Zs[rohit_id][center_line_point_number]
				XYZ = str(x_coord) + ", " + str(y_coord) + ", " + str(z_coord)
			cells.append(XYZ)
		rohit_mito_storage[len(rohit_mito_storage) -1] = cells
		if(len(rohit_mito_storage) % 500 == 0 or frame == frames):
			with open(filename, 'a') as csvfile:
				csvwriter = csv.writer(csvfile)
				for row in rohit_mito_storage:
					csvwriter.writerow(row)
			rohit_mito_storage = []
	
	if(plot_boolean):
		center_line = []
		curve_points = []
		center_line_plus_curve_points = []
		for mito in range(num_of_mitos):
			for point in range(len(center_line_Xs[mito])):
				x = center_line_Xs[mito][point] * 0.001
				y = center_line_Ys[mito][point] * 0.001
				center_line_plus_curve_points.append(x)
				center_line_plus_curve_points.append(y)
				center_line_plus_curve_points.append(800 * 0.001)
				center_line.append(x)
				center_line.append(y)
			for point in range(4):
				x = curve_pointXs[mito][point] * 0.001
				y = curve_pointYs[mito][point] * 0.001
				center_line_plus_curve_points.append(x)
				center_line_plus_curve_points.append(y)
				curve_points.append(x)
				curve_points.append(y)
				center_line_plus_curve_points.append(850 * 0.001)
		
		#plot(max_xy, center_line_plus_curve_points)
		plot(max_xy, curve_points)
		#plot(max_xy, center_line)
		#plot(max_xy, mitosPts)
	
	return mitosPts, curve_pointXs, curve_pointYs, mitoLength, recordcoordinates_mito_storage, rohit_mito_storage

#remove random photons from the surface of the sphare
def random_selector(data_x,data_y,data_z,emmiters_percentage):
	random_list=np.random.randint(0,len(data_x),int(len(data_x)*emmiters_percentage))
	final_data_x=np.take(data_x, random_list)
	final_data_y=np.take(data_y, random_list)
	final_data_z=np.take(data_z, random_list)
	return final_data_x,final_data_y,final_data_z

def get_vesicle_3D_points(r,cx,cy,cz,density_vesicle,emmiters_percentage):

	area=4*np.pi*r*r
	num_pts=int((area/1000)*density_vesicle)
	indices = arange(0, num_pts, dtype=float) + 0.5
	phi = arccos(1 - 2*indices/num_pts)
	theta = pi * (1 + 5**0.5) * indices
	xi, yi, zi = r*cos(theta) * sin(phi), r*sin(theta) * sin(phi), r*cos(phi);

	xi+=(cx)
	yi+=(cy)
	zi+=(cz)

	final_data_x,final_data_y,final_data_z=random_selector(xi,yi,zi,emmiters_percentage)
	datax=[]
	datay=[]
	dataz=[]
	for k in range(len(final_data_x)):
		datax.append(final_data_x[k])
		datay.append(final_data_y[k])
		dataz.append(final_data_z[k])

	data=[]
	data_x=[]
	data_y=[]
	data_z=[]
	for k in range(len(datax)):
		data.append(datax[k])
		data.append(datay[k])
		data.append(dataz[k])
	data_plot=np.array(data)
	data_plot = data_plot.reshape((int(len(data_plot)/3),3))
	return data, data_plot

#this is where you need work. print csv for vesicles. 

def setup_vesi_csv(sample, num_of_vesicles):
	filename = "output/sample_" + str(sample) + "/all_vesi_coordinates.csv"
	with open(filename, 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		headers = ["Frame"]
		for vesi in range(num_of_vesicles):
			headers.append("Vesi_"+str(vesi)+"_Center_Point_XYZ")
			headers.append("Vesi_"+str(vesi)+"_Radius")
			csvwriter.writerow(headers)

def generate_save_vesicles(sample, frame, max_xy, psftogether, zlow, zhigh, density, emmiters_percentage, recordcoordinates, printGt, printTif, 
				number_of_vesicles, min_r, max_r, radius = None, x_center = None, y_center = None, z_center = None):
	
	vesiPts = []
	vesiPts_separated_by_vesi = []
	firstFrame = False
	if(radius is None):
		firstFrame = True
		radius = []
		x_center, y_center, z_center = [], [], []
	else:
		number_of_vesicles = len(radius)
	for vesi_num in range(number_of_vesicles):
		if(firstFrame):
			radius.append(np.random.randint(int(min_r), int(max_r)))
			z_center.append(np.random.randint(int(zlow), int(zhigh)))
			x_center.append(0)
			y_center.append(0)
			x_center[vesi_num], y_center[vesi_num] = np.random.randint(-max_xy, max_xy, 2)
		if(printGt == 1 or printTif == 1):
			thisVesi_vesiPts, data_plot = get_vesicle_3D_points(radius[vesi_num], 
						x_center[vesi_num], y_center[vesi_num], z_center[vesi_num], density, emmiters_percentage)
			np_thisVesi_vesiPts=np.array(thisVesi_vesiPts)
			np_thisVesi_vesiPts*=0.001
		
			if(psftogether==0): # psf function will be run one mito at a time
				vesiPts = [] # So I need to submit a list of the lists of points making up each each mito
			
			# If I need all the points in one big list to run the psf on all vesicles at once, then I keep all points saved in vesiPts variable
			for pt in np_thisVesi_vesiPts:
				vesiPts.append(pt) # dump surfPts into vesiPts
		
			if(psftogether == 0):
				vesiPts_separated_by_vesi.append(vesiPts)#if I need pts organized by mito, then save each mito in here
	if(psftogether == 0):
		vesiPts = vesiPts_separated_by_vesi
	
	if(recordcoordinates == 1):#This will write the xyz coordinates of each vesicle and their radius to a csv file every single frame for every single vesicle 
		filename = "output/sample_" + str(sample) + "/all_vesi_coordinates.csv"
		if(os.path.exists(filename) == False):
			setup_vesi_csv(sample, number_of_vesicles)
		with open(filename, 'a') as csvfile:
			rows = []
			rows.append(str(frame))
			for vesi in range(number_of_vesicles):
				csvwriter = csv.writer(csvfile)
				rows.append(str(x_center[vesi]) + ", " + str(y_center[vesi]) + ", " + str(z_center[vesi]))
				rows.append(str(radius[vesi]))
			csvwriter.writerow(rows)
	
	return vesiPts, radius, x_center, y_center, z_center

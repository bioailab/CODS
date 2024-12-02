import numpy as np
import pandas as pd
import multiprocessing
from skimage.transform import downscale_local_mean
from skimage.util import random_noise
import skimage.io
import os
from skimage import filters
from pathlib import Path
import cv2
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import repeat
from helper_generator_v5 import generate_save_mito, arc_length_2d, arc_length, render, write_to_attributes, set_seed, generate_save_vesicles, write_to_iqra
import argparse
import math
import time
from scipy import interpolate
from multiprocessing import Pool
import signal
import sys
from traceback import format_exception
import linecache
import csv

def arg_parser():
	SAMPLE_DEFAULT = 1 #Samples are the number of times the program will run 
	FRAMES_DEFAULT = 1 #This is the number of images you want to generate. These images can be compiled into a video. 
	MITOPOP_DEFAULT = 2 #Mito Population is the total number of mitochondria which will exist in your video
	MODE_DEFAULT = 1 # if this is zero, then I'm in testing fast render mode. If this is 1, then I'm in production mode with full number of points and slow render
	max_xy_DEFAULT = 6000 #This is the width and height in nanometers of the images that will be produced 
	PSFTOGETHER_DEFAULT = 1 #1 = yes = together. 1 = no = one at a time. Whether or not the PSF function for the points of each mitochondria will be run all in one batch or seperately and added together afterward
	GT_DEFAULT = 1 #Should I print the gt images? 0 = no. 1 = yes. TIFF images are the blurry high quality ones. GT is the black and white.
	TIF_DEFAULT = 1 #Should I print the gt images? 0 = no. 1 = yes. TIFF images are the blurry high quality ones. GT is the black and white.
	PARALLELPSF_DEFAULT = 1 #0 = no = sequential. 1 = yes = parallel. If PSF function runs once for each mitochondria seperately then it can be parallel. 
	PARALLELSAMPLE_DEFAULT = 1 #0=no, 1=yes. If sequential PSF, then each sample can be in parallel. 
	WRITEDISK_DEFAULT = 0 #0 = no = don't write anything to disk. 1 = yes = write large variables in PSF function to disk so the program as a whole uses dramatically less RAM. If you use small enough values for max_xy and mitochondria population, or if you run with 18 and 20 for centermult and surf then the program can produce TIFF images on a computer with a typical amount of RAM, around 24 GB for example.
	probability_of_200 = 0.50#this is the probability that the wlow-whigh default range will be 200-300 instead of 300-400
	MITOWLOW_DEFAULT = 300 #This is the width of the mitochondria in nanometers
	MITOWHIGH_DEFAULT = 400 #When each mitochondria is generated on the first frame, it is given a random width in this range
	MITODENSITY_DEFAULT = 2 #the number of flourescing points in 100 nm × 100 nm area
	MITOZLOW_DEFAULT = 600 #The mitochondria's location in the Z axis is a line, one end is high and one end is low. This is the low Z value.
	MITOZHIGH_DEFAULT = 800
	RECORDCOORDINATES_DEFAULT = 0#This will write the xyz coordinates of all curve points and all center line skeleton points to a csv file every single frame for every single mitochondria 
	TERMINAL_DEFAULT = 0#This will turn on and off the GUI. setting this to 1 will turn off the GUI and only leave the terminal
	VESRLOW_DEFAULT = 50
	VESRHIGH_DEFAULT = 300
	NUMVESLOW_DEFAULT = 0
	NUMVESHIGH_DEFAULT = 0
	MAXLENGTH_DEFAULT = 5000
	ELASTICITY_DEFAULT = 8#This is the percentage of the original length that the mito length should be allowed to deviate from when wiggling
	ROTATESPEEDHIGH_DEFAULT = 0#Range of speeds that the mitochondria can be set to rotate at
	ROTATESPEEDLOW_DEFAULT = 0
	MITOWIGGLEHIGH_DEFAULT = 60# Range of intensities that mitochondria can be set to wiggle at
	MITOWIGGLELOW_DEFAULT = 30
	MITODRIFTHIGH_DEFAULT = 12#Range of speeds that mitochondria can be set to drift at
	MITODRIFTLOW_DEFAULT = 4
	MITOSEEKHIGH_DEFAULT = 231# Range of speeds that mitochondria can be set to seek at
	MITOSEEKLOW_DEFAULT = 110
	PROBABILITYMERGE_DEFAULT = 0.5# Probability that when a seeker has reached it's seeked mitochondria, they will merge into one
	PROBABILITYSPLIT_DEFAULT = 0.25#Upon a merge, the seeker and seeked each have this probability of splitting on the merge point. To "chop off" the hanging extra bit after a merge to form one long line 
	NETWORK_DEFAULT = 0
	DISSOLVE_DEFAULT = 0
	ROHITBEHAVIOR_DEFAULT = 0 #this turns on the random splitting and merging on average once every twenty frames and prints centerpoints to a csv file
	STATIONARYMITOS_DEFAULT = 0#stationary mitochondria will wiggle but they will not participate as seekers or seeked and they will not split. 
	ROHITFISSIONPERCENTAGE_DEFAULT = 3 #percentage chance that in one givin frame there will occur a split initiation as part of rohit behavior profile
	ROHITSEEKPERCENTAGE_DEFAULT = 3 #percentage chance that in one givin frame there will occur a seek initiation as part of rohit behavior profile
	
	parser = argparse.ArgumentParser(prog="node", description="Mitochondria Video!!")
	parser.add_argument("-s", "--samples", type=int, default=SAMPLE_DEFAULT,
			help="Number of times the program will run, default %d" % SAMPLE_DEFAULT)
	parser.add_argument("-f", "--frames", type=int, default=FRAMES_DEFAULT,
			help="How many frames do you want in the video??, default %d" % FRAMES_DEFAULT)
	parser.add_argument("-m", "--mitopop", type=int, default=MITOPOP_DEFAULT,
			help="How many mitochondria do you want?, default %d" % MITOPOP_DEFAULT)
	parser.add_argument("-mode", "--productionmode", type=int, default=MODE_DEFAULT,
			help="If this is 1, I'm in production mode. If this is 0, then I'm in testing mode where there's a super low number of points so render is super quick. Default %d" % MODE_DEFAULT)
	parser.add_argument("-c", "--max_xy", type=int, default=max_xy_DEFAULT,
			help="The x and y size of the max_xy. 20K is big. 8064 is a good size for several mitochondria to interact. Default: %d" % max_xy_DEFAULT)
	parser.add_argument("-g", "--printgt", type=int, default=GT_DEFAULT,
			help="Should I print gt images? 0=no, 1=yes. default %d" % GT_DEFAULT)
	parser.add_argument("-tif", "--printtif", type=int, default=TIF_DEFAULT,
			help="Should I print tiff images? 0=no, 1=yes. default %d" % TIF_DEFAULT)
	parser.add_argument("-pt", "--psftogether", type=int, default=PSFTOGETHER_DEFAULT,
			help="If this is set to 1, it will render all mitochondria at once by sending all of them to the PSF function in one big variable using lots of RAM. This is the behavior from the old code. If this is set to zero, then it will send one mitochondria at a time to be rendered by the PSF function. default %d" % PSFTOGETHER_DEFAULT)
	parser.add_argument("-pp", "--parallelpsf", type=int, default=PARALLELPSF_DEFAULT,
			help="If this is set to 1, a frame containing tif images will render the PSF of all mitochondria in a given frame in parallel with eachother if psftogether is not set to 1. default %d" % PARALLELPSF_DEFAULT)
	parser.add_argument("-ps", "--parallelsamples", type=int, default=PARALLELSAMPLE_DEFAULT,
			help="if set to 1, each sample will be done in parallel. default %d" % PARALLELSAMPLE_DEFAULT)
	parser.add_argument("-w", "--writedisk", type=int, default=WRITEDISK_DEFAULT,
			help="If set to 1, the PSF function will write it's largest variables to disk to avoid using tons of RAM. default %d" % WRITEDISK_DEFAULT)
	parser.add_argument("-mwl", "--mitowlow", type=int, default=MITOWLOW_DEFAULT,
			help="The lower end of the range of the possible width values. Width of the Mitochondria. Default %d" % MITOWLOW_DEFAULT)
	parser.add_argument("-mwh", "--mitowhigh", type=int, default=MITOWHIGH_DEFAULT,
			help="The upper end of the range of the mitochondria width. Default %d" % MITOWHIGH_DEFAULT)
	parser.add_argument("-md", "--mitodensity", type=int, default=MITODENSITY_DEFAULT,
			help="Mitochondria Density. default %d" % MITODENSITY_DEFAULT)
	parser.add_argument("-mzl", "--mitozlow", type=int, default=MITOZLOW_DEFAULT,
			help="The mitochondria's location in the Z axis is a line, one end is high and one end is low. This is the low Z value. default %d" % MITOZLOW_DEFAULT)
	parser.add_argument("-mzh", "--mitozhigh", type=int, default=MITOZHIGH_DEFAULT,
			help="High Mitochondria Z value. default %d" % MITOZHIGH_DEFAULT)
	parser.add_argument("-r", "--recordcoordinates", type=int, default=RECORDCOORDINATES_DEFAULT,
			help="If this is set to 1, then a csv file will be generated with the coordinates of each mitochondria's curve points as well as each of the points that make up their center line skeleton points for every single frame. default %d" % RECORDCOORDINATES_DEFAULT)
	parser.add_argument("-t", "--terminal", type=int, default=TERMINAL_DEFAULT,
			help="If terminal is set to 1, the GUI will be turned off. default %d" % TERMINAL_DEFAULT)
	parser.add_argument("-nvh", "--numveshigh", type=int, default=NUMVESHIGH_DEFAULT,
			help="Number of vesicles maximum. default %d" % NUMVESHIGH_DEFAULT)
	parser.add_argument("-nvl", "--numveslow", type=int, default=NUMVESLOW_DEFAULT,
			help="Number of vesicles minimum. default %d" % NUMVESLOW_DEFAULT)
	parser.add_argument("-vrl", "--vesrlow", type=int, default=VESRLOW_DEFAULT,
			help="Radius of vesical minimum. default %d" % VESRLOW_DEFAULT)
	parser.add_argument("-vrh", "--vesrhigh", type=int, default=VESRHIGH_DEFAULT,
			help="Radius of vesical maximum. default %d" % VESRHIGH_DEFAULT)
	parser.add_argument("-ml", "--maxlength", type=int, default=MAXLENGTH_DEFAULT,
			help="Maximum length of mitochondria. default %d" % MAXLENGTH_DEFAULT)
	parser.add_argument("-e", "--elasticity", type=int, default=ELASTICITY_DEFAULT,
			help="percentage of the original length that the mito length should be allowed to deviate from when wiggling. default %d" % ELASTICITY_DEFAULT)
	parser.add_argument("-rh", "--rotatespeedhigh", type=int, default=ROTATESPEEDHIGH_DEFAULT,
			help="Range of speeds that the mitochondria can be set to rotate at. default %d" % ROTATESPEEDHIGH_DEFAULT)
	parser.add_argument("-rl", "--rotatespeedlow", type=int, default=ROTATESPEEDLOW_DEFAULT,
			help="Range of speeds that the mitochondria can be set to rotate at. default %d" % ROTATESPEEDLOW_DEFAULT)
	parser.add_argument("-mwih", "--mitowiggleintensityhigh", type=int, default=MITOWIGGLEHIGH_DEFAULT,
			help="Range of intensities that mitochondria can be set to wiggle at. default %d" % MITOWIGGLEHIGH_DEFAULT)
	parser.add_argument("-mwil", "--mitowiggleintensitylow", type=int, default=MITOWIGGLELOW_DEFAULT,
			help="Range of intensities that mitochondria can be set to wiggle at. default %d" % MITOWIGGLELOW_DEFAULT)
	parser.add_argument("-mdh", "--mitodrifthigh", type=int, default=MITODRIFTHIGH_DEFAULT,
			help="Range of speeds that mitochondria can be set to drift at. default %d" % MITODRIFTHIGH_DEFAULT)
	parser.add_argument("-mdl", "--mitodriftlow", type=int, default=MITODRIFTLOW_DEFAULT,
			help="Range of speeds that mitochondria can be set to drift at. default %d" % MITODRIFTLOW_DEFAULT)
	parser.add_argument("-msh", "--mitoseekhigh", type=int, default=MITOSEEKHIGH_DEFAULT,
			help="Range of speeds that mitochondria can be set to seek at. default %d" % MITOSEEKHIGH_DEFAULT)
	parser.add_argument("-msl", "--mitoseeklow", type=int, default=MITOSEEKLOW_DEFAULT,
			help="Range of speeds that mitochondria can be set to seek at. default %d" % MITOSEEKLOW_DEFAULT)
	parser.add_argument("-pm", "--probabilitymerge", type=float, default=PROBABILITYMERGE_DEFAULT,
			help="Probability that when a seeker has reached it's seeked mitochondria, they will merge into one. default " + str(PROBABILITYMERGE_DEFAULT))
	parser.add_argument("-pos", "--probabilityofsplit", type=float, default=PROBABILITYSPLIT_DEFAULT,
			help="Upon a merge, the seeker and seeked each have this probability of splitting on the merge point. default " + str(PROBABILITYSPLIT_DEFAULT))
	parser.add_argument("-n", "--network", type=int, default=NETWORK_DEFAULT,
			help="Define behavior of mitochondria over time. default %d" % NETWORK_DEFAULT)
	parser.add_argument("-d", "--dissolve", type=int, default=DISSOLVE_DEFAULT,
			help="Define behavior of mitochondria over time. default %d" % DISSOLVE_DEFAULT)
	parser.add_argument("-rb", "--rohitbehavior", type=int, default=ROHITBEHAVIOR_DEFAULT,
			help="this turns on the random splitting and merging on average once every twenty frames and prints centerpoints to a csv file. default %d" % ROHITBEHAVIOR_DEFAULT)
	parser.add_argument("-sm", "--stationarymitos", type=int, default=STATIONARYMITOS_DEFAULT,
			help="This is the number of stationary mitochondria should exist. These won't participate in merging seeking or splitting or drifting. default %d" % STATIONARYMITOS_DEFAULT)
	parser.add_argument("-rfp", "--rohitfissionpercentage", type=int, default=ROHITFISSIONPERCENTAGE_DEFAULT,
			help="What percentage chance is there on one given frame that there will be a split initiated? default %d" % ROHITFISSIONPERCENTAGE_DEFAULT)
	parser.add_argument("-rsp", "--rohitseekpercentage", type=int, default=ROHITSEEKPERCENTAGE_DEFAULT,
			help="What percentage chance is there on one given frame that there will be a seek initiated? default %d" % ROHITSEEKPERCENTAGE_DEFAULT)
	
	
	#Do this, and also download from server there's a 1k test for rohit with GT!!
	return parser

def get_length(curve_pointXs_this_mito, curve_pointYs_this_mito, sample, frame):
	#gets the length of a line drawn smoothly curving between the points given
	if(len(curve_pointXs_this_mito) == 2):#If I'm getting the distance between two points, just return that instead of doing something complex
		return math.hypot(curve_pointXs_this_mito[0] - curve_pointXs_this_mito[1], curve_pointYs_this_mito[0] - curve_pointYs_this_mito[1])#distance between these two points
	elif(len(curve_pointXs_this_mito) < 2):
		msg = "cannot get the distance between less than two points\n"
		print(msg, end="")
		write_to_attributes(sample, msg, frame, True)#this writes the message to a text file
	mitozhighTemp = mitozhigh #the value that one end of the mitochondria sits at. mitozlow is the value that the other end sits at
	tck,u = interpolate.splprep([curve_pointXs_this_mito,curve_pointYs_this_mito], k=2,s=0)#This and everything below it is from the old code 
	unew = np.arange(0, 1, 0.01)
	out = interpolate.splev(unew, tck)
	x = out[0]
	y = out[1]
	arc_2d = arc_length_2d(x, y)
	if(arc_2d < (mitozhigh-mitozlow)):
		mitozhighTemp = mitozlow + int(arc_2d)
	z = np.linspace(mitozlow, mitozhighTemp, len(x))
	return arc_length(x, y, z)

def set_default_microscope_parameters():
	global num_basis
	global rho_samples
	global magnification
	global numerical_aperture
	global coverslip_RI_design_value
	global coverslip_RI_experimental_value
	global immersion_medium_RI_design_value
	global immersion_medium_RI_experimental_value
	global specimen_refractive_index_RI
	global microns_working_distance_immersion_medium_thickness_design_value
	global microns_coverslip_thickness_experimental_value
	global microns_coverslip_thickness_design_value
	global microscope_tube_length_in_microns
	
	num_basis = 100     # Number of rescaled Bessels that approximate the phase function.
	rho_samples = 1000  # Number of pupil sample along the radial direction.
	magnification = 100.0
	numerical_aperture = 1.4
	coverslip_RI_design_value = 1.515
	coverslip_RI_experimental_value = 1.515
	immersion_medium_RI_design_value = 1.515
	immersion_medium_RI_experimental_value = 1.515
	specimen_refractive_index_RI = 1.33
	microns_working_distance_immersion_medium_thickness_design_value = 150
	microns_coverslip_thickness_experimental_value = 170
	microns_coverslip_thickness_design_value = 170
	microscope_tube_length_in_microns = 200.0 * 1.0e+3
	
	# Microscope parameters.
	m_params = {"M" : magnification,					# magnification
			"NA" : numerical_aperture,					# numerical aperture
			"ng0" : coverslip_RI_design_value,			# coverslip RI design value
			"ng" : coverslip_RI_experimental_value,		# coverslip RI experimental value
			"ni0" : immersion_medium_RI_design_value,	# immersion medium RI design value
			"ni" : immersion_medium_RI_experimental_value,	# immersion medium RI experimental value
			"ns" : specimen_refractive_index_RI,		# specimen refractive index (RI)
			"ti0" : microns_working_distance_immersion_medium_thickness_design_value,	# microns, working distance (immersion medium thickness) design value
			"tg" : microns_coverslip_thickness_experimental_value,	# microns, coverslip thickness experimental value
			"tg0" : microns_coverslip_thickness_design_value,	# microns, coverslip thickness design value
			"zd0" : microscope_tube_length_in_microns}	# microscope tube length (in microns).
	return m_params

def gui(terminal, number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r, rohit_behavior, stationary_mitos):
	cancel = False
	if(terminal == 0): #If GUI is turned on 
		global centermult
		global surfDivisor
		global elasticity
		global rotate_speed_range_high
		global rotate_speed_range_low
		global mito_wiggle_high
		global mito_wiggle_low
		global mito_drift_high
		global mito_drift_low
		global mito_seek_high
		global mito_seek_low
		global probability_of_merge
		global probability_of_split_on_merge
		global network
		global dissolve
		global psftogether
		global samples
		global frames
		global max_xy
		global num_of_mitos
		global printGt
		global printTif
		global parallelPSF
		global parallelsamples
		global writedisk
		global mitowlow
		global mitowhigh
		global density_mitochondria
		global mitozlow
		global mitozhigh
		global recordcoordinates
		global num_basis
		global rho_samples
		global magnification
		global numerical_aperture
		global coverslip_RI_design_value
		global coverslip_RI_experimental_value
		global immersion_medium_RI_design_value
		global immersion_medium_RI_experimental_value
		global specimen_refractive_index_RI
		global microns_working_distance_immersion_medium_thickness_design_value
		global microns_coverslip_thickness_experimental_value
		global microns_coverslip_thickness_design_value
		global microscope_tube_length_in_microns
		global max_mito_length
		
		productionMode = False
		if(centermult == 1 and surfDivisor == 1):
			productionMode = True
		
		returned_values = gui_config(psftogether, samples, frames, max_xy, printTif, printGt, parallelPSF, 
						parallelsamples, writedisk, recordcoordinates, productionMode, rohit_behavior)#GUI for cmd line arguments
		cancel = returned_values[0]
		psftogether = returned_values[1]
		samples = returned_values[2]
		frames = returned_values[3]
		max_xy = returned_values[4]
		printTif = returned_values[5]
		printGt = returned_values[6]
		parallelPSF = returned_values[7]
		parallelsamples = returned_values[8]
		writedisk = returned_values[9]
		recordcoordinates = returned_values[10]
		productionMode = returned_values[11]
		rohit_behavior = returned_values[12]
		
		if(productionMode == False):
			centermult = 18
			surfDivisor = 20
		else:
			centermult = 1
			surfDivisor = 1
		
		if(cancel == False):
			returned_values = gui_mito(num_of_mitos, mitowlow, mitowhigh, max_mito_length, elasticity, density_mitochondria, mitozlow, mitozhigh, 
					rotate_speed_range_high, rotate_speed_range_low, mito_wiggle_high, mito_wiggle_low, 
					mito_drift_high, mito_drift_low, mito_seek_high, mito_seek_low, probability_of_merge, 
					probability_of_split_on_merge, network, dissolve, stationary_mitos)
			cancel = returned_values[0]
			num_of_mitos = returned_values[1]
			mitowlow = returned_values[2]
			mitowhigh = returned_values[3]
			max_mito_length = returned_values[4]
			elasticity = returned_values[5]
			density_mitochondria = returned_values[6]
			mitozlow = returned_values[7]
			mitozhigh = returned_values[8]
			rotate_speed_range_high = returned_values[9]
			rotate_speed_range_low = returned_values[10]
			mito_wiggle_high = returned_values[11]
			mito_wiggle_low = returned_values[12]
			mito_drift_high = returned_values[13]
			mito_drift_low = returned_values[14]
			mito_seek_high = returned_values[15]
			mito_seek_low = returned_values[16]
			probability_of_merge = returned_values[17]
			probability_of_split_on_merge = returned_values[18]
			network = returned_values[19]
			dissolve = returned_values[20]
			stationary_mitos = returned_values[21]
			
			if(cancel == False):
				
				returned_values = gui_vesi(number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r)
				cancel = returned_values[0]
				number_of_vesicles_min = returned_values[1]
				number_of_vesicles_max = returned_values[2]
				vesi_min_r = returned_values[3]
				vesi_max_r = returned_values[4]
				
				if(cancel == False): #If I didn't close out of the first GUI
					
					returned_values = gui_micro_pars(num_basis, 
								rho_samples, magnification, numerical_aperture, coverslip_RI_design_value, coverslip_RI_experimental_value, immersion_medium_RI_design_value, 
								immersion_medium_RI_experimental_value, specimen_refractive_index_RI, microns_working_distance_immersion_medium_thickness_design_value, 
								microns_coverslip_thickness_experimental_value, microns_coverslip_thickness_design_value, microscope_tube_length_in_microns) 
								#GUI for microscope parameters
					cancel = returned_values[0]
					num_basis = returned_values[1]
					rho_samples = returned_values[2]
					magnification = returned_values[3]
					numerical_aperture = returned_values[4]
					coverslip_RI_design_value = returned_values[5]
					coverslip_RI_experimental_value = returned_values[6]
					immersion_medium_RI_design_value = returned_values[7]
					immersion_medium_RI_experimental_value = returned_values[8]
					specimen_refractive_index_RI = returned_values[9]
					microns_working_distance_immersion_medium_thickness_design_value = returned_values[10]
					microns_coverslip_thickness_experimental_value = returned_values[11]
					microns_coverslip_thickness_design_value = returned_values[12]
					microscope_tube_length_in_microns = returned_values[13]
	
	return cancel, number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r, rohit_behavior, stationary_mitos

def prepare_filesystem(sample, rohit_behavior):#create folders needed to store output
	attributesTXT = "output/sample_"+str(sample)+'/attributes.txt'
	if os.path.exists(attributesTXT):#I won't be overwriting this file, so delete the old one before I start appending to it
		os.remove(attributesTXT)
	iqraJSON = "output/sample_"+str(sample)+'/iqra_record.json'
	if os.path.exists(iqraJSON):#I won't be overwriting this file, so delete the old one before I start appending to it
		os.remove(iqraJSON)
	vesiCoordsCSV = "output/sample_" + str(sample) + "/all_vesi_coordinates.csv"
	if os.path.exists(vesiCoordsCSV):#I won't be overwriting this file, so delete the old one before I start appending to it
		os.remove(vesiCoordsCSV)
	rohitCSV = "output/sample_" + str(sample) + "/rohit_centerpoints.csv"
	if((os.path.exists(rohitCSV)) == True):
		os.remove(rohitCSV)
	if(recordcoordinates == 1):
		for num in range(1000):#Hopefully there's never more than 1K mitochondria? 
			if(os.path.exists("output/sample_" + str(sample) + "/mito_" + str(num) + "_coordinates.csv")):#Delete all old csv files for these relevant samples
				os.remove("output/sample_" + str(sample) + "/mito_" + str(num) + "_coordinates.csv")
			else:
				break
	Path('output/sample_'+str(sample)).mkdir(parents=True, exist_ok=True)
	Path('output/sample_'+str(sample)+"/physics_gt").mkdir(parents=True, exist_ok=True)#ground truth
	Path('output/sample_'+str(sample)+"/tif").mkdir(parents=True, exist_ok=True)#blurred tiff images

def center_of_mass_for_one_mito(mito):
	#Takes the id number of a mito and uses that to get it's curve points and gets the average of the four curve points. 
	mass_center=[0, 0] #array with two elements, first is x, second is y coordinate
	curve_pt_total_num=0
	for point in range(len(curve_pointXs[mito])):#For each curve point in this mito
		curve_pt_total_num+=1
		mass_center[0]=mass_center[0]+curve_pointXs[mito][point]
		mass_center[1]=mass_center[1]+curve_pointYs[mito][point]#adding the x and y values of the curve points
	mass_center[0]=mass_center[0]/curve_pt_total_num#and dividing by the total, to get the average
	mass_center[1]=mass_center[1]/curve_pt_total_num#gets center of mass of curve points
	return mass_center

def rotatePoint(center, pointX, pointY, angleDegrees):#returns a point that has been rotated around a central axis
	#The center is the point that you want to rotate this point around. X and Y are the coordinates of the point I want rotated. Angle is the amount of degrees I want rotated
	angleRadians=math.radians(angleDegrees)#convert degrees into radians
	destinationX=center[0] + math.cos(angleRadians) * (pointX - center[0]) - math.sin(angleRadians) * (pointY - center[1])
	destinationY=center[1] + math.sin(angleRadians) * (pointX - center[0]) + math.cos(angleRadians) * (pointY - center[1])
	return destinationX, destinationY#Returns where the point would end up if it was rotated

def rotateMito(mito_nums):
	global curve_pointXs
	global curve_pointYs
	mass_center=[0, 0] #array with two elements, first is x, second is y. point which is the center/average of this mito's 4 curve points
	curve_pt_total_num=0 #It's important to count the total number curve points because I could be rotating more than one mitochondria around their shared center
	for mito_num in mito_nums:#For each mito
		for point in range(len(curve_pointXs[mito_num])):#For each curve point in this mito
			curve_pt_total_num+=1
			mass_center[0]=mass_center[0]+curve_pointXs[mito_num][point]
			mass_center[1]=mass_center[1]+curve_pointYs[mito_num][point]#adding the x and y values of all the curve points of all mitos 
	mass_center[0]=mass_center[0]/curve_pt_total_num#and dividing by the total, to get the average
	mass_center[1]=mass_center[1]/curve_pt_total_num#gets center of mass of curve points
	for mito_num in mito_nums:
		for point in range(len(curve_pointXs[mito_num])):		#rotates all points around the center of mitochondria
			curve_pointXs[mito_num][point], curve_pointYs[mito_num][point] = rotatePoint(mass_center, curve_pointXs[mito_num][point], 
						curve_pointYs[mito_num][point], rotate_speed[mito_num])

def wiggleMito(sample, mito_num, elasticity, frame):#All coordinates shifted randomly. But must stay within the elasticity percentage of the original length
	global curve_pointXs
	global curve_pointYs
	
	set_seed()#This takes in to account the time on the clock to best generate a different random number from the last time these random numbers were generated
	count_attempts = 0
	found_good_length = False
	while found_good_length == False: #Do this until I randomly get a good length within the elasticity percentage of the original length
		
		curve_pointXs_backup = curve_pointXs[mito_num].copy()#Make a backup of the original curve point values. The .copy() function must be used because if the equals sign = is used then the variables will be linked and changing one will change the other. 
		#Backup is needed because I might need to try multiple sets of values to get a length within the limits
		curve_pointYs_backup = curve_pointYs[mito_num].copy()
		
		for point in range(len(curve_pointXs[mito_num])):#For every point in this mito
			randomX=np.random.randint(-mito_wiggle[mito_num], mito_wiggle[mito_num] + 1)
			randomY=np.random.randint(-mito_wiggle[mito_num], mito_wiggle[mito_num] + 1)#choose a random distance within the wiggle intensity
			curve_pointXs_backup[point]=curve_pointXs_backup[point]+randomX#Move each curve point by this amount
			curve_pointYs_backup[point]=curve_pointYs_backup[point]+randomY
		
		mito_length_this_attempt = get_length(curve_pointXs_backup, curve_pointYs_backup, sample, frame)#Get length of the new mitochondria position after wiggling
		
		len_upper_lim = original_length[mito_num] + (elasticity * 0.01 * original_length[mito_num])#Calculate the range that the new length is allowed to be within
		len_lower_lim = original_length[mito_num] - (elasticity * 0.01 * original_length[mito_num])
		
		count_attempts = count_attempts + 1
		
		if(mito_length_this_attempt < len_upper_lim and mito_length_this_attempt > len_lower_lim):
			found_good_length = True
			curve_pointXs[mito_num] = curve_pointXs_backup#If length is within the range defined by the elasticity of the mitochondria, then save changes to the real mitochondria
			curve_pointYs[mito_num] = curve_pointYs_backup
		elif(count_attempts > 200):
			found_good_length = True #Set this to true because I need to exit from this while loop now. 
			msg = "Failed 200 times to find good length wiggle. \\\\\\ Sample: " + str(sample) + " Mito number " + str(mito_num) + " -  - original_length[mito_num: " + str(original_length[mito_num]) + "\n"
			if(frame < 900):
				print(msg, end="")
			write_to_attributes(sample, msg, frame, True)

def drift_to_connect_merged_points(sample, frame, plot_to_show_connect_merged = False, if_plotting_which_connection_to_show = 0):
	#Makes sure that all the points in the merged mitos that should be pairs of merge points have identical X/Y coordinates
	#This is done by chosing one mitochondria and moving all merged ones toward it so the merge points share coordinate values, and repeating this process until all merge points overlap
	merged_points_successfully_overlaping = False
	loop_count = 0
	while merged_points_successfully_overlaping == False and loop_count < 100:
		loop_count = loop_count + 1
		merged_range = range(len(merged))# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
		#If this is the first time in this loop, this is the value of merged_range. if there's 5 mitochondria, then it's 0-4
		if(loop_count > 90):
			merged_range = []
			for merged_range_num in range(2, len(merged), 3):#2, 5, 8, 0, 3, 6, 9, 1, 4, 7, 10
				merged_range.append(merged_range_num)
			for merged_range_num in range(0, len(merged), 3):#If I'm unable to bring all the merged points together, 
				merged_range.append(merged_range_num)		 #it could be because there's a repeating loop in trying to move them to bring them together
			for merged_range_num in range(1, len(merged), 3):#So this mixes up the order in which I try to bring them together to try to get around such loops 
				merged_range.append(merged_range_num)		 #This is mainly a worry in high numbers of complexly merged mitos
		elif(loop_count > 80):
			merged_range = []
			for merged_range_num in range(1, len(merged), 3):#1, 4, 7, 10, 2, 5, 8, 0, 3, 6, 9
				merged_range.append(merged_range_num)
			for merged_range_num in range(2, len(merged), 3):
				merged_range.append(merged_range_num)
			for merged_range_num in range(0, len(merged), 3):
				merged_range.append(merged_range_num)
		elif(loop_count > 70):
			merged_range = []
			for merged_range_num in range(1, len(merged), 3):#1, 4, 7, 10, 0, 3, 6, 9, 2, 5, 8
				merged_range.append(merged_range_num)
			for merged_range_num in range(0, len(merged), 3):
				merged_range.append(merged_range_num)
			for merged_range_num in range(2, len(merged), 3):
				merged_range.append(merged_range_num)
		elif(loop_count > 60):
			merged_range = []
			for merged_range_num in range(0, len(merged), 3):# 0, 3, 6, 9, 2, 5, 8, 1, 4, 7, 10
				merged_range.append(merged_range_num)
			for merged_range_num in range(2, len(merged), 3):
				merged_range.append(merged_range_num)
			for merged_range_num in range(1, len(merged), 3):
				merged_range.append(merged_range_num)
		elif(loop_count > 50):
			merged_range = []
			for merged_range_num in range(2, len(merged), 3):# 2, 5, 8, 1, 4, 7, 10, 0, 3, 6, 9
				merged_range.append(merged_range_num)
			for merged_range_num in range(1, len(merged), 3):
				merged_range.append(merged_range_num)
			for merged_range_num in range(0, len(merged), 3):
				merged_range.append(merged_range_num)
		elif(loop_count > 40):
			merged_range = []
			for merged_range_num in range(0, len(merged), 3):# 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8
				merged_range.append(merged_range_num)
			for merged_range_num in range(1, len(merged), 3):
				merged_range.append(merged_range_num)
			for merged_range_num in range(2, len(merged), 3):
				merged_range.append(merged_range_num)
		elif(loop_count > 30):
			merged_range = []
			for merged_range_num in range(1, len(merged), 2):# 1, 3, 5, 7, 9, 0, 2, 4, 6, 8, 10
				merged_range.append(merged_range_num)
			for merged_range_num in range(0, len(merged), 2):
				merged_range.append(merged_range_num)
		elif(loop_count > 20):
			merged_range = []
			for merged_range_num in range(0, len(merged), 2):# 0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9
				merged_range.append(merged_range_num)
			for merged_range_num in range(1, len(merged), 2):
				merged_range.append(merged_range_num)
		elif(loop_count > 10):
			merged_range = range(len(merged)-1, -1, -1)# 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
		
		if(plot_to_show_connect_merged and frame != 2):
			merged_range = range(if_plotting_which_connection_to_show, if_plotting_which_connection_to_show + 1)
		
		for mito in merged_range:#For each mito
			if(merged[mito][0] != -1):# -1 is often used as a placeholder when there should be no value
				for mergeRef in merged[mito]:#For each mito merged with this mito
					if(mergeRef[0] != -1):
						#Move this mito in the direction and distance needed to match up their merge points
						try:
							exact_drift_x = curve_pointXs[mito][mergeRef[1]] - curve_pointXs[mergeRef[0]][mergeRef[2]]#amount needed to move the merged mito's merge point so it lines up with the merge point of this mito along the x axis
						except:
							print(" ")
							print(" IndexError: list index out of range ")
							print("mito: ", mito, ", mergeRef[1]: ", mergeRef[1], ", mergeRef[0]: ", mergeRef[0], ", mergeRef[2]: ", mergeRef[2])
							print(" ")
						exact_drift_y = curve_pointYs[mito][mergeRef[1]] - curve_pointYs[mergeRef[0]][mergeRef[2]]#y axis
						if(exact_drift_x != 0 or exact_drift_y != 0):#Only drift if the merge point pair is not already the same point
							driftMito(mergeRef[0], 0, 0, 0, exact_drift_x, exact_drift_y)#Drift the entire mito such that their merge points overlap
			if(plot_to_show_connect_merged and frame != 2):
				break
		
		merged_points_successfully_overlaping = True#checks if I'm done making all the merge points overlap perfectly
		if(plot_to_show_connect_merged == False):
			for mito in range(len(merged)):#For each mito
				try:
					if(merged[mito][0] != -1):
						for mergeRef in merged[mito]:#For each mito which is merged with this mito
							if(mergeRef[0] != -1):
								if((curve_pointXs[mito][mergeRef[1]] != curve_pointXs[mergeRef[0]][mergeRef[2]]) or #Looking for an example of a merge point pair that don't have the same coordinates
												(curve_pointYs[mito][mergeRef[1]] != curve_pointYs[mergeRef[0]][mergeRef[2]])):
									merged_points_successfully_overlaping = False#Found an example. Try again
				except:
					print("")
	if(loop_count>98):
		#msg = "there's a looping issue here... can't bring wiggled mitos back in merge with eachother even after 100 tries//////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ \n"
		msg = "."#this means the merged mitochondria can't be brought perfectly back together so they're all touching. So one pair of the merged ones will be seperated until they lie in a different configuration for example by a split or maybe an additional merge into the motichondria network
		print(msg, end="")
		write_to_attributes(sample, "Connect Merge Points Error\n", frame)

def get_exact_x_and_y(directionX, directionY, distance):
	#takes direction x and y values which are not scaled to achieve the desired hypotenus distance
	#outputs the exact x and y values that will produce the desired distance (hypotenus)
	hypot = math.sqrt((directionX*directionX)+(directionY*directionY))#hypotenus of the given direction XY
	multiple=distance/hypot#how much to change the direction values by in order to result in the desired distance
	exact_directionX=directionX*multiple#getting the exact direction values that will perfectly match the distance I want!
	exact_directionY=directionY*multiple
	return exact_directionX, exact_directionY

def driftMito(mito_num, directionX, directionY, distance, exactX = None, exactY = None):
	#moves all curve points of a mitochondria in a given direction
	#direction is given by a combonation of x and y adjustments that can be mode to achieve a hypotenus of the correct desired direction
	#Exact X and Y values are those that produce not only the correct direction but also the correct distance desired
	#if exactX and Y are known, then direction X and Y are ignored
	global curve_pointXs
	global curve_pointYs
	
	if exactX is None:
		exactX, exactY = get_exact_x_and_y(directionX, directionY, distance)
		#exact x and y allows calling the function when you can already encode the distance into the x and y directions, 
		#instead of doing a bunch of conversion calculations with the hypotenus and everything
	for point in range(len(curve_pointXs[mito_num])):
		curve_pointXs[mito_num][point]=curve_pointXs[mito_num][point]+(exactX)#adjust point location by the exact amount desired
		curve_pointYs[mito_num][point]=curve_pointYs[mito_num][point]+(exactY)

def spread_endpoints(to_grow, x1, x2, y1, y2, sample):
	#move the end curve points at the two outter edges of a mitochondria away from eachother
	#takes the amount that the mitochondria is to grow in length, and the x and y values of it's two endpoints
	directionX1, directionY1 = get_exact_x_and_y(x1 - x2, y1 - y2, to_grow / 2)#Get the direction needed for point 1 to move in the direction of point 2 by half the distance I want the mitochondria to grow
	x1 = x1 + directionX1#Make the move
	y1 = y1 + directionY1
	directionX2, directionY2 = get_exact_x_and_y(x2 - x1, y2 - y1, to_grow / 2)
	x2 = x2 + directionX2
	y2 = y2 + directionY2
	return x1, x2, y1, y2

def initiate_seek(sample, frame, seeker, seeked):
	global iqra_seeker_start_frame
	text = "Frame " + str(frame) + "\n"
	text = text + "SEEK initiated. Mito " + str(seeker) + " seeking " + str(seeked) + "\n\n"
	write_to_attributes(sample, text, frame)
	mito_seek_target[seeker] = seeked
	if(iqra_record):
		iqra_seeker_start_frame[seeker] = frame

def cancel_seek(sample, frame, seeker, seeked, reason="other"):
	global iqra_sentence_id
	text = "Frame " + str(frame) + "\n"
	text = text + "SEEK Canceled. Mito " + str(seeker) + " no longer seeking " + str(seeked) + ". Reason: " + reason + "\n\n"
	write_to_attributes(sample, text, frame)
	mito_seek_target[seeker] = -1
	if(iqra_record):
		if(frame - iqra_seeker_start_frame[seeker] > 4):
			write_to_iqra(sample, '{"category": 1, "video_id": ' + str(sample) + ', "start_time": ' + str(iqra_seeker_start_frame[seeker]) + ', "end_time": ' + str(frame) + ', "split": "trained validation or test split", "id": ' + str(iqra_sentence_id) + '},\n{"caption": "Mitochondria ' + str(seeker) + ' is seeking mitochondria number ' + str(seeked) + '", "video_id": ' + str(sample) + ', "sentence_id": ' + str(iqra_sentence_id) + ', "id": ' + str(iqra_sentence_id) + '},\n')
			iqra_sentence_id = iqra_sentence_id + 1

def split(split_mito_num, split_point, frame, sample, widths, stationary_mitos, rohit_id_map, split_is_due_to_a_merge=False):#cuts a mito into two parts. 
	#The larger part takes any merged mitos who were merged exactly on the split point. 
	#The smaller part get's it's own new mito id (it's own spot in the merged array and other global arrays)
	global curve_pointXs
	global curve_pointYs
	actually_did_the_split = False
	
	set_seed()
	proceed = False#check for errors in input values before proceeding with the split
	minimum_mito_length = 290 # If a mitochondria is shorter than this, it won't be able to wiggle without exceeding the limits of the elasticity
	
	if(len(curve_pointXs) > split_mito_num):
		#id number of mito to split must be within existing mitos. 
		if(len(curve_pointXs[split_mito_num]) > split_point):
			#curve point to split on must be within the existing curve points of this mito
			proceed = True
		else:
			msg = "split point is larger than number of curve points on this mito? " + str(len(curve_pointXs[split_mito_num])) + str(split_point) + "\nERROR\n"
	else:
		msg = "mito number to split is larger than the number of existing mitos? " + str(len(curve_pointXs)) + ", " + str(split_mito_num) + "\nERROR\n"
	
	if((split_mito_num in stationary_mitos) == True):
		proceed = False
		msg = ""
	
	if(proceed):
		global merged
		global mito_wiggle
		global mito_seek_speed
		global mito_seek_target
		global mito_drift
		global mito_drift_directionX
		global mito_drift_directionY
		global rotate_speed
		global original_length
		global iqra_sentence_id
		
		text = "Frame " + str(frame) + "\n"
		text = text + "Splitting mito " + str(split_mito_num) + " on curve point number " + str(split_point) + "\n"
		text = text + str(split_mito_num)+ " center point: " + str(center_of_mass_for_one_mito(split_mito_num)) + "\n"
		
		split_mito_curve_point_num = len(curve_pointXs[split_mito_num])#The number of curve points which exist in this mito
		
		#If split point is an end point, then check for merged mitochondria on that split point. delete the mergers so the merged points drift apart
		if(split_point == split_mito_curve_point_num-1 or split_point == 0):#If split point is at one end of the mito!
			if(merged[split_mito_num][0] != -1):
				for merged_mito in range(len(merged[split_mito_num])):#go through each merged mito
					mm = merged[split_mito_num][merged_mito]#The reason I do this with the range(len( is because I need to modify mm when I need to delete a merge! and you can't modify it through a var 
					if((mm[1] == split_mito_curve_point_num-1 and split_point == split_mito_curve_point_num-1) or (mm[1] == 0 and split_point == 0)):#If the merged mito is merged at this same end of our mito where I want to split!
						merged[split_mito_num][merged_mito] = [-1, -1, -1] #Terminate the merger
						for merged_mitos_merged_mito in range(len(merged[mm[0]])):
							if(merged[mm[0]][merged_mitos_merged_mito][0] == split_mito_num): #find the element of the merged mito's array that references us
								merged[mm[0]][merged_mitos_merged_mito] = [-1, -1, -1] #Terminate the merger
								msg = "Split merged on end merge point\n"
								if(rohit_behavior == 0 and frame < 900):
									print(msg, end="")
								actually_did_the_split = True
								write_to_attributes(sample, msg, frame)
								text = text + str(split_mito_num) + " unmerged with " + str(mm[0]) + "\n"
								text = text + str(mm[0])+ " center point: " + str(center_of_mass_for_one_mito(mm[0])) + "\n"
								text = text + str(merged) + "\n\n"
								write_to_attributes(sample, text, frame)
								mito_drift_directionX[mm[0]] = np.random.randint(-1000, 1001) #Give the unmerged mitochondria a new random drift direction and speed
								mito_drift_directionY[mm[0]] = np.random.randint(-1000, 1001)
								mito_drift[mm[0]] = np.random.randint(mito_drift_low, mito_drift_high + 1)
								if(iqra_record):
									write_to_iqra(sample, '{"category": 1, "video_id": ' + str(sample) + ', "start_time": ' + str(frame) + ', "end_time": ' + str(frame) + ', "split": "trained validation or test split", "id": ' + str(iqra_sentence_id) + '},\n{"caption": "Mitochondria ' + str(split_mito_num) + ' has unmerged from mitochondria ' + str(mm[0]) +'", "video_id": ' + str(sample) + ', "sentence_id": ' + str(iqra_sentence_id) + ', "id": ' + str(iqra_sentence_id) + '},\n')
									iqra_sentence_id = iqra_sentence_id + 1
		#If the split is somewhere in the middle of the mitochondria, then split it on that point. One of the mitochondria will be given two of the original curve points and the other resulting mitochondria will be given three. The other two plus one which is the same so their ends are touching right after they are split. 
		#The merged mitochondria stay merged
		#If there's a merged mitochondria where the merge point is exactly on the split point, then that merge point is migrated to the bigger half (if the mitochondria has 4 curve points then the merged mitocondria stays with the half which includes 3 curve points, not the small one which is only left with two of the original curve points)
		else:
			actually_did_the_split = True
			merge_possibility = ""
			if(split_is_due_to_a_merge== True ):
				merge_possibility += " Due to a Merge"
			if(rohit_behavior == 0 and frame < 900):
				print(msg, end="")
			write_to_attributes(sample, "Split" + merge_possibility + "\n", frame)
			if(iqra_record):
				write_to_iqra(sample, '{"category": 1, "video_id": ' + str(sample) + ', "start_time": ' + str(frame) + ', "end_time": ' + str(frame) + ', "split": "trained validation or test split", "id": ' + str(iqra_sentence_id) + '},\n{"caption": "Mitochondria ' + str(split_mito_num) + ' has split on curve point ' + str(split_point) + str(merge_possibility) +'", "video_id": ' + str(sample) + ', "sentence_id": ' + str(iqra_sentence_id) + ', "id": ' + str(iqra_sentence_id) + '},\n')
				iqra_sentence_id = iqra_sentence_id + 1
			#this is supposed to terminate a merge which is on the split point. 						#it's actually broken and it terminates other merges also, so this should be looked into if you wanna use it.			#fix it so it only terminates merges which the merge point is exactly on the split point plssss
			#if(np.random.randint(0,2) == 0 and split_is_due_to_a_merge == False):#50 percent chance of this happening
			#	for merged_mito in range(len(merged[split_mito_num])):
			#		mm = merged[split_mito_num][merged_mito]
			#		if(mm != -1):
			#			merged[split_mito_num][merged_mito] = [-1, -1, -1]
			#			for merged_mitos_merged_mito in range(len(merged[mm[0]])):
			#				if(merged[mm[0]][merged_mitos_merged_mito] != -1):
			#					if(merged[mm[0]][merged_mitos_merged_mito][0] == split_mito_num): #find the element of the merged mito's array that references us
			#						merged[mm[0]][merged_mitos_merged_mito] = [-1, -1, -1] #Terminate the merger
			#						print("Split merged on split point")
			#						text = text + str(split_mito_num) + " unmerged with " + str(mm[0]) + "\n"
			#						text = text + str(mm[0])+ " center point: " + str(center_of_mass_for_one_mito(mm[0])) + "\n"
			#The bigger part of the mitochondria will be given the mitochondria id of the "old" mitochondria that I'm splitting
			#The smaller part of the mitochondia will be given the new mitochondria id below
			new_mito_num = len(curve_pointXs)#the id number of this newly generated mitochondria is one plus the highest id
			
			if(len(merged) <= new_mito_num):#if there's not enough items in merged to account for the newly created mitochondria, then add more (this is almost certainly the case)
				merged.append([-1])
			else:
				merged[new_mito_num] = [-1]# -1 is used as a placeholder. No mitochondria's are merged with this new one yet. 
			if(len(iqra_seeker_start_frame) <= new_mito_num):
				iqra_seeker_start_frame.append(-1)
			else:
				iqra_seeker_start_frame[new_mito_num] = -1
			
			curve_pointXs.append([])
			curve_pointYs.append([])#new mitochondria that's splitting off will be added as it's own element in curve_points
			x1s = []
			y1s = []
			x2s = []#the x and y coordinate values of the curve points of the new big mito and the new little mito will be stored in 1 or 2, 
			y2s = []#and then whichever one is shortest will be given the new mitochondria id and the longest one will keep the old id
			
			for curve_point_num in range(split_mito_curve_point_num):#for each curve point in the big old mito that I'm splitting (4 for now)
				curve_pointX = curve_pointXs[split_mito_num][curve_point_num]#deposit the values from the old curve points into the two new smaller arrays 1 and 2
				curve_pointY = curve_pointYs[split_mito_num][curve_point_num]# I don't know which one is short and which one is long yet
				if(curve_point_num <= split_point):# add the split point to 1 along with the lower curve points 
					x1s.append(curve_pointX)
					y1s.append(curve_pointY)
				if(curve_point_num >= split_point):# add the split point to 2 along with the higher curve points
					x2s.append(curve_pointX)
					y2s.append(curve_pointY)
			
			#If any of the resulting mitos are too small, spread apart their end points to make them 280 in length
			mito1_to_grow = minimum_mito_length - get_length(x1s, y1s, sample, frame)#If the resulting mito in array 1 is big enough, then this to_grow variable will be negative
			mito2_to_grow = minimum_mito_length - get_length(x2s, y2s, sample, frame)
			if(mito1_to_grow > 0):
				x1s[0], x1s[len(x1s)-1], y1s[0], y1s[len(y1s)-1] = spread_endpoints(mito1_to_grow, x1s[0], x1s[len(x1s)-1], y1s[0], y1s[len(y1s)-1], sample)
				#The coordinates of the endpoints of the small mitochondria are sent to have their endpoints stretched apart
			if(mito2_to_grow > 0):
				x2s[0], x2s[len(x2s)-1], y2s[0], y2s[len(y2s)-1] = spread_endpoints(mito2_to_grow, x2s[0], x2s[len(x2s)-1], y2s[0], y2s[len(y2s)-1], sample)
			
			if(split_mito_curve_point_num/2 >= split_point+1):#split point is 1, so the long mito gets curve points 1,2,3 and the short one gets 0,1
				curve_pointXs[split_mito_num] = x2s#the old existing mito id is updated to have the set of curve points which contains 3 points
				curve_pointYs[split_mito_num] = y2s
				curve_pointXs[new_mito_num] = x1s#the new mito id is populated with the set of curve points which contains 2 points
				curve_pointYs[new_mito_num] = y1s
				if(merged[split_mito_num][0] != -1):#if there exists merged mitos on the split mito
					for merged_mito in range(len(merged[split_mito_num])):#go through each merged mito
						mm = merged[split_mito_num][merged_mito]#The reason I do this with the range(len( is because I need to modify mm when I need to delete a merge! and you can't modify it through a var 
						if(mm[1] >= split_point):#if there's merged mitos on the big mito which will keep the same mito id as the split mito
							if(mm[1] != split_mito_curve_point_num-1):#if this isn't the last curve point on the big mito
								mm[1] = mm[1] - split_point#big mito is starting at old index 1 so must be bumped down
							for merged_mitos_merged_mito in merged[mm[0]]:#In a mito that I'm merged with, go through every mito that that mito is merged with
								if(merged_mitos_merged_mito != -1):
									if(merged_mitos_merged_mito[0] == split_mito_num and merged_mitos_merged_mito[2] != split_mito_curve_point_num-1): 
										#find the element of the merged mito's array that references us, but ignore mitos merged with our last curve point, because I want to 
										#reduce the curve point reference of all but that last curve point, since new curve points will be inserted to refil it up to 4 curve points,
										#And those will be inserted between the last and the second to last curve points
										merged_mitos_merged_mito[2] = merged_mitos_merged_mito[2] - split_point #reduce the reference to our curve point
						else:#if there's merged mitos on the little new mito
							if(mm[1] == split_mito_curve_point_num-1):#if this is the last curve point on the little mito
								mm[1] = mm[1] + big_mito_num_of_curve_pts#little mito should end on last curve point index so bumping that up now. 
							for merged_mitos_merged_mito in merged[mm[0]]:
								if(merged_mitos_merged_mito != -1):
									if(merged_mitos_merged_mito[0] == split_mito_num):
										merged_mitos_merged_mito[0] = new_mito_num #set the mito id to the newly created mito
										if(merged_mitos_merged_mito[2] == split_point): 
											#find the element of the merged mito's array that references us
											#bump up the curve point reference of the last point to make it the end curve point
											merged_mitos_merged_mito[2] = merged_mitos_merged_mito[2] + big_mito_num_of_curve_pts
							if(merged[new_mito_num][0] == -1):#If the merged list is empty
								merged[new_mito_num][0] = mm#add the reference to this merged mito as the only element in the merged list for the little one
							else:
								merged[new_mito_num].append(mm)#Add the reference to the list of elements
							merged[split_mito_num][merged_mito] = [-1, -1, -1]#remove the reference to this merged mito from the big mito, since the merged mito is now only part of the little one
				curve_pointXs[split_mito_num].append(x2s[2])#big mito gets one new curve point
				curve_pointYs[split_mito_num].append(y2s[2])
				curve_pointXs[split_mito_num][2] = (curve_pointXs[split_mito_num][1]+curve_pointXs[split_mito_num][3])/2#New midpoint lies between the last and second to last
				curve_pointYs[split_mito_num][2] = (curve_pointYs[split_mito_num][1]+curve_pointYs[split_mito_num][3])/2
				
				curve_pointXs[new_mito_num].append(x1s[1])#little mito gets two new curve points
				curve_pointXs[new_mito_num].append(x1s[1])
				curve_pointYs[new_mito_num].append(y1s[1])
				curve_pointYs[new_mito_num].append(y1s[1])
				curve_pointXs[new_mito_num][1] = (curve_pointXs[new_mito_num][3] - curve_pointXs[new_mito_num][0]) / 4 + curve_pointXs[new_mito_num][0]#new points are between existing two
				curve_pointYs[new_mito_num][1] = (curve_pointYs[new_mito_num][3] - curve_pointYs[new_mito_num][0]) / 4 + curve_pointYs[new_mito_num][0]
				curve_pointXs[new_mito_num][2] = (curve_pointXs[new_mito_num][3] - curve_pointXs[new_mito_num][0]) / 2 + curve_pointXs[new_mito_num][0]
				curve_pointYs[new_mito_num][2] = (curve_pointYs[new_mito_num][3] - curve_pointYs[new_mito_num][0]) / 2 + curve_pointYs[new_mito_num][0]
			else:#split point is 2, so long mito gets curve points 0,1,2 and short one gets 2,3
				curve_pointXs[split_mito_num] = x1s
				curve_pointYs[split_mito_num] = y1s
				curve_pointXs[new_mito_num] = x2s
				curve_pointYs[new_mito_num] = y2s
				big_mito_num_of_curve_pts = split_mito_curve_point_num - split_point
				if(merged[split_mito_num][0] != -1):#if there exists merged mitos on the split mito
					for merged_mito in range(len(merged[split_mito_num])):#go through each merged mito
						mm = merged[split_mito_num][merged_mito]#The reason I do this with the range(len( is because I need to modify mm when I need to delete a merge! and you can't modify it through a var 
						if(mm[1] >= split_point):#if there's merged mitos on the little new mito
							if(mm[1] != split_mito_curve_point_num-1):#if this isn't the last curve point on the little mito 
								#(this is explained further below where I "reduce the reference to our curve point")
								mm[1] = mm[1] - split_point#little mito is starting at old index 1 so must be bumped down
							for merged_mitos_merged_mito in merged[mm[0]]:  #In a mito that I'm merged with, go through every mito that that mito is merged with
								if(merged_mitos_merged_mito != -1):
									if(merged_mitos_merged_mito[0] == split_mito_num):
										merged_mitos_merged_mito[0] = new_mito_num #set the mito id to the newly created mito
										if(merged_mitos_merged_mito[2] != split_mito_curve_point_num-1):
											#find the element of the merged mito's array that references us, but ignore mitos merged with our last curve point, because I want to 
											#reduce the curve point reference of all but that last curve point, since new curve points will be inserted to refil it up to 4 curve points,
											#And those will be inserted between the last and the second to last curve points
											merged_mitos_merged_mito[2] = merged_mitos_merged_mito[2] - split_point #reduce the reference to our curve point
							if(merged[new_mito_num][0] == -1):
								merged[new_mito_num][0] = mm#add the reference to this merged mito to the merged list for the little one
							else:
								merged[new_mito_num].append(mm)
							merged[split_mito_num][merged_mito] = [-1, -1, -1]#remove the reference to this merged mito from the big long mito, since the merged mito is now only part of the little one
						else:#if there's merged mitos on the big mito which will keep the same mito id as the split mito
							if(mm[1] == split_mito_curve_point_num-1):#if this is the last curve point on the big mito
								mm[1] = mm[1] + big_mito_num_of_curve_pts#big mito should end on last curve point index so bumping that up now. 
							for merged_mitos_merged_mito in merged[mm[0]]:
								if(merged_mitos_merged_mito != -1):
									if(merged_mitos_merged_mito[0] == split_mito_num and merged_mitos_merged_mito[2] == split_point): 
										#find the element of the merged mito's array that references us - bump up the curve point reference of the last point to make it the end curve point
										merged_mitos_merged_mito[2] = merged_mitos_merged_mito[2] + big_mito_num_of_curve_pts
				curve_pointXs[split_mito_num].append(x1s[2])#big mito gets one new curve point
				curve_pointYs[split_mito_num].append(y1s[2])
				curve_pointXs[split_mito_num][2] = (curve_pointXs[split_mito_num][1]+curve_pointXs[split_mito_num][3])/2#New midpoint lies between the last and second to last
				curve_pointYs[split_mito_num][2] = (curve_pointYs[split_mito_num][1]+curve_pointYs[split_mito_num][3])/2
				
				curve_pointXs[new_mito_num].append(x2s[1])#little mito gets two new curve points
				curve_pointXs[new_mito_num].append(x2s[1])
				curve_pointYs[new_mito_num].append(y2s[1])
				curve_pointYs[new_mito_num].append(y2s[1])
				curve_pointXs[new_mito_num][1] = (curve_pointXs[new_mito_num][3] - curve_pointXs[new_mito_num][0]) / 4 + curve_pointXs[new_mito_num][0]#new points are between existing two
				curve_pointYs[new_mito_num][1] = (curve_pointYs[new_mito_num][3] - curve_pointYs[new_mito_num][0]) / 4 + curve_pointYs[new_mito_num][0]
				curve_pointXs[new_mito_num][2] = (curve_pointXs[new_mito_num][3] - curve_pointXs[new_mito_num][0]) / 2 + curve_pointXs[new_mito_num][0]
				curve_pointYs[new_mito_num][2] = (curve_pointYs[new_mito_num][3] - curve_pointYs[new_mito_num][0]) / 2 + curve_pointYs[new_mito_num][0]
			
			#Update rohit_id_map
			#highest_rohit_id = max(rohit_id_map) # the first element in this list coorisponds to mito id 1. The value there points to the id in rohit's id numbering schema
			#rohit_id_map[split_mito_num] = highest_rohit_id + 1 # when there's a split, the two new mitochondria are given new ids. 
					#Nobody gets the old id which the old larger mitochondria had
			#rohit_id_map.append(highest_rohit_id + 2)
			
			for rohit_id in range(len(rohit_id_map)):
				if(rohit_id_map[rohit_id] == split_mito_num):#Mapping id numbers to this new array so I can easily create csv for rohit 
					rohit_id_map[rohit_id] = -1
					rohit_id_map.append(split_mito_num)
					rohit_id_map.append(new_mito_num)
			
			#I needed to add this part with measuring the length again and spreading if too short because for some reason lengths were ending up smaller than the allowed minimum
			split_mito_to_grow = minimum_mito_length - get_length(curve_pointXs[split_mito_num], curve_pointYs[split_mito_num], sample, frame)#If the resulting mito in array 1 is big enough, then this to_grow variable will be negative
			if(split_mito_to_grow > 0):
				curve_pointXs[split_mito_num][0], curve_pointXs[split_mito_num][3], curve_pointYs[split_mito_num][0], curve_pointYs[split_mito_num][3] = spread_endpoints(split_mito_to_grow, 
							curve_pointXs[split_mito_num][0], curve_pointXs[split_mito_num][3], curve_pointYs[split_mito_num][0], curve_pointYs[split_mito_num][3], sample)
			original_length[split_mito_num] = get_length(curve_pointXs[split_mito_num], curve_pointYs[split_mito_num], sample, frame)
			#Update the length of the split mito now that it's shorter
			new_mito_to_grow = minimum_mito_length - get_length(curve_pointXs[new_mito_num], curve_pointYs[new_mito_num], sample, frame)
			if(new_mito_to_grow > 0):
				curve_pointXs[new_mito_num][0], curve_pointXs[new_mito_num][3], curve_pointYs[new_mito_num][0], curve_pointYs[new_mito_num][3] = spread_endpoints(new_mito_to_grow, 
							curve_pointXs[new_mito_num][0], curve_pointXs[new_mito_num][3], curve_pointYs[new_mito_num][0], curve_pointYs[new_mito_num][3], sample)
			new_mito_length = get_length(curve_pointXs[new_mito_num], curve_pointYs[new_mito_num], sample, frame)#get length of newly created mito
			if(len(original_length) <= new_mito_num):#if no spot for length
				original_length.append(new_mito_length)#add spot for length
			else:
				original_length[new_mito_num] = new_mito_length#update spot with length
			if(len(widths) <= new_mito_num):#if no spot for widths
				widths.append(widths[split_mito_num])#add spot with the same width as the old mito so it doesn't change width after splitting
			else:
				widths[new_mito_num] = widths[split_mito_num]#update spot with old width
			rand_rotate = np.random.randint(rotate_speed_range_low, rotate_speed_range_high + 1)#choose random rotation for new small mito
			if(np.random.randint(0,2) == 0):
				rand_rotate = rand_rotate * -1#50% chance that rotation is backwards
			if(len(rotate_speed) <= new_mito_num):#if there's not a spot in the rotate_speed variable for this new mito to fit
				rotate_speed.append(rand_rotate)#Add element for new mito
			else:
				rotate_speed[new_mito_num] = rand_rotate#update element for new mito
			if(len(mito_wiggle) <= new_mito_num):#If no spot for new mito in wiggle_intensity
				mito_wiggle.append(np.random.randint(mito_wiggle_low, mito_wiggle_high + 1))#add element for new small mito with random wiggle intensity 
			else:
				mito_wiggle[new_mito_num] = np.random.randint(mito_wiggle_low, mito_wiggle_high + 1)#update wiggle intensity with random value
			if(len(mito_seek_speed) <= new_mito_num):#If no spot in seek speed for new small mito
				mito_seek_speed.append(np.random.randint(mito_seek_low, mito_seek_high + 1))#add element for new small mito with random seek speed
			else:
				mito_seek_speed[new_mito_num] = np.random.randint(mito_seek_low, mito_seek_high + 1)#update seek speed with random value
			if(len(mito_drift) <= new_mito_num):#if no spot in drift_speed for new small mito
				mito_drift.append(np.random.randint(mito_drift_low, mito_drift_high + 1))#add element for drift_speed with random value
			else:
				mito_drift[new_mito_num] = np.random.randint(mito_drift_low, mito_drift_high + 1)#update drift_speed with random value
			if(len(mito_seek_target) <= new_mito_num):#if there's not enough seek_targets, then add more
				mito_seek_target.append(-1)
			else:
				cancel_seek(sample, frame, new_mito_num, mito_seek_target[new_mito_num], "new mito, old seek value?")#or update seek_targets with empty value so it doesn't immediately start seeking
			if(len(mito_drift_directionX) <= new_mito_num):#If no spot in drift direction, 
				mito_drift_directionX.append(np.random.randint(-1000, 1001))#add random value
			else:
				mito_drift_directionX[new_mito_num] = np.random.randint(-1000, 1001)#update with random value
			if(len(mito_drift_directionY) <= new_mito_num):#If no spot for drift direction
				mito_drift_directionY.append(np.random.randint(-1000, 1001)) #add random value
			else: 
				drift_directionY[new_mito_num] = np.random.randint(-1000, 1001) #update with random value
			text = text + str(split_mito_num)+ " new length: " + str(original_length[split_mito_num]) + "\n"
			text = text + "Newly created mito " + str(new_mito_num)+ " length: " + str(original_length[new_mito_num]) + "\n"#Interesting to note changes in length in text file
			text = text + "old and new mito's width: " + str(widths[split_mito_num]) + "\n"
			text = text + "Widths: " + str(widths) + "\n"
			text = text + "Who's merged with who? = " + str(merged) + "\n\n"
			write_to_attributes(sample, text, frame)
	else:
		msg = msg + ""
		if(rohit_behavior == 0 and frame < 900):
			print(msg, end="")
		write_to_attributes(sample, msg, frame)
	return widths, rohit_id_map, actually_did_the_split

def set_rotation_after_merge(rohit_behavior, seeked_num, seeker_num, frame):
	global rotate_speed
	already_merged = False#If the seekee or seeker is already merged with a third party, turn all three of these guys' rotation to zero pls
	if(rotate_speed[seeked_num] > 0):
		for mergeList in merged[seeked_num]:
			if(mergeList != -1):#If the seeked is merged with any other mitochondria
				if(mergeList[0] != -1 and mergeList[0] != seeker_num and mergeList[0] != seeked_num):#make sure it's not the seeker or itself that it's merged to
					already_merged = True
					rotate_speed[mergeList[0]] = 0
		for mergeList in merged[seeker_num]:#And repeat with the seeker, cancel all rotations if he's already merged with another mitochondria
			if(mergeList != -1):
				if(mergeList[0] != -1 and mergeList[0] != seeker_num and mergeList[0] != seeked_num):
					already_merged = True
					rotate_speed[mergeList[0]] = 0
		if(already_merged == True):
			rotate_speed[seeked_num] = 0
			rotate_speed[seeker_num] = 0 # I already set the rotation speeds of the merged mitos to zero above. So I can just set these two to zero
	
	if(already_merged == False):#If I didn't just set all rotation to zero above and there's only two merged total now, then set seeker to seeked rotation
		rotate_speed[seeker_num] = rotate_speed[seeked_num]

def split_on_merge(probability_of_split_on_merge, closest_points, stationary_mitos, rohit_id_map, seeker_num, seeked_num, num_of_splits, frame, sample, widths):
	if((np.random.randint(0,101) * 0.01) < probability_of_split_on_merge):
		if(closest_points[0] != 0 and closest_points[0] != len(curve_pointXs[seeker_num])-1 and (seeker_num in stationary_mitos) == False):#If the merge point is not on the end of the seeker mito
			widths, rohit_id_map, actually_did_the_split = split(seeker_num, 
						closest_points[0], frame, sample, widths, stationary_mitos, rohit_id_map, True)#split seeker mito at the merge point
			if(actually_did_the_split == True):
				num_of_splits += 1
	if((np.random.randint(0,101) * 0.01) < probability_of_split_on_merge):
		if(closest_points[1] != 0 and closest_points[1] != len(curve_pointXs[seeked_num])-1 and (seeked_num in stationary_mitos) == False):#If the merge point is not on the end of the seeked mito
			widths, rohit_id_map, actually_did_the_split = split(seeked_num, 
						closest_points[1], frame, sample, widths, stationary_mitos, rohit_id_map, True)#split seeked mito at the merged point
			if(actually_did_the_split == True):
				num_of_splits += 1
	return num_of_splits

def seek(seeker_num, frame, sample, probability_of_merge, probability_of_split_on_merge, widths, stationary_mitos, rohit_id_map):
	#seeker will drift toward the seeked mitochondria. The direction of travel optomizes bringing the closest two curve points closer together, one point in each mitochondria
	#chance of merge or kiss and run. In a merge, there's a chance that each mitochondria might be split on the merge point if the merge point is not an end point. 
	global rotate_speed
	global mito_wiggle
	global mito_drift
	global mito_drift_directionX
	global mito_drift_directionY
	global mito_seek_target
	global mito_seek_speed
	global iqra_seeker_start_frame
	global iqra_sentence_id
	merge_happened = False #This variable is printed if rohit_behavior==1 into a csv file custom made for rohit
	num_of_splits = 0
	
	set_seed()
	seeked_num = mito_seek_target[seeker_num]#gets the id of the mitochondria which is beeing seeked from the global seek_target variable
	if((seeked_num in stationary_mitos) == False and (seeker_num in stationary_mitos) == False): #Only perform seek if neither seeker nor seeked are part of stationary_mitos
		distance_to_drift=mito_seek_speed[seeker_num]#gets the distance to travel toward the seeked mitochondria from the global seek_speed variable
		closest_points=[0, 0] # variable to contain the curve point ids of the two curve points that are closest to eachother. one on the seeker and one on the seeked
		closest_distance = -1 # the distance between these two closest points
		for seeker_point in range(len(curve_pointXs[seeker_num])):#for each curve point in seeker
			for seeked_point in range(len(curve_pointXs[seeked_num])):#for each curve point in seeked
				seeked_and_seeker_point_distance = math.hypot(curve_pointXs[seeked_num][seeked_point] - curve_pointXs[seeker_num][seeker_point] , 
							curve_pointYs[seeked_num][seeked_point] - curve_pointYs[seeker_num][seeker_point])#distance between these curve points
				if(seeked_and_seeker_point_distance <= closest_distance) or (closest_distance < 0):#if these are the first two points, or if these points are closer than all previousely checked...
					closest_points[0]=seeker_point
					closest_points[1]=seeked_point # Finding the closest two points
					closest_distance=seeked_and_seeker_point_distance#updating the known distance between the two closest points found so far
		meet=False#Did the seeker reach the seeked yet? 
		if(closest_distance<=distance_to_drift):#Are they close enough that the seeker will reach the seeked in one hop? 
			distance_to_drift=closest_distance#Don't overshoot your goal. If there's a small distance left, don't jump past it
			meet=True#Seeker will reach the seeked within this function
		seeker_drift_directionX = curve_pointXs[seeked_num][closest_points[1]] - curve_pointXs[seeker_num][closest_points[0]]#Getting the direction needed to go, not taking into account the distance desired
		seeker_drift_directionY = curve_pointYs[seeked_num][closest_points[1]] - curve_pointYs[seeker_num][closest_points[0]]
		if(distance_to_drift > 0):
			driftMito(seeker_num, seeker_drift_directionX, seeker_drift_directionY, distance_to_drift) #drifting
		
		if(meet):#reached the goal, seek behavior needs to stop after this. seek_target variable will be updated with -1 meaning no value. 
			text = "Frame " + str(frame) + "\n"
			text = text + str(seeker_num)+ " seeked " + str(seeked_num) + " on curve points " + str(closest_points[0]) + " and " + str(closest_points[1]) + "\n"
			text = text + str(seeker_num)+ " center point: " + str(center_of_mass_for_one_mito(seeker_num)) + "\n"
			text = text + str(seeked_num)+ " center point: " + str(center_of_mass_for_one_mito(seeked_num)) + "\n"#good to note that seek has finished
			write_to_attributes(sample, text, frame)
			
			mito_drift[seeked_num] = 0 #Stop the drift of the seeked immediately. I want to reduce the gliding of big groups of mitos
			
			cancel_seek(sample, frame, seeker_num, seeked_num, "seeker reached seeked")#officially stop the seek
			
			if((np.random.randint(0,101) * 0.01) > probability_of_merge):#If probability_of_merge is 0.5 then should be close to 50/50
				msg = "Kiss and Run\n"
				if(rohit_behavior == 0 and frame < 900):
					print(msg, end="")
				write_to_attributes(sample, msg + "\n", frame)
				mito_drift_directionX[seeker_num]=np.random.randint(-1000, 1001) # Choose a random direction to go for the seek and run 
				mito_drift_directionY[seeker_num]=np.random.randint(-1000, 1001)
				mito_drift[seeker_num] = np.random.randint(mito_drift_low, mito_drift_high + 1)#pick a random speed for the runner
				if(iqra_record):
					write_to_iqra(sample, '{"category": 1, "video_id": ' + str(sample) + ', "start_time": ' + str(iqra_seeker_start_frame[seeker_num]) + ', "end_time": ' + str(frame) + ', "split": "trained validation or test split", "id": ' + str(iqra_sentence_id) + '},\n{"caption": "Mitochondria ' + str(seeker_num) + ' has performed a kiss and run on mitochondria ' + str(seeked_num) + '", "video_id": ' + str(sample) + ', "sentence_id": ' + str(iqra_sentence_id) + ', "id": ' + str(iqra_sentence_id) + '},\n')
					iqra_sentence_id = iqra_sentence_id + 1
			else:#Merge the mitos after one seeked the other successfully
				merge_happened = True
				SeekerMergedNewInfo=[seeked_num, closest_points[0], closest_points[1]]#These are the arrays that will be saved in the merged array
				SeekedMergedNewInfo=[seeker_num, closest_points[1], closest_points[0]]
				#The first element is the id of the mitochondria which this mito is merged with, and the next two elements are the curve points that are connected,
				#Beginning with the curve point of the mitochondria in question
				if(merged[seeker_num][0] == -1):#If this mito isn't currently merged with any other mito, then set the whole thing to this
					merged[seeker_num][0]=SeekerMergedNewInfo 
				else:
					merged[seeker_num].append(SeekerMergedNewInfo)#if this mito is already merged, then add this new one to the list of merged ones. 
				if(merged[seeked_num][0] == -1):#Do the same with the other one that it's merging to. add this one to that one's merge list
					merged[seeked_num][0]=SeekedMergedNewInfo
				else:
					merged[seeked_num].append(SeekedMergedNewInfo)
				
				set_rotation_after_merge(rohit_behavior, seeked_num, seeker_num, frame)#sets rotation to zero if seeked is already merged with another mito. If not, sets seeker to seeked rotation.
				
				mito_wiggle[seeker_num] = mito_wiggle[seeked_num]
				mito_drift_directionX[seeker_num] = mito_drift_directionX[seeked_num]# Give the seeker the same drift,wiggle intensity as the seeked
				mito_drift_directionY[seeker_num] = mito_drift_directionY[seeked_num]
				mito_drift[seeker_num] = mito_drift[seeked_num]
				
				split_on_merge(probability_of_split_on_merge, closest_points, stationary_mitos, rohit_id_map, seeker_num, seeked_num, num_of_splits, frame, sample, widths)#sometimes splits on merge point
				
				write_to_attributes(sample, "Merged\n" + str(merged) + "\n\n", frame)
				if(rohit_behavior == 0 and frame < 900):
					print("Merged")
				if(iqra_record):
					write_to_iqra(sample, '{"category": 1, "video_id": ' + str(sample) + ', "start_time": ' + str(iqra_seeker_start_frame[seeker_num]) + ', "end_time": ' + str(frame) + ', "split": "trained validation or test split", "id": ' + str(iqra_sentence_id) + '},\n{"caption": "Mitochondria ' + str(seeker_num) + ' has merged with mitochondria ' + str(seeked_num) + '", "video_id": ' + str(sample) + ', "sentence_id": ' + str(iqra_sentence_id) + ', "id": ' + str(iqra_sentence_id) + '},\n')
					iqra_sentence_id = iqra_sentence_id + 1
	return widths, rohit_id_map, merge_happened, num_of_splits #the list containing widths of all mitochondria must be returned because the main function command center must be updated about any changes to this variable and it could be changed within this function if a merge triggers a split and there's a new mitochondria born which needs it's width to be kept track of. 

def dissolve_func(frame, sample, widths, stationary_mitos, rohit_id_map):#This makes the mitochondria come together in an unorganized way and form something of a network, and then later on between frame 170-330 it makes the mitochondria start breaking apart randomly
	actually_did_the_split = False
	if(frame == 170):
		widths, rohit_id_map, actually_did_the_split = split(0, 0, frame, sample, widths, stationary_mitos, rohit_id_map)#these are arbitrarily chosen mitochondria ids to split on arbitrarily chosen curve points 
	if(frame == 180):
		widths, rohit_id_map, actually_did_the_split = split(1, 1, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 190):
		widths, rohit_id_map, actually_did_the_split = split(2, 2, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 200):
		widths, rohit_id_map, actually_did_the_split = split(3, 3, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 210):
		widths, rohit_id_map, actually_did_the_split = split(4, 0, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 220):
		widths, rohit_id_map, actually_did_the_split = split(5, 1, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 230):
		widths, rohit_id_map, actually_did_the_split = split(6, 3, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 240):
		widths, rohit_id_map, actually_did_the_split = split(7, 2, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 250):
		widths, rohit_id_map, actually_did_the_split = split(8, 1, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 260):
		widths, rohit_id_map, actually_did_the_split = split(0, 1, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 270):
		widths, rohit_id_map, actually_did_the_split = split(1, 2, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 280):
		widths, rohit_id_map, actually_did_the_split = split(2, 3, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 290):
		widths, rohit_id_map, actually_did_the_split = split(3, 0, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 300):
		widths, rohit_id_map, actually_did_the_split = split(4, 1, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 310):
		widths, rohit_id_map, actually_did_the_split = split(5, 2, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 320):
		widths, rohit_id_map, actually_did_the_split = split(6, 3, frame, sample, widths, stationary_mitos, rohit_id_map)
	if(frame == 330):
		widths, rohit_id_map, actually_did_the_split = split(7, 0, frame, sample, widths, stationary_mitos, rohit_id_map)
	return widths, rohit_id_map, actually_did_the_split

def network_func(frame, sample):
	global mito_seek_target
	if(frame == 30 and len(mito_seek_target) > 1):
		initiate_seek(sample, frame, 1, 0)
	if(frame == 60 and len(mito_seek_target) > 2):
		initiate_seek(sample, frame, 2, 1)
	if(frame == 90 and len(mito_seek_target) > 3):
		initiate_seek(sample, frame, 3, 2)
	if(frame == 120 and len(mito_seek_target) > 4):
		initiate_seek(sample, frame, 4, 3)
	if(frame == 135 and len(mito_seek_target) > 5):
		initiate_seek(sample, frame, 5, 4)
	if(frame == 150 and len(mito_seek_target) > 6):
		initiate_seek(sample, frame, 6, 5)
	if(frame == 165 and len(mito_seek_target) > 7):
		initiate_seek(sample, frame, 7, 6)
	if(frame == 180 and len(mito_seek_target) > 8):
		initiate_seek(sample, frame, 8, 7)

def rohit_behavior_func(frame, sample, widths, stationary_mitos, rohit_id_map, probability_of_split_per_frame, probability_of_seek_per_frame):
	global rohit_behavior_counter
	seeks_amount_this_frame = 0
	splits_amount_this_frame = 0
	active_mitos = num_of_mitos - len(stationary_mitos)
	probability_of_split_per_mito = probability_of_split_per_frame / active_mitos
	probability_of_seek_per_mito = probability_of_seek_per_frame / active_mitos
	for this_mito in range(num_of_mitos):
		if((np.random.randint(0,100000001) * 0.00000001) < probability_of_split_per_mito and splits_amount_this_frame < 5):#don't let more than 5 splits happen in one frame
			curve_point = np.random.randint(0,4)
			widths, rohit_id_map, actually_did_the_split = split(this_mito, curve_point, frame, sample, widths, stationary_mitos, rohit_id_map)
			if(actually_did_the_split == True):
				splits_amount_this_frame += 1
				if(rohit_behavior_limit > -1):
					rohit_behavior_counter = rohit_behavior_counter + 1
			#print to csv
		if((np.random.randint(0,100000001) * 0.00000001) < probability_of_seek_per_mito and seeks_amount_this_frame < 5):#don't let more than 5 seeks happen in one frame
			seeks_amount_this_frame += 1
			notMe_and_notAlreadySeeking = False
			tries = False
			rand_seeked = this_mito #just declare this var before I set it to what I want later in the while loop
			while(notMe_and_notAlreadySeeking == False and tries < 10):
				rand_seeked = np.random.randint(0,num_of_mitos)#Choose a random other mito to start seeking. As long as it's not seeking itself...
				if(rand_seeked != this_mito and mito_seek_target[rand_seeked] == -1):#And as long as the seeked is not a seeker of some other mito... I don't want that because a seeker seeking a seeker might just run off the screen. I saw it once. I have no idea what the original runner was seeking off the screen ... creepy. 
					notMe_and_notAlreadySeeking = True
				tries = tries + 1
			if(notMe_and_notAlreadySeeking == True):
				initiate_seek(sample, frame, this_mito, rand_seeked)# START SEEKING!!
				if(rohit_behavior_limit > -1):
					rohit_behavior_counter = rohit_behavior_counter + 1
	return widths, rohit_id_map, splits_amount_this_frame

def execute_behaviors(widths, frame, sample, probability_of_merge, probability_of_split_on_merge, stationary_mitos, rohit_id_map):
	merges_amount_this_frame = 0
	num_of_splits_on_merge = 0
	for mitoNumLoop in range(len(curve_pointXs)*2):
		#go through all the mitochondria and perform their manuevers 
		#But I need to go through the list of mitos twice. the first time around, I only do the ones who are currently seeking. Then on the second round I do the rest.
		#I made it work with the seekers first in response to a bug that was making the seeker never able to reach the destination 
		mitoNum=mitoNumLoop
		if(mitoNumLoop >= len(curve_pointXs)):
			mitoNum=mitoNumLoop-len(curve_pointXs)#mitoNumLoop goes to double the mito num. So when it get's big, gotta cut it in half
		if((mito_seek_target[mitoNum]!=-1 and mitoNumLoop < len(curve_pointXs)) or (mito_seek_target[mitoNum]==-1 and mitoNumLoop >= len(curve_pointXs))):
			#when I'm running through the mitochondrias the first time, and mitoNumLoop is less than the total mito num, then only calculate those mitos who are seeking
			#After mitoNumLoop gets bigger than the total number of mitochondria, then it's time to only do those mitos who are not currently seeking
			if((mitoNum in stationary_mitos) == False):#If this mito is stationary, don't do anything except wiggle
				seeking_soDontDrift=False
				if(mito_seek_speed[mitoNum] != 0 and mito_seek_target[mitoNum] != -1 and mito_seek_target[mitoNum] < len(curve_pointXs)):
					#Only do the seek if there's a seek_target which is not -1
					seeking_soDontDrift=True																				#SEEK
					widths, rohit_id_map, merge_happened, num_of_splits = seek(mitoNum, frame, sample, 
									probability_of_merge, probability_of_split_on_merge, widths, stationary_mitos, rohit_id_map) 
					num_of_splits_on_merge += num_of_splits
					if(merge_happened == True):
						merges_amount_this_frame += 1
				iamsmallest = True #If this mito is the smallest in the list of those which it's merged with, then it will do the rotating and wiggling, if not, nothing happens
				
				for mergeList in merged[mitoNum]:#mergeList is a list of mitos merged with me
					if(mergeList != -1):#If there's at least one merged with me
						if(mergeList[0] < mitoNum and mergeList[0] != -1):
							iamsmallest = False#I am the smallest mito number of the mito numbers who I'm merged with
				if(iamsmallest):#The rotate function takes a list of mitochondria and rotates them all together around their shared center of mass
					if(rotate_speed[mitoNum]!=0):
						toRotate = []
						toRotate.append(mitoNum)#this mito will always be sent to rotate and wiggle
						for mergeList in merged[mitoNum]:#gather up all the mitos that are merged with this mito, these will also be sent to rotateMito() and wiggleMito()
							if(mergeList != -1):
								if(mergeList[0] != -1):
									toRotate.append(mergeList[0])
						rotateMito(toRotate)																				#ROTATE
				if(mito_drift[mitoNum]!=0 and mito_seek_target[mitoNum] == -1 and seeking_soDontDrift == False):
					driftMito(mitoNum, mito_drift_directionX[mitoNum], mito_drift_directionY[mitoNum], mito_drift[mitoNum])			#DRIFT
			if(mito_wiggle[mitoNum] != 0):
				wiggleMito(sample, mitoNum, elasticity, frame)																		#WIGGLE
	return widths, rohit_id_map, merges_amount_this_frame, num_of_splits_on_merge

def command_center(sample, program_intro, centermult, surfDivisor, probability_of_merge, probability_of_split_on_merge, elasticity, m_params, 
			number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r, density_vesicle, max_mito_length, rohit_behavior, 
			pixels, nm_per_pixel, stationary_mitos, rohitfissionpercentage, rohitseekpercentage, plot_boolean):
	#This function contains the loop which creates one frame for each iteration of the loop. 
	#It calls the functions to determine what kind of behavior the mitochondria are partaking in on each frame.
	#And it calls the functions to generate the mitochondria and render them into images
	global curve_pointXs#Arrays that define the various mitochondria curve points, so the location and shape of the mitochondria are preserved from one frame to another. The first element in the curve_pointXs array is an array which contains all of the X values of the 2 dimensional curve points which define the position of the first mitochondria. The length of this array is the same as the number of mitochondria. 
	global curve_pointYs
	global original_length # This is a variable that defines the length of each mitochondria when they were first randomly generated. When mitochondrias wiggle their length changes. If the new length after the wiggle diverges from the value in original_length more than the set elasticity percentage difference from the original_length allowed then the new wiggled position is abandoned and a new one is calculated. 
	global rotate_speed# This is an array which contains a value for each mitochondria. The value is the angle out of 360 total degrees that each mitochondria will spin between each frame. rotate_speed_range is an array containing the low value and the high value for possible rotation speeds. When the program selects a random rotation speed for a mitochondria it will select a value between these two numbers. 
	global mito_wiggle# This is an array which contains a value for each mitochondria. The value is the number of coordinate points that each point in the curve points of the mitochondria will move between frames. Each curve point moves in a random direction independent of the directions of other curve points in the mitochondria. wiggle_intensity_range is an array containing the low and high values for possible wiggle intensities. These are used when generating a random wiggle intensity for a mitochondria. 
	global mito_drift# This is an array which contains a value for each mitochondria. The value is the number of coordinate points that each point in the curve points of the mitochondria will move between frames. Each curve point in one mitochondria will move in the same direction. At the start of the program, a random drift direction is chosen for each mitochondria and that stays the same between frame generations. drift_speed_range is an array containing the high and low values for possible random drift speeds when being chosen for a new mitochondria.
	global mito_seek_speed# This is an array which contains a value for each mitochondria. The value is the number of coordinate points that each point in the curve points of the mitochondria will move between frames. Each curve point in one mitochondria will move in the same direction. The direction the mitochondria will move in is decided by finding one point in the mitochondria which is the shortest distance to one other point in a different mitochondria and calculating the direction the mitochondria would need to go to shorten that distance. seek_speed_range is an array containing the high and low values for possible random seek speeds when they need to be generated
	global mito_seek_target# This is an array which contains a value for each mitochondria. The value is a number which represents a different mitochondria which this mitochondria is meant to be seeking. 
	global merged # This is an array which contains an array for each mitochondria. Each array contains an array of arrays which each represent a different mitochondria which this mitochondria is merged with. Each array contains three numbers. The first number is the id representing the other mitochondria. The second number is the id of the curve point on this mitochondria which is the merge point. The third number is the id of the curve point / merge point on the other mitochondria. 
	global mito_drift_directionX # This is a variable to keep track of the drift direction of each mitochondria. It does not take speed into account, so these values may be large or small, the only thing that matters is that the x and y gives the correct angle of the direction that the mitochondria should be drifting in. As usual these are arrays with each item corresponding to each mitochondria in the same order as they are defined in the curve_pointXs array. 
	global mito_drift_directionY
	global num_of_mitos
	global original_length
	
	frame = 1
	write_to_attributes(sample, program_intro, frame)# this prints the values from the command line arguments
	
	set_seed()#This sets the numpy random number generator seed. I incorporate the time on the clock to make sure two identical seeds won't be generated and I incorporate the process id number to be sure two different processes will never generate the same seed
	emmiters_percentage = 1
	widths = []# A list of the width of each mitochondria
	num_of_curve_pts = 4 #the x/y shape / curve of the mitochondria is defined by just 4 points. A smooth arc is drawn to connect these points.
	mito_drift_directionX, mito_drift_directionY, rotate_speed, mito_wiggle, mito_drift, mito_seek_speed, merged, mito_seek_target = [], [], [], [], [], [], [], []
	frameStartTime = int( time.time() )
	curve_pointXs, curve_pointYs, vesi_radius, vesi_x_center, vesi_y_center, vesi_z_center = None, None, None, None, None, None#These will be generated on the first frame
	original_num_of_mitos = num_of_mitos
	number_of_vesicles=np.random.randint(number_of_vesicles_min, number_of_vesicles_max + 1)
	
	for mitoNum in range(original_num_of_mitos):
		if(mito_wiggle_low == mito_wiggle_high):
			mito_wiggle.append(mito_wiggle_low)
		else:
			mito_wiggle.append(np.random.randint(mito_wiggle_low, mito_wiggle_high + 1))	#  Set random values for mitochondria speeds and directions within limits
		if(mito_seek_low == mito_seek_high):
			mito_seek_speed.append(mito_seek_low)
		else:
			mito_seek_speed.append(np.random.randint(mito_seek_low, mito_seek_high + 1))
		if(mito_drift_low == mito_drift_high):
			mito_drift.append(mito_drift_low)
		else:
			mito_drift.append(np.random.randint(mito_drift_low, mito_drift_high + 1))
		if(int(mitowlow) == int(mitowhigh)):
			widths.append(int(mitowlow))
		else:
			widths.append(np.random.randint(int(mitowlow), int(mitowhigh) + 1))
		if(rotate_speed_range_low == rotate_speed_range_high):
			rand_rotate = rotate_speed_range_low
		else:
			rand_rotate = np.random.randint(rotate_speed_range_low, rotate_speed_range_high + 1)
		if(np.random.randint(0,2) == 0):
			rand_rotate = rand_rotate * -1 # 50% chance that rotation is backwards
		rotate_speed.append(rand_rotate)
		mito_drift_directionX.append(np.random.randint(-1000, 1001)) # these X and Y values give us a direction of travel. to get the direction I use the angle of the hypotenus of the triangle that these two lines create
		mito_drift_directionY.append(np.random.randint(-1000, 1001)) # Get random drift direction
		mito_seek_target.append(-1)
		merged.append([-1])
		iqra_seeker_start_frame.append(-1)
	
	sample_intro = "number_of_vesicles: " + str(number_of_vesicles) + "\n"
	sample_intro = sample_intro + "mito_wiggle_intensity: " + str(mito_wiggle) + "\n"
	sample_intro = sample_intro + "mito_seek_speed: " + str(mito_seek_speed) + "\n"
	sample_intro = sample_intro + "mito_drift_speed: " + str(mito_drift) + "\n"
	sample_intro = sample_intro + "mito_widths: " + str(widths) + "\n"
	sample_intro = sample_intro + "rotate_speed: " + str(rotate_speed) + "\n"
	sample_intro = sample_intro + "SAMPLE " + str(sample) + "\n\n"
	write_to_attributes(sample, sample_intro, frame)
	print(sample_intro)
	
	#up/down Y is flipped minus is up
	
	#ToDo:
	
	#make it so you enter the psf when generating and then it prints that everywhere so you know what psf to make it at. 
	
	#you should make it so when rohit_behavior is enabled, it still prints out the time it takes to print a frame if we're generating images at all. 
	
	#Needs to be a centralized location for checks on a seek trigger. maybe initiate behavior can have a return success or not and the initiate func can do all checks. 
	
	#If mito 1 is merged with mito 2 and mito 1 seeks mito 2 and performs a merge, then there will be two entries in the merged list. Now that's not good.
	#Big problem is the connect merge points error happening when this happens ^ when there's only mito 1 merged with mito 2, but there's two records of it in the merged list
	#Connect merge points error actually causes a mitochondria to dissapear completely in the 4_9_glitch_bug_run_away in serverPics. I think it teleports off screen upward and that's where the mitos go when they sprint off screnn! 
	
	#make it so merges which are the cause of a split on merge are not terminated on the split behavior. 
	
	#Suyog suggests to Add half circle to end of mito. I think I took notes somewhere about his equation? like 2 lines or smth? 
	
	#Krishna says "10 milisecond frame rate" this what you've been doing is perfect! keep all settings the same. From the videos in the first project update with mostly the GT images. 
	
	#5 motion paterns in cvpr paper
	#abinanda has simulated 5 types of vesicle  motion
	#look at simulated matlab code for 5 types of motion in nanomotion github.
	
	#should we simulate photokinetics? Aby is wondering. This is the flickering ... but Aby had other ideas about what that would look like in my videos I think. 
	
	#golay equation smooth out merged mitochondria. This is one option (a bad one) to fix kinkiness in end to end joined mitos. need to merge into one long smooth mito!! I think there's an easier way, during wiggle only go in a direction closer to smoothness!! 
	
	#add noise
	
	#bug where mitos run off screen. could be caused by group of mitos where one seeks another in the same merged group. 
	
	#Need to make Z random
	#initial position
	
	#get data from ida to simulate nano tubes "highway" for mitochondria
	
	#Helper_generator.py get_mito_center_line_points() line 321 "I couldn't use the get_length function for this"      just do it anyway. import and export whatever vars you'll need you dumzo
	
	#Make a mode where they don't move, just do one of the connect merge points per frame? But somehow start merged? You can start merged actually!! Just edit the merged list :D :D :D
	
	
	#cannot seek yourself
	#cannot seek mito who you're merged with already
	#cannot seek anyone if you're merged with a mitochondria who is merged to another mitochondria?  
	
	#re-add plot code to grab nice screenshots of mitochondria points plotted for thesis. 
	
	#checking for wiggle cannot be done without changing length too much error. 
	#checking for seeking nothing error where runs off screen sometimes sprinting offscreen!
	
	#change color of mitos vs vesicles, and make a button to change mito color and btn for vesi color
	
	#make vesicles:
	#kiss and run example with vesical triggering fusion etc
	#try to get one example of vesical kiss and run triggering something on mitochondria. 
	#I should use that one flip example to make a flip behavior
	
	#1) kiss and run (KAR) with delay before run
	#2) vesical merges to mito or goes close to it for some time, spends some time rolling along it
	#3) vesical KAR triggers split
	#4) mito consumed by vesicle?
	#5) vesical merged with vesical
	#6) vesical that consumes mito needs to be 200 nm - 2 micrometers    2 microns
	#7) the ones that bite off and engulf should be 500 nm
	# num of vesicles should be added to GUI
	
	#should try to remove the bit about doing all the behaviors of the seekers before the other mitochondria. maybe that's not necessary. certainly makes it more complex code
	#add noise functionallity      Don't ask now. next meeting. Krishna should see if the noise is good for us. maybe I show her the old images from the paper.
	#right now i'm pretty sure a seek will fail if the seeker is merged to anyone due to the drift_to_connect_merged_points. I need to involve all merged in the seek. all should drift with the seeker. 
	#look in data segmentation and training directory in old code and see if they do stuff with labeling there? 
	#make sure on seek it drags everyone with and doesn't rely on connect function to do that dragging!
	
	#only if you have time, not primary work. secondary. 
	#start looking at segmentation models 
	#check what segmentation models are there in AI 
	#start with normal segmentation for example classifying a bycicle vs person in an image
	#end goal segmentation of biological organells 
	#EG "unit" is a model but there are more.
	#easier to train many kinds of things like motion fusion 
	#two kinds of segmentation chip and something 
	
	#Optomizations:
	#make them not drift automatically. instead, make a sprint behavior that happens when I need them seperated like after a split. sprint should have a set distance and then they stop. And default behavior will be sitting fixed in cement wiggling. 
	#make it so mitos which don't contain any on-canvas points are not rendered
	#make it so you can determine if a mito is attached to another mito in a chain network. If they're attached, and one is seeking, then don't make the other start seeking
	
	#check for and print about halt state / starvation:    from himanshu: It looks like 10 of your processes were executing but may be in halt state. Most probably it is the case of starvation where the process were not allowed to execute.
	
	rohit_id_map = [] # the first element in this list coorisponds to mito id 1. The value there points to the id in rohit's id numbering schema
	for mito_num in range(num_of_mitos):
		rohit_id_map.append(mito_num)#Populate it with 0 -> num_of_mitos
	probability_of_split_per_frame = rohitfissionpercentage*0.01 # 35% chance on each frame that one mito will split
	probability_of_seek_per_frame = rohitseekpercentage*0.01
	
	recordcoordinates_mito_storage = []
	rohit_mito_storage = []
	
	#curve_pointXs = [[1636, 2878, 2963, 4764], [-1071, -859, -232, -114], [-477, 279, 1097, 1372]]
	#curve_pointYs = [[3373, 1894, 1362, 2031], [2385, 3669, 3096, 1642], [-1536, -2927, -1276, -1032]]
	
	if(plot_to_show_connect_merged):
		#merged[0], merged[1], merged[2], merged[3], merged[4] = [[1, 1, 0]], [[0, 0, 1]], [[3, 0, 1]], [[2, 1, 0], [4, 2, 0]], [[3, 0, 2]]
		#curve_pointXs = [[-3671, -2731, -2096, -1144], [-2731, -1486, -1405, 370], [-875, -299, 1887, 2572], [-1995, -875, -234, 1403], [-234, -500, 362, 901]]#these are two together and three together
		#curve_pointYs = [[-3987, -2727, -2090, -1004], [-2727, -3608, -2534, -1701], [1784, 874, 510, 585], [1871, 1784, 2410, 2521], [2410, 3006, 4824, 4351]]
		
		#right now I've successfully built a loop that never successfully connects the merge points
		#One of the mitos starts far away, and we need to just move it's points so it's closer and connected to the others! 
		#the mitochondria in question is id number 0. move it so it's connected to mitochondria number 1. thanks
		
		merged[0], merged[1], merged[2], merged[3] = [[1, 3, 1]],          [[3, 0, 1], [0, 1, 3]],          [[3, 0, 2]],           [[1, 1, 0], [2, 2, 0]]
		curve_pointXs = [[-2826, -1886, -1251, -299], [-875, -299, 1887, 2572], [-234, -500, 362, 901], [-1995, -875, -234, 1403]]#these are two together and three together
		curve_pointYs = [[-2109, -849, -212, 874], [1784, 874, 510, 585], [2410, 3006, 4824, 4351], [1871, 1784, 2410, 2521]]
	
	while frame < (frames+1):#for each frame
		try:#This is to prevent the program from terminating when there's an exception. This is good when there's multiple samples so they don't all terminate because of an edge case in one of them. 
			timeOnPSF=0#keeps track of the amount of time spent specifically on the PSF function inside microscPSFmod.py
			frameStartTime = int( time.time() )#record the amount of time it takes to make this frame
			splits_amount_this_frame = 0#these two variables are printed if rohit_behavior == 1 in a csv file custom made for rohit.
			merges_amount_this_frame = 0
			
			if(frame > 1):
				if(network == 1):#these two functions contains a series of checks on which frame number I'm at. At certain frame numbers, different actions are taken. 
					network_func(frame, sample)
				if(dissolve == 1):
					widths, rohit_id_map, actually_did_the_split = dissolve_func(frame, sample, widths, stationary_mitos, rohit_id_map)
					if(actually_did_the_split == True):
						splits_amount_this_frame += 1
				if(rohit_behavior == 1 and (rohit_behavior_limit == -1 or rohit_behavior_counter < rohit_behavior_limit)):
					widths, rohit_id_map, splits_amount_this_frame = rohit_behavior_func(frame, 
								sample, widths, stationary_mitos, rohit_id_map, probability_of_split_per_frame, probability_of_seek_per_frame)
					if(frame % 500 == 0):
						local_time = time.localtime()
						timeOfDay = str(local_time.tm_hour) + ":" + str(local_time.tm_min)
						msg = "Frame " + str(frame) + " TimeStamp: " + str(timeOfDay) + "\n"
						print(str(sample) + " " + msg, end="")
						write_to_attributes(sample, msg, frame)
				
				if(plot_to_show_connect_merged == False or (frame == 3) or (frame == 8) or (frame == 13) or (frame == 18) or (frame == 23)):
					#loops through mitochondria calling each behavior function. EG wiggle, seek, drift, rotate...
					widths, rohit_id_map, merges_amount_this_frame, num_of_splits_on_merge = execute_behaviors(widths, 
									frame, sample, probability_of_merge, probability_of_split_on_merge, stationary_mitos, rohit_id_map)
					splits_amount_this_frame += num_of_splits_on_merge
				
				for target in range(len(mito_seek_target)):
					if(mito_seek_target[target] in stationary_mitos):#If I've set a mitochondria to be seeked who is in the stationary list, 
						cancel_seek(sample, frame, target, mito_seek_target[target], "stationary cannot be seeked")#then terminate that seek behavior
					if(mito_seek_target[target] != -1):
						if(mito_seek_target[mito_seek_target[target]] != -1): #If there exists a mitochondria seeking a mitochondria who is seeking another mitochondria,
							cancel_seek(sample, frame, target, mito_seek_target[target], "cannot seek a seeker")#then terminate that seek behavior
				
				if(plot_to_show_connect_merged==False or ((frame != 3) and (frame != 8) and (frame != 13) and (frame != 18) and (frame != 23))):
					if_plotting_which_connection_to_show = frame - 4
					if(frame>8):
						if_plotting_which_connection_to_show = frame - 9
					if(frame>13):
						if_plotting_which_connection_to_show = frame - 14
					if(frame>18):
						if_plotting_which_connection_to_show = frame - 19
					if(frame>23):
						if_plotting_which_connection_to_show = frame - 24
					drift_to_connect_merged_points(sample, frame, plot_to_show_connect_merged, if_plotting_which_connection_to_show)#Connect all the merged mitos back together
				num_of_mitos = len(curve_pointXs)#update the number of mitochondria because it might have changed after a split
			
			#			GENERATE MITOS
			mitosPts = []
			if(num_of_mitos > 0):
				mitosPts, curve_pointXs, curve_pointYs, current_len, recordcoordinates_mito_storage, rohit_mito_storage = generate_save_mito(sample, 
							frame, num_of_mitos, centermult, surfDivisor, max_xy, psftogether, 
							mitozlow, mitozhigh, max_mito_length, density_mitochondria, emmiters_percentage, widths, num_of_curve_pts, 
							recordcoordinates, printGt, printTif, curve_pointXs, curve_pointYs, rohit_behavior, rohit_id_map, 
							recordcoordinates_mito_storage, rohit_mito_storage, frames, splits_amount_this_frame, merges_amount_this_frame, plot_boolean)
			
			#			GENERATE VESICLES
			vesiclesPts= []
			if(number_of_vesicles > 0):
				vesiclesPts, vesi_radius, vesi_x_center, vesi_y_center, vesi_z_center = generate_save_vesicles(sample, frame, max_xy, psftogether, mitozlow, mitozhigh, 
								density_vesicle, emmiters_percentage, recordcoordinates, printGt, printTif, number_of_vesicles, vesi_min_r, vesi_max_r, 
						vesi_radius, vesi_x_center, vesi_y_center, vesi_z_center)
			
			if(frame == 1):#Print details about each mitochondria on the first frame to command line and also to txt file
				original_length = current_len#current_len was returned by generate_save_mito() 
				initial_location_and_size = "Frame 1\n"
				for mitoNum in range(original_num_of_mitos):
					initial_location_and_size += "Mito " + str(mitoNum) + " center point: " + str(center_of_mass_for_one_mito(mitoNum)) + "\n"
					initial_location_and_size += "Mito " + str(mitoNum) + " length: " + str(original_length[mitoNum]) + "\n"
				for vesiNum in range(number_of_vesicles):
					initial_location_and_size += "Vesicle " + str(vesiNum) + " center point: "
					initial_location_and_size += str(vesi_x_center[vesiNum]) + ", " + str(vesi_y_center[vesiNum]) + ", " + str(vesi_z_center[vesiNum]) + "\n"
					initial_location_and_size += "Vesicle " + str(vesiNum) + " radius: " + str(vesi_radius[vesiNum]) + "\n"
				initial_location_and_size += "\n"
				write_to_attributes(sample, initial_location_and_size, frame)#attributes.txt
				if(sample == 0):#to give some sign of life in a slow run
					print("Now finished defining the mitochondria's locations for Sample 1. Rendering them now ... ")
			
			if(printGt == 1 or printTif == 1):
				#combine vesicles and mitochondria into one list of entities
				entities = mitosPts
				del mitosPts#Large variable taking lots of RAM. Contains coordinates of every single emmiter 
				for eachVesi in vesiclesPts:
					entities.append(eachVesi)
				del vesiclesPts
				
							#RENDER
				timeOnPSF = render(entities, sample, frame, max_xy, psftogether, printTif, 
								printGt, parallelPSF, writedisk, m_params, num_basis, rho_samples, pixels, nm_per_pixel)
				del entities
			
			if(printTif == 1 and rohit_behavior == 0 and frame < 900): #If we're printing tifs then we have time to print these deets
				frameStatement="Frame " + str(frame)
				override = False # If this is true, then print_to_attributes will still print to attributes.txt some text even if frame number > 900
				if(frame == frames):
					frameStatement="Last Frame"
					override = True
				totalSeconds = int(int( time.time() )-frameStartTime)#Printing more details to cmd line and attributes.txt
				percentage_statement = ""
				if(totalSeconds != 0 and printTif == 1):
					percentage_statement = ", which is " + str(round((timeOnPSF / totalSeconds * 100), 2)) + "%"
				local_time = time.localtime()
				timeOfDay = str(local_time.tm_hour) + ":" + str(local_time.tm_min)
				msg = str(totalSeconds) + " seconds total. " + str(timeOnPSF) + " sec for PSF" + str(percentage_statement) + ". timestamp: " + timeOfDay + ". " + frameStatement + "\n"
				print(msg, end="")
				write_to_attributes(sample, msg, frame, override)
			elif(frame == frames):
				msgaaron = "Columns in Rohit CSV: " + str(len(rohit_id_map))
				print(msgaaron)
				write_to_attributes(sample, msgaaron, frame, True)
			
			frame=frame+1
		except KeyboardInterrupt: # handle KeyboardInterrupt exception
			local_time = time.localtime()
			timeOfDay = str(local_time.tm_hour) + ":" + str(local_time.tm_min)
			msg = "KeyboardInterrupt received, exiting. :D   timestamp: " + timeOfDay #Exception Last Line
			print(str(sample) + " " + msg)
			write_to_attributes(sample, msg, frame, True)
			sys.exit(0)
		except Exception as e:
			local_time = time.localtime()
			timeOfDay = str(local_time.tm_hour) + ":" + str(local_time.tm_min)
			frame=frame+1
			exc_type, exc_obj, tb = sys.exc_info()
			
			f = tb.tb_frame
			lineno = tb.tb_lineno
			filename = f.f_code.co_filename#complex code for grabbing the exception details for debugging
			linecache.checkcache(filename)
			line = linecache.getline(filename, lineno, f.f_globals)
			msg = "timestamp: " + timeOfDay + "\n"
			msg = msg + '\n First Line: EXCEPTION IN ({}, LINE {} "{}")'.format(filename, lineno, line.strip()) + "\n"#exception first line
			info, error = format_exception(exc_type, exc_obj, tb)[-2:]
			msg = msg + "Last line: " + str((f'Exception in:\n{info}{error}'))#Exception Last Line
			print(str(sample) + " " + msg)
			write_to_attributes(sample, error, frame, True)
	
	for mito in range(num_of_mitos):
		if(mito_seek_target[mito] != -1):
			cancel_seek(sample, frame, mito, mito_seek_target[mito], reason="End of video")
	
	if(writedisk == 1):
		h5py_file = 'output/sample_'+str(sample) + '/h5py_file.hdf5'
		if os.path.exists(h5py_file):#If I'm writting PSF vars to disk, clean them up after done running
			os.remove(h5py_file)
	
	local_time = time.localtime()
	today_date = time.strftime("%Y-%m-%d", time.localtime())
	timeOfDay = str(local_time.tm_hour) + ":" + str(local_time.tm_min)
	finishing_statement = "today's date: " + str(today_date) + "\nTimestamp: " + timeOfDay + "\n"
	write_to_attributes(sample, finishing_statement, frame, True)
	print(str(sample) + " " + finishing_statement)

def loop_through_samples(centermult, surfDivisor, program_intro, probability_of_merge, probability_of_split_on_merge, elasticity, m_params,
			number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r, density_vesicle, max_mito_length, rohit_behavior, 
			pixels, nm_per_pixel, stationary_mitos, rohitfissionpercentage, rohitseekpercentage, plot_boolean):
	if(parallelsamples):#run samples in parallel
		argsForParallel = []#To send variables to control_center(), I make a list of the variables I want to send in each process, then I put each list into a list and send it to pool.starmap
		for sample in range(samples):
			prepare_filesystem((sample + 1), rohit_behavior)
			argsForParallel.append([(sample + 1), program_intro, centermult, surfDivisor, probability_of_merge, probability_of_split_on_merge, elasticity, m_params, 
						number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r, density_vesicle, 
						max_mito_length, rohit_behavior, pixels, nm_per_pixel, stationary_mitos, rohitfissionpercentage, rohitseekpercentage, plot_boolean])
		with Pool(samples) as pool:
			pool.starmap(command_center, argsForParallel)  #PARALLEL SAMPLES
	else:#run samples sequentially
		for sample in range(samples):
			prepare_filesystem((sample + 1), rohit_behavior)
			command_center((sample + 1), program_intro, centermult, surfDivisor, probability_of_merge, probability_of_split_on_merge, elasticity, m_params, 
						number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r, density_vesicle, 
						max_mito_length, rohit_behavior, pixels, nm_per_pixel, stationary_mitos, rohitfissionpercentage, rohitseekpercentage, plot_boolean)

if __name__ == "__main__":
	global samples
	global frames
	global num_of_mitos
	global centermult
	global surfDivisor
	global max_xy
	global psftogether
	global printGt
	global printTif
	global parallelPSF
	global parallelsamples
	global writedisk
	global mitowlow
	global mitowhigh
	global density_mitochondria
	global mitozlow
	global mitozhigh
	global recordcoordinates
	global terminal
	global elasticity
	global rotate_speed_range_high
	global rotate_speed_range_low
	global mito_wiggle_high
	global mito_wiggle_low
	global mito_drift_high
	global mito_drift_low
	global mito_seek_high
	global mito_seek_low
	global probability_of_merge
	global probability_of_split_on_merge
	global network
	global dissolve
	global iqra_record
	global iqra_seeker_start_frame
	global iqra_sentence_id
	global rohit_behavior_limit
	global rohit_behavior_counter
	
	iqra_record = False #this is set to zero down below where the gui function is called
	iqra_seeker_start_frame = []
	iqra_sentence_id = 0
	rohit_behavior_limit = -1
	rohit_behavior_counter = 0
	
	parser = arg_parser()
	myArgs = parser.parse_args()#Stealing all the variables from the command line arguments. These can be found defined up at the top of this file. 
	samples = myArgs.samples
	frames = myArgs.frames
	num_of_mitos = myArgs.mitopop
	productionMode = myArgs.productionmode
	max_xy = myArgs.max_xy
	psftogether = myArgs.psftogether
	printGt = myArgs.printgt
	printTif = myArgs.printtif
	parallelPSF = myArgs.parallelpsf
	parallelsamples = myArgs.parallelsamples
	writedisk = myArgs.writedisk
	mitowlow = myArgs.mitowlow
	mitowhigh = myArgs.mitowhigh
	density_mitochondria = myArgs.mitodensity
	mitozlow = myArgs.mitozlow
	mitozhigh = myArgs.mitozhigh
	recordcoordinates = myArgs.recordcoordinates
	terminal = myArgs.terminal
	number_of_vesicles_min = myArgs.numveslow
	number_of_vesicles_max = myArgs.numveshigh
	vesi_min_r = myArgs.vesrlow
	vesi_max_r = myArgs.vesrhigh
	max_mito_length = myArgs.maxlength
	elasticity = myArgs.elasticity
	rotate_speed_range_high = myArgs.rotatespeedhigh
	rotate_speed_range_low = myArgs.rotatespeedlow
	mito_wiggle_high = myArgs.mitowiggleintensityhigh
	mito_wiggle_low = myArgs.mitowiggleintensitylow
	mito_drift_high = myArgs.mitodrifthigh
	mito_drift_low = myArgs.mitodriftlow
	mito_seek_high = myArgs.mitoseekhigh
	mito_seek_low = myArgs.mitoseeklow
	probability_of_merge = myArgs.probabilitymerge
	probability_of_split_on_merge = myArgs.probabilityofsplit
	network = myArgs.network
	dissolve = myArgs.dissolve
	rohit_behavior = myArgs.rohitbehavior
	stationary_mitos = myArgs.stationarymitos
	rohitfissionpercentage = myArgs.rohitfissionpercentage
	rohitseekpercentage = myArgs.rohitseekpercentage
	
	global plot_to_show_connect_merged
	plot_to_show_connect_merged = False
	plot_boolean = False
	if(plot_to_show_connect_merged):
		num_of_mitos = 4
		terminal = 1
		mito_wiggle_high = 400
		mito_wiggle_low = 400
		probability_of_split_on_merge = 0
		probability_of_merge = 1
		mito_drift_high = 0
		mito_drift_low = 0
	
	if(terminal == 0):
		import tkinter
		from tkinter import ttk as tkinter_ttk
		from tkinter import messagebox as tkinter_messagebox
		from mito_gui import gui_config, gui_mito, gui_vesi, gui_micro_pars
	
	m_params = set_default_microscope_parameters()
	density_vesicle = 4
	if(productionMode == False):
		centermult = 18#these reduce the number of points/emmittors in each mitochondria for testing purposes so run time is dramatically reduced. 
		surfDivisor = 20
	else:
		centermult = 1
		surfDivisor = 1
	
	nm_per_pixel = 42
	pixels = int(max_xy * 2 / nm_per_pixel)
	
	cancel, number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r, rohit_behavior, stationary_mitos = gui(terminal, number_of_vesicles_min, 
					number_of_vesicles_max, vesi_min_r, vesi_max_r, rohit_behavior, stationary_mitos) # cancel is a variable that cancels the program if the x button is pressed in the GUI
	
	if(printGt == 0 and printTif == 0):
		iqra_record = False #If we're running this at super speed, then disable writing to Iqra's JSON
	
	if(cancel == False):#This will be true if the user clicks the x button in any of the GUIs
		
		local_time = time.localtime()
		today_date = time.strftime("%Y-%m-%d", time.localtime())
		program_intro = "today's date: " + str(today_date) + "\n"
		timeOfDay = str(local_time.tm_hour) + ":" + str(local_time.tm_min)
		program_intro += "Timestamp: " + timeOfDay + "\n"
		program_intro = program_intro + "Coordinate Grid Center Point XYZ is always: [0, 0, 700]\n"
		if(psftogether == 0):
			program_intro = program_intro + "One entity after another-little PSFs\n"				#psftogether
		else:
			program_intro = program_intro + "All entity together in one big PSF\n"
		program_intro = program_intro + str(samples) + " Samples\n"								#samples
		program_intro = program_intro + str(frames) + " Frames\n"								#frames
		program_intro = program_intro + "Max X/Y: " + str(max_xy) + " Nanometers\n"				#max_xy
		program_intro = program_intro + "Canvas Width & Height: " + str(max_xy*2) + " Nanometers\n"	#canvas
		program_intro = program_intro + "Canvas Width & Height: " + str(pixels) + " Pixels\n"	#pixels
		if(printGt==1):
			program_intro = program_intro + "Printing GT images\n"								#printGt
		else:
			program_intro = program_intro + "Not printing GT images\n"
		if(printTif == 1):
			program_intro = program_intro + "Printing Tiff images\n"							#printTif
		else:
			program_intro = program_intro + "Not printing Tiff images\n"
		if(writedisk == 1):
			program_intro = program_intro + "WRITING PSF TO DISK\n"								#writedisk
		if(parallelsamples):
			program_intro = program_intro + "Parallel Samples\n"								#parallelsamples
		else:
			program_intro = program_intro + "Sequential Samples\n"
		if(parallelsamples == 1 and parallelPSF == 1):
			if(samples > 1):
				msg = "Parallel PSF is not possible due to parallel samples. Reverting to Sequential PSF mode."
				program_intro += msg + "\n"
				parallelPSF = 0
				if(terminal == 0):
					tkinter_messagebox.showwarning("Warning", msg)
			else:
				msg = "Parallel PSF and parallel samples cannot both be activated. samples = 1. Reverting to Sequential samples mode."
				program_intro += msg + "\n"
				parallelsamples = 0
				if(terminal == 0):
					tkinter_messagebox.showwarning("Warning", msg)
		if(writedisk == 1 and parallelPSF == 1):
			msg = "Parallel PSF is not possible due to writedisk. Reverting to Sequential PSF mode."
			program_intro += msg + "\n"		#Removing contradictions
			parallelPSF = 0
			if(terminal == 0):
				tkinter_messagebox.showwarning("Warning", msg)
		if(psftogether == 1 and parallelPSF == 1):
			msg = "Parallel PSF is not possible due to psftogether. Reverting to Sequential PSF mode."
			program_intro += msg + "\n"
			parallelPSF = 0
			if(terminal == 0):
				tkinter_messagebox.showwarning("Warning", msg)
		if(parallelPSF):
			program_intro = program_intro + "parallel PSF\n"									#parallelPSF
		else:
			program_intro = program_intro + "sequential PSF\n"
		if(recordcoordinates == 1):
			program_intro = program_intro + "Printing to CSV\n"									#recordcoordinates
		else:
			program_intro = program_intro + "Not printing to CSV\n"
		if(surfDivisor==1 and centermult == 1):
			program_intro = program_intro + "Production Mode (high number of points)\n"			#surfdivisor & centermult
		else:
			program_intro = program_intro + "Testing Mode (low num of pts)\n"
		if(rohit_behavior == 1):
			program_intro += "rohit_behavior is turned ON with a " + str(rohitfissionpercentage) + "% chance of split & " + str(rohitseekpercentage) + "% chance of seek\n"	#rohit_behavior
		
		if(terminal == 1):
			program_intro = program_intro + "Disabling GUI\n"									#terminal
		
		program_intro = program_intro + "\n"
		
		program_intro = program_intro + "Network: " + str(network) + " (1=yes, 0=no)\n"										#network
		program_intro = program_intro + "Dissolve: " + str(dissolve) + " (1=yes, 0=no)\n"									#dissolve
		program_intro = program_intro + str(num_of_mitos) + " Mitos\n"														#num_of_mitos
		program_intro = program_intro + "Width Range: " + str(mitowlow) + " - " + str(mitowhigh) + "\n"						#mitowlow & mitowhigh
		program_intro = program_intro + "max_mito_length: " + str(max_mito_length) + "\n"									#max_mito_length
		program_intro = program_intro + "Mitochondria Density: " + str(density_mitochondria) + "\n"							#density_mitochondria
		program_intro = program_intro + "Mitochondria Z: " + str(mitozlow) + " - " + str(mitozhigh) + "\n"					#zlow & zhigh
		program_intro = program_intro + "Elasticity: " + str(elasticity) + "%\n"											#elasticity
		program_intro = program_intro + "Rotate Speed Range: " + str(rotate_speed_range_low) + " - " + str(rotate_speed_range_high) + "\n"		#rotate_speed_range_high & rotate_speed_range_low
		program_intro = program_intro + "Wiggle Intensity Range: " + str(mito_wiggle_low) + " - " + str(mito_wiggle_high) + "\n"	#wiggle_intensity_range_high & wiggle_intensity_range_low
		program_intro = program_intro + "Drift Speed Range: " + str(mito_drift_low) + " - " + str(mito_drift_high) + "\n"			#drift_speed_range_high & drift_speed_range_low
		program_intro = program_intro + "Seek Speed Range: " + str(mito_seek_low) + " - " + str(mito_seek_high) + "\n"			#seek_speed_range_high & seek_speed_range_low
		program_intro = program_intro + "Probability of Merge: " + str(probability_of_merge*100) + "%" + "\n"				#probability_of_merge
		program_intro = program_intro + "Probability of Split on Merge: " + str(probability_of_split_on_merge*100) + "%\n"	#probability of split on merge
		if(stationary_mitos > 0):
			program_intro += "stationary_mitos: " + str(stationary_mitos) + "\n"											#stationary_mitos
		
		program_intro = program_intro + "\n"
		
		program_intro = program_intro + "number_of_vesicles range: " + str(number_of_vesicles_min) + " - " + str(number_of_vesicles_max) + "\n"		#vesicle number
		program_intro = program_intro + "vesi radius range: " + str(vesi_min_r) + " - " + str(vesi_max_r) + "\n"									#vesicle radius
		
		program_intro = program_intro + "\n"
		
		program_intro = program_intro + "num_basis: " + str(num_basis) + "\n"
		program_intro = program_intro + "rho_samples: " + str(rho_samples) + "\n"
		program_intro = program_intro + "magnification: " + str(magnification) + "\n"
		program_intro = program_intro + "numerical_aperture: " + str(numerical_aperture) + "\n"
		program_intro = program_intro + "coverslip_RI_design_value: " + str(coverslip_RI_design_value) + "\n"
		program_intro = program_intro + "coverslip_RI_experimental_value: " + str(coverslip_RI_experimental_value) + "\n"
		program_intro = program_intro + "immersion_medium_RI_design_value: " + str(immersion_medium_RI_design_value) + "\n"
		program_intro = program_intro + "immersion_medium_RI_experimental_value: " + str(immersion_medium_RI_experimental_value) + "\n"
		program_intro = program_intro + "specimen_refractive_index_RI: " + str(specimen_refractive_index_RI) + "\n"
		program_intro = program_intro + "microns_working_distance_immersion_medium_thickness_design_value: " + str(microns_working_distance_immersion_medium_thickness_design_value) + "\n"
		program_intro = program_intro + "microns_coverslip_thickness_experimental_value: " + str(microns_coverslip_thickness_experimental_value) + "\n"
		program_intro = program_intro + "microns_coverslip_thickness_design_value: " + str(microns_coverslip_thickness_design_value) + "\n"
		program_intro = program_intro + "microscope_tube_length_in_microns: " + str(microscope_tube_length_in_microns) + "\n"
		
		print(program_intro)
		
		stationary_mitos_temp = []
		for mito in range(stationary_mitos):#reformatting. if there's 2 stationary mitos, then stationary_mitos will equal this list: [0, 1]
			stationary_mitos_temp.append(mito)
		stationary_mitos = stationary_mitos_temp
		
		loop_through_samples(centermult, surfDivisor, program_intro, probability_of_merge, probability_of_split_on_merge, elasticity, m_params, 
					number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r, 
					density_vesicle, max_mito_length, rohit_behavior, pixels, nm_per_pixel, stationary_mitos, rohitfissionpercentage, rohitseekpercentage, plot_boolean)
	if(terminal == 0):
		tkinter_messagebox.showinfo("Information", "This program has finished running. Thank you.")

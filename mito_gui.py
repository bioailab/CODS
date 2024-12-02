import tkinter
from tkinter import ttk as tkinter_ttk
from tkinter import messagebox as tkinter_messagebox

def capture_gui_config(root, psftogether_value, samples_user_input, frames_user_input, canvas_user_input, printTif_value, printGt_value, parallelPSF_value,
					parallelsamples_value, writedisk_value, recordcoordinates_value, productionMode_value, rohit_behavior_value):
	#This function runs when the submit button is pressed on the first settings window
	#It captures the current values of the input boxes and saves them to be outputted to the main program
	global cancel
	global psftogether
	global samples
	global frames
	global max_xy
	global printTif
	global printGt
	global parallelPSF
	global parallelsamples
	global writedisk
	global recordcoordinates
	global productionMode
	global rohit_behavior
	
	psftogether = int(psftogether_value)
	printTif = int(printTif_value)
	printGt = int(printGt_value)
	parallelPSF = int(parallelPSF_value)
	parallelsamples = int(parallelsamples_value)
	writedisk = int(writedisk_value)
	recordcoordinates = int(recordcoordinates_value)
	frames = int(frames_user_input)
	samples = int(samples_user_input)
	max_xy = int(int(canvas_user_input)/2)
	productionMode = int(productionMode_value)
	rohit_behavior = int(rohit_behavior_value)
	root.destroy()
	cancel = False#When the submit button is pressed, that means the user did not click the x button to close the window. In this case, the program should be allowed to continue running normally

def gui_config(psftogether_default, samples_default, frames_default, max_xy_default, printTif_default, printGt_default, 
				parallelPSF_default, parallelsamples_default, writedisk_default, recordcoordinates_default, productionMode_default, rohit_behavior_default):
	#GUI to collect values for command line arguments
	global cancel
	global psftogether
	global samples
	global frames
	global max_xy
	global printTif
	global printGt
	global parallelPSF
	global parallelsamples
	global writedisk
	global recordcoordinates
	global productionMode
	global rohit_behavior
	
	psftogether, samples, frames, max_xy, printTif, printGt, parallelPSF, parallelsamples, writedisk, recordcoordinates, productionMode, rohit_behavior = [], [], [],[],[],[],[],[],[],[],[],[]
	
	cancel = True # this variable is true when the user clicks the 'x' to close the GUI. If this happens, don't continue with the execution of the program. 
	tkinter_window = tkinter.Tk()
	tkinter_window.geometry('580x320')
	tkinter_window.title("Simulated Complex Dynamic Environment")
	
	tkinter.Label(tkinter_window, text = "(42 nm per pixel)").grid(column = 1, row = 1)#info
	tkinter.Label(tkinter_window, text = "Canvas width and height in nm").grid(column = 1, row = 2)#max_xy
	canvas_widget = tkinter.Spinbox(tkinter_window, from_ = 500, to = 50000)
	canvas_widget.grid(column = 2, row = 2)
	canvas_widget.delete(0, "end")
	canvas_widget.insert(0, max_xy_default * 2)
	
	frames_label = tkinter.Label(tkinter_window, text = "Frames").grid(column = 1, row = 3)#num of Frames
	frames_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 10000)
	frames_widget.grid(column = 2, row = 3)
	frames_widget.delete(0, "end")
	frames_widget.insert(0, frames_default)
	
	tkinter.Label(tkinter_window, text = "Samples").grid(column = 1, row = 4)#num of Samples 
	samples_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 50000)
	samples_widget.grid(column = 2, row = 4)
	samples_widget.delete(0, "end")
	samples_widget.insert(0, samples_default)
	
	tkinter.Label(tkinter_window, text = "PSF").grid(column = 1, row = 5)
	parallelPSF_widget = tkinter_ttk.Combobox(tkinter_window, values=['Sequential', 'Parallel'])# Parallel PSF?
	parallelPSF_widget.grid(column = 2, row = 5)
	parallelPSF_widget.current(parallelPSF_default)
	
	tkinter.Label(tkinter_window, text = "Samples").grid(column = 1, row = 6)
	parallelsamples_widget = tkinter_ttk.Combobox(tkinter_window, values=['Sequential', 'Parallel'])# Parallel Samples?
	parallelsamples_widget.grid(column = 2, row = 6)
	parallelsamples_widget.current(parallelsamples_default)
	
	tkinter.Label(tkinter_window, text = "Record Mitochondria Coordinates CSV").grid(column = 1, row = 7)
	recordcoordinates_widget = tkinter_ttk.Combobox(tkinter_window, values=['No', 'Yes'])# record coordinates?
	recordcoordinates_widget.grid(column = 2, row = 7)
	recordcoordinates_widget.current(recordcoordinates_default)
	
	tkinter.Label(tkinter_window, text = "Print Tiff Images?").grid(column = 1, row = 8)
	printTif_widget = tkinter_ttk.Combobox(tkinter_window, values=['No', 'Yes'])# Print Tiff images?
	printTif_widget.grid(column = 2, row = 8)
	printTif_widget.current(printTif_default)
	
	tkinter.Label(tkinter_window, text = "Print GT images?").grid(column = 1, row = 9)
	printGt_widget = tkinter_ttk.Combobox(tkinter_window, values=['No', 'Yes'])# Print GT images?
	printGt_widget.grid(column = 2, row = 9)
	printGt_widget.current(printGt_default)
	
	tkinter.Label(tkinter_window, text = "RAM usage reduction").grid(column = 1, row = 10)
	writedisk_widget = tkinter_ttk.Combobox(tkinter_window, values=['Off', 'On'])# write to disk?
	writedisk_widget.grid(column = 2, row = 10)
	writedisk_widget.current(writedisk_default)
	
	tkinter.Label(tkinter_window, text = "PSF for each mitochondria in one frame").grid(column = 1, row = 11)
	psftogether_widget = tkinter_ttk.Combobox(tkinter_window, values=['One at a time', 'Together'])# psf together?
	psftogether_widget.grid(column = 2, row = 11)
	psftogether_widget.current(psftogether_default)
	
	tkinter.Label(tkinter_window, text = "Mode").grid(column = 1, row = 12)
	productionMode_widget = tkinter_ttk.Combobox(tkinter_window, values=['Low num of points (fast render)', 'Full Production Mode'], width=30)# production mode?
	productionMode_widget.grid(column = 2, row = 12)
	productionMode_widget.current(productionMode_default)
	
	tkinter.Label(tkinter_window, text = "Random Behavior + Record Centerpoints").grid(column = 1, row = 13)
	rohit_behavior_widget = tkinter_ttk.Combobox(tkinter_window, values=['Off', 'On'], width=30)# Rohit_behavior
	rohit_behavior_widget.grid(column = 2, row = 13)
	rohit_behavior_widget.current(rohit_behavior_default)
	
	submit_button = tkinter.Button(tkinter_window, text = "Submit", command = lambda: capture_gui_config(tkinter_window, psftogether_widget.current(), 
				samples_widget.get(), frames_widget.get(), canvas_widget.get(), printTif_widget.current(), printGt_widget.current(), parallelPSF_widget.current(),
				parallelsamples_widget.current(), writedisk_widget.current(), recordcoordinates_widget.current(), productionMode_widget.current(), 
				rohit_behavior_widget.current())).grid(column = 1, row = 14)
	
	tkinter_window.mainloop()
	return [cancel, psftogether, samples, frames, max_xy, printTif, printGt, parallelPSF, parallelsamples, writedisk, recordcoordinates, productionMode, rohit_behavior]

def capture_gui_mito(root, num_of_mitos_user_input, wlow_user_input, whigh_user_input, max_mito_length_user_input, elasticity_user_input, 
				density_mitochondria_user_input, zlow_user_input, zhigh_user_input, rotate_speed_range_high_user_input, 
				rotate_speed_range_low_user_input, wiggle_intensity_range_high_user_input, wiggle_intensity_range_low_user_input, 
				drift_speed_range_high_user_input, drift_speed_range_low_user_input, seek_speed_range_high_user_input, seek_speed_range_low_user_input, 
				probability_of_merge_user_input, probability_of_split_on_merge_user_input, network_user_input, dissolve_user_input, stationary_mitos_user_input):
	global cancel
	global num_of_mitos
	global wlow
	global whigh
	global max_mito_length
	global elasticity
	global density_mitochondria
	global zlow
	global zhigh
	global rotate_speed_range_high
	global rotate_speed_range_low
	global wiggle_intensity_range_high
	global wiggle_intensity_range_low
	global drift_speed_range_high
	global drift_speed_range_low
	global seek_speed_range_high
	global seek_speed_range_low
	global probability_of_merge
	global probability_of_split_on_merge
	global network
	global dissolve
	global stationary_mitos
	
	num_of_mitos = int(num_of_mitos_user_input)
	wlow = int(wlow_user_input)
	whigh = int(whigh_user_input)
	zlow = int(zlow_user_input)
	zhigh = int(zhigh_user_input)
	density_mitochondria = int(density_mitochondria_user_input)
	max_mito_length = int(max_mito_length_user_input)
	elasticity = int(elasticity_user_input)
	rotate_speed_range_low = int(rotate_speed_range_low_user_input)
	rotate_speed_range_high = int(rotate_speed_range_high_user_input)
	wiggle_intensity_range_low = int(wiggle_intensity_range_low_user_input)
	wiggle_intensity_range_high = int(wiggle_intensity_range_high_user_input)
	drift_speed_range_low = int(drift_speed_range_low_user_input)
	drift_speed_range_high = int(drift_speed_range_high_user_input)
	seek_speed_range_low = int(seek_speed_range_low_user_input)
	seek_speed_range_high = int(seek_speed_range_high_user_input)
	probability_of_merge = float(probability_of_merge_user_input)
	probability_of_split_on_merge = float(probability_of_split_on_merge_user_input)
	network = network_user_input
	dissolve = dissolve_user_input
	stationary_mitos = int(stationary_mitos_user_input)
	root.destroy()
	cancel = False#When the submit button is pressed, that means the user did not click the x button to close the window. In this case, the program should be allowed to continue running normally

def gui_mito(num_of_mitos_default, wlow_default, whigh_default, max_mito_length_default, elasticity_default, density_mitochondria_default, zlow_default, zhigh_default, 
					rotate_speed_range_high_default, rotate_speed_range_low_default, wiggle_intensity_range_high_default, wiggle_intensity_range_low_default, 
					drift_speed_range_high_default, drift_speed_range_low_default, seek_speed_range_high_default, seek_speed_range_low_default, probability_of_merge_default, 
					probability_of_split_on_merge_default, network_default, dissolve_default, stationary_mitos_default):
	global cancel
	global num_of_mitos
	global wlow
	global whigh
	global max_mito_length
	global elasticity
	global density_mitochondria
	global zlow
	global zhigh
	global rotate_speed_range_high
	global rotate_speed_range_low
	global wiggle_intensity_range_high
	global wiggle_intensity_range_low
	global drift_speed_range_high
	global drift_speed_range_low
	global seek_speed_range_high
	global seek_speed_range_low
	global probability_of_merge
	global probability_of_split_on_merge
	global network
	global dissolve
	global stationary_mitos
	
	num_of_mitos, wlow, whigh, max_mito_length, elasticity, density_mitochondria, zlow, zhigh, rotate_speed_range_high, rotate_speed_range_low, wiggle_intensity_range_high, wiggle_intensity_range_low, drift_speed_range_high, drift_speed_range_low, seek_speed_range_high, seek_speed_range_low, probability_of_merge, probability_of_split_on_merge, network, dissolve, stationary_mitos = [],[],[],[],[],[],[],[], [],[],[],[], [],[],[],[],[], [],[],[], []
	
	cancel = True # this variable is true when the user clicks the 'x' to close the GUI. If this happens, don't continue with the execution of the program. 
	tkinter_window = tkinter.Tk()
	tkinter_window.geometry('700x500')
	tkinter_window.title("Mitochondria Attributes")
	
	tkinter.Label(tkinter_window, text = "Number of mitochondria").grid(column = 1, row = 1)#num of mitos
	num_of_mitos_widget = tkinter.Spinbox(tkinter_window, from_ = 0, to = 10Z00)
	num_of_mitos_widget.grid(column = 2, row = 1)
	num_of_mitos_widget.delete(0, "end")
	num_of_mitos_widget.insert(0, num_of_mitos_default)
	
	tkinter.Label(tkinter_window, text = "Mitochondria Width Range").grid(column = 1, row = 2)#min mito width
	wlow_widget = tkinter.Spinbox(tkinter_window, from_ = 50, to = 1000)
	wlow_widget.grid(column = 2, row = 2)
	wlow_widget.delete(0, "end")
	wlow_widget.insert(0, wlow_default)
	tkinter.Label(tkinter_window, text = " - ").grid(column = 3, row = 2)#max mito width
	whigh_widget = tkinter.Spinbox(tkinter_window, from_ = 50, to = 1000)
	whigh_widget.grid(column = 4, row = 2)
	whigh_widget.delete(0, "end")
	whigh_widget.insert(0, whigh_default)
	
	tkinter.Label(tkinter_window, text = "Z depth of mitochondria Range").grid(column = 1, row = 3)# zlow
	zlow_widget = tkinter.Spinbox(tkinter_window, from_ = 100, to = 2000)
	zlow_widget.grid(column = 2, row = 3)
	zlow_widget.delete(0, "end")
	zlow_widget.insert(0, zlow_default)
	tkinter.Label(tkinter_window, text = " - ").grid(column = 3, row = 3)# zhigh
	zhigh_widget = tkinter.Spinbox(tkinter_window, from_ = 100, to = 2000)
	zhigh_widget.grid(column = 4, row = 3)
	zhigh_widget.delete(0, "end")
	zhigh_widget.insert(0, zhigh_default)
	
	tkinter.Label(tkinter_window, text = "Max Length").grid(column = 1, row = 4)# max_mito_length
	max_mito_length_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 50000)
	max_mito_length_widget.grid(column = 2, row = 4)
	max_mito_length_widget.delete(0, "end")
	max_mito_length_widget.insert(0, max_mito_length_default)
	
	tkinter.Label(tkinter_window, text = "Density").grid(column = 1, row = 5)# density_mitochondria
	density_mitochondria_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 500)
	density_mitochondria_widget.grid(column = 2, row = 5)
	density_mitochondria_widget.delete(0, "end")
	density_mitochondria_widget.insert(0, density_mitochondria_default)
	
	tkinter.Label(tkinter_window, text = "elasticity percentage").grid(column = 1, row = 6)# elasticity
	elasticity_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 99)
	elasticity_widget.grid(column = 2, row = 6)
	elasticity_widget.delete(0, "end")
	elasticity_widget.insert(0, elasticity_default)
	
	tkinter.Label(tkinter_window, text = "Rotate Speed Range").grid(column = 1, row = 7)# rotate_speed_range
	rotate_speed_range_low_widget = tkinter.Spinbox(tkinter_window, from_ = 0, to = 359)
	rotate_speed_range_low_widget.grid(column = 2, row = 7)
	rotate_speed_range_low_widget.delete(0, "end")
	rotate_speed_range_low_widget.insert(0, rotate_speed_range_low_default)
	tkinter.Label(tkinter_window, text = " - ").grid(column = 3, row = 7)
	rotate_speed_range_high_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 359)
	rotate_speed_range_high_widget.grid(column = 4, row = 7)
	rotate_speed_range_high_widget.delete(0, "end")
	rotate_speed_range_high_widget.insert(0, rotate_speed_range_high_default)
	
	tkinter.Label(tkinter_window, text = "Wiggle Intensity Range").grid(column = 1, row = 8)# wiggle_intensity_range
	wiggle_intensity_range_low_widget = tkinter.Spinbox(tkinter_window, from_ = 0, to = 999)
	wiggle_intensity_range_low_widget.grid(column = 2, row = 8)
	wiggle_intensity_range_low_widget.delete(0, "end")
	wiggle_intensity_range_low_widget.insert(0, wiggle_intensity_range_low_default)
	tkinter.Label(tkinter_window, text = " - ").grid(column = 3, row = 8)
	wiggle_intensity_range_high_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 1000)
	wiggle_intensity_range_high_widget.grid(column = 4, row = 8)
	wiggle_intensity_range_high_widget.delete(0, "end")
	wiggle_intensity_range_high_widget.insert(0, wiggle_intensity_range_high_default)
	
	tkinter.Label(tkinter_window, text = "Drift Speed Range").grid(column = 1, row = 9)# drift_speed_range
	drift_speed_range_low_widget = tkinter.Spinbox(tkinter_window, from_ = 0, to = 999)
	drift_speed_range_low_widget.grid(column = 2, row = 9)
	drift_speed_range_low_widget.delete(0, "end")
	drift_speed_range_low_widget.insert(0, drift_speed_range_low_default)
	tkinter.Label(tkinter_window, text = " - ").grid(column = 3, row = 9)
	drift_speed_range_high_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 1000)
	drift_speed_range_high_widget.grid(column = 4, row = 9)
	drift_speed_range_high_widget.delete(0, "end")
	drift_speed_range_high_widget.insert(0, drift_speed_range_high_default)
	
	tkinter.Label(tkinter_window, text = "Seek Speed Range").grid(column = 1, row = 10)# seek_speed_range
	seek_speed_range_low_widget = tkinter.Spinbox(tkinter_window, from_ = 0, to = 999)
	seek_speed_range_low_widget.grid(column = 2, row = 10)
	seek_speed_range_low_widget.delete(0, "end")
	seek_speed_range_low_widget.insert(0, seek_speed_range_low_default)
	tkinter.Label(tkinter_window, text = " - ").grid(column = 3, row = 10)
	seek_speed_range_high_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 1000)
	seek_speed_range_high_widget.grid(column = 4, row = 10)
	seek_speed_range_high_widget.delete(0, "end")
	seek_speed_range_high_widget.insert(0, seek_speed_range_high_default)
	
	tkinter.Label(tkinter_window, text = "Probability of Merge").grid(column = 1, row = 11)# probability_of_merge
	probability_of_merge_widget = tkinter.Entry(tkinter_window)
	probability_of_merge_widget.grid(column = 2, row = 11)
	probability_of_merge_widget.delete(0, "end")
	probability_of_merge_widget.insert(0, probability_of_merge_default)
	
	tkinter.Label(tkinter_window, text = "Probability of Split (on merge)").grid(column = 1, row = 12)# probability_of_split_on_merge
	probability_of_split_on_merge_widget = tkinter.Entry(tkinter_window)
	probability_of_split_on_merge_widget.grid(column = 2, row = 12)
	probability_of_split_on_merge_widget.delete(0, "end")
	probability_of_split_on_merge_widget.insert(0, probability_of_split_on_merge_default)
	
	tkinter.Label(tkinter_window, text = "Build Network?").grid(column = 1, row = 13)# Network
	network_widget = tkinter_ttk.Combobox(tkinter_window, values=['No', 'Yes'])
	network_widget.grid(column = 2, row = 13)
	network_widget.current(network_default)
	
	tkinter.Label(tkinter_window, text = "Dissolve at the End?").grid(column = 1, row = 14)# Dissolve
	dissolve_widget = tkinter_ttk.Combobox(tkinter_window, values=['No', 'Yes'])
	dissolve_widget.grid(column = 2, row = 14)
	dissolve_widget.current(dissolve_default)
	
	tkinter.Label(tkinter_window, text = "Number of Stationary Mitochondria").grid(column = 1, row = 15)#Stationary Mitos
	stationary_mitos_widget = tkinter.Spinbox(tkinter_window, from_ = 0, to = 100)
	stationary_mitos_widget.grid(column = 2, row = 15)
	stationary_mitos_widget.delete(0, "end")
	stationary_mitos_widget.insert(0, stationary_mitos_default)
	
	submit_button = tkinter.Button(tkinter_window, text = "Submit", command = lambda: capture_gui_mito(tkinter_window, num_of_mitos_widget.get(), wlow_widget.get(),
				whigh_widget.get(), max_mito_length_widget.get(), elasticity_widget.get(), density_mitochondria_widget.get(), zlow_widget.get(), zhigh_widget.get(), 
				rotate_speed_range_high_widget.get(), rotate_speed_range_low_widget.get(), wiggle_intensity_range_high_widget.get(), wiggle_intensity_range_low_widget.get(), 
				drift_speed_range_high_widget.get(), drift_speed_range_low_widget.get(), seek_speed_range_high_widget.get(), 
				seek_speed_range_low_widget.get(), probability_of_merge_widget.get(), probability_of_split_on_merge_widget.get(),
				network_widget.current(), dissolve_widget.current(), stationary_mitos_widget.get())).grid(column = 1, row = 16)
	
	tkinter_window.mainloop()
	return [cancel, num_of_mitos, wlow, whigh, max_mito_length, elasticity, density_mitochondria, zlow, zhigh, 
					rotate_speed_range_high, rotate_speed_range_low, wiggle_intensity_range_high, wiggle_intensity_range_low, 
					drift_speed_range_high, drift_speed_range_low, seek_speed_range_high, seek_speed_range_low, probability_of_merge, 
					probability_of_split_on_merge, network, dissolve, stationary_mitos]

def capture_gui_vesi(root, number_of_vesicles_min_user_input, number_of_vesicles_max_user_input, vesi_min_r_user_input, vesi_max_r_user_input):
	#This function runs when the submit button is pressed on the Second settings window with the behavior settings
	#It captures the current values of the input boxes and saves them to be outputted to the main program
	global cancel
	global number_of_vesicles_min
	global number_of_vesicles_max
	global vesi_min_r
	global vesi_max_r
	
	number_of_vesicles_min = int(number_of_vesicles_min_user_input)
	number_of_vesicles_max = int(number_of_vesicles_max_user_input)
	vesi_min_r = int(vesi_min_r_user_input)
	vesi_max_r = int(vesi_max_r_user_input)
	
	root.destroy()
	cancel = False#When the submit button is pressed, that means the user did not click the x button to close the window. In this case, the program should be allowed to continue running normally

def gui_vesi(number_of_vesicles_min_default, number_of_vesicles_max_default, vesi_min_r_default, vesi_max_r_default): 
	#GUI to collect values for vesicles
	global cancel
	global number_of_vesicles_min
	global number_of_vesicles_max
	global vesi_min_r
	global vesi_max_r
	
	number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r = [],[],[],[]
	
	cancel = True # this variable is true when the user clicks the 'x' to close the GUI. If this happens, don't continue with the execution of the program. 
	tkinter_window = tkinter.Tk()
	tkinter_window.geometry('700x300')
	tkinter_window.title("Vesicle Attributes")
	
	tkinter.Label(tkinter_window, text = "Number of Vesicles Range").grid(column = 1, row = 1)# num vesicles
	number_of_vesicles_min_widget = tkinter.Spinbox(tkinter_window, from_ = 0, to = 100)
	number_of_vesicles_min_widget.grid(column = 2, row = 1)
	number_of_vesicles_min_widget.delete(0, "end")
	number_of_vesicles_min_widget.insert(0, number_of_vesicles_min_default)
	tkinter.Label(tkinter_window, text = " - ").grid(column = 3, row = 1)
	number_of_vesicles_max_widget = tkinter.Spinbox(tkinter_window, from_ = 0, to = 100)
	number_of_vesicles_max_widget.grid(column = 4, row = 1)
	number_of_vesicles_max_widget.delete(0, "end")
	number_of_vesicles_max_widget.insert(0, number_of_vesicles_max_default)
	
	tkinter.Label(tkinter_window, text = "Vesicle Radius").grid(column = 1, row = 2)# vesicle radius
	vesi_min_r_widget = tkinter.Spinbox(tkinter_window, from_ = 0, to = 10000)
	vesi_min_r_widget.grid(column = 2, row = 2)
	vesi_min_r_widget.delete(0, "end")
	vesi_min_r_widget.insert(0, vesi_min_r_default)
	tkinter.Label(tkinter_window, text = " - ").grid(column = 3, row = 2)
	vesi_max_r_widget = tkinter.Spinbox(tkinter_window, from_ = 1, to = 10000)
	vesi_max_r_widget.grid(column = 4, row = 2)
	vesi_max_r_widget.delete(0, "end")
	vesi_max_r_widget.insert(0, vesi_max_r_default)
	
	submit_button = tkinter.Button(tkinter_window, text = "Submit", command = lambda: capture_gui_vesi(tkinter_window, 
					number_of_vesicles_min_widget.get(), number_of_vesicles_max_widget.get(), vesi_min_r_widget.get(), 
					vesi_max_r_widget.get())).grid(column = 2, row = 3)
	
	tkinter_window.mainloop()
	return [cancel, number_of_vesicles_min, number_of_vesicles_max, vesi_min_r, vesi_max_r]

def capture_gui_microscope_values(root, num_basis_user_input, rho_samples_user_input, magnification_user_input, numerical_aperture_user_input, 
				coverslip_RI_design_value_user_input, coverslip_RI_experimental_value_user_input, immersion_medium_RI_design_value_user_input, 
				immersion_medium_RI_experimental_value_user_input, specimen_refractive_index_RI_user_input, 
				microns_working_distance_immersion_medium_thickness_design_value_user_input, 
				microns_coverslip_thickness_experimental_value_user_input, microns_coverslip_thickness_design_value_user_input,
				microscope_tube_length_in_microns_user_input):
	#This function runs when the submit button is pressed on the Third settings window with microscope parameters
	#It captures the current values of the input boxes and saves them to be outputted to the main program
	global cancel
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
	
	num_basis = int(num_basis_user_input)
	rho_samples = int(rho_samples_user_input)
	magnification = float(magnification_user_input)
	numerical_aperture = float(numerical_aperture_user_input)
	coverslip_RI_design_value = float(coverslip_RI_design_value_user_input)
	coverslip_RI_experimental_value = float(coverslip_RI_experimental_value_user_input)
	immersion_medium_RI_design_value = float(immersion_medium_RI_design_value_user_input)
	immersion_medium_RI_experimental_value = float(immersion_medium_RI_experimental_value_user_input)
	specimen_refractive_index_RI = float(specimen_refractive_index_RI_user_input)
	microns_working_distance_immersion_medium_thickness_design_value = float(microns_working_distance_immersion_medium_thickness_design_value_user_input)
	microns_coverslip_thickness_experimental_value = float(microns_coverslip_thickness_experimental_value_user_input)
	microns_coverslip_thickness_design_value = float(microns_coverslip_thickness_design_value_user_input)
	microscope_tube_length_in_microns = float(microscope_tube_length_in_microns_user_input)
	root.destroy()
	cancel = False#When the submit button is pressed, that means the user did not click the x button to close the window. In this case, the program should be allowed to continue running normally

def gui_micro_pars(num_basis_default, rho_samples_default, magnification_default, numerical_aperture_default, coverslip_RI_design_value_default, 
				coverslip_RI_experimental_value_default, immersion_medium_RI_design_value_default, immersion_medium_RI_experimental_value_default, 
				specimen_refractive_index_RI_default, microns_working_distance_immersion_medium_thickness_design_value_default, 
				microns_coverslip_thickness_experimental_value_default, microns_coverslip_thickness_design_value_default, microscope_tube_length_in_microns_default): 
	#GUI to collect values for Microscope Parameters
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
	global cancel
	
	num_basis, rho_samples, magnification, numerical_aperture, coverslip_RI_design_value, coverslip_RI_experimental_value, immersion_medium_RI_design_value, immersion_medium_RI_experimental_value, specimen_refractive_index_RI, microns_working_distance_immersion_medium_thickness_design_value, microns_coverslip_thickness_experimental_value, microns_coverslip_thickness_design_value, microscope_tube_length_in_microns = [],[],[],[],[],[],[],[],[],[],[],[],[]
	
	cancel = True#this variable is true when the user clicks the 'x' to close the GUI. If this happens, don't continue with the execution of the program. 
	tkinter_window = tkinter.Tk()
	tkinter_window.geometry('850x500')
	tkinter_window.title("Microscope Parameters")
	
	tkinter.Label(tkinter_window, text = "Num Basis (Number of Rescaled Bessels That Approximate the Phase Function)").grid(column = 1, row = 1)#num_basis
	num_basis_widget = tkinter.Entry(tkinter_window)
	num_basis_widget.grid(column = 2, row = 1)
	num_basis_widget.delete(0, "end")
	num_basis_widget.insert(0, num_basis_default)
	
	rho_samples_label = tkinter.Label(tkinter_window, text = "Rho Samples (Number of Pupil Sample Along the Radial Direction)").grid(column = 1, row = 2)#rho_samples
	rho_samples_widget = tkinter.Entry(tkinter_window)
	rho_samples_widget.grid(column = 2, row = 2)
	rho_samples_widget.delete(0, "end")
	rho_samples_widget.insert(0, rho_samples_default)
	
	tkinter.Label(tkinter_window, text = "Magnification").grid(column = 1, row = 3)#magnification
	magnification_widget = tkinter.Entry(tkinter_window)
	magnification_widget.grid(column = 2, row = 3)
	magnification_widget.delete(0, "end")
	magnification_widget.insert(0, magnification_default)
	
	tkinter.Label(tkinter_window, text = "Numerical Aperture").grid(column = 1, row = 4)#numerical_aperture
	numerical_aperture_widget = tkinter.Entry(tkinter_window)
	numerical_aperture_widget.grid(column = 2, row = 4)
	numerical_aperture_widget.delete(0, "end")
	numerical_aperture_widget.insert(0, numerical_aperture_default)
	
	tkinter.Label(tkinter_window, text = "Coverslip RI Design Value").grid(column = 1, row = 5)#coverslip_RI_design_value
	coverslip_RI_design_value_widget = tkinter.Entry(tkinter_window)
	coverslip_RI_design_value_widget.grid(column = 2, row = 5)
	coverslip_RI_design_value_widget.delete(0, "end")
	coverslip_RI_design_value_widget.insert(0, coverslip_RI_design_value_default)
	
	tkinter.Label(tkinter_window, text = "Coverslip RI Experimental Value").grid(column = 1, row = 6)#coverslip_RI_experimental_value
	coverslip_RI_experimental_value_widget = tkinter.Entry(tkinter_window)
	coverslip_RI_experimental_value_widget.grid(column = 2, row = 6)
	coverslip_RI_experimental_value_widget.delete(0, "end")
	coverslip_RI_experimental_value_widget.insert(0, coverslip_RI_experimental_value_default)
	
	tkinter.Label(tkinter_window, text = "Immersion Medium RI Design Value").grid(column = 1, row = 7)# immersion_medium_RI_design_value
	immersion_medium_RI_design_value_widget = tkinter.Entry(tkinter_window)
	immersion_medium_RI_design_value_widget.grid(column = 2, row = 7)
	immersion_medium_RI_design_value_widget.delete(0, "end")
	immersion_medium_RI_design_value_widget.insert(0, immersion_medium_RI_design_value_default)
	
	tkinter.Label(tkinter_window, text = "Immersion Medium RI Experimental Value").grid(column = 1, row = 8)# immersion_medium_RI_experimental_value
	immersion_medium_RI_experimental_value_widget = tkinter.Entry(tkinter_window)
	immersion_medium_RI_experimental_value_widget.grid(column = 2, row = 8)
	immersion_medium_RI_experimental_value_widget.delete(0, "end")
	immersion_medium_RI_experimental_value_widget.insert(0, immersion_medium_RI_experimental_value_default)
	
	tkinter.Label(tkinter_window, text = "Specimen Refractive Index (RI)").grid(column = 1, row = 9)# specimen_refractive_index_RI
	specimen_refractive_index_RI_widget = tkinter.Entry(tkinter_window)
	specimen_refractive_index_RI_widget.grid(column = 2, row = 9)
	specimen_refractive_index_RI_widget.delete(0, "end")
	specimen_refractive_index_RI_widget.insert(0, specimen_refractive_index_RI_default)
	
	tkinter.Label(tkinter_window, text = "Microns, Working Distance (Immersion Medium Thickness) Design Value").grid(column = 1, row = 10)# microns_working_distance_immersion_medium_thickness_design_value
	microns_working_distance_immersion_medium_thickness_design_value_widget = tkinter.Entry(tkinter_window)
	microns_working_distance_immersion_medium_thickness_design_value_widget.grid(column = 2, row = 10)
	microns_working_distance_immersion_medium_thickness_design_value_widget.delete(0, "end")
	microns_working_distance_immersion_medium_thickness_design_value_widget.insert(0, microns_working_distance_immersion_medium_thickness_design_value_default)
	
	tkinter.Label(tkinter_window, text = "Microns, Coverslip Thickness Experimental Value").grid(column = 1, row = 11)# microns_coverslip_thickness_experimental_value
	microns_coverslip_thickness_experimental_value_widget = tkinter.Entry(tkinter_window)
	microns_coverslip_thickness_experimental_value_widget.grid(column = 2, row = 11)
	microns_coverslip_thickness_experimental_value_widget.delete(0, "end")
	microns_coverslip_thickness_experimental_value_widget.insert(0, microns_coverslip_thickness_experimental_value_default)
	
	tkinter.Label(tkinter_window, text = "Microns, Coverslip Thickness Design Value").grid(column = 1, row = 12)# microns_coverslip_thickness_design_value
	microns_coverslip_thickness_design_value_widget = tkinter.Entry(tkinter_window)
	microns_coverslip_thickness_design_value_widget.grid(column = 2, row = 12)
	microns_coverslip_thickness_design_value_widget.delete(0, "end")
	microns_coverslip_thickness_design_value_widget.insert(0, microns_coverslip_thickness_design_value_default)
	
	tkinter.Label(tkinter_window, text = "Microscope Tube Length (In Microns)").grid(column = 1, row = 13)# microscope_tube_length_in_microns
	microscope_tube_length_in_microns_widget = tkinter.Entry(tkinter_window)
	microscope_tube_length_in_microns_widget.grid(column = 2, row = 13)
	microscope_tube_length_in_microns_widget.delete(0, "end")
	microscope_tube_length_in_microns_widget.insert(0, microscope_tube_length_in_microns_default)
	
	submit_button = tkinter.Button(tkinter_window, text = "Submit", command = lambda: capture_gui_microscope_values(tkinter_window, num_basis_widget.get(), 
				rho_samples_widget.get(), magnification_widget.get(), numerical_aperture_widget.get(), coverslip_RI_design_value_widget.get(), 
				coverslip_RI_experimental_value_widget.get(), immersion_medium_RI_design_value_widget.get(), immersion_medium_RI_experimental_value_widget.get(), 
				specimen_refractive_index_RI_widget.get(), microns_working_distance_immersion_medium_thickness_design_value_widget.get(), 
				microns_coverslip_thickness_experimental_value_widget.get(), microns_coverslip_thickness_design_value_widget.get(),
				microscope_tube_length_in_microns_widget.get())).grid(column = 1, row = 14)
	
	tkinter_window.mainloop()
	return [cancel, num_basis, rho_samples, magnification, numerical_aperture, coverslip_RI_design_value, coverslip_RI_experimental_value, immersion_medium_RI_design_value, 
				immersion_medium_RI_experimental_value, specimen_refractive_index_RI, microns_working_distance_immersion_medium_thickness_design_value, 
				microns_coverslip_thickness_experimental_value, microns_coverslip_thickness_design_value, microscope_tube_length_in_microns]

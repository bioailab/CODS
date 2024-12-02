May 2023 Aaron Celeste

Documentation: https://munin.uit.no/bitstream/handle/10037/29335/thesis.pdf?sequence=2&isAllowed=y

CODS Installation

Original Development Environment:
Linux Ubuntu 22.04
Python 3.10.6
pip pandas (Version: 1.4.4)
pip scikit-image (Version: 0.19.3)
pip matplotlib (Version: 3.7.1)
pip opencv-python (Version: 4.6.0.66)
pip joblib (Version: 1.2.0)
pip tqdm (Version: 4.64.1)

Optional (Allows Low Memory Usage Mode):
pip h5py (Version: 3.7.0)

Optional (Allows GUI):
sudo apt-get install python3-tk                (tkinter.TkVersion: 8.6)
If getting numpy errors:          pip install numpy==1.21

For Headless Server Environment:
pip opencv-python-headless (Version: 4.7.0.72)    (instead of opencv-python)





NOTES:

Output will be stored in a directory called output in the same location as the CODS code.

To run:
python generator_batch_parallel.py

DEMO (quick run in testing mode without gui just to get some fast imagery output (beware: very reduced image quality)):
python generator_batch_parallel.py -t 1 -mode 0 -pt 0 -c 2688 -pp 0 -ps 0 

Command line argument to disable GUI:
-t 1





WARNINGS:

Be sure to monitor RAM usage. 
Available command line arguments to decrease RAM usage:
-w 1  (writes large variables to disk)
-pt 0  (splits up organelles into different functions)
-pp 0  (Makes sure organelles aren't calculated in parallel)
-ps 0  (Makes sure samples aren't parallel)
-mode 0  (Testing Mode! Dramatically reduces the number of emitter points (not realistic images))
-c 2688  (This can be anything, but 2688 means 128 pixels high and 128 pixels wide video. This is small. This uses less RAM. 128*42=5376 (42 nanometers per pixel) and divided by 2 because the origin is in the center and this -c parameter is actually the max XY value so the canvas stretches from negative 2688 to positive 2688, so 5376 total nanometers wide)
-m 1  (lower number of mitochondria reduces RAM usage)

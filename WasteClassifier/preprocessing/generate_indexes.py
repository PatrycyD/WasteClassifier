import numpy
import sys
import WasteClassifier.config as config

files_in_folder = sys.argv[1]

indexes = numpy.random.randint(1, files_in_folder, config.TEST_PERCENT*files_in_folder)
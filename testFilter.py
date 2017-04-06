# Email Spam Filter - Naive Bayes Implementation Test
# Used to determine the accuracy of the filter

# run as: testFilter.py TRAINING_SIZE NUM_ITER
# ie: python2.7 testFilter.py 10 100

import sys
import subprocess
import numpy as np

TRAINING_SIZE = int(sys.argv[1])
MAX_TRAINING_SIZE = 4326
BUILD_DIR = "build/"
BAYES_FILTER = BUILD_DIR + "spam_filter_results" + str(TRAINING_SIZE) + ".npz"
EMAIL_START_NUM = MAX_TRAINING_SIZE - 1100
NUM_ITER = int(sys.argv[2])

DATA_DIR  = "data/"
SPAM_LABEL_FILE = DATA_DIR + "SPAMTrain.label"

# load the spam labels into array (col 0 = label, col 2 = emailName)
def loadSpamLabel():
	spamLabel = np.loadtxt(SPAM_LABEL_FILE, dtype = {
		"names": ("label", "emailName"),
		"formats": ("i4", "S15",)})

	# limit the size of our training set
	spamLabel = np.take(spamLabel, range(MAX_TRAINING_SIZE))

	return spamLabel

if EMAIL_START_NUM + NUM_ITER > MAX_TRAINING_SIZE:
	print "Max iterations has been set too high"
	exit()


print "Test Emails"
spamLabel = loadSpamLabel()
totalCorrect = 0

# go through each email in the available range
for i in range(EMAIL_START_NUM, EMAIL_START_NUM + NUM_ITER):
	print "data/training/TRAIN_" + format(i, "05") + ".eml:"

	# run filter program
	output = subprocess.check_output("python2.7 spamFilter.py " +
		BAYES_FILTER + " data/training/TRAIN_" + format(i, "05") +
		".eml", shell=True)
	output =  output.rstrip('\n')

	# print results
	if "SPAM" in output:
		print "predicition: SPAM"
	if "HAM" in output:
		print "predicition: HAM"

	if spamLabel[i]["label"] == 0:
		print "actual:      SPAM\n"
	else:
		print "actual:      HAM\n"

	# if the predtiction was the same as the actual increase totalCorrect
	if ("SPAM" in output and spamLabel[i]["label"] == 0) or \
	   ("HAM" in output  and spamLabel[i]["label"] == 1):
		totalCorrect += 1

# output the final result
print "\nTotal Correct:", totalCorrect
print "Percent Correct:", float(totalCorrect)/ NUM_ITER

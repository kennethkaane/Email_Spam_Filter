# Email Spam Filter - Naive Bayes Implementation Trainer

# run as: trainBayes.py TRAINING_SIZE VOCABULARY_LIMIT
# argument TRAINING_SIZE : integer between 1 and 4327
# ie: python2.7 trainBayes.py 10 100

# Note: uses python 2.7
#!/usr/bin/python

import sys
import argparse
import numpy as np

# the number of emails we want to train our spam filter on (max 4327)
# 10  should take ~0.4 seconds
# 100 should take ~44  seconds
# 500 should take ~19  minutes
TRAINING_SIZE = int(sys.argv[1])
MAX_TRAINING_SIZE = 4326 # size of training data set

# limit of the size of bag of words
VOCABULARY_LIMIT = 100
if sys.argv[2] > 1:
	VOCABULARY_LIMIT = int(sys.argv[2])

#data directories
DATA_DIR  = "data/"
TRAIN_DIR = DATA_DIR + "training/"
TEST_DIR  = DATA_DIR + "testing/"

SPAM_LABEL = 0
HAM_LABEL  = 1

# this file contains the labels of the emails
#	 spam = 0, and ham = 1
SPAM_LABEL_FILE = DATA_DIR + "SPAMTrain.label"

BUILD_DIR = "build/"

# the title includes the size of the results
RESULT_FILE = BUILD_DIR + "spam_filter_results" + str(TRAINING_SIZE)

# load the spam labels into array (col 0 = label, col 2 = emailName)
def loadSpamLabel():
	print "Loading spam labels"

	spamLabel = np.loadtxt(SPAM_LABEL_FILE, dtype = {
		"names": ("label", "emailName"),
		"formats": ("i4", "S15",)})

	# limit the size of our training set
	spamLabel = np.take(spamLabel, range(TRAINING_SIZE))

	return spamLabel


# calculate counts for pSpam, and pHam
def calculateLabelCounts(spamLabel):
	# P(spam) = total number of spam labels
	pSpam = np.sum([1 for i in spamLabel if i["label"] == SPAM_LABEL])
	# P(ham) = total number of ham labels
	pHam = np.sum([1 for i in spamLabel if i["label"] == HAM_LABEL])

	return (pSpam, pHam)


# build a vocabulary of available words
def buildVocabulary(spamLabel):
	print "Building vocabulary"

	vocabulary = np.array([])
	frequency  = np.array([])

	# go through the first TRAINING_SIZE emails in the dataset
	for email in spamLabel:
		# open the current email
		iFile = open(TRAIN_DIR + email["emailName"])

		# a list of words in the current email
		for word in iFile.read().split():
			# contains fails if the np.array is empty
			if len(vocabulary) == 0:
				vocabulary = np.append(vocabulary, word)
				frequency  = np.append(frequency, 1)

			# if the word is already in the vocabulary, add one to frequency
			elif vocabulary.__contains__(word):
				frequency[np.where(vocabulary==word)] += 1

			# if the word isn't already in the vocabulary, add it
			else:
				vocabulary = np.append(vocabulary, word)
				frequency  = np.append(frequency, 1)


		# close the input file
		iFile.close()

	return vocabulary, frequency


# determine the probability that a word is in spam, and ham
def getWordProbabilities(spamLabel, spamCount, hamCount, vocabulary):
	# these arrays should be the same shape as the vocabulary, and start at zero
	spamWordCount = np.zeros(len(vocabulary))
	hamWordCount  = np.zeros(len(vocabulary))

	# To find word probabilities we must count the number of occurances of each word
	print "Finding spam, and ham word counts"

	for email in spamLabel:
		# open the current email
		iFile = open(TRAIN_DIR + email["emailName"])

		# form a list of words found in this email (no duplicates)
		emailVocabulary = np.array([])

		# a list of words in the current email
		for word in iFile.read().split():
			# contains fails if the np.array is empty
			if len(emailVocabulary) == 0:
				emailVocabulary = np.append(emailVocabulary, word)
			# if the word isn't already in the vocabulary, add it
			elif not emailVocabulary.__contains__(word):
				emailVocabulary = np.append(emailVocabulary, word)

		# if the email is spam, add the words to spamWordCount
		if email["label"] == SPAM_LABEL:
			for word in emailVocabulary:
				spamWordCount[np.where(vocabulary==word)] += 1
		# if the email is ham, add the words to hamWordCount
		elif email["label"] == HAM_LABEL:
			for word in emailVocabulary:
				hamWordCount[np.where(vocabulary==word)] += 1

		# close the input file
		iFile.close()

	# probability a word is in spam, and ham
	print "Determining spam and ham word probabilities"

	pSpamWord = [wordCount/spamCount for wordCount in spamWordCount]
	pHamWord  = [wordCount/hamCount for wordCount in hamWordCount]

	return (pSpamWord, pHamWord)


# clean up the results to avoid a zero frequency problem
def cleanResults(pSpamWord, pHamWord):
	pSpamWord = [i if (i != 0.0) and
	                  (i != 1.0) else 0.5 for i in pSpamWord]
	pHamWord  = [i if (i != 0.0) and
	                  (i != 1.0) else 0.5 for i in pHamWord]

	return pSpamWord, pHamWord


# run tests to make sure our results make sense
def verifyResults(pSpamWord, pHamWord):
	print "\nRunning tests"
	# check that no probability is greater than one
	pSpamWordCheck = np.sum([1 for i in pSpamWord if i > 1.0])
	if pSpamWordCheck == 0.0:
		print "\tPASSED: pSpamWord Check"
	else:
		print "\tFAILED: pSpamWord Check with value: ", pSpamWordCheck

	pHamWordCheck = np.sum([1 for i in pHamWord if i > 1.0])
	if pHamWordCheck == 0.0:
		print "\tPASSED: pHamWord Check"
	else:
		print "\tFAILED: pHammWord Check with value: ", pHamWordCheck


# save the results to an output file
def saveResults(pSpam, pHam, vocabulary, pSpamWord, pHamWord):
	np.savez(RESULT_FILE, pSpam = pSpam,
	                      pHam = pHam,
	                      vocabulary = vocabulary,
	                      pSpamWord = pSpamWord,
	                      pHamWord = pHamWord)


# load spam labels into array
spamLabel = loadSpamLabel()

# calculate counts of spam labels and ham labels
spamCount, hamCount = calculateLabelCounts(spamLabel)
pSpam = float(spamCount) / TRAINING_SIZE
pHam = float(hamCount) / TRAINING_SIZE

# store vocabulary of available words
vocabulary, vocabularyFrequency = buildVocabulary(spamLabel)

# only except the words with the highest frequency
vocabulary = vocabulary[np.argsort(vocabularyFrequency)[::-1][:VOCABULARY_LIMIT]]

# FOR DEBUGGING
# print vocabularyFrequency[np.argsort(vocabularyFrequency)[::-1][:VOCABULARY_LIMIT]]

# determine the probability of a word is in spam and ham
pSpamWord, pHamWord = getWordProbabilities(spamLabel, spamCount, hamCount, vocabulary)

# clean up the results to avoid a zero frequency problem
pSpamWord, pHamWord = cleanResults(pSpamWord, pHamWord)


# verify and save the results
verifyResults(pSpamWord, pHamWord)
saveResults(pSpam, pHam, vocabulary, pSpamWord, pHamWord)

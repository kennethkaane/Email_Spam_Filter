# Email Spam Filter - Naive Bayes Implementation Filter
# determines if the given email is spam or not, by using the Bayes filter from trainBayes.py

# run as: spamFilter.py BAYES_FILTER EMAIL_NAME
# ie:python2.7 spamFilter.py build/spam_filter_results100.npz data/training/TRAIN_04232.eml


# Note: uses python 2.7

import sys
import numpy as np

DATA_DIR  = "data/"
TRAIN_DIR = DATA_DIR + "training/"
TEST_DIR  = DATA_DIR + "testing/"

BUILD_DIR = "build/"

# input arguments: file location of bayes results
# BAYES_FILTER = BUILD_DIR + "spam_filter_results" + str(10) + ".npz"
# EMAIL_NAME   = TRAIN_DIR + "TRAIN_04326.eml"
BAYES_FILTER = sys.argv[1]
EMAIL_NAME   = sys.argv[2]

# load the filter that was computed in trainBayes.py
def loadFilter(filterName):
	with np.load(filterName) as data:
		return (data["pSpam"],
		        data["pHam"],
		        data["vocabulary"],
		        data["pSpamWord"],
		        data["pHamWord"])

# get vocabulary of input email, that are available in Bayes vocabulary
def getEmailVocabulary(emailName, vocabulary):
	if len(vocabulary) == 0:
		return np.array([])

	emailVocabulary = np.array([])
	iFile = open(emailName)

	for word in iFile.read().split():
		if vocabulary.__contains__(word):
			emailVocabulary = np.append(emailVocabulary, word)

	# close the input file
	iFile.close()

	return emailVocabulary


def getAttributeValue(emailVocabulary, vocabulary):
	attributeValue = np.zeros(len(vocabulary))

	for word in emailVocabulary:
		attributeValue[np.where(vocabulary==word)] = 1

	return attributeValue


# get the P(Email|Spam)
def getPEmailGivenSpam(vocabulary, attributeValue, pSpamWord):
	pEmailGivenSpam = 1.0
	for i in range(len(vocabulary)):
		if attributeValue[i] == 0:
			pEmailGivenSpam *= 1-pSpamWord[i]
		else:
			pEmailGivenSpam *= pSpamWord[i]
	return pEmailGivenSpam


# get the P(Email|Ham)
def getPEmailGivenHam(vocabulary, attributeValue, pHamWord):
	pEmailGivenHam = 1.0
	for i in range(len(vocabulary)):
		if attributeValue[i] == 0:
			pEmailGivenHam *= 1-pHamWord[i]
		else:
			pEmailGivenHam *= pHamWord[i]
	return pEmailGivenHam


(pSpam, pHam, vocabulary, pSpamWord, pHamWord) = loadFilter(BAYES_FILTER) #load the filter result from trainBayes.py
emailVocabulary = getEmailVocabulary(EMAIL_NAME, vocabulary) #get vocabulary of input email
attributeValue = getAttributeValue(emailVocabulary, vocabulary) 

pEmailGivenSpam = getPEmailGivenSpam(vocabulary, attributeValue, pSpamWord) #get the P(Email|Spam)
pEmailGivenHam = getPEmailGivenHam(vocabulary, attributeValue, pHamWord) # get the P(Email|Ham)

#computes and prints probability of input email being Ham and Spam
if (pEmailGivenHam*pHam + pEmailGivenSpam*pSpam) != 0:
	pHamGivenEmail = pEmailGivenHam*pHam / (pEmailGivenHam*pHam + pEmailGivenSpam*pSpam)
	pSpamGivenEmail=1-pHamGivenEmail
	print "Probability of", EMAIL_NAME, "being Ham is", pHamGivenEmail
	print "Probability of", EMAIL_NAME, "being Spam is", pSpamGivenEmail
else:
	print "Error: pEmailGivenHam or pEmailGivenSpam is equal to zero"

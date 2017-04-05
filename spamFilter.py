# AI Final Project
# determines is the given email is spam or not, by using the Bayes filter from trainBayes.py
# Note: uses python 2.7

import numpy as np

DATA_DIR  = "data/"
BUILD_DIR = "build/"
TRAIN_DIR = DATA_DIR + "training/"
TEST_DIR  = DATA_DIR + "testing/"

# take in file location of bayes results
BAYES_FILTER = BUILD_DIR + "spam_filter_results" + str(10) + ".npz"
EMAIL_NAME   = TRAIN_DIR + "TRAIN_04326.eml"

EPSILON = 0.5

# load the filter that was made in trainBayes.py
def loadFilter(filterName):
	with np.load(filterName) as data:
		return (data["pSpam"],
		        data["pHam"],
		        data["vocabulary"],
		        data["pSpamWord"],
		        data["pHamWord"])

# get vocabulary of email, that are available in Bayes vocabulary
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

def getPEmailGivenSpam(vocabulary, attributeValue, pSpamWord):
	pEmailGivenSpam = 1.0
	for i in range(len(vocabulary)):
		if attributeValue[i] == 0:
			pEmailGivenSpam *= 1-pSpamWord[i]
		else:
			pEmailGivenSpam *= pSpamWord[i]
	return pEmailGivenSpam

def getPEmailGivenHam(vocabulary, attributeValue, pHamWord):
	pEmailGivenHam = 1.0
	for i in range(len(vocabulary)):
		if attributeValue[i] == 0:
			pEmailGivenHam *= 1-pHamWord[i]
		else:
			pEmailGivenHam *= pHamWord[i]
	return pEmailGivenHam

(pSpam, pHam, vocabulary, pSpamWord, pHamWord) = loadFilter(BAYES_FILTER)
emailVocabulary = getEmailVocabulary(EMAIL_NAME, vocabulary)
attributeValue = getAttributeValue(emailVocabulary, vocabulary)

pEmailGivenSpam = getPEmailGivenSpam(vocabulary, attributeValue, pSpamWord)
print pEmailGivenSpam
pEmailGivenHam = getPEmailGivenHam(vocabulary, attributeValue, pHamWord)
print pEmailGivenHam

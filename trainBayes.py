# Note: uses python 2.7

import numpy as np

# the number of emails we want to train our spam filter on (max 4327)
TRAINING_SIZE = 10

# this file contains the labels of the emails
#	 spam = 0, and ham = 1
DATA_DIR = "data/"
TRAIN_DIR = DATA_DIR + "training/"
TEST_DIR = DATA_DIR + "testing/"
SPAM_LABEL_FILE = DATA_DIR + "SPAMTrain.label"
SPAM_LABEL = 0
HAM_LABEL  = 1

# load the spam labels into array (col 0 = label, col 2 = emailName)
spamLabel = np.loadtxt(SPAM_LABEL_FILE, dtype = {
	"names": ("label", "emailName"),
	"formats": ("i4", "S15",)})

# limit the size of our training set
spamLabel = np.take(spamLabel, range(TRAINING_SIZE))

# P(spam) = total number of spam labels
pSpam = np.sum([1 for i in spamLabel if i["label"] == SPAM_LABEL])
# P(ham) = total number of ham labels
pHam = np.sum([1 for i in spamLabel if i["label"] == HAM_LABEL])


# build a vocabulary of available words
vocabulary = np.array([])

# go through the first TRAINING_SIZE emails in the dataset
for email in spamLabel:
	# open the current email
	oFile = open(TRAIN_DIR + email["emailName"])

	# a list of words in the current email
	for word in oFile.read().split():
		# contains fails if the np.array is empty
		if len(vocabulary) == 0:
			vocabulary = np.append(vocabulary, word)
		# if the word isn't already in the vocabulary, add it
		elif not vocabulary.__contains__(word):
			vocabulary = np.append(vocabulary, word)

# determine the probability that a word is in spam, and ham
# these arrays should be the same shape as the vocabulary, and start at zero
spamWordCount = np.zeros(len(vocabulary))
hamWordCount  = np.zeros(len(vocabulary))

# To find word probabilities we must count the number of occurances each word
for email in spamLabel:
	# open the current email
	oFile = open(TRAIN_DIR + email["emailName"])

	# form a list of words found in this email (no duplicates)
	emailVocabulary = np.array([])

	# a list of words in the current email
	for word in oFile.read().split():
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

pSpamWord = [i/pSpam for i in spamWordCount]
pHamWord  = [i/pHam for i in hamWordCount]



# DEBUGING
# check that no probability is greater than one
print "pSpamWord check:", np.sum([1 for i in pSpamWord if i > 1.0])
print "pHamWord check:", np.sum([1 for i in pHamWord if i > 1.0])


# DEBUGING
print "spamLabel:", spamLabel
print "pSpam:", pSpam
print "pHam:", pHam
#print "vocabulary:", vocabulary
#print "pSpamWord:", pSpamWord
#print "pHamWord:", pHamWord

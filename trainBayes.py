# Note: uses python 2.7

import numpy as np

# the number of emails we want to train our spam filter on (max 4327)
TRAINING_SIZE = 10

# this file contains the labels of the emails
#	 spam = 0, and ham = 1
SPAM_LABEL_FILE = "data/SPAMTrain.label"
SPAM_LABEL = 0
HAM_LABEL  = 1

# load the spam labels into array (col 0 = label, col 2 = emailName)
spamLabel = np.loadtxt(SPAM_LABEL_FILE, dtype = {
	"names": ("label", "emailName"),
	"formats": ("i4", "S15",)})

# Limit the size of our training set
spamLabel = np.take(spamLabel, range(TRAINING_SIZE))


# P(spam) = total number of spam labels
pSpam = np.sum([1 for i in spamLabel if i["label"] == SPAM_LABEL])
# P(ham) = total number of ham labels
pHam = np.sum([1 for i in spamLabel if i["label"] == HAM_LABEL])

print spamLabel
print pSpam
print pHam

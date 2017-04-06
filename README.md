# Email_Spam_Filter

- **trainBayes.py**: is used to train the spam filter, by using Bayes algorithm, which is done in the following steps:
	1. load the spam labels to know if each training email is ham or spam
	2. count occurrences of spam, and ham in the training set
	3. build a vocabulary of all words in the emails, and their frequencies
	4. take the n most frequent words
	5. determine the probability a word is in spam and ham
		- for example: "deal" might occur in 63 of the 100 spam emails, giving it a probability of 0.63
	6. clean the resulting probabilities of 0.0s and 1.0s, because they will cause a zero-frequency problem later on
	7. verify the results make sense (for example. no probability should be less than zero)
	8. save the results as our spam filter

- **spamFilter.py**: determines if the given email is spam or not, by using the Bayes filter from trainBayes.py, which is done in the following steps:
	1. load the Bayes filter
	2. get the vocabulary of the current email
	3. convert that into an attribute list in the Bayes vocabulary
	4. calculate P(Email|Spam), and P(Email|Ham)
	5. use Baye's theorem to find p(Spam|Email)

- **testFilter.py**: runs the spamFilter an given number of times to determine its accuracy.
	- currently this is at 88%


## Demo
``` bash
# train the Bayes filter
python2.7 trainBayes.py 50 100

# check if a given email is spam
python2.7 spamFilter.py build/spam_filter_results50.npz data/training/TRAIN_04232.eml

# determine the accuracy of the filter
python2.7 testFilter.py 50 100
```

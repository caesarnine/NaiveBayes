# import the required libs
import pandas as pd
from collections import Counter
import math

# read in the tweets
apptweets = pd.read_csv('apptweets.csv')
othertweets = pd.read_csv('othertweets.csv')

# drop any empty rows that are read in
apptweets = apptweets.dropna()
othertweets = othertweets.dropna()

# a function to clean up the tweets up a bit, by making them lowercase and replacing common punctuation with spaces
def cleantext(string):
    string = string.lower()
    string = string.replace(". ", " ")
    string = string.replace(": ", " ")
    string = string.replace("?"," ")
    string = string.replace("!"," ")
    string = string.replace(";", " ")
    string = string.replace(",", " ")
    return string

# apply the function to all the tweets using list comphrehension
apptweets['Tweet'] = [cleantext(i) for i in apptweets['Tweet']]
othertweets['Tweet'] = [cleantext(i) for i in othertweets['Tweet']]

# create two Counter objects to keep track of the number of times any token shows up in each dataset
appcnt = Counter()
othercnt = Counter()

# iterate through both dataframes, splitting the Tweet into separate words, and using the Counter objects to sum up the
# frequencies of the tokens
for index, row in apptweets.iterrows():
    for word in row.Tweet.split(' '):
        appcnt[word] += 1

for index, row in othertweets.iterrows():
    for word in row.Tweet.split(' '):
        othercnt[word] += 1

# copy the two previous Counter objects; this allows us to change them while iterating through the originals
appcntclean = appcnt.copy()
othercntclean = othercnt.copy()

# delete any tokens that are less than 4 characters in length
for i in appcnt:
    if len(i) <= 3:
        del appcntclean[i]
        
for i in othercnt:
    if len(i) <= 3:
        del othercntclean[i]

# add 1 to every token count, this is Additive Smoothing (Laplace Smoothing)
# this ensures that later on, if we encounter a word, we can act like we saw it once before
# so we add one to every other token to account for that, and give them more weight
for i in appcntclean:
    appcntclean[i] = appcnt[i] + 1
    
for i in othercntclean:
    othercntclean[i] = othercnt[i] + 1

# create two dict objects to contain the probabilities of the words given that they are an app tweet or other tweet
probapp = {}
probother = {}

# add in the logged probabilities into the dict for each respective dataset, the probability being:
# (count of how many times we've seen this token)/(count of all the times we've seen all tokens)
# we log them so we don't run into the floating point underflow error when trying to multiply together many small probablities
# instead by logging them we can just add them since log(a*b) = log(a) + log(b)
for i in appcntclean:
    probapp[i] = math.log(float(appcntclean[i])/sum(appcntclean.values()))

for i in othercntclean:
    probother[i] = math.log(float(othercntclean[i])/sum(othercntclean.values()))

# read in the test dataset that we'll validate against, and clean the tweets using the function we generatd before
testtweets = pd.read_csv('testtweets.csv')
testtweets['Tweet'] = [cleantext(i) for i in testtweets['Tweet']]

# create an empty list to hold the predicted values for the test tweets
results = []

# interate through every row in the test tweets dataframe
for index, row in testtweets.iterrows():
    # to hold the summation of the logged probabilities
    appsum = 0
    othersum = 0
    # tokenize the tweets, and iterate through each word
    for word in row.Tweet.split(' '):
        # ignore any token that's shorter than 4 characters
        if len(word) <= 3:
            appsum += 0
            othersum += 0
        # otherwise, look up the word's probability of being an app or an other tweet, and add it up
        # if we've never seen the word before, just assume that we've seen it once before, and add that logged probability to the sum
        else:
            if word in probapp:
                appsum += probapp[word]
            else:
                appsum += math.log(float(1)/sum(appcntclean.values()))
                
            if word in probother:
                othersum += probother[word]
            else:
                othersum += math.log(float(1)/sum(othercntclean.values()))
    # if the sum of the logged probabilities for app is greater than other, than we'll predict it's an App
    if appsum > othersum:
        results.append('App')
    # otherwise we'll predict that it's an Other
    else:
        results.append('Other')

# append the results onto the dataframe
testtweets['Prediction'] = results

# view the dataframe to see how we did, we predicted every class class correctly except for one
testtweets

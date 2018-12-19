# -*- coding: utf-8 -*-

import re


def word_count(input_txt, pattern):
    
def word_count(input_txt, pattern):
    counts = dict()
    
    r= re.findall(pattern,input_txt)
    s = ' '.join(r)
    regexp = re.compile(r'@[\w]+')
    if regexp.search(s):
        if s in counts:
            counts[s] += 1
        else:
            counts[s] = 1
    return counts


#t1 = remove_pattern(['@manohar keep going don''t stop','asd as @madmax belive! that you\'s next'], '.*@.*')
#t1 = word_count([
#'@manohar keep going don''t stop','asd as @madmax @belive! that you\'s next'
#,'manohar is working in natural language processing',
#'asda dsdaae efasfadad ad asdsadasd@TerrenceJ106  U should have told her and that \
#voice box gone b bad 2 if ur ass keep on smokin them \
#cancer sticks  lol',' @HollywoodHansM lol like how gud kobe iz   lol and \
#its gunn be the lakers  amp  cavs in the finals ',' @alanasachi hahaha aww I wish I could \
#come but I have work  boooo'], '@[\w]+')

t1 = word_count('@Rove1974 @Pink is our hero too  Love her  Wanna go see her in Brisvegas again but sadly my bank balance wont let me','@[\w]+')


word = 'fubar'
regexp = re.compile(r'@[\w]+')
if regexp.search(word):
    print ('matched')
else:
    print('not matched')
    
    
    
    
    
    
    
    
    
# Twitter Sentiment Analysis using nltk
'''
   1) Tweets Preprocessing 
          i) Removing Punctuations, Numbers, and Special Characters
         ii) Removing Short Words
        iii) Tokenization
         iv) Stemming

   2) Findout Insights
          i) Get Maximum tweets done by user 
         ii) Get Maximum Hashtag of fields
        iii) common words used in the tweets
         iv) Explore 0 Polarity tweets 
          v) Explore 4 Polarity tweets 
'''

# Load the Libraries 
import re                   # Pattern finding using Regular Expressions
import pandas as pd         # Data Manipulation 
import numpy as np          # Numerical computation for arrays
import matplotlib.pyplot as plt # simple data visualization tools 
import seaborn as sns           # highly intutively and more supported visualization tools
import nltk                     # dictionary used for data preprocessing eg,. Parts Of Speech
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)


# read csv file using pandas dataframe
twitter_dataset  = pd.read_csv('twitter_data.csv',sep='"""')   
twitter_dataset['id;'] = twitter_dataset['Unnamed: 0'][22]   

# Remove unwanted column from dataset 
del twitter_dataset['";']
del twitter_dataset[',,,,,,,"']
del twitter_dataset['Unnamed: 0']
del twitter_dataset['Unnamed: 5']

# Rename columns in dataframe
twitter_dataset.columns = ['id','polarity','tweets']


# Remove twitter handles for vectorization we don't need special character 
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt  

#twitter_dataset['tidy_tweets'] = np.vectorize(remove_pattern)(twitter_dataset['tweets'], "@[\w]*")


# Remove special characters, numbers, punctuations
twitter_dataset['id'] = twitter_dataset['id'].str.replace("[^0-9]", " ")
twitter_dataset['polarity'] = twitter_dataset['polarity'].str.replace("[^0-9]", " ")
# Exclude tweets handle and #tag for Data Exploration 
twitter_dataset['tidy_tweets'] = twitter_dataset['tweets'].str.replace("[^a-zA-Z0-9_#@]", " ")

               
# Before Stemming we have to get all twitter handles and hashtag 
'''
      find the count of each twitter handle    
      eg,.  ['@Elonmusk Tesla going to change remarkable ', '@Elonmusk  Telsa new S series Model'] ---> {'@Elonmusk':2}
''' 
def word_count(input_txt, pattern):
    print(type(input_txt))
    r= re.findall(pattern,input_txt)
    s = ' '.join(r)
    regexp = re.compile(pattern)
    if regexp.search(s):
        if s in counts:
           counts[s] += 1
        else:
           counts[s] = 1       
    return counts


# Get Maximum tweets done by user             
tweets_handle = '@[\w]+'
counts = dict()
for i in range(len(twitter_dataset['tidy_tweets'])):
    word_count(twitter_dataset['tidy_tweets'][i],tweets_handle)

max_tweets_handle = max(counts.keys(), key=(lambda k: counts[k]))
counts[max_tweets_handle]
print('Maximum number of tweets done by {} and frequency is {}'.format(max_tweets_handle,counts[max_tweets_handle]))


# Get Maximum Hashtag of fields
hashtag = '#[\w]+'
counts = dict()
for i in range(len(twitter_dataset['tidy_tweets'])):
    word_count(twitter_dataset['tidy_tweets'][i],hashtag)

max_hashtag = max(counts.keys(), key=(lambda k: counts[k]))
print('Maximum number of #tag appears for the product {} and frequency is {}'.format(max_hashtag,counts[max_hashtag]))


# Now remove tweets handle 
twitter_dataset['tidy_tweets'] = np.vectorize(remove_pattern)(twitter_dataset['tweets'], "@[\w]*")


while {} in result:
    result.remove({})


# Removing Short Words if word less than 4   eg. hmm, yup, lol
twitter_dataset['tidy_tweets'] = twitter_dataset['tidy_tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
twitter_dataset.head()

# Tokenization - for given sentence split into words  eg. ['good working night until'] --> good, working, night, until
tokenized_tweet = twitter_dataset['tidy_tweets'].apply(lambda x: x.split())
tokenized_tweet.head()


# Stemming - Removing the suffixes eg. “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”.
from nltk.stem.porter import *
# Creating stemmer obj
stemmer = PorterStemmer()   

stemmed_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
stemmed_tweet.head()

# Integrate stemmed tokens  into our twitter_dataset dataframe
for i in range(len(stemmed_tweet)):
    stemmed_tweet[i] = ' '.join(stemmed_tweet[i])

twitter_dataset['tidy_tweets'] = stemmed_tweet


''' 
    Find out meaningful insights in tweets
        1) common words used in the tweets using WordCloud
        2) Explore 0 Polarity words 
        3) Explore 4 Polarity words 
'''

# 1) Visualize common words used in the tweets using WordCloud
all_words = ' '.join([text for text in twitter_dataset['tidy_tweets']])
from wordcloud import WordCloud
wordcloud = WordCloud(background_color = "white", width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# 2) Visualize O polarity from the given dataset
_0_polarity =' '.join([text for text in twitter_dataset['tidy_tweets'][twitter_dataset['polarity'].astype(str).astype(int) == 0]])

wordcloud = WordCloud(background_color = "white", width=800, height=500, random_state=21, max_font_size=110).generate(_0_polarity)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# 3) Visualize 4 polarity from the given dataset 
_4_polarity =' '.join([text for text in twitter_dataset['tidy_tweets'][twitter_dataset['polarity'].astype(str).astype(int) == 4]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(_4_polarity)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# 2) impact of Hashtags on tweets sentiment
# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# Extracting hashtags from 0 Polarity Tweets
HT_regular = hashtag_extract(twitter_dataset['tidy_tweets'][twitter_dataset['polarity'].astype(str).astype(int) == 0])

# Extracting hashtags from 4 Polarity Tweets
HT_games = hashtag_extract(twitter_dataset['tidy_tweets'][twitter_dataset['polarity'].astype(str).astype(int) == 4])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_games = sum(HT_games,[])



a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


b = nltk.FreqDist(HT_games)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()



    

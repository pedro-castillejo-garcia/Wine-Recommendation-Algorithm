"Project: Wine Recommendation System"
"Author: Pedro Castillejo Garcia"
"Created on: June 2023"

"""
Description: This is a recommendation system which helps buyers to find wine bottles from a data set to save their time and money. The python algorithms build co-occurrence matrix and based on different parameters such as wine description, country and province of origin, variety, reviews (in a scale from 0-100 points) and price segment, the most similar products are returned from the database as recommendation.
"""


# Loading basic packages

import pandas as pd
import numpy as np
import matplotlib as plt
pd.set_option("display.max.columns", None)

# Load the data set winemag-data-130k-v2.csv
data = pd.read_csv('winemag-data-130k-v2.csv', low_memory = False)
data.head()

# We have some "nan" string values, those are cells that are empty in our table, let's remove them and copy the columns which will be needed to build the engine

wine = data[['title', 'country', 'description', 'points', 'price', 'province', 'variety', 'winery']]
wine = wine.query("title != 'NaN' and country != 'NaN' and description != 'NaN' and points != 'NaN' and price != 'NaN' and province != 'NaN' and variety != 'NaN' and winery != 'NaN'")
wine = wine.dropna()

# Now sort the wines according to their points (with most points on top)

wine = wine.sort_values('points', ascending=False)
wine.head()
wine.shape

# We have 120915 observations in 8 columns which is too much to calculate cosine similarity on any normal computer so let's use only part of the original database. 

wine['points'].describe()
quantile = wine["points"].quantile(0.90)
wine = wine.copy().loc[wine["points"] >= quantile]
wine.shape

# Now a 10% of the original database is remainining (As we are using a quantile of 0.9. That means that the rest up to 1 is what we will be using, that is 0.1, then a 10%, which in this case is around 20500 wines. This amount of data should be enough to prepare reliable recommendation engine. However, if you have more powerful computer, feel free to change the used quantile.

# Let's build the first engine which will be based on description of the wines bottles so we put that column the first one in the table

wine["description"].head()

# Import class from scikitlearn

from sklearn.feature_extraction.text import TfidfVectorizer

# To increase accuracy, we remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(wine['description'])

# Import class from scikitlearn

from sklearn.metrics.pairwise import linear_kernel

#The code from below will create the cosine similarity matrix which be needed to build the engine.
# If you get memory error, increase the quantile value used to divide dataframe.

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(wine.index, index=wine['title']).drop_duplicates()

def wine_recommendations(title, cosine_simil=cosine_simil):
    
    # Fit index of the wine to each wine title
    index = indices[title]
    
    # Calculate the similarity score between wine which you selected and the others wines in database
    simil_scores = list(enumerate(cosine_simil[index]))
    
    # Sort the results by the similarity scores
    simil_scores = sorted(simil_scores, key=lambda x: x[1], reverse=True)
    
    # Show the top 5 results with the highest similarity score
    simil_scores = simil_scores[1:6]

    # Extract indices of the recommended wines
    wine_indices = [i[0] for i in simil_scores]

    # Create new dataframe
    recom1 = wine[['title', 'country', 'description', 'points', 'price', 'province', 'variety', 'winery']].iloc[wine_indices]

    # Combine selected wine and recommendations in this new dataframe
    frames = [wine[wine["title"] == title], recom1]
    recommendation = pd.concat(frames, keys=['x', 'y'])
    
    # Return the result recommendation
    return recommendation

# To get recommendation, you have to use wine title which index is not higher than size of wine dataframe (for quantile = 90 the max index is 20461)

Recommendation = wine_recommendations('Château Mouton Rothschild 2009  Pauillac')
Recommendation

# I don't think these are the worthful recommendations. Let's try to fix it and create new column with metadata of wine bottles

# Before that, we have to change price and points to strings for easier categorising

wine['price'] = wine['price'].astype(str) 
wine['points'] = wine['points'].astype(str)

# And remove duplicates in titles

wine['title'].value_counts()
wine[wine["title"] == "Charles Heidsieck NV Brut Réserve  (Champagne)"] 

# We have to be careful because some of the titles are the two or more different wine types with different prices or points

# The best way to do it is add to metadata title of the wine and the rest of the variables 

def metadata(x):
    return ''.join(x['country']) + '' + '' .join(x['title']) + ''.join(x['points']) + ' ' + x['price'] + ' ' + ''.join(x['province'] + ' ' + x['variety'])

wine['metadata'] = wine.apply(metadata, axis=1)
wine['metadata'].value_counts() 

# As expected, we have a couple of duplicates of not only wine title but also its price, points, etc. which have to be removed from the dataset

wine = wine.drop_duplicates('metadata')
wine['metadata'].value_counts()

# We can remove the title from metadata column

def metadata(x):
    return ''.join(x['country']) + ' ' + ''.join(x['points']) + ' ' + x['price'] + ' ' + ''.join(x['province'] + ' ' + x['variety'])


wine['metadata'] = wine.apply(metadata, axis=1)
wine['metadata'].head() 

# And we get all the information in one column only

from sklearn.feature_extraction.text import CountVectorizer

# Once again we will remove from the column all english stop words and repeat previous steps to create the cosine similarity matrix

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(wine['metadata']) 

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix, count_matrix)

wine = wine.reset_index()
indices = pd.Series(wine.index, index=wine['title'])
wine = wine[['title', 'country', 'description', 'points', 'price', 'province', 'variety', 'winery', 'metadata']]

# We were based on different similary matrix so we have to rebuild our recommendation's def

def wine_recommendations(title, cosine_sim=cosine_sim):
    
    # Fit index of the wine to the title
    idx = indices[title]
    
    # Calculate the similarity score between wine which you selected and the others wines in database
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the results by the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Show the top 5 results with the highest similarity score
    sim_scores = sim_scores[1:6]

    # Extract indices of the recommended wines
    wine_indices = [i[0] for i in sim_scores]

    # Create new dataframe
    recommendation = wine[['title', 'country', 'description', 'points', 'price', 'province', 'variety', 'winery', 'metadata']].iloc[wine_indices]

    # Combine selected wine and recommendations in this new dataframe
    frames = [wine[wine["title"] == title], recommendation]
    recommendation = pd.concat(frames, keys=['x', 'y'])
    
    # Return the result recommendation
    return recommendation

Recommendation = wine_recommendations('Château Mouton Rothschild 2009  Pauillac')
Recommendation

# Ok, as expected, we have much better recommendations but I think they are too accurate. It's works more like showing the substitute of the wines rather than showing similar wines. We can try to use less metadata and add price segment instead of the actual price.

# Price segment - What is it? : Instead of having dataframe prices such as 30.00 USD or 31.00 USD, I prefer to describe it as one of the segment, in this case it would be "Low-Price".


# Before we create price segments, we have to change type of price from string to float, as we will need the numerical values to calculate the segments

wine['price'] = wine['price'].astype(float)
wine["price"].dtype


wine["price"].plot(kind='hist')

# As seen on the graph we have a lot of "cheap" wine bottles (less than 250 $) and only a few of prestige. Let's create 8 price segments.

pd.qcut(wine["price"], q=8)

# Above code suggests us a range of prices for each segment. It would work if we want to have the same number of observations in every segment but we don't so let's modify it a little but based on it

""" 
Therefore we will be using these price ranges to define the segments:
    - Low-Price [0 - 30.0]
    - Reasonable (30.0 - 55.0]
    - Standard (55.0 - 80.0]
    - High-Price (80.0 - 120.0]
    - Premium (120.0 - 250.0]
    - Super Premium (250.0 - 500.0]
    - Prestige (500.0 - 1000.00]
    - Super Prestige (1000.0 - 2500.0]
"""

# We will work on copy of price column

wine["price segment"] = wine["price"]

# Loop to sort each wine to its right price segment

segment = []

for row in wine["price segment"]:
    if row < 30:
        segment.append('LowPrice')
    elif row < 55:
        segment.append('Reasonable')
    elif row < 80:
        segment.append('Standard')
    elif row < 120:
        segment.append('High-Price')
    elif row < 250:
        segment.append('Premium')
    elif row < 500:
        segment.append('SuperPremium')
    elif row < 1000:
        segment.append('Prestige')
    elif row >= 1000:
        segment.append('SuperPrestige')
    else:
        segment.append('Error')
        
wine['price segment'] = segment

# Here we can see how many wines we have for each of the segments we set

wine['price segment'].value_counts()

# We have to rebuild our metadata and we will limit it to the most important variables to have more varied recommendations.

def metadata(x):
    return ''.join(x['points']) + ' ' + x['price segment'] + ' ' + x['variety']

wine['metadata'] = wine.apply(metadata, axis=1)

# Once again we have to create the new similarity matrix as done in previous steps

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(wine['metadata']) 

cosine_sim = cosine_similarity(count_matrix, count_matrix)

wine = wine.reset_index()
indices = pd.Series(wine.index, index=wine['title'])

wine = wine[['title', 'country', 'description', 'points', 'price', 'price segment', 'province', 'variety', 'winery']]

def wine_recommendations(title, cosine_sim=cosine_sim):
    
    # Fit index of the wine to the title
    idx = indices[title]
    
    # Calculate the similarity score between wine which you selected and the others wines in database
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the results by the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Show the top 10 results with the highest similarity score
    sim_scores = sim_scores[1:10]

    # Extract indices of the recommended wines
    wine_indices = [i[0] for i in sim_scores]

    # Create new dataframe
    recommendation = wine[['title', 'country', 'description', 'points', 'price', 'price segment', 'province', 'variety', 'winery']].iloc[wine_indices]

    # Combine selected wine and recommendations in this new dataframe
    frames = [wine[wine["title"] == title], recommendation]
    recommendation = pd.concat(frames, keys=['x', 'y'])
    
    # Return the result recommendation
    return recommendation

Recommendation = wine_recommendations('Château Mouton Rothschild 2009  Pauillac')
Recommendation


"""
Now these recommendations which we created are better. As you see, "Château Mouton Rothschild 2009 Pauillac" has a lot of positive reviews (96 points) but it is a really expensive wine (1300 USD per bottle) and not so many people could afford to buy it. Thanks to the latest version of the engine, we found alternatives which are "Portfolio 2013 Limited Edition Red (Napa Valley)" for 155 USD per bottle and "Château Mouton Rothschild 2014 Pauillac" for 400 USD per bottle. So before you buy the "Château Mouton Rothschild 2009 Pauillac" for 1300 USD, take a look on the same wine from 2010 which has the best reviews (100 points!) and the price is "only" 200 USD higher.
"""
# Machine Learning & Data Science

This repository contains all the projects/challenges that I have done when learning Machine Learning and Data Science.

## Table of Contents
- [AirBnb Recommender Chatbot](#airbnb-recommender-chatbot)
    - [Sameple Chatbot Conversation](#sample-front-end-chatbot-conversation)
    - [AirBnb Chatbot Program Flow](#airbnb-recommender-chatbot-program-flow)
    - [Natural Language Processing](#natural-language-processing)
    - [Sentiment Analysis on reviews](#sentiment-analysis-on-reviews)
    - [Topic Analysis on reviews](#topic-analysis-on-reviews-of-recommended-airbnb-listing)
- [Kaggle - Titanic Classification Model](#kaggle---titanic)
- [Kaggle - Ames Housing Regression Model](#kaggle---ames-housing)

## AirBnb Recommender Chatbot

An AirBnb recommender chatbot that is powered by DialogFlow for a very natural front-end conversation experience, while picking up intents and entities from user's sentences.

The backend recommender engine make use of Natural Language Processing to generate TF-IDF vectors and cosine similarity of user inputs with the rest of AirBnb dataset to recommend the 2 AirBnb listings that is most similar to user's description.

The backend recommender engine also performs Sentiment Analysis on thousands of reviews of the recommended listings to generate mean scores for user's reference. Furthermore, Topic Analysis is conducted on the reviews, in order for user to quickly grasp the main topics of reviews and make an informed booking decision.

### Sample Front-end Chatbot Conversation

![](AirBnb%20Recommender%20Chatbot/screenshots/conversation1.png)
![](AirBnb%20Recommender%20Chatbot/screenshots/conversation3.png)
![](AirBnb%20Recommender%20Chatbot/screenshots/conversation2.png)

### AirBnb Recommender Chatbot Program Flow

- Using DialogFlow chatbot to first ask for user's preferences for filter columns e.g. budget (price), property_type, room_type) whose values will then be used to filter out AirBnBs listings that do not meet the hard requirement.
- Then, use the chatbot to ask what are users' expectations of the AirBnb they want to stay in. E.g. host personality, ambience, interior design, location, neighbourhood, ...
- Concatenate the potential description columns to produce a document for each Airbnb listing
- Natural Language processing of the document of each Airbnb
- Based on user's description, insert it into the Airbnb listing dataframe, act as if it is a listing itself
- Generate the TF-IDF vectors for the whole Airbnb listing dataframe, which considers the importance of words.
- Compute the cosine similarity of user description's vector array with all the other listing vector arrays
- Recommend the top 2 similar AirBnb listing to user

### Natural Language Processing

- Visualise text columns using WordCloud.
- Text preprocessing (tokenising, stemming, removing stop words & punctuation)
- Generate TF-IDF vectors for each AirBnb listing, while taking into account of all listings in the dataframe
- After generating the TF-IDF vectors of every Airbnb listing, this is equvalient to representing each listing in a multi-dimensional space with each dimension being a word.
- Generate the cosine similarity matrix for every AirBnb listing vector against every other listing vector and recommend the two most similar AirBnb listing vectors compared to user input's listing vector in this multi-dimensional space. 

### Sentiment Analysis on Reviews

- To generate a mean score for each AirBnb listing, instead of using the score in original dataset which is highly skewed.
- Data cleaning of reviews dataset to remove very short reviews which are not very helpful
- Language detection of reviews to remove non-english reviews
- VADER library is used for valence-based sentiment analysis where it is able to differentiate postive and negative words while capturing the intensity of the words to be taken into account when generating a score for a review. For example, the word ‘excellent’ would be treated as more positive than ‘good’.
- So if an AirBnb listing has a higher mean sentiment score, means its reviews are generally more postive and reviewers are more happy with the stay.

### Sample User Input and Recommended Listings

Sample User Inputs captured from DialogFlow:
![Sample User Inputs](AirBnb%20Recommender%20Chatbot/screenshots/sample-userinput.png)

Sample Recommended AirBnb listings from Engine:
![](AirBnb%20Recommender%20Chatbot/screenshots/sample-recommendedlistings.png)

Same Recommended AirBnb listings' details:
![](AirBnb%20Recommender%20Chatbot/screenshots/sample-recommendedlistings-details.png)

- As we can see our recommendations's "description", "neighborhood_overview", "transit", "host_about" is similar to our user's description of his or her dream AirBnB
- Besides that, both the "review_scores_rating" and the "review_senti_score" of the 1st listing outscored the 2nd recommended listing, meaning the first 1 is probably a better choice.

### Topic Analysis on Reviews of Recommended AirBnb Listing

- The recommended AirBnb listings might have hundreds of reviews which will be cumbersome for users to read through.
- Topic analysis is then performed on the recommended AirBnb listings' reviews to allow the users to quickly identify the main topics | features | highlights present in the reviews. Definitely helpful for customers to make an more informative choice.
- The topics identified from the reviews are displayed together with sample reviews they appear in, so it is easier for user to understand the topic without losing the context.
- Sample Topics identified from recommended listings's reviews:
    ![](AirBnb%20Recommender%20Chatbot/screenshots/sample-topicanalysis.png)


## Kaggle - Titanic 

A classification model built to predict the survivability of a passenger on Titanic as part of Kaggle online ML competition.

```bash
Kaggle-Titanic (Classification)/Titanic.ipynb
```

In terms of Data Science, example of:
- Exploratory Data Analysis
- Data Cleaning (e.g. input missing values)
- Feature Engineering (e.g. encoding, binning, scaling)

In terms of Machine Learning, example of:
- Classification Models (e.g. randomforest, KNeighbors, Gaussian, SVC)
- Model Evaluation (e.g. cross-validation, confusion matrix) 
- Model Tuning (e.g. hyperparameters)

> Model's score in Kaggle online ML competition: 82% accuracy.
![](Kaggle-Titanic%20(Classification)/Titanic_Kaggle_Score.JPG)

## Kaggle - AMES Housing

A regression model built to predict the sale price of given houses in Ames Iowa, USA as part of Kaggle online ML competition.

```bash
Kaggle-Ames Housing (Regression)/Ames Housing Regression Model1.ipynb
Kaggle-Ames Housing (Regression)/Ames Housing Regression Model2.ipynb
```

In terms of Data Science, example of:
- Correlation between variables and target variable (e.g. select useful variables from 79 variables using Microsoft Azure and seaborn's heat map)
- Exploratory Data Analysis
- Data Cleaning (e.g. removing outliers, skewness, normal distribution)
- Numeric Variable & Categorical Variable (ordinal and nominal categorical variable)
- Feature Engineering (e.g. create a lot of new features, normalisation via log transformation)

In terms of Machine Learning, example of:
- Regression Models (e.g. linear regression, non-linear regression, polynomial regression)
- Regularization (e.g. auto feature selection and best for dataset with many features, no need EDA to select your features, can feature engineer even more features and let regularization models select the features themselves)
- Model Evaluation (e.g. mean square error (MSE))

> Model's score in Kaggle online ML competition: 0.117 MSE.
![](Kaggle-Ames%20Housing%20(Regression)/Housing_V2_Kaggle_Score.JPG)
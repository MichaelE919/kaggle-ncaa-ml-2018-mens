
# NCAA March Madness (MoneyBall edition)

## Description
This is the code I wrote and used to enter a submission into [Kaggle's 2018 Men's NCAA Machine Learning competition](https://www.kaggle.com/c/mens-machine-learning-competition-2018).
#### This notebook will cover the complete process of building a classifier machine learning model to predict the win probability of each matchup in the 2018 NCAA March Madness Tournament.
### There are four major milestones:
1. Create training and test datasets
2. Create a machine learning model and train using the training set
3. Test the model using the test sets and create a submission file for Stage 1 of the Kaggle competition
4. Update datasets with 2018 data and create predictions for the 2018 NCAA March Madness Tournament

### Part 1: Creating the training/test set
#### The dataset is generated from the following features:
* [The Four Factors](https://www.nbastuffer.com/analytics101/four-factors/)
* [Player Impact Estimate (PIE)](https://masseybasketball.blogspot.com/2013/07/player-impact-estimate.html)
* [Adjusted Offensive Efficiency](https://cbbstatshelp.com/efficiency/adjusted-efficiency/)
* [Adjusted Defensive Efficiency](https://cbbstatshelp.com/efficiency/adjusted-efficiency/)
* [Adjusted Efficiency Margin](https://cbbstatshelp.com/ratings/adjem/)
* [Defensive Rebounding Percentage](https://www.nbastuffer.com/analytics101/defensive-rebounding-percentage/)
* Offensive Rebound to Turnover Margin
* [Assist Ratio](https://www.nbastuffer.com/analytics101/assist-ratio/)
* Free Throw Percentage
* Score Differential
* [Rating Percentage Index (RPI)](https://en.wikipedia.org/wiki/Rating_Percentage_Index)
* Tournament Seed
* Win Percentage


### Part 2: Create and train the machine learning model
#### Workflow:
* Create training set of data from 2013 and prior
* 2014-2017 data will be a true test set
* Split training set into a separate training and test set
* Initiate classifiers, create parameter and pipeline objects
* Use best performing classifer and fit with full training set
* Create data to input into the model
* Create predictions

## Disclaimer
The majority of the code in this repository was used to create the Kaggle submission; however, for my features, I used offensive and defensive efficiency numbers that were *not* adjusted for competition and I did not have the Offensive Rebound to Turnover Margin. While my models had a decent accuracy, they did a poor job of predicting upsets. After the competition closed and I watched more and more upsets that my models did not correctly predict, I replaced the offensive and defensive efficiency numbers with the adjusted versions from [kenpom.com](https://kenpom.com/) and added the offensive rebound to turnover margin, after some additional reading.
Not only did I correctly "predict" some of the upsets, the accuracy of my models increased.

# Project_McNulty - Classifying Stackoverflow Questions

**Code can be found in the classification folder**

**Presentation slides can be found in the presentation folder**

## How can you get your stackoverflow questions answered quickly?
The modeling process, which is found in the jupyter notebook in the classification folder, goes in-depth into how a stackoverflow can be classified.  A question that is able to be answered _"quickly"_ is one that has an answer within 30 minutes of posting.  A question that is able to be answered _"slowly"_ is one that has an answer after 30 minutes of posting.

## Brief overview of the modeling process
The modeling process began with sentiment analysis of the question title and answer.  The features were fed into a variety of models including Logistic Regression, Gradient Descent, and Naive Bayes.  Additionally, the question title and body were vectorized by finding the **term frequency–inverse document frequency (tf-idf)**.  The resulting dataframe was a very sparse matrix. Therefore, the Multinomial Naive Bayes model was used to fit the sparse matrix and predict the classes.  This model performed fairly well with a precision score of ~0.67. 

The final step was to use an ensemble method called **stacking**.  The Multinomial Naive Bayes model and a Random Forest model were stacked together so that their individual predictions became features for the final stacked model.  The stacked model was a Logistic Regression model and the precision score went up to ~0.70.  

## Takeaways
Stacking helps, but the complexity and computation time offsets the benefit in this case. Classifying text is difficult because the vectorized matrix is very sparse so most models other than Naive Bayes will not work very well.

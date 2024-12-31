# Project Journal

## Context
We had access to a training set and a test set from an insurance company. These dataframes had 20 columns listing the characteristics of the insurance accounts of different customers. These characteristics were, for example, the number of dependants, the health score, or even whether the person holding the account smoked or not. The aim was to predict the ‘Premium Amount’, the price paid by each customer in the ‘test’ set, based on the training set.

## Initial Approach
I first tried to run a linear regression on different columns, using the `statsmodels.api` models. I was able to test different sets of parameters, which enabled me to use the p-values to see which columns played an important role in the value of the Premium Amount. Unfortunately, the results were not very conclusive (which is logical given that the relationship is not linear). However, initially, this allowed me to run regressions only on the columns I had identified. This considerably reduced the calculation time.

## Data Cleaning
In the first part, I tried to carry out tests to determine how to clean my data properly and which models to use. I filled in the NaNs in the test set with `0`s at the beginning and removed those in the training set.

## Model Development
### Initial Models
As I had to run a regression, I opted for classic regression models, starting with RandomForest with fixed parameters. My first instinct was to visualize the distribution of the data and separate the columns into two categories:
- Those with objects (e.g., type of property, address).
- Those with numbers (e.g., health score, annual income).

As I couldn't use the object columns as they were, I used one-hot encoding with `LabelEncoder` from `sklearn.preprocessing`. The only exception was the column with the start dates of the insurance contracts, which I translated into seconds to quantify it.

I split the training set into a training part and a validation part (80-20 split) to evaluate the MSE and R2 of the model and see if my changes improved the results.

### Improvements
- I normalized my data using `StandardScaler` from `sklearn.preprocessing`. This reduced calculation time and improved accuracy.
- I decided to fill the NaNs in the test file with the median of the column values. I did the same for the training set, to train on cleaned data like the test data. The training R2 and MSE were reduced, but the Kaggle score for my prediction improved, showing that this approach was effective.

## Advanced Models
- I explored better regression models and found that `HistGradientBoostingRegressor` worked well. I conducted a grid search to try different parameters and find the best set of hyperparameters.

## Adjusting for the Kaggle Metric
At this point, I realized that the evaluation metric for the Kaggle competition was RMSLE, whereas my model was training to improve R2. I applied the `log(x+1)` function to the ‘Premium Amount’ column and changed the model's scoring parameter to `neg_root_mean_squared_error`. This led to a considerable improvement in my score.

## Final Training
I trained my model over the entire training set using an extensive grid search. This took around 5 hours and improved my ranking, enabling me to reach the top 30%.

## Further Improvements
### 1. Creating New Features
- I created columns based on dates (e.g., day, month, `sin(year)`).
- Combined several columns (e.g., `income / number of dependents`).

### 2. Improving Encoding
- I implemented one-hot encoding based on the frequency of occurrence of the objects.

### 3. Trying New Models
- I used `AutoGluon`, which tries different models and combinations of models, then selects the best one.

Combining these efforts allowed me to reach the top 15% of competitors.

## Addressing Calculation Time Issues
Initially, I trained my model on 20% of the training data to reduce computation time. Once I had a well-functioning model with clean data, I trained it on the full dataset. Additionally, as the training set was large (1.2 million rows), I chose to remove rows with NaN values.

When my code worked well but was slow, I used Google Colab, which provided faster computation than my local machine.

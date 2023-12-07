## My Project

I applied machine learning techniques to investigate the demographic factors most closely associated with personal income for a dataset of American adults. Below is my report.

***

## Introduction 

For this problem, I chose to use the “Adult” dataset from the UC Irvine Machine Learning Repository. This dataset is composed of 15 common demographic factors for 48,842 adults taken from the 1994 US Census. The 14 features in this dataset include age, education level, race, sex, occupation, marital-status, among others, while the target is a binary variable specifying whether an individual makes over $50,000 per year or not. The goal with this dataset was to predict whether an individual makes more than $50,000 annually based on the 14 census variables.

This dataset naturally lends itself to a binary classification problem, and since the target variable is known (income level), I chose to use a supervised machine learning model. More specifically, I wanted to apply decision trees to this problem, as their property of being a whitebox machine learning algorithm meant that I could gain more valuable insights about the dataset’s features and analyze their relative importance in predicting the target variable. In other words I wanted to explore the question: What demographic factors are most influential in determining an American adult’s income?

To solve this problem, I trained three types of decision tree based models from the scikit-learn Python library–a single decision tree, a random forest model with 100 estimators, and a boosting random forest model with 100 classifiers. After testing, I found that the single decision tree had the best accuracy rate and f1-score of 0.821 and 0.829, respectively, though both of the other models were not far behind, also close to 0.8 for both metrics. After conducting feature ranking on each of the models, I found that the most influential features in the models were marital status, number of years of education, age, and capital gains. Though I fell short of making any broad conclusions about the influence of these demographic features on an individual's income due to the large influence of sample selection on these models.

## Data

The Adult dataset includes 14 features and the target variable “income” for 48,842 American adults. Below is a table of all of the variables included in the dataset, as well as the type of data (binary, categorical, numeric) of each feature. According to the UCI Machine Learning Repository, the dataset had already been cleaned, so the amount of preprocessing necessary on my end was somewhat limited. Though, one small issue that came up was that when reading the data into Google Colab, the target variable “income” was being detected as having four categories, with some of the values ending in commas while others do not (for example, “>50K,” vs. “>50K”). This required a simple fix of removing all of the commas in this field with string manipulation functions, after which Python was able to detect “income” as a binary variable. This was done with the code below:

```python
y['income'] = y['income'].str.replace('.', '')
mapping_dict = {'>50K': 1, '<=50K': 0}
y['income'] = y['income'].replace(mapping_dict) #turn target values into binary
```

The largest data preprocessing step taken in this project involved balancing the target variable. More specifically, around 76% of the records in the original dataset had annual incomes less than or equal to $50,000, so a machine learning model trained on that dataset would be at risk of having a bias towards predicting that value. To eliminate this risk, the “<=50K” income category was downsampled so that there were equal numbers of samples with incomes “>50K” and “<=50K”. This involved randomly selecting “>50K” income records to be included in the new dataset, as done in the code chunk below. 

This method can also be used to correct imbalances in the feature variables, which I did explore for the “race” feature. This involved subsetting the data for each race category, and randomly downsampling them all to the size of the category with the least number of samples, as seen below. Given that over 80% of the data falls into the “White” race category (see Figure 1), this step significantly reduced the size of the data to under 600 samples for the training dataset. This introduces more variability in the models, which I will explore further in the discussion section of this report.

FIGURE 1 HERE

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Samples in original Adult dataset according to "race" feature.*

Another data preprocessing step undertaken was the separation of training and testing datasets, which was done with scikit-learn’s train_test_split() function, as seen in the code chunk below. I assigned 80% of the data into the training dataset and 20% of the data into the testing dataset. This is a crucial step, as having a separate testing dataset that the model isn’t trained on means that we can accurately test the model’s performance and avoid overfitting.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Modelling

Decision trees are a versatile supervised learning model that can be used for both classification and regression problems. At its core, the decision tree algorithm attempts to create logical pathways for predicting the target variable by “splitting” entries based on their feature values. The decision tree algorithm used in this model splits on the feature with the lowest Gini impurity, which is a measure of the likelihood that an observation is incorrectly classified. Decision trees were chosen for this model primarily because of their intuitive nature and structure. The splits that the algorithm chooses to make are indicative of the relative influences of each feature on the target variable, which allows us to make greater insights into what demographic factors have the largest influence on an individual’s income. More specifically, decision trees include a metric called feature importance, which measures each feature’s contribution to the final predictions made by the model. These values can also be used to filter out extraneous features to construct a more refined model, though the dataset in this problem has few enough features that doing that is somewhat unnecessary. The first model I trained was a single decision tree with a maximum depth (number of layers/splits) of 5.

In addition to a base decision tree model, I also constructed two decision tree models that also use ensemble methods, namely a random forest and boosting. A random forest trains a large number of decision trees (100 in the case of my model) and aggregates their results to create a better prediction. Finally, I trained a boosting model, also with 100 estimators. Boosting is similar to a random forest in that it is composed of many decision trees, but it differs in that the model goes through multiple iterations of training, with each subsequent iteration focusing on the samples that the previous model misclassified, in a way correcting for the past model’s weaknesses. Both of these ensemble methods used trees with a maximum depth of 2, which can be that low because of the large number of estimators involved in the model.

Each of these three models was trained using a function from the scikit-learn library, namely DecisionTreeClassifier, RandomForestClassifier, and GradientBoostingClassifier. These functions handle all of the calculations involved in training the models, so on my part I simply had to input my training dataset into each model and set the chosen model parameters, as seen with the code for a single decision tree below:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

max_depth = 5
tree = DecisionTreeClassifier(max_depth = max_depth)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
```

The code for the random forest model and boosting model was structured similarly. After training the models, they were each used to predict the target variables for the testing dataset. These predictions were then compared with the true target values from the testing dataset by calculating an accuracy score (the rate of getting predictions correct) and a f1-score (a more complex metric that seeks to balance precision and true positive rate). Finally, the scikit-learn functions all calculate the attribute feature_importances_, which gives an indication of the relative contribution of each feature in the model predictions. 

Due to the relatively small training dataset and the use of random sampling to balance the race feature and income target variable, the models can produce slightly different results each time we train them. To account for this, I trained each model type five times and collected the accuracy score, f1-score, and feature importances for each training cycle. The accuracy and f1-scores are summarized in Table 1 and 2 below.

| Training Run | Decision Tree | Random Forest | Boosting |
| --- | --- | --- | --- |
| 1 | 0.864 |	0.823 | 0.782 |
| 2 | 0.799 |	0.805 |	0.844 |
| 3 | 0.821 |	0.775 |	0.795 |
| 4 | 0.836 |	0.829 |	0.816 |
| 5 | 0.784 |	0.734 |	0.755 |
| Mean | 0.821 |	0.793 |	0.798 |

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)


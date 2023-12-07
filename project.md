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

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

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


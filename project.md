## My Project

I applied machine learning techniques to investigate the demographic factors most closely associated with personal income for a dataset of American adults. Below is my report.

***

## Introduction 

For this problem, I chose to use the “Adult” dataset from the UC Irvine Machine Learning Repository. This dataset is composed of 15 common demographic factors for 48,842 adults taken from the 1994 US Census. The 14 features in this dataset include age, education level, race, sex, occupation, marital-status, among others, while the target is a binary variable specifying whether an individual makes over $50,000 per year or not. The goal with this dataset was to predict whether an individual makes more than $50,000 annually based on the 14 census variables.

This dataset naturally lends itself to a binary classification problem, and since the target variable is known (income level), I chose to use a supervised machine learning model. More specifically, I wanted to apply decision trees to this problem, as their property of being a whitebox machine learning algorithm meant that I could gain more valuable insights about the dataset’s features and analyze their relative importance in predicting the target variable. In other words I wanted to explore the question: What demographic factors are most influential in determining an American adult’s income?

To solve this problem, I trained three types of decision tree based models from the scikit-learn Python library–a single decision tree, a random forest model with 100 estimators, and a boosting random forest model with 100 classifiers. After testing, I found that the single decision tree had the best accuracy rate and f1-score of 0.821 and 0.829, respectively, though both of the other models were not far behind, also close to 0.8 for both metrics. After conducting feature ranking on each of the models, I found that the most influential features in the models were marital status, number of years of education, age, and capital gains. Though I fell short of making any broad conclusions about the influence of these demographic features on an individual's income due to the large influence of sample selection on these models.

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

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


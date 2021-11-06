# Note
+ Cross-validation is the first and most essential step when it comes to building machine learning models.
If you have a `good cross-validation scheme in which validation data is representative of training and real-world data` you
will be able to build a good machine learning model which is highly generalizable.
+ Before creating any kind of machine learning model, we must know what cross-validation is and how to choose the best
cross-validation depending on your datasets.
+ **What is cross-validation?**
  + **Cross-validation** is a step in process of building a machine learning model which helps us ensure that our
  models fit the data accurately and also ensures that we do not overfit.(**overfitting**)
+ **Decision trees** classifier.
  + Max_depth.
+ **Overfitting**: `Another definition of overfitting would be when the test loss increases as we keep improving training loss.`
+ A few types of cross-validation techniques:
  + k-fold.
  + stratified k-fold.
  + hold-out.
  + leave-one-out
  + group k-fold.

+ When you get a dataset to build machine learning models, you separate them into two different sets: **training and validation**.
## K-fold cross-validation
+ We can divide the data into k different sets which are exclusive of each other.
+ We can split any data into k-equal parts using KFold from scikit-learn. Each sample is assigned a value from 0 to k-1
when using k-fold cross-validation.
## Stratified k-fold
+ If you have skewed dataset (example: binary classification with 90% positive samples and only 10% negative samples),
using simple k-fold cross-validation for a dataset like this can result in folds with all negative samples.
In these cases, we prefer using stratified k-fold cross-validation.
+ Stratified k-fold cross-validation keeps the ratio of labels in each fold constant.(So, with in each fold, you will 
have the same 90% positive and 10% negative samples).
+ Easy to modify the code by only changing from KFold to StratifiedKFold.
+ The rule is simple. If it's a standard classification problem. Choose stratified k-fold blindly.
## Hold-out based validation
+ Suppose we have a large amount of data, depending on which algorithm we choose, training and even validation can be very
expensive for a dataset. In these cases, we can opt for a hold-out based validation.
+ The process for creating the hold-out remains the same as stratified k-fold. For a dataset which has 1 million samples, we can
create ten folds instead of 5 and keep one of those folds as hold-out. This means we will have 100k sample in the hold-out,
and we will always calculate loss, accuracy and other metrics `on this set` and train on 900k samples.
+ Hold-out is also used very frequently with time-series data.
## Leave-one-out cross-validation
+ In many cases. we have to deal with small datasets and creating big validation sets means losing a lot of data for
the model to learn. In those cases, we can opt for a type of k-fold cross-validation where k=N. where N is the number of 
samples in the dataset.
+ This mean that in all folds of training, we will be training on all data samples except 1.
+ The number of folds for this type of cross-validation is the same as the number of samples that we have in the dataset.

## Use stratified k-fold for a regression problem.
+ The good thing about regression problems is that we can use all the cross-validation techniques mentioned above for regression
problems except  for stratified k-fold.
+ We have first to divide the target into bins(because no have yet class) and then we can use stratified k-fold in the same way as for classification problems.
+ There are several choices for selecting the appropriate number of bins. You can use a simple rule like **Sturge's Rule**
  + `Number of bins = 1 + log2(N)`

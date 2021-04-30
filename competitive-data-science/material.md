# Week 1
## Tools
* [Scikit-Learn](http://scikit-learn.org/)
* [H2O](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html)
* [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit)
* [XGBoost](https://github.com/dmlc/xgboost)
* [LightGBM](https://github.com/Microsoft/LightGBM)
* [Keras](https://keras.io/)
* [PyTorch](http://pytorch.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [LibFM](http://www.libfm.org/)
* [LibFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/)
* [Arbitrary order FM](https://github.com/geffy/tffm)

## Feature preprocessing
* [Preprocessing in Sklearn](http://scikit-learn.org/stable/modules/preprocessing.html)
* [Andrew NG about gradient descent and feature scaling](https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling)
* [Feature Scaling and the effect of standardization for machine learning algorithms](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)

## Feature generation
* [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
* [Discussion of feature engineering on Quora](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)

## Feature extraction from text
### Bag of words
* [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)
* [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)

### Word2vec
* [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)
* [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)
* [Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
* [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)

### NLP Libraries
* [NLTK](http://www.nltk.org/)
* [TextBlob](https://github.com/sloria/TextBlob)

## Feature extraction from images
### Pretrained models
* [Using pretrained models in Keras](https://keras.io/applications/)
* [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)

### Finetuning
* [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)
* [Fine-tuning Deep Learning Models in Keras](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)

# Week2
## Visualization Tools
* [Seaborn](https://seaborn.pydata.org/)
* [Plotly](https://plot.ly/python/)
* [Bokeh](https://github.com/bokeh/bokeh)
* [ggplot](http://ggplot.yhathq.com/)
* [Graph visualization with NetworkX](https://networkx.github.io/)
* [Biclustering algorithms for sorting corrplots](http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)

## Validation strategies
> The main rule you should know — never use data you train on to measure the quality of your model. The trick is to split all your data into training and validation parts.

* Holdout scheme:
 1. Split train data into two parts: partA and partB.

 2. Fit the model on partA, predict for partB.

 3. Use predictions for partB for estimating model quality. Find such hyper-parameters, that quality on partB is maximized.


* K-Fold scheme:

 1. Split train data into K folds.

 2. Iterate though each fold: retrain the model on all folds except current fold, predict for the current fold.

 3. Use the predictions to calculate quality on each fold. Find such hyper-parameters, that quality on each fold is maximized. You can also estimate mean and variance of the loss. This is very helpful in order to understand significance of improvement.

* LOO (Leave-One-Out) scheme:

 1. Iterate over samples: retrain the model on all samples except current sample, predict for the current sample. You will need to retrain the model N times (if N is the number of samples in the dataset).

 2. In the end you will get LOO predictions for every sample in the trainset and can calculate loss.  

> Notice, that these are validation schemes are supposed to be used to estimate quality of the model. When you found the right hyper-parameters and want to get test predictions don't forget to retrain your model using all training data.

* [Validation in Sklearn](http://scikit-learn.org/stable/modules/cross_validation.html)
* [Advices on validation in a competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)

## Data leakage
* [Perfect score script by Oleg Trott -- used to probe leaderboard](https://www.kaggle.com/olegtrott/the-perfect-score-script)
* [Page about data leakages on Kaggle](https://www.kaggle.com/docs/competitions#leakage)
* [A​nother page about data leakages on Kaggle](https://www.kaggle.com/dansbecker/data-leakage)

# Week 3
## Evaluation metrics
### Classification

* [Evaluation Metrics for Classification Problems: Quick Examples + References](http://queirozf.com/entries/evaluation-metrics-for-classification-quick-examples-references)
* [Decision Trees: “Gini” vs. “Entropy” criteria](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria)
* [Understanding ROC curves](http://www.navan.name/roc/)

### Ranking
* [Learning to Rank using Gradient Descent](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) -- original paper about pairwise method for AUC optimization
* [Overview of further developments of RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
* [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) (implemtations for the 2 papers from above)
* [Learning to Rank Overview](https://wellecks.wordpress.com/2015/01/15/learning-to-rank-overview)

### Clustering
* [Evaluation metrics for clustering](http://nlp.uned.es/docs/amigo2007a.pdf)

# Week 4
## Hyperparameter tunning
* [Tuning the hyper-parameters of an estimator (sklearn)](http://scikit-learn.org/stable/modules/grid_search.html)
* [Optimizing hyperparameters with hyperopt](http://fastml.com/optimizing-hyperparams-with-hyperopt/)
* [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

## Pipeline
* [Far0n's framework for Kaggle competitions "kaggletils"](https://github.com/Far0n/kaggletils)
* [28 Jupyter Notebook tips, tricks and shortcuts](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)

## Advanced feature engineering
### Matrix Factorization:

* [Overview of Matrix Decomposition methods (sklearn)](http://scikit-learn.org/stable/modules/decomposition.html)

### t-SNE:

* [Multicore t-SNE implementation](https://github.com/DmitryUlyanov/Multicore-TSNE)
* [Comparison of Manifold Learning methods (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html)
* [How to Use t-SNE Effectively (distill.pub blog)](https://distill.pub/2016/misread-tsne/)
* [tSNE homepage (Laurens van der Maaten)](https://lvdmaaten.github.io/tsne/)
* [Example: tSNE with different perplexities (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py)

### Interactions:

* [Facebook Research's paper about extracting categorical features from trees](https://research.fb.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)
* [Example: Feature transformations with ensembles of trees (sklearn)](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)

## Validation schemes for 2-nd level models

> There are a number of ways to validate second level models (meta-models). If not specified, we assume that the data does not have a time component. We also assume we already validated and fixed hyperparameters for the first level models (models).

### a)  Simple holdout scheme

1. Split train data into three parts: partA and partB and partC.

2. Fit N diverse models on partA, predict for partB, partC, test_data getting meta-features partB_meta, partC_meta and test_meta respectively.

3. Fit a metamodel to a partB_meta while validating its hyperparameters on partC_meta.

4. When the metamodel is validated, fit it to [partB_meta, partC_meta] and predict for test_meta.

### b)  Meta holdout scheme with OOF meta-features

 1. Split train data into K folds. Iterate though each fold: retrain N diverse models on all folds except current fold, predict for the current fold. After this step for each object in train_data we will have N meta-features (also known as out-of-fold predictions, OOF). Let's call them train_meta.

 2. Fit models to whole train data and predict for test data. Let's call these features test_meta.

 3. Split train_meta into two parts: train_metaA and train_metaB. Fit a meta-model to train_metaA while validating its hyperparameters on train_metaB.

 4. When the meta-model is validated, fit it to  train_meta and predict for test_met

### c) Meta KFold scheme with OOF meta-features

 1. Obtain OOF predictions train_meta and test metafeatures test_meta using b.1 and b.2.

 2. Use KFold scheme on train_meta to validate hyperparameters for meta-model. A common practice to fix seed for this KFold to be the same as seed for KFold used to get OOF predictions.

 3. When the meta-model is validated, fit it to train_meta and predict for test_meta.

### d)  Holdout scheme with OOF meta-features

 1. Split train data into two parts: partA and partB.

 2. Split partA into K folds. Iterate though each fold: retrain N diverse models on all folds except current fold, predict for the current fold. After this step for each object in partA we will have N meta-features (also known as out-of-fold predictions, OOF). Let's call them partA_meta.

 3. Fit models to whole partA and predict for partB and test_data, getting partB_meta and test_meta respectively.

 4. Fit a meta-model to a partA_meta, using partB_meta to validate its hyperparameters.

 5. When the meta-model is validated basically do 2. and 3. without dividing train_data into parts and then train a meta-model. That is, first get out-of-fold predictions train_meta for the train_data using models. Then train models on train_data, predict for test_data, getting  test_meta. Train meta-model on the train_meta and predict for test_meta.

### e) KFold scheme with OOF meta-features

 1. To validate the model we basically do d.1 -- d.4 but we divide train data into parts partA and partB M times using KFold strategy with M folds.

 2. When the meta-model is validated do d.5.

## Validation in presence of time component

### f) KFold scheme in time series

> In time-series task we usually have a fixed period of time we are asked to predict. Like day, week, month or arbitrary period with duration of T.

 1. Split the train data into chunks of duration T. Select first M chunks.

 2. Fit N diverse models on those M chunks and predict for the chunk M+1. Then fit those models on first M+1 chunks and predict for chunk M+2 and so on, until you hit the end. After that use all train data to fit models and get predictions for test. Now we will have meta-features for the chunks starting from number M+1 as well as meta-features for the test.

 3. Now we can use meta-features from first K chunks [M+1,M+2,..,M+K] to fit level 2 models and validate them on chunk M+K+1. Essentially we are back to step 1. with the lesser amount of chunks and meta-features instead of features.

### g) KFold scheme in time series with limited amount of data
We may often encounter a situation, where scheme f) is not applicable, especially with limited amount of data. For example, when we have only years 2014, 2015, 2016 in train and we need to predict for a whole year 2017 in test. In such cases scheme c) could be of help, but with one constraint: KFold split should be done with the respect to the time component. For example, in case of data with several years we would treat each year as a fold.

## Stacking
* [Kaggle ensembling guide at MLWave.com (overview of approaches)](https://mlwave.com/kaggle-ensembling-guide/)
* [StackNet — a computational, scalable and analytical meta modelling framework (by KazAnova)](https://github.com/kaz-Anova/StackNet)
* [Heamy — a set of useful tools for competitive data science (including ensembling)](https://github.com/rushter/heamy)


# Week 5
## Past solutions

* http://ndres.me/kaggle-past-solutions/
* https://www.kaggle.com/wiki/PastSolutions
* http://www.chioka.in/kaggle-competition-solutions/
* https://github.com/ShuaiW/kaggle-classification/

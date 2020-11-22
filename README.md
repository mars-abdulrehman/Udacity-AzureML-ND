# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**The dataset contains data related direct marketing campaign of a banking institute. The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).**

**As part of the project, I performed 2 tasks:**
- Tune a LogisticRegression model hyperpartameters using hyperdrive. And achived accuracy of 90.9 for a **C** value of **0.001** and **max_iter** value of **50**.
- Used AutoML to find the best algorithm and got accuracy of **91.11** for **VotingEnsemble**.

**Though marginally better, AutoML achieved a better accuracy than the manually executed pipeline Logistic Regression**

## Scikit-learn Pipeline
**The dataset contains 20 features for the binary classification ML problem. Target variable is named 'y' with values 'yes' or 'no' indicating if a client will subscribe to term deposit or not.**
- **As the data contains lots of categorical features, the first step in the pipeline is data cleaning step. One hot encoding and other methods are used to convert categorical columns to numerical values. Similarly target variable is also converted to numerical using custom function.**
- **Next I define the parameter space for 2 paramerters of LogisticRegression:- 'C' and 'max_iter', which we are going to optimise using hyperdrive. For 'C' - uniform parameter space is defined between 0.001 and 100. And for max_iter, I have chosen a discrete choice based parameter space among [10, 50, 100, 150, 200].** 
- **I am using RandomParameterSampling as it gives almost the same performance as Grid sampling, without taking as much time and compute as Grid sampling takes. It also supports discrete and continuous hyperparameters and early termination of low-performance runs**
- **I have used BanditPolicy as early termination policy to terminate the poorly performing runs. This helps in improving computational efficiancy.**
- **The primary metrics to maximise during training in the pipeline is deficned as Accuracy.**
- **This pipeline was submitted and I achived a maximum accuracy of -- for hyper parameter values of "C" - "" and "max_iter" - "".** 

## AutoML
**For the same classification problem, I submitted an AutoML run as well for comparison. It ran multiple algorithms and performed data cleaning operations with minimal programmmer inputs, and produced best model as "VotingEnsemble" with maximum accuracy of "91.11".**
**Also, I performed AutoML using cross validations and training and test set and obtained similar accuracy.**

## Pipeline comparison

**Though we only get a difference of about 1% in accuracy between manually trained model, and model produced by AutoML, AutoML gives us a better accuracy of 91.11% and it took comparatively the same time to run multiple models as the LogisticRegression pipeline took to come up with best hyperparameters.**

**Also, AutoML performed all the data cleaning and preprocessing steps automatically, which saved a lot of programmers time. The saved time can be utilised to work on actual buiseness problem than spending on cleaning data and other mundane tasks.**

## Future work

**As future experiments, we can try and use different classification metrics like, F1 score, AUC, precision, recall scores, etc as Accuracy may not reflect the true model performance in real time conditions/buiseness scenarios.**
**Also, AutoML can be allowed to run for much longer duration to evaluate other models and come up with better performace metrics.**

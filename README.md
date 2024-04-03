# README table of contents
- [ml_packaging_classification](#ml-packaging-classification)
- [Use case description:](#use-case-description-)
- [Structure of the showcase](#structure-of-the-showcase)


# ml_packaging_classification
This repo is meant as a showcase to demonstrate a data science workflow for a multi-class classification (see use case description below) use case in business context. Data science use cases may have very different nature in terms of how the result/solution is used by the business. This use case has the characteristic of providing one-time insights and a one-time ouput of the results for further usage by the business. Therefore the repo focuses on analytics and limits operational aspects for analytical reusability.

The repo focuses on the following aspects:
- Build a simple ETL pipeline to prepare the raw data for analysis and classification.
- Conduct general data analysis for data quality investigation under consideration of the business goal.
- Conduct data analysis to get an understandig how to handle the data for multi-class classification, including a naive benchmark model using sklearn (DummyClassifier & a custom classifier).
- Builde multiple machine learning pipelines to evaluate best classification performance. The following aspects are considered within those pipelines:
  - Benchmarking pipelines to compare performance of multiple different types of models:
    - A basic benchmarking pipeline using naive classifiers as a base line.
    - A AutoML (automated machine learning) pipeline using PyCaret to compare a "large" variety of machine learning algorithms, considering:
      - including and excluding custom data pre-processing
      - pre-defined hyper-parameter set for each algorithm by PyCaret
      - using random search for HPO (hyper-parameter optimization) with a pre-defined hyper-parameter search space for each algorithm by PyCaret
    - A AutoML (automated machine learning) pipeline using AutoGluon.Tablular, considering:
      - including and excluding custom data pre-processing
      - including auto pre-processing by AutoGluon.Tabular
      - including auto feature engineering by AutoGluon.Tabular
      - including multiple classifiers by using:
        - multiple ml algorithms
        - "standard" HPO for each algorithm defined by AutoGluon.Tabular
        - ensables of algorithms (bagging and stacking with possible multiple layers)
    - A benchmarking pipeline for multiple tree-based algorithms, considering:  
    (since AutoML indicates a good performance of tree-based algorithms for the given use case; as well as showing that no single tree-based algorithm significantly outperforms others)
      - Tree-based classifiers: DecisionTree, RandomForest, LightGBM.
      - Model hyper-parameter optimization.
      - Class imbalance.
    - A benchmarking pipeline for Neural Network using Pytorch/Lightning, considering:  
    (AutoML shows relative low performance of NN with defined time contrains for the given use case. Double-check the AutoML-NN results with individual constrains)
      - Model hyper-parameter optimization.
      - Class imbalance.
  - A custom pipeline for the best performing model based on benchmarking, considering:
    - Model hyper-parameter optimization.
    - Class imbalance.
    - Oversamlping.
  - Business decision optimization for best model.
    - Threshold analysis (since best model provides probabilistic forecasts).
    - Consideration of class values from a business perspective using profit curves and under consideration of thresholds.
- Build a production pipeline (training & inference) for best model to provide final results.

**Python:** [sklearn](https://scikit-learn.org/stable/) | [PyCaret](https://pycaret.gitbook.io/docs) | [AutoGluon.Tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html) | [LightGBM](https://lightgbm.readthedocs.io/en/stable/) | [PyTorch/Lightning](https://lightning.ai/pytorch-lightning) | [MLflow](https://mlflow.org/) | [Optuna](https://optuna.org/)

# Use case description:
To reach sustainability goals for the packaging of products, the company needs to know to which packaging categories the single items belong to. Since this information is not given for 7.058 items of the total 90.035, the goal is to provide the categories for the items with missing ones based on a data-driven approach.  
First analysis has shown that simple 1:1 relationships and rule-based approaches do not lead to proper results. Therefore, a machine learning approach was used. The goal is to build a solution that is capable of doing a highly accurate prediction for as many packaging categories as possible. Meaning that on the one side predictions need to meet a certain threshold for accuracy to be usefull for the business (a small amount of wrong classifications can be tolerated but low classification accuracy does not help the business). On the other hand, a threshold for a minimum number of products needs to be covered (it is not mendatory to provide good forecasts for all items, but providing good forecasts for only a small amount of items also does not help the business a lot). Finally the machine learning solution should consider business decision optimization (cost optimization) based on different individual packaging (class) values.


# Structure of the showcase
As the showcase is intended to reflect the data science process to taggle the use case, the structure builds up on this.

**Code structure**

# ml_packaging_classification
This repo is meant as a showcase for multi-class classification. It focuses on the following aspects:
- Build a simple ETL pipeline to prepare the data for classification
- Conduct data analysis to get an understandig how to handle the data for multi-class classification
- Builde multiple machine learning pipelines to evaluate best classification performance. The following aspects are considered within those pipelines:
  - Different models (Naive, DecisionTree, RandomForest, LightGBM, XGBoost, Neural Networks)
  - Model hyper-parameter optimization
  - Class imbalance
  - Business decision optimization for best model
    - Threshold analysis (since best model provides probabilistic forecasts)
    - Consideration of class values from a business perspective using profit curves and under consideration of thresholds
- Build a production pipeline (training & inference) for best model


# Use case description:
To reach sustainability goals for the packaging of products, the company needs to know to which packaging categories the single items belong to. Since this information is not given for 7.058 items of the total 90.035, the goal is to provide the categories for the items with missing ones based on a data-driven approach.  
First analysis has shown that simple 1:1 relationships and rule-based approaches do not lead to proper results. Therefore, a machine learning approach was used. To goal is to build a solution that is capable of doing a highly accurate prediction for as many packaging categories as possible. Meaning that on the one side predictions need to meet a certain threshold for accuracy. On the other hand, a threshold for a minimum number of products needs to be covered. And final the machine learning solution should consider business decision optimization (cost optimization) based on different individual packaging (class) values.

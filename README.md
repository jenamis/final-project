![](Resources/logo.png)

# Machine Learning for SPE & LCMS Method Prediction

## Project Overview

### Topic
This project developed two machine learning (ML) models.
1. A model to predict which of two sample preparation and purification methods, also known as Solid Phase Extraction (SPE) methods, is optimal to use for a chemical compound based on properties of that compound's structure. 
2. A model to predict which of two methods for separating and analyzing sample components, also known as Liquid-Chromatography Mass-Spectrometry (LCMS) methods, is optimal to use for a chemical compound based on properties of that compound's structure.

### Reason for Selection
The team at an automated chemistry platform that works to automate the process of making small chemical compounds to be used in research and development for medicinal purposes is seeking ML models that can be used to select the best SPE and LCMS methods to test for purification and analysis of each chemical compound in a large library of compounds. Without ML models that can effectively predict the optimal SPE and LCMS methods to use, the team must make a best guess of which methods to test based on a subset of properties of each compound’s structure (also known as molecular descriptors). This process can be time consuming and expensive, especially if the wrong SPE and/or LCMS method(s) end up being selected and testing must be repeated using other methods. Development of these two ML models has the potential to improve the time and cost efficiency of the automated chemistry platform’s process.

### Data Source
This project utilized datasets provided by the data team at the automated chemistry platform. The first dataset listed compounds tested by the platform over the past two years and included molecular descriptors such as molecular weight, topological polar surface area (TPSA), quantitative estimate of drug-likeness (QED), among many others believed to be potentially relevant to predicting the appropriate SPE and LCMS methods to use for compound purification and analysis. These *molecular descriptors* were generated from input SMILES strings using the Python library [RDKit](https://www.rdkit.org/), which provides an extensive [list of available descriptors](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors). The file for these calculations can be found at [chemCalculate.py](database/chemCalculate.py). 

The second dataset included the status of testing for each compound and the SPE and LCMS methods used for each compound that had completed the purification stage. Each compound was identified by a unique structure ID, and proprietary information about the actual structure of the compound was excluded from the datasets.

### Data ETL Process

![The Data ETL Pipeline used for this project](Resources/PurifAI_ETL_pipeline.png)


1. **Raw data was extracted from chemistry platform database as CSV files. These are accessible from AWS S3 buckets:**
   
   - https://purifai.s3.us-west-1.amazonaws.com/data/outcomes.csv
   - https://purifai.s3.us-west-1.amazonaws.com/data/structures.csv
   
2. **Data was transformed and cleaned using pandas in this file [clean_dataset.ipynb](database/clean_dataset.ipynb) and new CSV files were generated. These were uploaded to and are accessible from AWS S3 buckets:**
   
   - https://purifai.s3.us-west-1.amazonaws.com/clean-data/cleaned-outcomes.csv
   - https://purifai.s3.us-west-1.amazonaws.com/clean-data/cleaned-structures.csv
   
     The cleaning of the structures data was straight forward. Due to the structure data’s sensitive and private nature, all that was required was to drop the structure representation fields (SMILES) and encode the `STRUCTURES_ID`s via substring replacements.
   
     The outcomes table cleaning was more cumbersome as the dataset was already in need of thorough cleaning due to a lot of unmaintained manual data entry from eariler entries of the source data. Namely, the `spe_method` and `preferred_lcms_method` fields needed to be accurately null for the model to work properly. This required a manual review and transfer from several other offline data sources. Other transformations were made to the actual data including normalizing all percent or decimal values to be consistent with each field.
   
3. **Data was loaded into the AWS database (*purifai.ceoinb9nwfxg.us-west-1.rds.amazonaws.com*) using PySpark using this file [purifAI_database.ipynb](database/purifAI_database.ipynb).**

[Here is the database diagram](database/DBD%20Diagram.png).

4. **Data for SPE analysis was extracted as a merged table (`spe_analysis_df`) using SQLAlchemy and pandas. This DataFrame for analysis was obtained using the code from [spe_analysis_data.ipynb](database/spe_analysis_data.ipynb). Data for LCMS analysis (`lcms_analysis_df`) was extracted in the same way using the code from [lcms_analysis_data.ipynb](database/lcms_analysis_data.ipynb).**

### Questions to Answer

This project sought to answer the following questions:
- Which molecular descriptors are relevant to include as features in a ML model to predict the optimal SPE method for compound purification?
- Can a ML model be developed that has sufficiently high accuracy, precision, and sensitivity for predicting optimal SPE method for compound purification?
- Which ML model will perform best for predicting optimal SPE method for compound purification?
- Which molecular descriptors are relevant to include as features in a ML model to predict the optimal LCMS method for compound analysis?
- Can a ML model be developed that has sufficiently high accuracy, precision, and sensitivity for predicting optimal LCMS method for compound analysis?
- Which ML model will perform best for predicting optimal LCMS method for compound analysis?

## Machine Learning Model
Python scripts with pandas in Jupyter Notebook were used to test the performance of supervised ML models using the following algorithms and resampling methods:
- Imbalanced-learn's `BalancedRandomForestClassifier` (Balanced Random Forest)
- Imbalanced-learn's `EasyEnsembleClassifier` (Easy Ensemble AdaBoost)
- XGBoost's `XGBClassifier` (Extreme Gradient Boosting, or XGBoost)
- Scikit-learn's `LogisticRegression` with the following imbalanced-learn sampling modules:
   - `RandomOverSampler` (LR with Random Oversampling) 
   - `SMOTE` (LR with SMOTE Oversampling)
   - `RandomUnderSampler` (LR with Random Undersampling)
   - `ClusterCentroids` (LR with Cluster Centroids Undersampling)
   - `SMOTEENN` (LR with SMOTEENN Over and Undersampling)

One set of models was developed to predict optimal SPE method for compound purification, and a second set was developed to predict optimal LCMS method for compound analysis. All models used molecular descriptors from the `structures` dataset as features (see details below under Feature Engineering & Selection). The listed ML algorithms and sampling methods were selected for testing due to a class imbalance for both binary target variables (SPE method and LCMS method). Model performance was evaluated using scikit-learn's `balanced_accuracy_score` and `confusion_matrix` modules and imbalanced-learn's `classification_report_imbalanced` module. Model comparisons were based primarily on balanced accuracy score and weighted F1 score.

### Data Preprocessing
The ML model development scripts connected to the AWS database using SQLAlchemy. An inner join between the `outcomes` and `structures` tables on the `structure_id` column was used to merge the two cleaned datasets. Since the goal of both ML models is to predict optimal methods related to compound purification, only rows where the compound successfully completed the purification stage of testing (indicated by the value "true" in the `spe_successful` column) were retained in the data for model development. For development of the LCMS ML model, the data was further limited to include only rows with one of the main LCMS methods in the `preferred_lcms_method` column (rows with the LunaOmega LpH method were excluded because it is very rarely used).

Data preprocessing continued as outlined below.
- After the merged dataset was loaded into the model development scripts as a pandas DataFrame, all columns from the original `outcomes` dataset **except for** the structure ID (`structure_id`) and columns containing the SPE or LCMS method were dropped from the DataFrame. 
   - The dropped columns contained additional outcome data from the compound testing process that may be of interest for future analysis but were extraneous to the current objective of predicting optimal SPE and LCMS methods. 
- Duplicate rows were dropped from the DataFrame. 
   
**For SPE ML model development:**
- If a structure ID was tested multiple times with same SPE method, only one row was retained for that structure ID and SPE method combination.
- If a structure ID was tested successfully with both SPE methods, rows for that structure ID with each SPE method were retained. 
   
**For LCMS ML model development:**
- If a structure ID was tested multiple times with same LCMS method, only one row was retained for that structure ID and LCMS method combination.
- If a structure ID was tested successfully with both LCMS methods, rows for that structure ID with each LCMS method were retained. 
- Scikit-learn's `LabelEncoder` module was used to transform the SPE or LCMS method (target) from string to numerical data in some of the model testing scripts. All features were already numerical. **Note:** This preprocessing step was skipped when training the final saved models.

### Feature Engineering & Selection
The original `structures` dataset included 45 molecular descriptors believed to be potentially relevant for predicting the optimal SPE and LCMS methods to use for compound purification and analysis. The base version of each ML model (described under Model Testing & Training) was tested using all 45 properties as features in the model. In addition, all models except Easy Ensemble AdaBoost were tested with a subset of selected features. 
- For the Balanced Random Forest and XGBoost models, feature importances were retrieved and Scikit-learn's `SelectFromModel` module was used to select the features to include in the subset.
- For the Logistic Regression models, features were sorted in descending order by the absolute value of their coefficient as a proxy for feature importance. For the SPE models, the top 20 features were selected to include in the subset (aligning with the number of selected features for the SPE Balanced Random Forest model, since the distributions of feature importance were similar). For the LCMS models, the top 14 features were selected to include in the subset (aligning with the number of selected features for the LCMS Balanced Random Forest model). With additional time for model testing and training, a more systematic approach to feature selection for the Logistic Regression models could be undertaken.

The tables below show the balanced accuracy score (abbreviated BA) and weighted F1 score for the base version of each model with all features included compared with only selected features. Model versions with selected features performed worse or only slightly better than the versions with all features, indicating that the full set of 45 features did not contain extraneous features creating significant "noise" in the model. Additionally, none of the models using only a subset of selected features had a higher balanced accuracy score or weighted F1 score than the best performing models using all features. Therefore, all features were retained when model testing advanced to the hyperparameter tuning stage.

***Comparison of Base Model Performance for Predicting SPE Method with All Features and Selected Features***

<img src="Resources/spe_features.png" height="225">


***Comparison of Base Model Performance for Predicting LCMS Method with All Features and Selected Features***

<img src="Resources/lcms_features.png" height="225">


Since feature values ranged from less than 1 to greater than 700, scikit-learn's `StandardScaler` module was used to scale all features after completing the train-test split.

### Train-Test Split
The data was split using scikit-learn's `train_test_split` module with default parameters. The original training and testing sets were used for the Balanced Random Forest, Easy Ensemble AdaBoost, and XGBoost models. Due to the class imbalance for the target variables, each of the Logistic Regression models utilized a resampling method for the training data, as indicated in the name given to the model. 

### Model Testing & Training
Two stages of model testing and training were performed for both SPE model development and LCMS model development. 

#### Initial Testing
Base versions of the ML algorithms and resampling methods listed above were tested and balanced accuracy scores and weighted F1 scores were compared. 
- For Balanced Random Forest, Easy Ensemble AdaBoost, and XGBoost base versions, n_estimators was set equal to 100 and all other hyperparameters were default. 
- For LR base versions, all hyperparameters were default. 

The tables below show a comparison of base model performance sorted from highest to lowest balanced accuracy score. For the model predicting SPE method, Balanced Random Forest and XGBoost had the highest balanced accuracy and weighted F1 scores. For the model predicting LCMS method, Balanced Random Forest had the highest balanced accuracy and weighted F1 scores, while LR with Cluster Centroids Undersampling and XGBoost had the second highest balanced accuracy score and weighted F1 score, respectively.

***Base Model Performance for Predicting SPE Method***

<img src="Resources/spe_base.png" height="225">


***Base Model Performance for Predicting LCMS Method***

<img src="Resources/lcms_base.png" height="225">


Base model testing scripts can be accessed here:
- [SPE model testing script](machine_learning/SPE/ML_testing/spe_ML_base_model_testing_updated.ipynb)
- [LCMS model testing script](machine_learning/LCMS/ML_testing/lcms_ML_base_model_testing_updated.ipynb)

#### Hyperparameter Tuning 
Hyperparameter tuning was performed for all of the ML algorithms listed above.
- For Balanced Random Forest, a two-step process for hyperparameter tuning using both scikit-learn's `RandomizedSearchCV` and `GridSearchCV` was performed. A random search for best parameters was performed first, followed by a grid search with hyperparameter values set based on the random search results. The following subset hyperparameters were tuned:
   - `n_estimators`
   - `min_samples_split`
   - `min_samples_leaf`
   - `max_features`
   - `max_depth`
   - `bootstrap`
   - `replacement`
- For XGBoost, `GridSearchCV` was used to search for the best values for the following subset of hyperparameters:
   - `n_estimators`
   - `colsample_bytree`
   - `gamma`
   - `learning_rate`
   - `max_depth`
   - `min_child_weight`
- Since the Easy Ensemble AdaBoost and LR base models did not perform as well overall, less attention was given to their hyperparameter tuning and `GridSearchCV` was used to search for the best values for only a small subset of hyperparameters. 
   - For Easy Ensemble AdaBoost: `n_estimators`
   - For LR: `C`, `penalty`

For all models, the balanced accuracy and weighted F1 scores for the version with the best identified hyperparameter values were compared with the scores for the base version. The tables below show these comparisons, and are sorted from highest to lowest balanced accuracy score after hyperparameter tuning. For predicting SPE method, Balanced Random Forest had the highest balanced accuracy score after hyperparameter tuning, but the XGBoost base model still had the highest weighted F1 score. For predicting LCMS method, XGBoost had the highest balanced accuracy score and weighted F1 score after hyperparameter tuning. 

***Comparison of Base and Grid Search Model Performance for Predicting SPE Method***

<img src="Resources/spe_grid.png" height="225">


***Comparison of Base and Grid Search Model Performance for Predicting LCMS Method***

<img src="Resources/lcms_grid.png" height="225">


Hyperparameter tuning scripts for **SPE models** can be accessed here:
- [Balanced Random Forest](machine_learning/SPE/ML_testing/spe_balanced_random_forest_param_search_updated.ipynb)
- [XGBoost](machine_learning/SPE/ML_testing/Grid%20search%20on%20XGBoost_spe.ipynb)
- [Easy Ensemble AdaBoost](machine_learning/SPE/ML_testing/Grid%20search%20on%20Easy%20Ensemble%20AdaBoost%20Classifier_spe.ipynb)
- [LR](machine_learning/SPE/ML_testing/Grid%20search%20on%20Logistic%20Regression_spe.ipynb)

Hyperparameter tuning scripts for **LCMS models** can be accessed here:
- [Balanced Random Forest](machine_learning/LCMS/ML_testing/lcms_balanced_random_forest_param_search_updated.ipynb)
- [XGBoost](machine_learning/LCMS/ML_testing/Grid%20search%20on%20XGBoost_lcms.ipynb)
- [Easy Ensemble AdaBoost](machine_learning/LCMS/ML_testing/Grid%20search%20on%20Easy%20Ensemble%20AdaBoost%20Classifier_lcms.ipynb)
- [LR](machine_learning/LCMS/ML_testing/Grid%20search%20on%20Logistic%20Regression_lcms.ipynb)

### Final Model Selection & Performance

Balanced Random Forest and XGBoost algorithms both produced high-performing ML models for predicting optimal SPE method and optimal LCMS method. Balanced Random Forest was originally selected for both final models. However, after further comparison of weighted F1 scores and precision and recall values, XGBoost was selected instead for both final models. Details of each final model and its performance are discussed separately below.

#### SPE ML Model
Although Balanced Random Forest had a slightly higher best balanced accuracy score than XGBoost for predicting SPE method , XGBoost was selected for this model due to its higher weighted F1 score. In particular, the precision for predicting HLB (the minority class) was much higher with XGBoost than with Balanced Random Forest. The difference in recall for predicting HLB, where XGBoost performed the worst, was smaller and XGBoost still performed fairly well. Performance metrics for both models are shown below.

***XGBoost Performance Metrics***

<img src="Resources/spe_performance_xgb.png" width="650"> 


***Balanced Random Forest Performance Metrics***

<img src="Resources/spe_performance_brf.png" width="650">


Performance metrics for the selected XGBoost model are explained below. 
- Balanced Accuracy Score: The model's SPE method predictions for the testing data were correct **89%** of the time. 
- Precision for Predicting MCX: When the method was predicted as MCX, it actually was MCX **95%** of the time.
- Precision for Predicting HLB: When the method was predicted as HLB, it actually was HLB **90%** of the time. 
- Recall for Predicting MCX: When the method was actually MCX, the model correctly predicted it as such **98%** of the time.
- Recall for Predicting  HLB: When the method was actually HLB, the model correctly predicted it as such **81%** of the time. 

The default hyperparameters used for the base version of the model were used for the final model since performance did not improve through hyperparameter tuning. They are shown below.

<img src="Resources/spe_final_params.png" width="450">


#### LCMS ML Model

XGBoost and Balanced Random Forest had nearly equal best balanced accuracy scores for predicting LCMS method, so XGBoost was once again selected for this model due to its higher weighted F1 score. As with the SPE model, XGBoost had a higher precision than Balanced Random Forest for predicting the minority LCMS method class (Gemini). Performance metrics for both models are shown below. 

***XGBoost Performance Metrics***

<img src="Resources/lcms_performance_xgb.png" width="650"> 


***Balanced Random Forest Performance Metrics***

<img src="Resources/lcms_performance_brf.png" width="650">


Performance metrics for the selected XGBoost model are explained below.  
- Balanced Accuracy Score: The model's LCMS method predictions for the testing data were correct **89%** of the time. 
- Precision for Predicting Xbridge: When the method was predicted as Xbridge, it actually was Xbridge **93%** of the time.
- Precision for Predicting Gemini: When the method was predicted as Gemini, it actually was Gemini **88%** of the time. 
- Recall for Predicting Xbridge: When the method was actually Xbridge, the model correctly predicted it as such **95%** of the time.
- Recall for Predicting  Gemini: When the method was actually Gemini, the model correctly predicted it as such **82%** of the time. 

The hyperparameters for the final model are shown below.

<img src="Resources/lcms_final_params.png" width="350">


In general, XGBoost has several benefits that make it a strong choice for these models.
- It can perform well with imbalanced classes like SPE method and LCMS method.
- It can run efficiently with large datasets. 

A limitation of XGBoost is that, as a boosting algorithm, it can be more prone to overfitting.

## Website & Dashboard

Using HTML/CSS, a website was created with two main components: a dashboard and method prediction application.  

Tableau was integrated to create a fully functioning and interactive dashboard. The dashboard includes the performance metrics for our final SPE and LCMS models, both of which used XGBoost. 

![performance_metrics_dashboard](Resources/performance_metrics_dashboard.png)

The dashboard also gives users the ability to explore the SPE and LCMS methods by feature (chemical descriptor) for the structures in the dataset. 

![method_vs_features](Resources/method_vs_features.png)

Users are able see the ranking of feature importances for the final SPE and LCMS ML models. 

![feature_importances](Resources/feature_importances.png)

Flask and Dash was used to create a web application that allows users to input a SMILES string or list of SMILES strings and predict the optimal SPE and LCMS methods using the final ML models developed through this project. 

![enter_smiles](Resources/enter_smiles.png)

Click here for a [demonstration]() of the dashboard.

## purifAI Python Package

The **purifAI** package was designed to use for bulk input prediction of optimal SPE and LCMS methods for chemical compounds. The package uses the final XGBoost ML models developed through this project to generate these predictions.

### Installation

```
pip install purifAI
```

### Usage

The user can input a CSV file of SMILES strings and the first function, `calculate_descriptors(self, smiles, ipc_avg=False)`, will calculate from these SMILES the molecular descriptors to be put into the prediction model as features. The user can then call `RunSPEPrediction(self, smiles)` for SPE method predictions and `RunLCMSPrediction(self, smiles)` for LCMS method predictions. 

```python
# import dependencies
from purifai.methodSelection import model_selection
from tkinter import filedialog as fd
from tkinter.filedialog import asksaveasfile
from tkinter.messagebox import showinfo
import pandas as pd
import os
import wget
```

```python
# Download saved model.pkl and scaler.pkl
cwd = os.getcwd()
    
url = 'https://github.com/jenamis/purifAI/raw/main/machine_learning/SPE/models/'
if not os.path.exists(os.getcwd() + '/spe_xgb_model.pkl'):
    wget.download(url+ 'spe_xgb_model.pkl')
if not os.path.exists(os.getcwd() + '/spe_scaler.pkl'):
    wget.download(url+ 'spe_scaler.pkl')
    
url= 'https://github.com/jenamis/purifAI/raw/main/machine_learning/LCMS/models/'
if not os.path.exists(os.getcwd() + '/lcms_xgb_model.pkl'):
    wget.download(url+ 'lcms_xgb_model.pkl')
if not os.path.exists(os.getcwd() + '/lcms_scaler.pkl'):
    wget.download(url+ 'lcms_scaler.pkl')
```

```python
# set up model and scaler file path    
spe_xgb_model = cwd + '/spe_xgb_model.pkl'
spe_scaler = cwd + '/spe_scaler.pkl'
lcms_xgb_model = cwd + '/lcms_xgb_model.pkl'
lcms_scaler = cwd + '/lcms_scaler.pkl'
```

```python
# set up the model predictor by calling the model_selection function in purifAI
model_predictor = model_selection(spe_xgb_model, 
                            spe_scaler,
                            lcms_xgb_model,
                            lcms_scaler)
```

```python
# Get the input smiles 
showinfo(title="Select SMILES List (CSV)", message="Select the list of structures' SMILES to process. NOTE: Column header must be 'SMILES'.")
inputfile = fd.askopenfilename()

# Created descriptors_df
df = pd.read_csv(inputfile)
df = df.dropna(subset=['SMILES'])
smiles = df['SMILES'].to_list()
```

```python

# iterate through the smiles list and perform ml perdiction 

df["PREDICTED_SPE_METHOD"] = ''
df["PREDICTED_LCMS_METHOD"] = ''

for i in range(len(df)):
    smile = df.loc[i, 'SMILES']
    # perform prediction on SPE method
    predicted_SPE_method = model_predictor.RunSPEPrediction(smile)
    df.loc[i, "PREDICTED_SPE_METHOD"] = str(predicted_SPE_method)
    print("RunSPEPrediction succesful...")
    
    # perform prediction on LCMS method
    predicted_LCMS_method = model_predictor.RunLCMSPrediction(smile)
    df.loc[i, "PREDICTED_LCMS_METHOD"] = str(predicted_LCMS_method)
    print("RunLCMSprediction succesful...")
```

```python
# save the prediction results 
showinfo(title="Save results", message="Save the prediction results")
prediction_result = asksaveasfile()
df.to_csv(prediction_result, index=False)
```

```python
# Generate structure data (features)
descriptors_results = []
for smile in smiles:
    descriptors = model_predictor.calculate_descriptors(smile)
    descriptors_results.append(descriptors)
    # print('smile = \n', smile, 'descriptors = \n', descriptors)
print(descriptors_results)

names = ['MolWt', 'ExactMolWt', 'qed', 'TPSA', 'HeavyAtomMolWt', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'FpDensityMorgan1', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'Ipc', 'Kappa2', 'LabuteASA', 'PEOE_VSA10', 'PEOE_VSA2', 'SMR_VSA10', 'SMR_VSA4', 'SlogP_VSA2', 'SlogP_VSA6','MaxEStateIndex', 'MinEStateIndex', 'EState_VSA3', 'EState_VSA8', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount']
descriptors_df = pd.DataFrame(descriptors_results, columns=names)
descriptors_df['SMILES'] = df['SMILES']
result_df = pd.concat([df, descriptors_df], axis=1)
```
```python
# Save the prediction and descriptors dataframe
showinfo(title="Save Results", message="Save a summary dataframe with prediction and descriptors")
summary_with_descriptors = asksaveasfile()
result_df.to_csv(summary_with_descriptors, index=False)
```

The testing code above can be used directly to perform bulk input prediction of SPE and LCMS methods. Users can also change the model and/or scaler by simply changing the file paths.

```python
# set up the file path
spe_xgb_model = cwd + '/spe_xgb_model.pkl'
spe_scaler = cwd + '/spe_scaler.pkl'
lcms_xgb_model = cwd + '/lcms_xgb_model.pkl'
lcms_scaler = cwd + '/lcms_scaler.pkl'
```
Click here for a [demonstration](https://www.youtube.com/watch?v=b7e8zEA0pMY&t=4s) of the purifAI Python package.

## Conclusion

All-in-all, the scores of the final ML models seem to encourage an optimistic implementation on real structure predictions. This will hopefully begin to guide automated decisions on purification and analysis methods for compounds passing through the automated chemistry platform.

### Positive Outcomes

A highly predictive ML model implemented in this way would improve the automated chemistry platform by:
- Increasing success quantity metrics and minimizing failed sample counts (better fitted methods on a per-sample-basis will inherently improve purity and sample recovery results);
- Limiting the amount of time and resources taken by compound samples needing to be re-tested through the platform;
- Eventually being able to highlight what types of compound structures are inevitably unsuited for the automated platform, thus indicating for which structures new methods should be developed; and
- Opening many other doors for similar models to be applied to other obstacles that challenge the automation of the platform.

### Potential Limitations

Some readily acknowledged limitations of these (and other) ML models could impact the breadth of their impact.
- The model will only be able to predict accurately for compound structures that are similar enough to the training set. In the confinements of targeting small compound drug targets, these are generally all within a certain scope of alikeness.
- As compounds evolve with the growing knowledge of chemistry and drug discovery as time passes, this (or any) model will need to be reoptimized with new and additional data to retain accuracy.
- Only when the predictive model is correctly and consistently indicating successful methods over many different types of samples can this model be determined helpful for predicting optimal methods accurately.

### Expansions & Future Endeavours 

To address the issue of structural likeness between input structures of the model and those that it was trained upon is a relevant, but complex question.

An immediate next development to this tool would be to develop a way of measuring chemico-statistical likeness of an input set of chemical compounds compared with the set of compounds used to train and test the ML models. This analysis would require assessing for a statistical difference between the array of molecular descriptors of compounds in the input set and the array of descriptors for the training set.

Additionally, this predictive work could be expanded to encompass more fields of data relevent to measuring a given compound’s success throught the automated platform. This could be done by implementing some deep learning strategies to play with models using a greater coverage of the relevant data. In the scope of this work, an additional model could be implemented using deep learning towards the resolution of chemical challenges. Common prepackaged tools, like [deepchem](https://deepchem.io/) or [chainer-chemistry](https://github.com/chainer/chainer-chemistry), may aid those investigations and prove useful specifically for the unique challenges that chemistry provides.

Should this project’s work prove successful for predicting purification methods, it may prove useful to expand these tactics to the realm of automated synthesis and retrosynthetic AI (at least explorationally).

----------------------------------------------------------------
### [Presentation Slides](https://docs.google.com/presentation/d/1imPsfB5MEK9zgKy1yReFI4jfK9sjPSk96bwV_MYEqJc/edit?usp=sharing)

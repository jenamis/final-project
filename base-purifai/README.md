# purifAI

The __purifAI__ package is designed for bulk input prediction about what chemical purification method should be used for a particular chemical. 

The ML model used in the package is  Extreme Gradient Boosting (XGBoost). Purification methods we included in machine learning are solid phase extraction (SPE) methods and liquid chromatography–mass spectrometry (LCMS) methods.


## Installation

```
pip install purifAI
```

## Usage

User can input a csv file of chemical formulars as smiles, our first function `calculate_descriptors(self, smiles, ipc_avg=False)` will calculate and convert these chemicla formulars into descriptors and features to be put into the prediction model. Then, user can call `RunSPEPrediction(self, smiles)` for SPE methods prediction; or `RunLCMSPrediction(self, smiles)` for LCMS methods prediction. 

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

The testing code above can be use direltly to perform bulk input prediction on SPE and LCMS methods. Users can also change the model and or scaler by simply changing the file paths.

```python
# set up the file path
spe_xgb_model = cwd + '/spe_xgb_model.pkl'
spe_scaler = cwd + '/spe_scaler.pkl'
lcms_xgb_model = cwd + '/lcms_xgb_model.pkl'
lcms_scaler = cwd + '/lcms_scaler.pkl'
```


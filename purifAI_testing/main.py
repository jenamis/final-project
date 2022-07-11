# import dependendies
from purifai.methodSelection import model_selection
import pandas as pd
from tkinter import filedialog as fd
from tkinter.filedialog import asksaveasfile
from tkinter.messagebox import showinfo
import os
import wget

# download model.pkl and scaler.pkl
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
    
spe_xgb_model = cwd + '/spe_xgb_model.pkl'
spe_scaler = cwd + '/spe_scaler.pkl'
lcms_xgb_model = cwd + '/lcms_xgb_model.pkl'
lcms_scaler = cwd + '/lcms_scaler.pkl'

# call the model_selection class from the package
model_predictor = model_selection(spe_xgb_model, 
                            spe_scaler,
                            lcms_xgb_model,
                            lcms_scaler)


# Get the input SMILES
# smiles = "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N"
showinfo(title="Select SMILES List (CSV)", message="Select the list of structures' SMILES to process. NOTE: Column header must be 'SMILES'.")
inputfile = fd.askopenfilename()
df = pd.read_csv(inputfile)
df = df.dropna(subset=['SMILES'])
print(df)
smiles = df['SMILES'].to_list()

# iterate through the smiles list and perform ml perdiction 
df["PREDICTED_SPE_METHOD"] = ''
df["PREDICTED_LCMS_METHOD"] = ''

for i in range(len(df)):
    smile = df.loc[i, 'SMILES']
    # call the function RunSPEPrediction()
    predicted_SPE_method = model_predictor.RunSPEPrediction(smile)
    df.loc[i, "PREDICTED_SPE_METHOD"] = str(predicted_SPE_method)
    print("RunSPEPrediction succesful...")
    
    # call the function RunLCMSPrediction()
    predicted_LCMS_method = model_predictor.RunLCMSPrediction(smile)
    df.loc[i, "PREDICTED_LCMS_METHOD"] = str(predicted_LCMS_method)
    print("RunLCMSprediction succesful...")
    
# save the results
showinfo(title="Save results", message="Save the prediction results")
prediction_result = asksaveasfile()
df.to_csv(prediction_result, index=False)

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

# print(result_df)
showinfo(title="Save Results", message="Save a summary dataframe with prediction and descriptors")
summary_with_descriptors = asksaveasfile()
result_df.to_csv(summary_with_descriptors, index=False)
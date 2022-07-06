import time
import pandas as pd
from purifai.methodSelection import model_selection
import os
import wget

cwd = os.getcwd()
url = 'https://github.com/jenamis/purifAI/raw/main/machine_learning/SPE/models/'
if not os.path.exists(os.getcwd() + '/spe_xgb_model.pkl'):
    wget.download(url+ 'spe_xgb_model.pkl')
if not os.path.exists(os.getcwd() + '/spe_scaler.pkl'):
    wget.download(url+ 'spe_scaler.pkl')
url= 'https://github.com/jenamis/purifAI/raw/main/machine_learning/LCMS/models/'
if not os.path.exists(os.getcwd() + '/lcms_xgbmodel.pkl'):
    wget.download(url+ 'lcms_xgb_model.pkl')
if not os.path.exists(os.getcwd() + '/lcms_scaler.pkl'):
    wget.download(url+ 'lcms_scaler.pkl')
spe_xgb_model = cwd + '/spe_xgb_model.pkl'
spe_scaler = cwd + '/spe_scaler.pkl'
lcms_xgb_model = cwd + '/lcms_xgb_model.pkl'
lcms_scaler = cwd + '/lcms_scaler.pkl'

model_predictor = model_selection(spe_xgb_model, 
                            spe_scaler,
                            lcms_xgb_model,
                            lcms_scaler)

print("PurifAI Application Initialzing.......")

inputfile = model_predictor.
# inputfile = "/Users/yycheung/Desktop/resources/smiles_list.csv"
df = pd.read_csv(inputfile)
df = df.dropna(subset=['SMILES'])
print(df)
smiles = df['SMILES'].to_list()


df["PREDICTED_SPE_METHOD"] = ''
df["PREDICTED_LCMS_METHOD"] = ''

for i in range(len(df)):
    smiles = df.loc[i, 'SMILES']
    
    predicted_SPE_method = model_predictor.RunSPEPrediction(smiles)
    df.loc[i, "PREDICTED_SPE_METHOD"] = str(predicted_SPE_method)
    print("RunSPEPrediction succesful...")
    
    
    predicted_LCMS_method = model_predictor.RunLCMSPrediction(smiles)
    df.loc[i, "PREDICTED_LCMS_METHOD"] = str(predicted_LCMS_method)
    print("RunLCMSprediction succesful...")

# # Generate structure data (features)
# descriptors_results = []
# for smile in smiles:
#     descriptors = model_predictor.calculate_descriptors_df(smile)
#     descriptors_results.append(descriptors)
#     # print(descriptors)
# # print(descriptors_results)
# names = ['MolWt', 'exactMolWt', 'qed', 'TPSA', 'HeavyAtomMolWt', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'FpDensityMorgan1', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'Ipc', 'Kappa2', 'LabuteASA', 'PEOE_VSA10', 'PEOE_VSA2', 'SMR_VSA10', 'SMR_VSA4', 'SlogP_VSA2', 'SlogP_VSA6','MaxEStateIndex', 'MinEStateIndex', 'EState_VSA3', 'EState_VSA8', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount']
# descriptors_df = pd.DataFrame(descriptors_results, columns=names)
    
# # print(descriptors_df)

# descriptors_df['SMILES'] = df['SMILES']
# result_df = pd.concat([df, descriptors_df], axis=1)
# print(result_df)
# outputfile = model_predictor.output_file_popup()
# outputfile = "/Users/yycheung/Desktop/resources/result.csv"
df.to_csv("result.csv", index=False)
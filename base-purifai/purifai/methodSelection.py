from importlib.util import spec_from_file_location
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.filedialog import asksaveasfile
from tkinter.messagebox import showinfo
import pickle
import wget
class model_selection:
    def __init__(self, 
                 saved_spe_model , 
                 saved_spe_scaler,
                 saved_lcms_model ,
                 saved_lcms_scaler):
        #if the saved_model is not empty load the saved_model to self.model
        if saved_spe_model != None:
            self.spe_model = pickle.load(open(saved_spe_model,'rb'))
        else:
            print('Where is the saved spe model!!!???')
            return
        if saved_spe_scaler != None:
            self.spe_scaler = pickle.load(open(saved_spe_scaler,'rb'))
        else:
            print('Where is the saved spe scaler!!!?')
            return
        
        if saved_lcms_model != None:
            self.lcms_model = pickle.load(open(saved_lcms_model,'rb'))
        else:
            print('Where is the saved lcms model!!!???')
            return
        if saved_lcms_scaler != None:
            self.lcms_scaler = pickle.load(open(saved_lcms_scaler,'rb'))
        else:
            print('Where is the saved lcms scaler!!!?')
            return
        self.descs = []
    
    # def calculate_descriptors(self, smiles, ipc_avg=False):
    #     mol = Chem.MolFromSmiles(smiles)
    #     names = ['MolWt', 'exactMolWt', 'qed', 'TPSA', 'HeavyAtomMolWt', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'FpDensityMorgan1', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'Ipc', 'Kappa2', 'LabuteASA', 'PEOE_VSA10', 'PEOE_VSA2', 'SMR_VSA10', 'SMR_VSA4', 'SlogP_VSA2', 'SlogP_VSA6','MaxEStateIndex', 'MinEStateIndex', 'EState_VSA3', 'EState_VSA8', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount']
    #     if names is None:
    #         names = [d[0] for d in Descriptors._descList]
    #     calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)

    #     self.descs = [calc.CalcDescriptors(mol)]
    #     descs_df = pd.DataFrame(self.descs, columns=names)
    #     # print(descs_df)
    #     if 'Ipc' in names and ipc_avg:
    #         self.descs['Ipc'] = Descriptors.Ipc(mol, avg=True)
    #     return descs_df
    
    def calculate_descriptors(self, smiles, ipc_avg=False):
        mol = Chem.MolFromSmiles(smiles)
        names = ['MolWt', 'ExactMolWt', 'qed', 'TPSA', 'HeavyAtomMolWt', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'FpDensityMorgan1', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'Ipc', 'Kappa2', 'LabuteASA', 'PEOE_VSA10', 'PEOE_VSA2', 'SMR_VSA10', 'SMR_VSA4', 'SlogP_VSA2', 'SlogP_VSA6','MaxEStateIndex', 'MinEStateIndex', 'EState_VSA3', 'EState_VSA8', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount']
        if names is None:
            names = [d[0] for d in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
        descriptors = calc.CalcDescriptors(mol)
        if 'Ipc' in names and ipc_avg:
            descriptors['Ipc'] = [Descriptors.Ipc(mol, avg=True)]
        return descriptors
    
    def RunSPEPrediction(self, smiles):
        features = [self.calculate_descriptors(smiles)]
        features_scaled = self.spe_scaler.transform(features)
        features_scaled_df = pd.DataFrame(features_scaled)
        features_scaled_df.to_csv('features.csv')
        y = self.spe_model.predict(features_scaled_df)
        return y
    
    def RunLCMSPrediction(self, smiles):
        features = [self.calculate_descriptors(smiles)]
        features_scaled = self.lcms_scaler.transform(features)
        features_scaled_df = pd.DataFrame(features_scaled)
        y = self.lcms_model.predict(features_scaled_df)
        return y
    
if __name__ == '__main__':
    
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
    # spe_scaler = cwd + '/spe_scaler.pkl'
    spe_scaler = '/Users/yycheung/Analysis project/purifAI/MachineLearning/SPE/spe_scaler1.pkl'
    lcms_xgb_model = cwd + '/lcms_xgb_model.pkl'
    # lcms_scaler = cwd + '/lcms_scaler.pkl'
    lcms_scaler = '/Users/yycheung/Analysis project/purifAI/MachineLearning/LCMS/lcms_scaler1.pkl'

    model_predictor = model_selection(spe_xgb_model, 
                                spe_scaler,
                                lcms_xgb_model,
                                lcms_scaler)
    
    # smiles = "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N"
    showinfo(title="Select SMILES List (CSV)", message="Select the list of structures' SMILES to process. NOTE: Column header must be 'SMILES'.")
    inputfile = fd.askopenfilename()

    # created descriptors_df
    df = pd.read_csv(inputfile)
    df = df.dropna(subset=['SMILES'])
    print(df)
    smiles = df['SMILES'].to_list()
    
    df["PREDICTED_SPE_METHOD"] = ''
    df["PREDICTED_LCMS_METHOD"] = ''
    
    # iterate through the smiles list and perform ml perdiction 
    for i in range(len(df)):
        smile = df.loc[i, 'SMILES']
        
        predicted_SPE_method = model_predictor.RunSPEPrediction(smile)
        df.loc[i, "PREDICTED_SPE_METHOD"] = str(predicted_SPE_method)
        print("RunSPEPrediction succesful...")
        
        
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


    # descs = [model_selection.calculate_descriptors(smiles)]
    
    # print(f'The SPE method you should use is : {model_selection.RunSPEPrediction(smiles)}')
    # print(f'The LCMS method you should use is : {model_selection.RunLCMSPrediction(smiles)}')
        


# import os
# os.system("pip install -r requirements.txt")
import time
import pandas as pd
# from lcms_ml_functions import *
from graphical_interface import gui_input_file, gui_outputfile_file
from methodPredictor import RunSPEPrediction, RunLCMSPrediction
from chemCalculate import calculate_descriptors


if __name__ == "__main__":

    print("PurifAI Application Initialzing.......")

    # inputfile = gui_input_file()
    inputfile = "/Users/lukeperrin/Downloads/smiles_list.csv"
    df = pd.read_csv(inputfile)
    df = df.dropna(subset=['SMILES'])
    print(df)
    smiles = df['SMILES'].to_list()


    df["PREDICTED_SPE_METHOD"] = ""
    df["PREDICTED_LCMS_METHOD"] = ""

    for i in range(len(df)):
        smiles = df.loc[i, 'SMILES']
        print(smiles)
        predicted_SPE_method = RunSPEPrediction(smiles)
        df.loc[i, "PREDICTED_SPE_METHOD"] = str(predicted_SPE_method)
        print("RunSPEPrediction succesful...")
        predicted_LCMS_method = RunLCMSPrediction(smiles)
        df.loc[i, "PREDICTED_LCMS_METHOD"] = str(predicted_LCMS_method)

    # Generate structure data (features)
    descriptors_results = []
    for smile in smiles:
        descriptors = calculate_descriptors(smile)
        descriptors_results.append(descriptors)
        print(descriptors)
    print(descriptors_results)
    names = ['MolWt', 'ExactMolWt', 'qed', 'TPSA', 'HeavyAtomMolWt', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'FpDensityMorgan1', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'Ipc', 'Kappa2', 'LabuteASA', 'PEOE_VSA10', 'PEOE_VSA2', 'SMR_VSA10', 'SMR_VSA4', 'SlogP_VSA2', 'SlogP_VSA6','MaxEStateIndex', 'MinEStateIndex', 'EState_VSA3', 'EState_VSA8', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount']
    descriptors_df = pd.DataFrame(descriptors_results, columns=names)
    print(descriptors_df)

    descriptors_df['SMILES'] = df['SMILES']
    result_df = pd.concat([df, descriptors_df], axis=1)
    print(result_df)
    # outputfile = gui_outputfile_file()
    outputfile = "/Users/lukeperrin/Downloads/predictor_results.csv"
    result_df.to_csv(outputfile, index=False)



from purifai.methodSelection import model_selection


model_predictor = model_selection("/Users/yycheung/Analysis project/purifAI/spe_brf_model.pkl", 
                            "/Users/yycheung/Analysis project/purifAI/spe_scaler.pkl",
                            "/Users/yycheung/Analysis project/purifAI/lcms_brf_model.pkl",
                            "/Users/yycheung/Analysis project/purifAI/lcms_scaler.pkl")
smiles = "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N"
descs = [model_predictor.calculate_descriptors(smiles)]
print(f'The SPE method you should use is : {model_predictor.RunSPEPrediction(smiles)}')
print(f'The LCMS method you should use is : {model_predictor.RunLCMSPrediction(smiles)}')
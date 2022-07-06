
from purifai.methodSelection import model_selection
import os
import wget

# Auto download the saved model and scaler files
cwd = os.getcwd()
url = 'https://github.com/jenamis/purifAI/raw/main/machine_learning/SPE/models/'
if not os.path.exists(os.getcwd() + '/spe_brf_model.pkl'):
    wget.download(url+ 'spe_xgb_model.pkl')
if not os.path.exists(os.getcwd() + '/spe_scaler.pkl'):
    wget.download(url+ 'spe_scaler.pkl')
url= 'https://github.com/jenamis/purifAI/raw/main/machine_learning/LCMS/models/'
if not os.path.exists(os.getcwd() + '/lcms_brf_model.pkl'):
    wget.download(url+ 'lcms_xgb_model.pkl')
if not os.path.exists(os.getcwd() + '/lcms_scaler.pkl'):
    wget.download(url+ 'lcms_scaler.pkl')

# define the file path
spe_xgb_model = cwd + '/spe_xgb_model.pkl'
spe_scaler = cwd + '/spe_scaler.pkl'
lcms_xgb_model = cwd + '/lcms_xgb_model.pkl'
lcms_scaler = cwd + '/lcms_scaler.pkl'

# load the function
model_predictor = model_selection(spe_xgb_model, 
                            spe_scaler,
                            lcms_xgb_model,
                            lcms_scaler)

# define the input smiles
smiles = "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N"

# calcualting the chemical features and result in a list of list
descs = [model_predictor.calculate_descriptors(smiles)]

# call the function and perform prediction on the input smiles
print(f'The SPE method you should use is : {model_predictor.RunSPEPrediction(smiles)}')
print(f'The LCMS method you should use is : {model_predictor.RunLCMSPrediction(smiles)}')
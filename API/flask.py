
from methodSelection import model_selection
from chemCalculate import calculate_descriptors

spe_model = model_selection()
smiles = "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N"
descs = [calculate_descriptors(smiles)]
print(f'The SPE method you should use is : {spe_model.RunSPEPrediction(smiles)}')
print(f'The LCMS method you should use is : {spe_model.RunLCMSPrediction(smiles)}')
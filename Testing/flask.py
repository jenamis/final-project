from purifai.methodSelection import model_selection

spe_model = model_selection('/Users/yycheung/Analysis project/purifAI/Testing/spe_brf_model.pkl', 
                            '/Users/yycheung/Analysis project/purifAI/Testing/spe_scaler.pkl',
                            '/Users/yycheung/Analysis project/purifAI/Testing/lcms_brf_model.pkl',
                            '/Users/yycheung/Analysis project/purifAI/Testing/lcms_scaler.pkl')
<<<<<<< Updated upstream
smiles = "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N"
=======
smiles = "ClC1=C(Cl)C=C(N2CCN(C(C3=CC(NC(N3)=O)=O)=O)CC2)C=C1"
>>>>>>> Stashed changes
descs = [spe_model.calculate_descriptors(smiles)]
print(f'The SPE method you should use is : {spe_model.RunSPEPrediction(smiles)}')
print(f'The LCMS method you should use is : {spe_model.RunLCMSPrediction(smiles)}')
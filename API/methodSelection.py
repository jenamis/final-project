from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from chemCalculate import calculate_descriptors
import pickle
class model_selection:
    def __init__(self, saved_spe_model = 'API/spe_brf_model.pkl', 
                 saved_spe_scaler= 'API/spe_scaler.pkl',
                 saved_lcms_model = 'API/lcms_brf_model.pkl',
                 saved_lcms_scaler= 'API/lcms_scaler.pkl'):
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

    def RunSPEPrediction(self, smiles):
        features = calculate_descriptors(smiles)
        features_scaled = self.spe_scaler.transform(features)
        y = self.spe_model.predict(features_scaled)
        return y
    
    def RunLCMSPrediction(self, smiles):
        features = calculate_descriptors(smiles)
        features_scaled = self.lcms_scaler.transform(features)
        y = self.lcms_model.predict(features_scaled)
        return y
    
if __name__ == '__main__':
    model_object = model_selection()
    smiles = "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N"
    descs = [calculate_descriptors(smiles)]
    print(f'The SPE method you should use is : {model_object.RunSPEPrediction(smiles)}')
    print(f'The LCMS method you should use is : {model_object.RunLCMSPrediction(smiles)}')
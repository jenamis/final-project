from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from chemCalculate import calculate_descriptors
import pickle



def RunSPEPrediction(self):
    saved_spe_model = '/Users/lukeperrin/Documents/UCB Data Bootcamp/group-7-project/purifAI_luke/purifAI_predictor_app/models/spe_brf_model.pkl'
    saved_spe_scaler = '/Users/lukeperrin/Documents/UCB Data Bootcamp/group-7-project/purifAI_luke/purifAI_predictor_app/models/spe_scaler.pkl'
    if saved_spe_model != None:
        spe_model = pickle.load(open(saved_spe_model, 'rb'))
    else:
        print('Where is the saved spe model!!!???')

    if saved_spe_scaler != None:
        spe_scaler = pickle.load(open(saved_spe_scaler, 'rb'))
        return spe_scaler
    else:
        print('Where is the saved spe scaler!!!?')
    features = calculate_descriptors(self)
    features_scaled = spe_scaler.transform(features)
    predicted_SPE_method = spe_model.predict(features_scaled)
    return predicted_SPE_method


def RunLCMSPrediction(self):
    saved_lcms_model = '/Users/lukeperrin/Documents/UCB Data Bootcamp/group-7-project/purifAI_luke/purifAI_predictor_app/models/lcms_brf_model.pkl'
    saved_lcms_scaler = '/Users/lukeperrin/Documents/UCB Data Bootcamp/group-7-project/purifAI_luke/purifAI_predictor_app/models/lcms_scaler.pkl'
    if saved_lcms_model != None:
        lcms_model = pickle.load(open(saved_lcms_model, 'rb'))
    else:
        print('Where is the saved lcms model!!!???')
    if saved_lcms_scaler != None:
        lcms_scaler = pickle.load(open(saved_lcms_scaler, 'rb'))
    else:
        print('Where is the saved lcms scaler!!!?')
    features = calculate_descriptors(self)
    features_scaled = lcms_scaler.transform(features)
    features_scaled.reshape(1, -1)
    predicted_LCMS_method = lcms_model.predict(features_scaled)

    return predicted_LCMS_method



# smiles = "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N"
# prediction = predict_from_smiles(smiles)
# print(prediction)


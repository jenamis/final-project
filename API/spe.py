from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from chemCalculate import calculate_descriptors
import pickle
class spe:
    def __init__(self, saved_model_filename = None):
        
        #if the saved_model is not empty load the saved_model to self.model
        if saved_model_filename != None:
            self.model = pickle.load(open(saved_model_filename, 'rb'))

    def RunPrediction(self, smiles):
        features = calculate_descriptors(smiles)
        y = self.model.predict(features)
        return y
if __name__ == '__main__':
    model_object = spe('API/spe_brf_model.sav')
    smiles = "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N"
    # print(model_object.RunPrediction([[-0.23601421, -0.23375002,  0.48955629, -0.8890365 , -0.17761245,
    #     0.1709978 , -0.90123393,  0.10786976, -0.1741022 ,  1.29234626,
    #    -0.15320403,  0.13847805,  1.66615658, -0.41771504,  0.5498574 ,
    #    -0.0618161 , -0.36739205, -0.54264441, -0.36450782,  0.00952119,
    #    -1.06737328, -0.45597801, -0.50077368, -0.80388822,  0.51854072,
    #    -1.48409244, -0.08814202, -0.87484169, -0.32171426, -0.32951952,
    #    -1.12285645, -0.23349792, -0.91183957, -0.95433182, -0.8439449 ,
    #     0.62572119, -0.22570688, -0.7941993 , -0.3141551 ,  0.6681677 ,
    #    -0.10242044, -0.20506081, -0.59315847, -0.64066596, -1.15652533]]))
    model_object.RunPrediction(smiles)

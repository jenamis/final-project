from purifai.methodSelection import model_selection
import pandas as pd
from graphical_interface import gui_input_file, gui_outputfile_file
from contextlib import redirect_stdout
import warnings
from datetime import datetime


def predict_methods_from_smiles(self):

    model_predictor = model_selection("static/models/spe_brf_model.pkl",
                                "static/models/spe_scaler.pkl",
                                "static/models/lcms_brf_model.pkl",
                                "static/models/lcms_scaler.pkl")

    predicted_spe_model = model_predictor.RunSPEPrediction(self)
    predicted_lcms_model = model_predictor.RunLCMSPrediction(self)
    return predicted_spe_model, predicted_lcms_model

def evaluate_structures(self):

    df = pd.read_csv(self)
    df = df.dropna(subset=['SMILES'])

    df["PREDICTED_SPE_METHOD"] = ""
    df["PREDICTED_LCMS_METHOD"] = ""

    N = len(df['SMILES'])

    for i in range(N):

        smiles = df.loc[i, 'SMILES']

        result = predict_methods_from_smiles(smiles)

        predicted_SPE_method = result[0]
        df.loc[i, "PREDICTED_SPE_METHOD"] = predicted_SPE_method

        predicted_LCMS_method = result[1]
        df.loc[i, "PREDICTED_LCMS_METHOD"] = predicted_LCMS_method

    return df


if __name__ == "__main__":

    timestamp = datetime.now()

    timestamp = str(timestamp).replace(" ", "_@T=")

    with open(f"log_outfiles/{timestamp}_log.txt", 'w') as f:

        print("File began running:")
        print(timestamp)

        with redirect_stdout(f):

            print('data')

            warnings.filterwarnings("ignore")

            file_select = gui_input_file()
            predictions_df = evaluate_structures(file_select)

            try:
                predictions_df.to_csv(file_select, index=False)
                print("Application completed without errors.")

            except:
                print("Application completed with errors.")




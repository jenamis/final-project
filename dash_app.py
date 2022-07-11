from operator import index
import dash
import purifai
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
import base64
import datetime
import dash_bootstrap_components as dbc     
from purifai.methodSelection import model_selection


names = ['MolWt', 'exactMolWt', 'qed', 'TPSA', 'HeavyAtomMolWt', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'FpDensityMorgan1', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'Ipc', 'Kappa2', 'LabuteASA', 'PEOE_VSA10', 'PEOE_VSA2', 'SMR_VSA10', 'SMR_VSA4', 'SlogP_VSA2', 'SlogP_VSA6','MaxEStateIndex', 'MinEStateIndex', 'EState_VSA3', 'EState_VSA8', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount']

model_predictor = model_selection("static/models/spe_brf_model.pkl",
                            "static/models/spe_scaler.pkl",
                            "static/models/spe_brf_model.pkl",
                            "static/models/lcms_scaler.pkl")

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Textarea(
        id='textarea-input',
        value='Enter a list of SMILES here \n(one per line)',
        style={'width': '100%', 'height': 300},
    ),
    # html.Div(id='output-data', style={'whiteSpace': 'pre-line'}),
    # html.Div(dcc.Input(id='input-data', type='text')),
    dbc.Button('Submit', id='submit-button', className="me-2", size="lrg", n_clicks=0),
    html.Div(id='submit-text',
             children='Click here to submit list'),
    html.Div(id='textarea-output', style={'whiteSpace': 'pre-line'}),
    html.Div(id='table-output')
])

@app.callback(
    Output("table-output", "children"), [Input("submit-button", "n_clicks"), Input("textarea-input", "value")],
)
def on_button_click(n_clicks, contents):
    if n_clicks > 0:
        table = html.Div()
        columns = ['SMILES', 'PREDICTED_SPE_METHOD', 'PREDICTED_SPE_METHOD']
        prediction_df = pd.DataFrame(columns=columns)
        descriptors_df = pd.DataFrame(columns=names)
        df = pd.DataFrame(columns=[columns + names])
            
        
        print(contents)
        smiles_list = contents.split('\n')
        # smiles_list.pop()
        prediction_df['SMILES'] = smiles_list
        print(smiles_list)
        
        x = 0
        for i in range(len(prediction_df['SMILES'])):
            smiles = prediction_df.loc[i, 'SMILES']
            print(f"SMILES entry received: {smiles}")
            
            descriptors = model_selection.calculate_descriptors(self=model_selection, smiles=smiles)
            descriptors_temp = pd.DataFrame(descriptors,columns=names)
            descriptors_df = pd.concat([descriptors_df, descriptors_temp])
            print(f"Molecular descriptors calculated for entry {i}:")
            print(descriptors_df.head())
                            
            predicted_SPE_method = model_predictor.RunSPEPrediction(features=descriptors)
            prediction_df.loc[i, "PREDICTED_SPE_METHOD"] = str(predicted_SPE_method[0])
            print(f"SPE prediction succesful...\n Predicted {predicted_SPE_method} SPE method for entry {i}")
            
            predicted_LCMS_method = model_predictor.RunLCMSPrediction(features=descriptors)
            prediction_df.loc[i, "PREDICTED_LCMS_METHOD"] = str(predicted_LCMS_method[0])
            print(f"LCMS prediction succesful...\n Predicted {predicted_LCMS_method} LCMS method for entry {i}")
            
            df = prediction_df.merge(descriptors_df, left_index=True, right_index=True)

        table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

        return table
    

if __name__ == '__main__':
    
    app.run_server(debug=True)
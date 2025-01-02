"""
Docker Container: https://hub.docker.com/r/continuumio/anaconda3
RDKit Installation: https://www.rdkit.org/docs/Install.html
"""
import pandas as pd
import streamlit as st
import time
import numpy as np
import streamlit.components.v1 as components
from PIL import Image
import streamlit.components.v1 as components
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import pickle
import keras
import tensorflow as tf
import base64
from pandas.io.formats.style import Styler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, random_split
import torch_geometric
from torch_geometric.loader import DataLoader

from Bio import SeqIO 
from io import StringIO

from dataprocessing import sequenceEncoding, peptide_sequence_to_graph, dBLOSUM
from model import DeepCPP

st.set_page_config(
    page_title="DeepCPP",
    page_icon="icon.ico",
    layout="centered",
    initial_sidebar_state="collapsed"
    
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
        visibility: hidden;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# image = Image.open('site_name.png')

# st.image(image, caption='')

st.markdown(
    """
    <style>
    .css-6qob1r{
        background-color: #0A7090; /* Replace with your desired background color */
    }
    .css-17lntkn{
        color: #E4F7FD;
        font-size: 18px;
    }
    .css-nziaof{
        background-color: #063649;
    }
    .css-nziaof:hover{
        background-color: #195A74;
    }
    .css-pkbazv{
        color:#C3D0D5;
        font-size: 18px;
    }
    .st-dk{
        color: orange;
    }
    .st-emotion-cache-phe2gf, .st-emotion-cache-0{
        text-align:center;
    }
    .st-emotion-cache-78kz3v{
        text-align:left;
    }
    </style>
    """,
    unsafe_allow_html=True
)
weel1aimg="""
    <style> 
        .my_site_name{
            text-align:center
        }
    </style>
    <div class='my_site_name'>
       <img src='https://deepcpp.streamlit.app:443/~/+/media/dac66dfcb8b2835a251b3ed21fdb583fc9ad67af00f508b9007f5a9b.png'>
    </div>
    """
# st.image("site_name.png", caption="")

st.write(weel1aimg, unsafe_allow_html=True)
weel1a2="""
    <style> 
        .mywell{
            display:block;
            font-size: 15px;
            text-align:justify;
            border-radius: 4px;
            box-shadow: 0 1px 1px rgb(0 0 0 / 5%);
        }
        .mywell {
            font-family: "Times",Helvetica,Arial,sans-serif;
            border:1px solid #ddd;
            border-radius: 4px;
            padding-left: 10px;
            padding-right:10px;
            padding-bottom:10px;
            background-color:#FAFBFC;
            box-shadow: 0 1px 1px rgb(0 0 0 / 5%);
        }
    </style>
    <div class='mywell'>
        <p>
            A web service of the DeepCPP model was build to predict the permation ability of pepties determining if a peptide is a 
            cell penetranating peptide (CPP) or not (non-CPP). 
        </p>
        <h6 style='padding-bottom:0rem;'>Input: </h6>
        <p>
            <ol>
              <li>sigle line prediction: the user can provide peptide sequence in the input box.</li>
              <li>Batch prediction: Provide a list of peptide sequence in FASTA file </li>
            </ol>
            <b>Note:</b> peptide sequences 5 - 30 residues in length
        </p>
        <p>
            <h6 style='padding-bottom:0rem;'>Output: </h6>
            The web service tool predict whether the query peptide is a CPP or non-CPP, and provides the confidence score of prediction, a value
            between 0 and 1, which denotes the level of likeliness of the peptide to be cell penetrating peptide. where a score close to 1 denotes a strong
            confidence from the tool that the peptide is a CPP, and a score close to 0 a strong confidence that the peptide is a non-CPP.
        </p>
        Click <a href="Help" target="_blank">here</a> For more information.
    </div>
    """

st.write(weel1a2, unsafe_allow_html=True)

font_css = """
<style>
button[data-baseweb="tab"] {
    width:50%;
    font-size:17px;
}
.css-16idsys p{
    font-size:17px;
}
.stButton{
    text-align:left;
}
.stDownloadButton{
    text-align:left;
}
</style>
"""

st.write(font_css, unsafe_allow_html=True)
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

def valid_peptideseq(cpp_sequence):

    # Define the set of conventional amino acids
    valid_amino_acids = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
    
    # Check sequence length
    if not (5 <= len(cpp_sequence) <= 30):
        return False
    
    # Check if all characters in the sequence are valid amino acids
    for residue in cpp_sequence:
        if residue.upper() not in valid_amino_acids:
            return False
    return True

# def valid_fasta(fasta_input):
#     valid_amino_acids = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
#                          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
    
#     # Dictionary to store validation results
#     validation_results = {}
    
#     try:
#         # # Handle file-like object or string content
#         # if hasattr(fasta_input, 'read'):  # Uploaded file-like object
#         #     fasta_content = fasta_input.read().decode('utf-8')  # Decode to string
#         # elif isinstance(fasta_input, list):  # List of lines
#         fasta_content = "\n".join(fasta_input)
#         # elif isinstance(fasta_input, str):  # Direct string content
#         #     fasta_content = fasta_input
#         # else:
#         #     raise ValueError("Unsupported input type for FASTA file")

#         # Parse the FASTA content
#         for record in SeqIO.parse(fasta_content.splitlines(), "fasta"):
#             sequence = str(record.seq).upper()
#             # Validate sequence length and amino acids
#             if 5 <= len(sequence) <= 30 and all(residue in valid_amino_acids for residue in sequence):
#                 validation_results[record.id] = True
#             else:
#                 validation_results[record.id] = False
#     except Exception as e:
#         raise ValueError(f"Error reading FASTA file: {e}")
    
#     return validation_results

def valid_fasta(sequence):

    valid_amino_acids = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}

    validation_result = 4

    try:
        if 5 <= len(sequence) <= 30 and all(residue in valid_amino_acids for residue in sequence):
            validation_result = 0
        if (5 > len(sequence) or len(sequence)> 30) and all(residue in valid_amino_acids for residue in sequence) == False:
            validation_result = 1    
        elif 5 > len(sequence) or len(sequence)> 30:
            validation_result = 2
        elif all(residue in valid_amino_acids for residue in sequence) == False:
            validation_result = 3
    except Exception as e:
        raise ValueError(f"Error reading FASTA file: {e}")

    return validation_result


def color_prediction(val):
            color = '#ED554E' if val>=0.5 else ''
            return f'background-color: {color}'

##Building model
def build_model_single(df_single):
    device = torch.device("cpu")
    data_test_graph = peptide_sequence_to_graph(df_single)
    X_data_test_one_hot_encoded = sequenceEncoding(dBLOSUM, df_single) #seq_to_categorical_new(test_d)

    X_test_one_hot_tensor = torch.tensor(X_data_test_one_hot_encoded, dtype=torch.float32)
    test_seq_dataset = TensorDataset(X_test_one_hot_tensor)

    test_graph_loader = DataLoader(data_test_graph, batch_size=64)
    test_seq_loader = DataLoader(test_seq_dataset, batch_size=64)

    loaded_model = pickle.load(open('DeepCPPpredNew.pkl', 'rb')) 

    # Evaluate the model
    loaded_model.eval()
    with torch.no_grad():
        for (graph_batch, seq_batch) in zip(test_graph_loader, test_seq_loader):
            graph_data = graph_batch.to(device)
            seq_data = seq_batch[0].to(device)

            outputs = loaded_model(graph_data, seq_data).view(-1, 1)

            preds = torch.sigmoid(outputs).cpu().numpy()

    if preds >= 0.5:
        emoticon = "✔️"
    else:
        emoticon = "✖️"
    prediction_probability = pd.Series(preds.squeeze(), name='Probability')
    prediction_output = pd.Series(emoticon, name='CPP')
    molecule_name = pd.Series(df_single.sequence, name='Peptide Sequence')
    df = pd.concat([molecule_name, prediction_probability,prediction_output], axis=1)
    return df

def predict_peptide(cpp_sequence):
    device = torch.device("cpu")
    mol_class = ""
    molecule_name = pd.Series(cpp_sequence, name='sequence')
    mol_df = pd.concat([molecule_name], axis=1)
    
    data_test_graph = peptide_sequence_to_graph(mol_df)
    X_data_test_one_hot_encoded = sequenceEncoding(dBLOSUM, mol_df) 
    
    X_test_one_hot_tensor = torch.tensor(X_data_test_one_hot_encoded, dtype=torch.float32)
    test_seq_dataset = TensorDataset(X_test_one_hot_tensor)
    
    test_graph_loader = DataLoader(data_test_graph, batch_size=64)
    test_seq_loader = DataLoader(test_seq_dataset, batch_size=64)
    loaded_model = pickle.load(open('DeepCPPpred.pkl', 'rb')) 
    
    # Evaluate the model
    loaded_model.eval()
    with torch.no_grad():
        for (graph_batch, seq_batch) in zip(test_graph_loader, test_seq_loader):
            graph_data = graph_batch.to(device)
            seq_data = seq_batch[0].to(device)

            outputs = loaded_model(graph_data, seq_data).view(-1, 1)

            mol_pred = torch.sigmoid(outputs).cpu().numpy()
            
    if mol_pred >= 0.5:
        mol_class = "CPP"
    else:
        mol_class = "Non-CPP"
    return mol_pred, mol_class

tab1, tab2= st.tabs(["Single Prediction", "Batch Prediction"])


with tab1:
    cpp_sequence = st.text_input("Enter peptide sequence", key="cpp_sequence")
    # You can access the value at any point with:
    if st.button('Prediction', key='prediction1'):
            molecule_name = pd.Series(cpp_sequence, name='sequence')

            if cpp_sequence =="":
                import streamlit as st
                st.warning('Empty field Error: Please Enter a Peptide sequence before pressing the "Prediction" button', icon="⚠️")
            else:
                if valid_peptideseq(cpp_sequence) == False:
                    st.warning('input Error: Please Enter a valid Peptide sequence', icon="⚠️")
                else:
                    df = pd.concat([molecule_name], axis=1)
                    data = build_model_single(df)
                    # data = data.applymap(lambda x: x.item() if isinstance(x, (np.ndarray, np.generic)) else x)
                    table_title="<style> </style><div class='css-x9krnl ';style='display:block;'><p style='text-align:center;font-weight:bold;font-family:times;'>Cell Penetrating Peptide prediction output</p></div>"

                    st.markdown(table_title, unsafe_allow_html=True)
                    # AgGrid(data)            
                    st.dataframe(data)
            

with tab2:
    with open('test.fasta') as f:
        st.download_button('Download Example input file', f,'test.fasta')
    # st.markdown('Upload FASTA file containing list of peptide sequences', unsafe_allow_html=True)
    st.header('Upload FASTA file of peptide sequences')
    uploaded_file = st.file_uploader("", type=["fasta"])

    if st.button('Prediction', key='prediction2'):


        if uploaded_file:
            try:
                with st.spinner('Prediction...'):
                    fasta_content = uploaded_file.read().decode("utf-8")

                    fasta_content_in = []
                    all_preds = []
                    all_probabilities = []
                    all_class = []
                    emoticon = []
                    all_pepsequence = []
                    # st.write(fasta_content)

                    fasta_io = StringIO(fasta_content)
                    for record in SeqIO.parse(fasta_io, "fasta"):
                        # Validate the FASTA content

                        peptide_sequence = str(record.seq).upper()
                        is_valid = valid_fasta(peptide_sequence)

                        if is_valid == 0:
                            prediction_score, mol_class = predict_peptide(peptide_sequence)

                            if mol_class == 'CPP':
                                emoticon.append("✔️")
                            else:
                                emoticon.append("✖️")
                        else:
                            prediction_score = [0]  # Wrap in a list to make it iterable

                            mol_class = "Invalid - "+ str(is_valid)
                            emoticon.append("⚠️")

                        all_pepsequence.append(peptide_sequence)
                        all_class.append(mol_class)
                        all_probabilities.extend(prediction_score)
                    prediction_output = pd.Series(emoticon, name='CPP')
                    prediction_classes = pd.Series(all_class, name='Class')
                    prediction_prob = pd.Series(all_probabilities,name='Probability')
                    molecule_seq = pd.Series(all_pepsequence, name='Peptide Sequence')
                    df = pd.concat([molecule_seq, prediction_prob, prediction_output], axis=1)
                    df['Probability'] = df['Probability'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
                    df_download = pd.concat([molecule_seq, prediction_prob, prediction_classes], axis=1)
                    df_download['Probability'] = df_download['Probability'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

                    table_title="<style> </style><div class='css-x9krnl ';style='display:block;'><p style='text-align:center;font-weight:bold;font-family:times;'>Cell Penetrating Peptide prediction output</p></div>"

                    st.markdown(table_title, unsafe_allow_html=True)
                    styler = Styler(df)

                    styler.set_table_styles(
                        [{'selector': 'Inhibitor', 'props': [('text-align', 'center')]}]
                    )

                    st.dataframe(styler)
                    #st.write(df)
                    st.markdown(filedownload(df_download), unsafe_allow_html=True)
            except ValueError as e:
                st.error(f"Error: {e}")

        else:
            st.info("Please upload a FASTA file.")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

footer="<style> </style><div class='css-x9krnl footer';style='display:block;text-align:center;'><p style='text-align:center;'>Copyright©2024 <a href='https://sites.google.com/sunmoon.ac.kr/dslab/research?authuser=0' target='_blank'> D&S Lab, Sunmoon University</a></p></div>"

st.markdown(footer, unsafe_allow_html=True)

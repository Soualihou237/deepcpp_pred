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
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import PandasTools, Draw
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

from dataprocessing import sequenceEncoding, peptide_sequence_to_graph
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

image = Image.open('site_name.png')

st.image(image, caption='')

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
            <b>DeepCPPpred</b> is a web service for the DeepCPP model, build to predict the permation ability of pepties determining if a peptide is a 
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
            The DeepCPPpred tool predict whether the query peptide is a CPP or non-CPP, and provides the confidence score of prediction, a value
            between 0 and 1, which denotes the level of likeliness of the peptide to be cell penetrating peptide. where a score close to 1 denotes a strong
            confidence from the DeepCPPpred that the peptide is a CPP, and a score close to 0 a strong confidence that the peptide is a non-CPP.
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

def valid_fasta(fasta_content):
    """
    Validate sequences in a FASTA file.

    Parameters:
        fasta_content: A string representing the FASTA content.

    Returns:
        dict: A dictionary with sequence IDs as keys and True/False as values,
              indicating whether each sequence is valid.
    """
    valid_amino_acids = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}

    validation_results = {}
    try:
        # Use StringIO to handle the content as a file-like object
        fasta_io = StringIO(fasta_content)
        for record in SeqIO.parse(fasta_io, "fasta"):
            sequence = str(record.seq).upper()
            if 5 <= len(sequence) <= 30 and all(residue in valid_amino_acids for residue in sequence):
                validation_results[record.id] = 1
            if 5 < len(sequence) > 30 and all(residue in valid_amino_acids for residue in sequence) == False:
                validation_results[record.id] = 2    
            elif 5 > len(sequence) or len(sequence)> 30:
                validation_results[record.id] = 3
            elif all(residue in valid_amino_acids for residue in sequence) == False:
                validation_results[record.id] = 4
            # else:
            #     validation_results[record.id] = 4
    except Exception as e:
        raise ValueError(f"Error reading FASTA file: {e}")

    return validation_results


dBLOSUM = { # CPP_sequence_length x 20
    'A':[ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],
    'C':[ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    'D':[-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],
    'E':[-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],
    'F':[-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],
    'G':[ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],
    'H':[-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],
    'I':[-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],
    'K':[-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],
    'L':[-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],
    'M':[-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],
    'N':[-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],
    'P':[-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],
    'Q':[-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],
    'R':[-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],
    'S':[ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],
    'T':[ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],
    'V':[ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],
    'W':[-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],
    'Y':[-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],
    'p':[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],#padding   
}





def color_prediction(val):
            color = '#ED554E' if val>=0.5 else ''
            return f'background-color: {color}'

##Building model
def build_model_single(df_single):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_test_graph = peptide_sequence_to_graph(df_single)
    X_data_test_one_hot_encoded = sequenceEncoding(dBLOSUM, df_single) #seq_to_categorical_new(test_d)

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

def build_model_batch(fasta_bach_content):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fasta_content_in = []
    all_preds = []
    probabilities = []
    all_class = []
    fasta_io = StringIO(fasta_content)
    for record in SeqIO.parse(fasta_io, "fasta"):
        fasta_content_in.append(str(record.seq).upper())

    dataFrame = pd.DataFrame(fasta_content_in, columns=['sequence']) #pd.Series(fasta_content_in, name='sequence')

    data_test_graph = peptide_sequence_to_graph(dataFrame)
    X_data_test_one_hot_encoded = sequenceEncoding(dBLOSUM, dataFrame) #seq_to_categorical_new(test_d)

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
            preds = torch.sigmoid(outputs).cpu().numpy() 
            probabilities.extend(preds)
            all_preds.extend(preds)

    emoticon = []
    for mol_pred in all_preds:
        if mol_pred >= 0.5:
            all_class.append("CPP")
            emoticon.append("✔️")
        else:
            all_class.append("Non-CPP")
            emoticon.append("✖️")

    prediction_output = pd.Series(emoticon, name='CPP')
    prediction_classes = pd.Series(all_class, name='Class')
    prediction_prob = pd.Series(probabilities,name='Probability')
    molecule_seq = pd.Series(dataFrame.sequence, name='Peptide Sequence')
    df = pd.concat([molecule_seq, prediction_prob, prediction_output], axis=1)
    df_download = pd.concat([molecule_seq, prediction_prob, prediction_classes], axis=1)

    return df,df_download
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
                fasta_content = uploaded_file.read().decode("utf-8")

                # Validate the FASTA content
                results = valid_fasta(fasta_content)
                fasta_correct = True

                for seq_id, is_valid in results.items():
                    status = ""
                    if is_valid !=1:
                        fasta_correct = False
                        break
                if fasta_correct == False:
                    st.subheader("Result of FASTA File examination")
                    for seq_id, is_valid in results.items():
                        # st.write(seq_id, is valid)
                        status = ""
                        if is_valid ==1:
                            status = "✅ Valid"
                        elif is_valid == 2 :
                            fasta_correct= False
                            "❌ Invalid: peptide sequence length is not respected and contains non-valid residues"
                        elif is_valid == 3:
                            fasta_correct= False
                            "❌ Invalid: peptide sequence length is not respected"
                        elif is_valid == 4:
                            fasta_correct= False
                            "❌ Invalid: peptide sequence contains non-valid residues"
                        st.write(f"**{seq_id}**: {status}")

                else:
                    with st.spinner('Prediction...'):
                        table_title="<style> </style><div class='css-x9krnl ';style='display:block;'><p style='text-align:center;font-weight:bold;font-family:times;'>Cell Penetrating Peptide prediction output</p></div>"

                        st.markdown(table_title, unsafe_allow_html=True)

                        prediction_output,prediction_output_download = build_model_batch(fasta_content)
                        # st.write(df)

                        styler = Styler(prediction_output)

                        # Apply CSS styling to center the cells in the 'Emoji' column
                        styler.set_table_styles(
                            [{'selector': 'Inhibitor', 'props': [('text-align', 'center')]}]
                        )

                        st.dataframe(styler)
                        #st.write(df)
                        st.markdown(filedownload(prediction_output_download), unsafe_allow_html=True)


            except ValueError as e:
                st.error(f"Error: {e}")

        else:
            st.info("Please upload a FASTA file.")


        # if uploaded_file is not None:

        #     # try:
        #     #     results = valid_fasta(uploaded_file)
        #     #     st.write("Validation Results:")
        #     #     st.write(results)
        #     # except ValueError as e:
        #     #     st.error(f"Error: {e}")


        #     non_valid_file = False
        #     # st.write(uploaded_file.read())
        #     results = valid_fasta(uploaded_file.read())

        #     # Print the validation results
        #     for seq_id, is_valid in results.items():
        #         st.write("Sequence ID"+seq_id+"Valid"+is_valid)
        #         if is_valid == True:
        #             non_valid_file = True

        #     if non_valid_file == True:
        #         st.warning('Invalid file content: Please provide a FASTA file with valid Peptide sequences', icon="⚠️")

        #     else:
        #         data = pd.read_csv(uploaded_file_1a2)
                
        #         with st.spinner('Prediction...'):
        #             prediction_output = build_model_batch(data)

        #             styler = Styler(prediction_output)

        #             # Apply CSS styling to center the cells in the 'Emoji' column
        #             styler.set_table_styles(
        #                 [{'selector': 'Inhibitor', 'props': [('text-align', 'center')]}]
        #             )

        #             st.dataframe(styler)
        #             #st.write(df)
        #             st.markdown(filedownload(prediction_output), unsafe_allow_html=True)

        # else:
        #      st.write('Upload a FASTA file....')


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

footer="<style> </style><div class='css-x9krnl footer';style='display:block;text-align:center;'><p style='text-align:center;'>Copyright©2024 <a href='https://sites.google.com/sunmoon.ac.kr/dslab/research?authuser=0' target='_blank'> D&S Lab, Sunmoon University</a></p></div>"

st.markdown(footer, unsafe_allow_html=True)






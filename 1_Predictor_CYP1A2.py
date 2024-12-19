"""
Docker Container: https://hub.docker.com/r/continuumio/anaconda3
RDKit Installation: https://www.rdkit.org/docs/Install.html
"""
import mols2grid
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
from rdkit.Chem import Draw
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



st.set_page_config(
    page_title="MuMCyp_Net",
    page_icon="picon.ico",
    layout="centered",
    
)


image = Image.open('name.png')

st.image(image, caption='Multimodal CYP450 Network predictor')

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
        .mywell > p{
            font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
            border:1px solid #ddd;
            border-radius: 4px;
            padding-left: 10px;
            padding-right:10px;
            padding-top: 10px;
            padding-bottom:10px;
            background-color:#EDF8F9;
            box-shadow: 0 1px 1px rgb(0 0 0 / 5%);
        }
    </style>
    <div class='mywell'>
        <p >
            <b style='border-bottom: 4px double;'>CYP1A2 Predictor:</b> Users have the flexibility to input either a single molecule's SMILES 
            sequence or a CSV file containing a comprehensive list of molecules, each potentially exhibiting 
            hibitory activities on the <b>CYP1A2</b> isozyme.
        </p>
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

#Finger print calculation
class Make_Descriptors:
    
    def __init__(self,df):
        #self.label = df['label']  # Smiles can be change to the name of your label
        self.molecules = [Chem.MolFromSmiles(smiles) for smiles in df.Smiles]
        self.df = df
        
    def _morgan_fingerprints(self, radius, size):
            morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size) for mol in self.molecules]
            return morgan_fps
    
    def calculate_morgan_fp(self, radius=2, size=2048):
        morgan_fps = self._morgan_fingerprints(radius=2, size=2048)
        
        Morgan_fpts = np.array(morgan_fps)
        morgan_fp_df = pd.DataFrame(Morgan_fpts,columns=['Col_{}'.format(i) for i in range(Morgan_fpts.shape[1])])
        #morgan_fp_df['label'] = self.label
        return morgan_fp_df

    def _maccs_fingerprints(self):
            macc_fps = [MACCSkeys.GenMACCSKeys(mol) for mol in self.molecules]
            return macc_fps
        
    def calculate_maccs_fp(self):
        maccs_fps = self._maccs_fingerprints()
        
        Maccs_fpts = np.array(maccs_fps)
        Maccs_fp_df = pd.DataFrame(np.array(Maccs_fpts),columns=['Col_{}'.format(i) for i in range(Maccs_fpts.shape[1])]) 
        #Maccs_fp_df['label'] = self.label
        return Maccs_fp_df
       
    def generate_descriptors(self):
        mols = self.molecules
        
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        descriptor_names = calc.GetDescriptorNames()

        Mol_descriptors =[]
        for mol in mols:
            mol=Chem.AddHs(mol)
            descriptors = calc.CalcDescriptors(mol)
            Mol_descriptors.append(descriptors)
            
        rdkit_df_descriptors = pd.DataFrame(Mol_descriptors, columns=descriptor_names)
        #rdkit_df_descriptors['label'] = self.label
        
        # #Drop highly correlate features
        # descriptors_df = rdkit_df_descriptors.drop(columns='label')
        # correlated_matrix = descriptors_df.corr().abs()

        # # Upper triangle of correlation matrix
        # upper_triangle = correlated_matrix.where(np.triu(np.ones(correlated_matrix.shape),k=1).astype(bool))

        # #drop highly correlated features based on set threshold=0.9
        # to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= 0.9)]
        # rdkit_df_descriptors_correlation_dropped = descriptors_df.drop(columns=to_drop, axis=1)
        
        
        
        # #make sure there are no NaN values in the rdkit descriptors dataframe
        # rdkit_df_descriptors_correlation_dropped.fillna(rdkit_df_descriptors_correlation_dropped.mean(), inplace=True )

        # #perform datascalling to improve the model performance
        # scalar = StandardScaler()
        # scaled_rdkit = scalar.fit_transform(rdkit_df_descriptors_correlation_dropped) 
        # scaled_rdkit_df_descriptors_correlation_dropped = pd.DataFrame(scaled_rdkit)
        # scaled_rdkit_df_descriptors_correlation_dropped['label'] = self.label
        # scaled_rdkit_df_descriptors_correlation_dropped['label'] = self.label
        return rdkit_df_descriptors 

def descriptors_generation(df):
    
    descriptor_maker = Make_Descriptors(df)
    morgan_df = descriptor_maker.calculate_morgan_fp()
    macc_df = descriptor_maker.calculate_maccs_fp()
    rdkit_descriptors = descriptor_maker.generate_descriptors()
    
    return morgan_df, macc_df,rdkit_descriptors

##End of fingerprint calculation

##Begin SMILE embedding
def get_unique_elements_as_dict(list_):
    """
    Given a list, obtain a dictionary with unique elements as keys and integer as values.
    Parameters
    ----------
    list_: list
        A generic list of strings.
    Returns
    -------
    dict
        Unique elements of the sorted list with assigned integer.
    """
    all_elements = set.union(*[set(smiles) for smiles in list_])
    unique_elements = sorted(list(all_elements))
    return {unique_elem: i for i, unique_elem in enumerate(unique_elements)}

def one_hot_encode(sequence, dictionary):
    """
    Creates the one-hot encoding of a sequence given a dictionary.
    Parameters
    ----------
    sequence: str
        A sequence of charaters.
    dictionary: dict
        A dictionary which comprises of characters.
    Returns
    -------
    np.array
        The binary matrix of shape `(len(dictionary), len(sequence))`,
        the one-hot encoding of the sequence.
    """
    ohe_matrix = np.zeros((len(dictionary), len(sequence)))
    for i, character in enumerate(sequence):
        ohe_matrix[dictionary[character], i] = 1
    return ohe_matrix

def pad_matrix(matrix, max_pad):
    """
    Pad matrix with zeros for shape dimension.
    Parameters
    ----------
    matrix: np.array
        The one-hot encoded matrix of a smiles.
    max_pad: int
        Dimension to which the padding should be done with zeros.
    Returns
    -------
    np.array
        The padded matrix of shape `(matrix.shape[0], max_pad)`.
    """
    if max_pad < matrix.shape[1]:
        return matrix
    else:
        return np.pad(matrix, ((0, 0), (0, max_pad - matrix.shape[1])))
def char_replacement(list_smiles):
    """
    Replace the double characters into single character in a list of SMILES string.
    Parameters
    ----------
    list_smiles: list
        list of SMILES string describing a compound.
    Returns
    -------
    list
        list of SMILES with character replacement.
    """
    return [
        smile.replace("Cl", "L")
        .replace("Br", "R")
        .replace("Se", "E")
        .replace("Zn", "Z")
        .replace("Si", "T")
        .replace("@@", "$")
        for smile in list_smiles
    ]


atom_to_index =  {
    "-":  0,
    "=":  1,
    "#":  2,
    "/":  3,
    "\\":  4,
    # Chirality
    "@":  5,
    "$":  6,
    # Formal charge
    "+":  7,
    "-":  8,
    # Branches
    "(":  9,
    ")":  10,
    # Rings
    "0":  11,
    "1":  12,
    "2":  13,
    "3":  14,
    "4":  15,
    "5":  16,
    "6":  17,
    "7":  18,
    "8":  19,
    "9":  20,
    "%":  21,
    # Atoms
    "B": 22,
    "C":  23,
    "E":  24,   
    "F":  25,
    "H":  26,
    "I":  27,
    "K":  28,
    "L":  29,
    "N":  30,
    "O":  31,
    "P":  32,
    "R":  33,
    "S":  34,
    "T":  35,
    "Z":  36,
    # aromatic atoms
    "b":  37,
    "c":  38,
    "e":  39,
    "i":  40,
    "n":  41,
    "o":  42,
    "s":  43,
    # Extra
    ".":  44,
    "*":  45,
    ":":  46,
    "[":  47,
    "]":  48,
}

def drop_unused_columns(d1, d2, d3):
    df1 = d1.drop(d1.columns.difference(['Col_1', 'Col_80', 'Col_294', 'Col_314', 'Col_322', 'Col_378',
       'Col_389', 'Col_561', 'Col_650', 'Col_656', 'Col_675', 'Col_695',
       'Col_718', 'Col_725', 'Col_802', 'Col_807', 'Col_841', 'Col_875',
       'Col_926', 'Col_935', 'Col_1019', 'Col_1028', 'Col_1039', 'Col_1057',
       'Col_1060', 'Col_1066', 'Col_1088', 'Col_1097', 'Col_1114', 'Col_1145',
       'Col_1152', 'Col_1160', 'Col_1199', 'Col_1325', 'Col_1357', 'Col_1385',
       'Col_1452', 'Col_1480', 'Col_1535', 'Col_1536', 'Col_1683', 'Col_1722',
       'Col_1745', 'Col_1750', 'Col_1754', 'Col_1816', 'Col_1855', 'Col_1866',
       'Col_1917', 'Col_1928']),axis=1)
    df2 = d2.drop(d2.columns.difference(['Col_36', 'Col_38', 'Col_42', 'Col_47', 'Col_52', 'Col_57', 'Col_59',
       'Col_62', 'Col_64', 'Col_65', 'Col_66', 'Col_69', 'Col_72', 'Col_73',
       'Col_74', 'Col_75', 'Col_77', 'Col_78', 'Col_79', 'Col_80', 'Col_81',
       'Col_82', 'Col_83', 'Col_85', 'Col_86', 'Col_87', 'Col_88', 'Col_89',
       'Col_90', 'Col_91', 'Col_92', 'Col_93', 'Col_94', 'Col_95', 'Col_96',
       'Col_97', 'Col_98', 'Col_100', 'Col_101', 'Col_102', 'Col_103',
       'Col_104', 'Col_105', 'Col_106', 'Col_107', 'Col_108', 'Col_109',
       'Col_110', 'Col_111', 'Col_112', 'Col_113', 'Col_114', 'Col_115',
       'Col_116', 'Col_117', 'Col_118', 'Col_119', 'Col_120', 'Col_121',
       'Col_122', 'Col_123', 'Col_124', 'Col_125', 'Col_126', 'Col_127',
       'Col_128', 'Col_129', 'Col_130', 'Col_131', 'Col_132', 'Col_133',
       'Col_134', 'Col_135', 'Col_136', 'Col_137', 'Col_138', 'Col_139',
       'Col_140', 'Col_141', 'Col_142', 'Col_143', 'Col_144', 'Col_145',
       'Col_146', 'Col_147', 'Col_148', 'Col_149', 'Col_150', 'Col_151',
       'Col_152', 'Col_153', 'Col_154', 'Col_155', 'Col_156', 'Col_157',
       'Col_158', 'Col_159', 'Col_160']),axis=1)
    df3 = d3.drop(d3.columns.difference(["MaxAbsEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "MolWt", "BCUT2D_MWHI", "BCUT2D_MRHI",
    "AvgIpc", "BalabanJ", "HallKierAlpha", "Ipc", "Kappa1", "Kappa2", "Kappa3", "PEOE_VSA1", "PEOE_VSA10",
    "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4",
    "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2",
    "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10",
    "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA7",
    "SlogP_VSA8", "TPSA", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3", "EState_VSA4",
    "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "VSA_EState10", "VSA_EState2",
    "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9",
    "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHeteroatoms",
    "NumRotatableBonds", "NumSaturatedHeterocycles", "NumSaturatedRings", "RingCount", "MolLogP", "fr_Al_OH",
    "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_C_O", "fr_NH0", "fr_NH1", "fr_NH2", "fr_Ndealkylation2",
    "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide", "fr_aniline", "fr_aryl_methyl", "fr_bicyclic", "fr_ester",
    "fr_ether", "fr_halogen", "fr_ketone", "fr_methoxy", "fr_para_hydroxylation", "fr_phenol", "fr_piperdine",
    "fr_pyridine"
    ]),axis=1)
    return df1, df2, df3
##End SMILE embedding

# Convert SMILES strings to integer sequences
def smiles_to_int(smiles_str):
    return [atom_to_index[atom] for atom in smiles_str if atom in atom_to_index]
def calculate_descriptors(df):
    Morgan_descriptors_df, Maccs_descriptors_df, rdkit_descriptors_df = descriptors_generation(df)

    d1, d2, d3 = drop_unused_columns(Morgan_descriptors_df,Maccs_descriptors_df,rdkit_descriptors_df)
    df_merged =  pd.concat([d1,d2,d3],axis=1, ignore_index=True )
    df_merged = df_merged.fillna(df_merged.mode().iloc[0])

    scalar = StandardScaler()
    scaled_rdkit = scalar.fit_transform(df_merged)
    X2_test = pd.DataFrame(scaled_rdkit)

    X1_ = df['Smiles']

    X1_ = char_replacement(X1_.tolist())

    #X_test = char_replacement(X_test.tolist())
    # Find the maximum sequence length
    

    # Create the atom-to-index dictionary


    X1_int = list(map(smiles_to_int, X1_))
    
    #compute the length at which i would want my smile to be
    char_lens = [len(smile) for smile in df['Smiles'].tolist()]
    mean_char_len = np.mean(char_lens)
    output_seq_char_len = int(np.percentile(char_lens, 95))

    # Pad the integer sequences
    X1_padded = pad_sequences(X1_int, maxlen=70, padding='post')


    # One-hot encode the integer sequences
    X1_test = np.array([to_categorical(seq, num_classes=len(atom_to_index) + 1) for seq in X1_padded])

    return X1_test,X2_test

def color_prediction(val):
            color = '#ED554E' if val>=0.5 else ''
            return f'background-color: {color}'

##Building model
def build_model_batch(dataframe):


    input_1,input_2 = calculate_descriptors(dataframe)

    # Reads in saved regression model
    load_model = pickle.load(open('model1a2.pkl', 'rb'))

    mol_name =[]
    mol_predictions =[]
    for index, row in dataframe.iterrows():
        molecule_name = pd.Series(row['Smiles'], name='Smiles')
        df = pd.concat([molecule_name], axis=1)
        x1,x2 = calculate_descriptors(df)
        
        pred = load_model.predict([x1,x2])
        mol_name.append(row['Smiles'])
        mol_predictions.append(pred)

    emoticon = []
    for mol_pred in mol_predictions:
        if mol_pred >= 0.5:
                emoticon.append("✔️")
        else:
            emoticon.append("✖️")

    prediction_output = pd.Series(emoticon, name='Inhibitor')
    molecule_name = pd.Series(mol_name, name='Molecule SMILES Sequence')
    df = pd.concat([molecule_name, prediction_output], axis=1)

    return df

# def predict_single_smile(df_single):

#     molecule_name = pd.Series(smiles_sequence, name='Smiles')
#     df = pd.concat([molecule_name], axis=1)

def build_model_single(df_single):

    input_1,input_2= calculate_descriptors(df_single)

    load_model = pickle.load(open('model1a2.pkl', 'rb'))
    # Apply model to make predictions
    prediction = load_model.predict([input_1,input_2])
    #prediction = np.squeeze(prediction)

    if prediction >= 0.5:
        emoticon = "✔️"
    else:
        emoticon = "✖️"

    prediction_output = pd.Series(emoticon, name='Inhibitor')
    molecule_name = pd.Series(df_single.Smiles, name='Molecule SMILES Sequence')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    return df

tab1, tab2= st.tabs(["Single Prediction", "Batch Prediction"])


with tab1:
    smiles_sequence = st.text_input("Enter SMILES sequence", key="smiles_sequence")
    # You can access the value at any point with:
    if st.button('Prediction', key='prediction1'):
            molecule_name = pd.Series(smiles_sequence, name='Smiles')
            df = pd.concat([molecule_name], axis=1)
            data = build_model_single(df)
            table_title="<style> </style><div class='css-x9krnl ';style='display:block;'><p style='text-align:center;'>CYP450 1A2 Inhibition prediction output</p></div>"

            st.markdown(table_title, unsafe_allow_html=True)
            st.dataframe(data)
            

with tab2:
    with open('example_molecules1a2.csv') as f:
        st.download_button('Download Example input file', f,'example_molecules1a2.csv')

    

    st.header('Upload CSV file')
    uploaded_file_1a2 = st.file_uploader("Upload your input file", type=['CSV'])

    if st.button('Prediction', key='prediction2'):
        if uploaded_file_1a2 is not None:
            #get the data

            data = pd.read_csv(uploaded_file_1a2)
            
            
            with st.spinner('Prediction...'):
                prediction_output = build_model_batch(data)

                styler = Styler(prediction_output)

                # Apply CSS styling to center the cells in the 'Emoji' column
                styler.set_table_styles(
                    [{'selector': 'Inhibitor', 'props': [('text-align', 'center')]}]
                )

                st.dataframe(styler)
                #st.write(df)
                st.markdown(filedownload(prediction_output), unsafe_allow_html=True)

        else:
             st.write('Upload CSV file....')


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

footer="<style> </style><div class='css-x9krnl footer';style='display:block;text-align:center;'><p style='text-align:center;'>Copyright©2023 <a href='https://sites.google.com/sunmoon.ac.kr/dslab/research?authuser=0' target='_blank'> D&S Lab, Sunmoon University</a></p></div>"

st.markdown(footer, unsafe_allow_html=True)
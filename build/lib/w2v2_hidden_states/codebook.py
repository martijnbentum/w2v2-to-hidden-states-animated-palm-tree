import torch
from . import load
import numpy as np
import torch

def outputs_to_codebook_indices(outputs, model_pt):
    '''map wav2vec2 outputs to codebook indices'''
    cv = outputs_to_codevectors(outputs, model_pt)
    codebook = load_codebook(model_pt)
    ci = codevectors_to_codebook_indices(cv, codebook)
    return ci

def load_model_pt(checkpoint = None, gpu = False):
    '''wrapper function to load the model_pt'''
    return load.load_model_pt(checkpoint, gpu)

def cnn_output_to_codebook_indices(cnn_output, model_pt, codebook=None):
    '''map cnn output to codebook indices'''
    cnn_output = torch.from_numpy(cnn_output)
    if len(cnn_output.shape) == 1:seq_len = 1
    elif len(cnn_output.shape) == 2:seq_len = cnn_output.shape[0]
    else: raise ValueError('cnn_output should be 1D or 2D')
    cnn_output = cnn_output.view(1,seq_len,-1)
    codevectors, tensor = model_pt.quantizer(cnn_output)
    cv = codevectors.detach().numpy()[0]
    if codebook is None:
        codebook = load_codebook(model_pt)
    ci = codevectors_to_codebook_indices(cv, codebook)
    return ci

def outputs_to_codevectors(outputs, model_pt):
    '''map cnn outputs to codevectors
    outputs     is the hidden states output of the wav2vec2 model
    model_pt    is the Wav2Vec2ForPreTraining model which has the codebook
                and quantizer loaded
    '''
    if type(outputs.extract_features) == np.ndarray:
        cnn_output = torch.from_numpy(outputs.extract_features)
    else:
        cnn_output = outputs.extract_features
    codevectors, tensor = model_pt.quantizer(cnn_output)
    return codevectors.detach().numpy()[0]

def load_codebook(model_pt):
    '''load the codebook from the model'''
    codebook = model_pt.quantizer.codevectors
    return codebook.detach().numpy()[0]

def codevectors_to_codebook_indices(codevectors, codebook):
    '''map codevectors to codebook indices
    codevectors     is a list of codevectors a codevector is a quantized
                    representation of the cnn output (i.e. extract_features)
    codebook        a matrix of codevectors, each quantized representation
                    can be found in the codebook, there are two codebooks
                    so a complete codevector can be represented with two indices
                    i.e. the locations in the codebooks
    '''
    codebook_indices = []
    for codevector in codevectors:
        ci = codevector_to_codebook_indices(codevector, codebook)
        codebook_indices.append(ci)
    return codebook_indices

def codevector_to_codebook_indices(codevector, codebook):
    '''map a codevector to codebook indices
    codevector      a codevector is a quantized
                    representation of the cnn output (i.e. extract_features)
    codebook        a matrix of codevectors, each quantized representation
                    can be found in the codebook, there are two codebooks
                    so a complete codevector can be represented with two indices
                    i.e. the locations in the codebooks
    '''
    slice_index = codebook.shape[-1]
    q1, q2 = codevector[:slice_index], codevector[slice_index:]
    index1 = get_row_index_of_vector_in_matrix(q1, codebook)
    index2 = get_row_index_of_vector_in_matrix(q2, codebook)
    codebook_indices = (index1, index2)
    return codebook_indices

def multiple_codebook_indices_to_codevectors(codebook_indices, codebook):
    if codebook is None:
        raise ValueError('please provide codebook')
    cv = []
    for ci in codebook_indices:
        cv.append(codebook_indices_to_codevector(ci, codebook))
    return np.array(cv)

def codebook_indices_to_codevector(codebook_indices, codebook):
    if codebook is None:
        raise ValueError('please provide codebook')
    a = codebook[codebook_indices[0]]
    b = codebook[codebook_indices[1]]
    return np.hstack((a,b))


def get_row_index_of_vector_in_matrix(vector, matrix):
    '''find the row index of a vector in a matrix'''
    return np.argwhere((vector == matrix).all(1)).flatten()[0]

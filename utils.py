import numpy as np
import pickle
import os
def cdist(XA, XB):
    m, n = XA.shape
    p, q = XB.shape
    
    if n != q:
        raise ValueError("XA and XB must have the same number of columns")
    
    XA_squared = np.sum(XA**2, axis=1).reshape(-1, 1)
    XB_squared = np.sum(XB**2, axis=1).reshape(1, -1)
    cross_term = np.dot(XA, XB.T)
    D = np.sqrt(np.maximum(XA_squared + XB_squared - 2 * cross_term, 0))
    
    return D


def save_obj(obj, name, dir_path='./data'):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, dir_path='./data'):
    with open(os.path.join(dir_path, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def load_sysdata(data_name):
	W = list()
	X = list()

	if 'ImgGendata' in data_name:
		data = load_obj(data_name)
		X.append(np.array(data['X1']))
		X.append(np.array(data['X2']))
		X.append(np.array(data['Y']))

		return X, W





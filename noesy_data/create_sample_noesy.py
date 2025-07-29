import numpy as np

data = np.load('noesy_data/output/1a6d.A.npz')['noesy_data']

with open('noesy_data/sample_noesy.txt', 'w') as f:
    for i in range(10): # Just take the first 10 for a sample
        res1_num = data[i]['res1_num']
        res2_num = data[i]['res2_num']
        dist = data[i]['distance']
        atom1 = data[i]['atom1_name']
        atom2 = data[i]['atom2_name']
        f.write(f"{res1_num} {res2_num} 0 {dist:.1f} {atom1} {atom2}\n")

import pickle
import numpy as np
p = r'd:\\roo\\ddos-protection-system\\models\\minmax_scaler.pkl'
with open(p, 'wb') as f:
    pickle.dump({'feature_names_in_': np.array(['Flow Duration','Total Fwd Packets','Total Backward Packets'])}, f)
print('Wrote', p)

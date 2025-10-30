import pickle
import numpy as np
import matplotlib.pyplot as plt


model_path = '/home/mcw/durga/kaldi/egs/gop_speechocean762/s5/exp/gop_test/model.pkl' 
with open(model_path, 'rb') as f:
    model_of = pickle.load(f)

print(f"\nLoaded model for {len(model_of)} phones.\n")


try:
    from utils import load_phone_symbol_table
    _, phone_int2sym = load_phone_symbol_table('/home/mcw/durga/kaldi/egs/gop_speechocean762/s5/exp/gop_test/phones-pure.txt')  # Update path
except:
    phone_int2sym = {}


for ph, (coef, intercept) in model_of.items():
    label = phone_int2sym.get(ph, str(ph))
    print(f"{label}: Coef={coef.flatten()}, Intercept={intercept.flatten()}")


selected_phone = 'AH'
ph_id = next((ph for ph in model_of if phone_int2sym.get(ph, str(ph)) == selected_phone), None)

if ph_id is not None:
    coef, intercept = model_of[ph_id]
    poly = np.linspace(-5, 5, 100).reshape(-1, 1)
    poly_features = np.hstack([np.ones_like(poly), poly, poly**2])
    predicted = poly_features @ coef.T + intercept

    plt.plot(poly, predicted, label=f'Regression for {selected_phone}')
    plt.xlabel('GOP Score')
    plt.ylabel('Predicted Human Score')
    plt.title(f'Polynomial Regression for {selected_phone}')
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print(f"Phone '{selected_phone}' not found in model.")

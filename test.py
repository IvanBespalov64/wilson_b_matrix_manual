from rdkit import Chem
import numpy as np

from wilson_b_matrix import get_full_derivative
from wilson_b_matrix import Angle
from wilson_b_matrix import get_current_derivative

mol = Chem.MolFromMolFile('water.mol', removeHs=False)
cart_derivatives = np.array([[9.8036462291277e-11, 3.3758505580049e-12, -1.1842328627332E-02],
			     [-6.3571835660352e-11, 3.1612236176228e-04, 5.9211642889396e-03],
			     [-3.4464626630925e-11, -3.1612236513812e-04, 5.9211643383920e-03]]).flatten() # Производные по декартовым координатам

print(get_full_derivative(mol, cart_derivatives)) # Вернет список внутренних координат и список соответствующих им производных

cur_coord = Angle(1, 0, 2)
print(get_current_derivative(mol, cart_derivatives, cur_coord)) # Вернет производную по двугранному углу 0-1-2-3

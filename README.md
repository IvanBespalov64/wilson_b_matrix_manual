# Wilson B Matrix

Файлы для мануала по Wilson B Matrix.

Для проверки запустить (test.py):

```python
from rdkit import Chem
import numpy as np

from wilson_b_matrix import get_full_derivative
from wilson_b_matrix import Angle
from wilson_b_matrix import get_current_derivative
from wilson_b_matrix import parse_grads_from_grads_file

mol = Chem.MolFromMolFile('water.mol', removeHs=False)

cart_derivatives = parse_grads_from_grads_file(len(mol.GetAtoms()), 
                                               grads_filename='gradient',
                                               soft='xtb').flatten() # Производные по декартовым координатам

print(get_full_derivative(mol, cart_derivatives)) # Вернет список внутренних координат и список соответствующих им производных

cur_coord = Angle(1, 0, 2)
print(get_current_derivative(mol, cart_derivatives, cur_coord)) # Вернет производную по валентному углу H-O-H
```

### Requirements
* numpy
* rdkit
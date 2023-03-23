# Wilson B Matrix

Файлы для мануала по Wilson B Matrix.

Для проверки запустить (test.py):

```python
from rdkit import Chem
import numpy as np

from wilson_b_matrix import get_full_derivative
from wilson_b_matrix import Dihedral
from wilson_b_matrix import get_current_derivative

mol = Chem.MolFromMolFile('test.mol')
cart_derivatives = np.ones(len(mol.GetAtoms()) * 3) # Производные по декартовым координатам

print(get_full_derivative(mol, cart_derivatives)) # Вернет список внутренних координат и список соответствующих им производных

cur_coord = Dihedral(0, 1, 2, 3)
print(get_current_derivative(mol, cart_derivatives, cur_coord)) # Вернет производную по двугранному углу 0-1-2-3
```

### Requirements
* numpy
* rdkit
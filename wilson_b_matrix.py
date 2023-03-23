import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from itertools import combinations

vec3d = np.ndarray

class LinAlgException(Exception):
    pass

class InternalCoord:
    pass

class Bond(InternalCoord):
    
    def __init__(self, i : int, j : int):
        self.i, self.j = sorted([i, j])
        
    def __eq__(self, bond : InternalCoord) -> bool:
        return (self.i, self.j) == (bond.i, bond.j)

    def __repr__(self):
        return f"Bond {self.i, self.j}"
        
class Angle(InternalCoord):
    
    def __init__(self, i : int, j : int, k : int):
        self.j = j
        self.i, self.k = sorted([i, k])

    def __eq__(self, angle: InternalCoord) -> bool:
        return (self.i, self.j, self.k) == (angle.i, angle.j, angle.k)

    def __repr__(self):
        return f"Angle {self.i, self.j, self.k}"
        
class Dihedral(InternalCoord):
    
    def __init__(self, i : int, j : int, k : int, l : int):
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        
        if l < i:
            self.i, self.l = l, i
            self.j, self.k = k, j
        
    def __eq__(self, dihedral : InternalCoord) -> bool:
        return (self.i, self.j, self.k, self.l) == (dihedral.i, dihedral.j, dihedral.k, dihedral.l)

    def __repr__(self):
        return f"Dihedral {self.i, self.j, self.k, self.l}"

class LinearAngle(InternalCoord):
    
    def __init__(self, i : int, j : int, k : int, orth_dir : vec3d):
        self.j = j
        self.i, self.k = sorted([i, k])
        self.orth_dir = orth_dir
        
    def __eq__(self, linang : InternalCoord) -> bool:
        return (self.i, self.j, self.k, self.orth_dir) == (linang.i, linang.j, linang.k, linang.orth_dir)

    def __repr__(self):
        return f"Linear angle {self.i, self.j, self.k} with orth_dir {self.orth_dir} "

class OutOfPlaneBend(InternalCoord):
    
    def __init__(self, c : int, i : int, j : int, k : int):
        self.c = c
        self.i, self.j, self.k = sorted([i, j, k])

    def __eq__(self, oop_bend : InternalCoord) -> bool:
        return (self.c, self.i, self.j, self.k) == (oop_bend.c, oop_bend.i, oop_bend.j, oop_bend.k)

    def __repr__(self):
        return f"OutOfPlaneBend on central atom {self.c} with {self.i, self.j, self.k}"

def bond_gradient(p1 : vec3d, 
                  p2 : vec3d) -> tuple[vec3d, vec3d]:
    
    dist = np.sqrt((p1 - p2).dot(p1 - p2))
    u = (p1 - p2) / dist
    
    return u, -u

def collinear(v1 : vec3d, 
              v2 : vec3d, 
              tol : float = 1e-6) -> bool:
    
    l1 = np.sqrt(v1.dot(v1))
    l2 = np.sqrt(v2.dot(v2))
    
    angle = np.arccos((v1 / l1).dot(v2 / l2))
    
    if(np.abs(angle) < tol):
        return True
    elif(np.abs(angle - np.pi) < tol):
        return True
    elif(np.abs(angle - 2 * np.pi) < tol):
        return True
    
    return False

def angle_gradient(p1 : vec3d, 
                   p2 : vec3d, 
                   p3 : vec3d, 
                   tol : float = 1e-6) -> tuple[vec3d, vec3d, vec3d]:
    
    u = p1 - p2
    v = p3 - p2
    
    angle = np.arccos(u.dot(v) / np.sqrt(u.dot(u) * v.dot(v)))
    
    bond21 = np.sqrt(u.dot(u))
    bond23 = np.sqrt(v.dot(v))
        
    w = np.zeros(3)
    pmp = np.array([1., -1., 1.])
    mpp = np.array([-1., 1., 1.])
    
    if np.abs(angle - np.pi) > tol:
        w = np.cross(u, v)
    elif collinear(u, pmp, tol) and collinear(v, pmp, tol):
        w = np.cross(u, pmp)
    elif collinear(u, mpp, tol) and collinear(v, mpp, tol):
        w = np.cross(u, mpp)
    else:
        raise LinAlgException
    
    w = w / np.sqrt(w.dot(w))
    
    v1 = np.cross(u, w) / bond21
    v3 = np.cross(w, v) / bond23
    v2 = -v1 - v3
    
    return v1, v2, v3

def dihedral_gradient(p1 : vec3d,
                      p2 : vec3d,
                      p3 : vec3d,
                      p4 : vec3d) -> tuple[vec3d, vec3d, vec3d, vec3d]:
    
    angle123 = np.arccos((p1 - p2).dot(p3 - p2) / np.sqrt((p1 - p2).dot(p1 - p2) * (p3 - p2).dot(p3 - p2)))
    sin_angle123 = np.sin(angle123)
    cos_angle123 = np.cos(angle123)
    
    angle234 = np.arccos((p2 - p3).dot(p4 - p3) / np.sqrt((p2 - p3).dot(p2 - p3) * (p4 - p3).dot(p4 - p3)))
    sin_angle234 = np.sin(angle234)
    cos_angle234 = np.cos(angle234)
    
    b12 = p2 - p1
    b23 = p3 - p2
    b34 = p4 - p3
    
    bond12 = np.sqrt(b12.dot(b12))
    bond23 = np.sqrt(b23.dot(b23))
    bond34 = np.sqrt(b34.dot(b34))
    
    b12 = b12 / bond12
    b23 = b23 / bond23
    b34 = b34 / bond34
    
    b32 = -b23
    b43 = -b34
    
    v1 = -np.cross(b12, b23) / (bond12 * sin_angle123 * sin_angle123)
    
    vc1 = (bond23 - bond12 * cos_angle123) / (bond12 * bond23 * sin_angle123)
    vc2 = cos_angle234 / (bond23 * sin_angle234)
    vv1 = np.cross(b12, b23) / sin_angle123
    vv2 = np.cross(b43, b32) / sin_angle234
    
    v2 = vc1 * vv1 + vc2 * vv2
    
    vc1 = (bond23 - bond34 * cos_angle234) / (bond23 * bond34 * sin_angle234)
    vc2 = cos_angle123 / (bond23 * sin_angle123)
    vv1 = np.cross(b43, b32) / sin_angle234
    vv2 = np.cross(b12, b23) / sin_angle123
    
    v3 = vc1 * vv1 + vc2 * vv2
    
    v4 = -np.cross(b43, b32) / (bond34 * sin_angle234 * sin_angle234)
    
    return v1, v2, v3, v4

def out_of_plane_gradient(vc : vec3d, 
                          v1 : vec3d, 
                          v2 : vec3d, 
                          v3 : vec3d) -> tuple[vec3d, vec3d, vec3d, vec3d]:
    b1 = v1 - vc
    b2 = v2 - vc
    b3 = v3 - vc
    
    e1 = b1 / np.sqrt(b1.dot(b1))
    e2 = b2 / np.sqrt(b2.dot(b2))
    e3 = b3 / np.sqrt(b3.dot(b3))
    
    a1 = np.arccos((v2 - vc).dot(v3 - vc) / np.sqrt((v2 - vc).dot(v2 - vc) * (v3 - vc).dot(v3 - vc)))
    a2 = np.arccos((v3 - vc).dot(v1 - vc) / np.sqrt((v3 - vc).dot(v3 - vc) * (v1 - vc).dot(v1 - vc)))
    a3 = np.arccos((v1 - vc).dot(v2 - vc) / np.sqrt((v1 - vc).dot(v1 - vc) * (v2 - vc).dot(v2 - vc)))

    sin_a1 = np.sin(a1)
    cos_a1 = np.cos(a1)
    cos_a2 = np.cos(a2)
    cos_a3 = np.cos(a3)
    
    ir1 = 1 / np.sqrt(b1.dot(b1))
    ir2 = 1 / np.sqrt(b2.dot(b2))
    ir3 = 1 / np.sqrt(b3.dot(b3))
    
    t1 = np.cross(e2, e3) / np.sin(a1)
    
    angle = np.arcsin(t1.dot(e1))
    
    cos_angle = np.cos(angle)
    tan_angle = np.tan(angle)
    
    s1 = ir1 * (t1 / cos_angle - tan_angle * e1)
    denominator = cos_angle * sin_a1 * sin_a1
    s2 = ir2 * t1 * (cos_a1 * cos_a2 - cos_a3) / denominator
    s3 = ir3 * t1 * (cos_a1 * cos_a3 - cos_a2) / denominator
    sc = -s1 - s2 - s3
    
    return sc, s1, s2, s3

def linear_angle_gradient(p1 : vec3d, 
                          p2 : vec3d,
                          p3 : vec3d,
                          orthogonal_direction : vec3d,
                          tol : float = 1e-6) -> tuple[vec3d, vec3d, vec3d]:
    pOrth = p2 + orthogonal_direction
    
    v1, v2, vOrth = angle_gradient(p1, p2, pOrth, tol)
    vOrth, v2, v3 = angle_gradient(pOrth, p2, p3)
    
    return v1, -(v1 + v3), v3

def wilson_b_matrix(x_cartesian : np.ndarray, 
                    bonds : list[Bond],
                    angles : list[Angle],
                    dihedrals : list[Dihedral],
                    linear_angles : list[LinearAngle],
                    out_of_plane_bends : list[OutOfPlaneBend]) -> np.ndarray:
    
    n_atoms = x_cartesian.size // 3
    n_irc = len(bonds) + len(angles) + len(dihedrals) +\
            len(linear_angles) + len(out_of_plane_bends)
    
    B = np.zeros((n_irc, 3 * n_atoms))
    offset = 0
    
    # Populate B matrix's rows corresponding to bonds 
    
    for i, bond in enumerate(bonds):
        p1, p2 = np.zeros(3), np.zeros(3)
        for j in range(3):
            p1[j] = x_cartesian[3 * bond.i + j]
            p2[j] = x_cartesian[3 * bond.j + j]
        g1, g2 = bond_gradient(p1, p2)
        for j in range(3):
            B[i, 3 * bond.i + j] = g1[j]
            B[i, 3 * bond.j + j] = g2[j]
            
    offset += len(bonds)
            
    # Populate B matrix's rows corresponding to angles 
    
    for i, angle in enumerate(angles):
        p1, p2, p3 = np.zeros(3), np.zeros(3), np.zeros(3)
        for j in range(3):
            p1[j] = x_cartesian[3 * angle.i + j]
            p2[j] = x_cartesian[3 * angle.j + j]
            p3[j] = x_cartesian[3 * angle.k + j]
        g1, g2, g3 = angle_gradient(p1, p2, p3)
        for j in range(3):
            B[i + offset, 3 * angle.i + j] = g1[j]
            B[i + offset, 3 * angle.j + j] = g2[j]
            B[i + offset, 3 * angle.k + j] = g3[j]
            
    offset += len(angles)
            
    # Populate B matrix's rows corresponding to dihedrals
    
    for i, dihedral in enumerate(dihedrals):
        p1, p2, p3, p4 = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for j in range(3):
            p1[j] = x_cartesian[3 * dihedral.i + j]
            p2[j] = x_cartesian[3 * dihedral.j + j]
            p3[j] = x_cartesian[3 * dihedral.k + j]
            p4[j] = x_cartesian[3 * dihedral.l + j]
        g1, g2, g3, g4 = dihedral_gradient(p1, p2, p3, p4)
        for j in range(3):
            B[i + offset, 3 * dihedral.i + j] = g1[j]
            B[i + offset, 3 * dihedral.j + j] = g2[j]
            B[i + offset, 3 * dihedral.k + j] = g3[j]
            B[i + offset, 3 * dihedral.l + j] = g4[j]
            
    offset += len(dihedrals)
    
    #  Populate B matrix's rows corresponding to linear angles
    
    for i, lin_ang in enumerate(linear_angles):
        p1, p2, p3 = np.zeros(3), np.zeros(3), np.zeros(3)
        for j in range(3):
            p1[j] = x_cartesian[3 * lin_ang.i + j]
            p2[j] = x_cartesian[3 * lin_ang.j + j]
            p3[j] = x_cartesian[3 * lin_ang.k + j]
        g1, g2, g3 = linear_angle_gradient(p1, p2, p3, lin_ang.orth_dir)
        for j in range(3):
            B[i + offset, 3 * lin_ang.i + j] = g1[j]
            B[i + offset, 3 * lin_ang.j + j] = g2[j]
            B[i + offset, 3 * lin_ang.k + j] = g3[j]
            
    offset += len(linear_angles)
    
    # Populate B matrix's rows corresponding to out of plane bends
    
    for i, bend in enumerate(out_of_plane_bends):
        p1, p2, p3, p4 = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for j in range(3):
            p4[j] = x_cartesian[3 * bend.c + j]
            p1[j] = x_cartesian[3 * bend.i + j]
            p2[j] = x_cartesian[3 * bend.j + j]
            p3[j] = x_cartesian[3 * bend.k + j]
        g4, g1, g2, g3 = out_of_plane_gradient(p4, p1, p2, p3)
        for j in range(3):
            B[i + offset, 3 * bend.c + j] = g4[j]
            B[i + offset, 3 * bend.i + j] = g1[j]
            B[i + offset, 3 * bend.j + j] = g2[j]
            B[i + offset, 3 * bend.k + j] = g3[j]
            
    return B

def non_parallel_direction(d : vec3d) -> vec3d:
    
    dirs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    return sorted(dirs, key=lambda v: d.dot(v) ** 2)[0]

def orthogonal_axis(d : vec3d,
                    axis : vec3d) -> tuple[vec3d]:
    first = np.cross(d, axis)
    first /= np.sqrt(first.dot(first))
    second = np.cross(d, first)
    second /= np.sqrt(second.dot(second))
    return first, second

def parse_to_internal(mol : Chem.rdchem.Mol,
                      coords : np.ndarray) -> tuple[list]:
    """
        returns a list of Bonds, Angles, Torisons, Linear Angles and Out-of-Plane Bends
    """

    #Bonds

    bonds = []

    for bond in mol.GetBonds():
        bonds.append(Bond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    #Angles

    angles_lst = []
    angles, linear_angles = [], []

    for bond in mol.GetBonds():
        beg_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for adj_bond in mol.GetBonds():
            if adj_bond.GetBeginAtomIdx() == beg_idx:
                if len({beg_idx, end_idx, adj_bond.GetEndAtomIdx()}) == 3:
                    current_angle = adj_bond.GetEndAtomIdx(), beg_idx, end_idx
                    if not (current_angle in angles_lst or current_angle[::-1] in angles_lst):
                        angles_lst.append(current_angle)
            elif adj_bond.GetEndAtomIdx() == beg_idx:
                if len({beg_idx, end_idx, adj_bond.GetBeginAtomIdx()}) == 3:
                    current_angle = adj_bond.GetBeginAtomIdx(), beg_idx, end_idx
                    if not (current_angle in angles_lst or current_angle[::-1] in angles_lst):
                        angles_lst.append(current_angle)
            elif adj_bond.GetBeginAtomIdx() == end_idx:
                if len({beg_idx, end_idx, adj_bond.GetEndAtomIdx()}) == 3:
                    current_angle = beg_idx, end_idx, adj_bond.GetEndAtomIdx()
                    if not (current_angle in angles_lst or current_angle[::-1] in angles_lst):
                        angles_lst.append(current_angle)
            elif adj_bond.GetEndAtomIdx() == end_idx:
                if len({beg_idx, end_idx, adj_bond.GetBeginAtomIdx()}) == 3:
                    current_angle = beg_idx, end_idx, adj_bond.GetBeginAtomIdx()
                    if not (current_angle in angles_lst or current_angle[::-1] in angles_lst):
                        angles_lst.append(current_angle)

    tol = 1e-2            

    for angle in angles_lst:
        if np.abs(rdMolTransforms.GetAngleRad(mol.GetConformer(), *angle) - np.pi) < tol:
            d = coords[angle[2], :] - coords[angle[0], :]
            axis = non_parallel_direction(d)
            first, second = orthogonal_axis(d, axis)
            linear_angles.append(LinearAngle(*angle, first))
            linear_angles.append(LinearAngle(*angle, second))
        else:
            angles.append(Angle(*angle))

    # Torsions    

    dihedrals = []

    for bond in mol.GetBonds():
        beg_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for bond_left in mol.GetBonds():
            if bond_left.GetBeginAtomIdx() != beg_idx and bond_left.GetEndAtomIdx() != beg_idx:
                continue
            left_atom = bond_left.GetBeginAtomIdx() if beg_idx == bond_left.GetEndAtomIdx() else bond_left.GetEndAtomIdx()
            for bond_right in mol.GetBonds():
                if bond_right.GetBeginAtomIdx() != end_idx and bond_right.GetEndAtomIdx() != end_idx:
                    continue
                right_atom = bond_right.GetBeginAtomIdx() if end_idx == bond_right.GetEndAtomIdx() else bond_right.GetEndAtomIdx()

                if len({left_atom, beg_idx, end_idx, right_atom}) != 4:
                    continue
                dihedrals.append(Dihedral(left_atom, beg_idx, end_idx, right_atom))

    # Out-of-plane angles

    out_of_plane_bends = []

    for atom in mol.GetAtoms():
        adj_bonds = atom.GetBonds()

        if len(adj_bonds) < 3:
            continue

        for cur_bonds in combinations(adj_bonds, 3):
            set_of_atoms = set()
            for cur_bond in cur_bonds:
                set_of_atoms.add(cur_bond.GetBeginAtomIdx())
                set_of_atoms.add(cur_bond.GetEndAtomIdx())
            set_of_atoms -= {atom.GetIdx()}
            out_of_plane_bends.append(OutOfPlaneBend(atom.GetIdx(), *set_of_atoms)) 

    return bonds, angles, dihedrals, linear_angles, out_of_plane_bends

def parse_coords_from_mol(mol : Chem.rdchem.Mol) -> np.ndarray:
    """
        Returns np.ndarray of coords with shape [N, 3], where N - number of atoms
    """

    parsed_xyz = list(map(lambda s: s.split(), Chem.MolToXYZBlock(mol).split("\n")[2:-1]))
    coords = np.array(list(map(lambda lst: list(map(float, lst[1:])), parsed_xyz)))
    return coords

def parse_atoms_from_mol(mol : Chem.rdchem.Mol) -> list[str]:
    """
        Returns list of atoms' symbols
    """

    return list(map(lambda s: s.split()[0], Chem.MolToXYZBlock(mol).split("\n")[2:-1]))

def get_wilson_b_matrix(mol : Chem.rdchem.Mol) -> np.ndarray:
    """
        Returns list of internal coords Wilson B-Matrix
    """

    coords = parse_coords_from_mol(mol).flatten()
    int_coords_list = parse_to_internal(mol, coords)
    return int_coords_list, wilson_b_matrix(coords.flatten(), *int_coords_list)

def gradient_to_internal(mol : Chem.rdchem.Mol,
                         cart_grad : vec3d) -> vec3d:
    """
        Converts cartesian gradint to internal, returns int_coords and internal gradient
    """

    int_coords, B = get_wilson_b_matrix(mol)
    return int_coords, np.linalg.pinv(B @ np.eye(B.shape[-1]) @ B.T) @ B @ np.eye(B.shape[-1]) @ cart_grad

def get_current_derivative(mol : Chem.rdchem.Mol,
                           cart_grad : vec3d,
                           cur_coord : InternalCoord) -> float:
    
    int_coords, int_grad = gradient_to_internal(mol, cart_grad)
    for idx, cur in enumerate(np.hstack(int_coords)):
        if cur == cur_coord:
            return int_grad[idx]
        
def get_full_derivative(mol : Chem.rdchem.Mol,
                        cart_grad : vec3d) -> vec3d:
    int_coords, int_grad = gradient_to_internal(mol, cart_grad)
    return int_grad

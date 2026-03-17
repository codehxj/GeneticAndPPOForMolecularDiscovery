from rdkit import Chem

mol = Chem.MolFromMolFile("10182418.mol")
writer = Chem.SDWriter("10182418.sdf")
writer.write(mol)
writer.close()
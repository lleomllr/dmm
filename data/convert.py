import os
import pickle
import torch
import scipy.io as sio
import numpy as np

# === paramètres ===
mat_path = "./nottingham.mat"   # chemin vers ton fichier .mat
out_path = "./nottingham.pkl"   # sortie souhaitée (.pkl)
min_note = 21
note_range = 88  # 21 -> 108 (piano standard)

# === chargement du fichier MATLAB ===
print(f"Chargement de {mat_path} ...")
mat_data = sio.loadmat(mat_path)

# Les .mat contiennent typiquement des clés : train, valid, test
splits = ["traindata", "validdata", "testdata"]
processed_dataset = {}

for split in splits:
    if split not in mat_data:
        print(f"Split '{split}' introuvable dans le fichier .mat")
        continue

    sequences = mat_data[split][0]
    print(f"→ {split}: {len(sequences)} séquences")
    seq_tensors = []
    seq_lengths = torch.zeros(len(sequences), dtype=torch.long)

    for i, seq in enumerate(sequences):
        # seq est souvent une matrice sparse de notes (format scipy.sparse)
        s = seq.toarray() if hasattr(seq, "toarray") else np.array(seq)
        # s a souvent shape (T, note_range)
        tensor = torch.tensor(s, dtype=torch.float32)
        seq_lengths[i] = tensor.shape[0]
        seq_tensors.append(tensor)

    processed_dataset[split] = {
        "sequences": seq_tensors,
        "sequence_lengths": seq_lengths
    }

# === sauvegarde pickle ===
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(processed_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Dataset converti et sauvegardé dans : {out_path}")

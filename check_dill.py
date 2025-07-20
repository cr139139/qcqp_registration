import dill

with open('asm_double.pkl', 'rb') as f:
    pca = dill.load(f)
print(pca.explained_variance_ratio)



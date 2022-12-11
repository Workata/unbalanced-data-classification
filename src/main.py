from loaders import KeelDatasetLoader

dataset_loader = KeelDatasetLoader()

df = dataset_loader.load("ecoli1.dat")
print(df)
print(type(df))



from DataPrep import DataPrep
from Eda import Eda
from Model import Model

dp = DataPrep()
eda = Eda()
model = Model()

data = dp.initiate()
data = eda.initiate(data)
model.initiate(data)


print("done")


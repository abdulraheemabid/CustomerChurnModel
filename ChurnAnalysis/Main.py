from DataPrep import DataPrep
from Eda import Eda
from Model import Model

dp = DataPrep()
eda = Eda()
model = Model()

# initiate data preparation by getting all 3 csvs and merging them
data = dp.initiate()

# initiate EDA process
data = eda.initiate(data)

# train ML models
model.initiate(data)



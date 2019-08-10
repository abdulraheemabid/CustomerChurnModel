from DataPrep import DataPrep
from Eda import Eda

dp = DataPrep()
eda = Eda()

data = dp.prep_data()
eda.initiate_eda(data)
print("done")


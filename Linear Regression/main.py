from include.loadData import *
from include.createPytorchTensors import createPytorchTensors
from include.visualizeData import visualizeData
from include.linearRegressionModel import *

def prepareDataSet():
    test,train = load_data()
    column_mappings = createMapping()
    apply_mappings(train, column_mappings)
    apply_mappings(test, column_mappings)
    save_data(test, "test")
    save_data(train, "train")

prepareDataSet()

#test,train = load_data()
#print_data(test)


# Load data
#test, train = load_data("modified_test.csv","modified_train.csv")

#print(train)
#print_data(test)
#print(column_mappings)
#test, train = load_data(test_new_name, train_new_name)

#visualizeData(train)

# Prepare data for PyTorch
#train_loader, test_loader = createPytorchTensors(train)

#startLinearRegressionModel(train_loader,test_loader, 100, 0.01)
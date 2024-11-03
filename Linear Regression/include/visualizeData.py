import seaborn as sns
import matplotlib.pyplot as plt

def visualizeData( csvData , columns_to_plot=None ):
    if columns_to_plot is None:
        columnsToPlot = ['MSSubClass','LotArea','OverallCond','SalePrice']
    elif not all(col in csvData.columns for col in columns_to_plot):
        print("One or more columns are missing from the DataFrame.")
        return
    sns.pairplot(csvData[columnsToPlot])
    plt.show()
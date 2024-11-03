import pandas as pd

def print_data(data):
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None) 
    print(data)
    print("\n\n")
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    #print(data.describe())

#replaces column values with numerical values based on a mapping  called column_mappings
def apply_mappings(df, column_mappings):
    for column, mapping in column_mappings.items():
        if column in df.columns:  # Ensure the column exists in the DataFrame
            df[column] = df[column].map(mapping).fillna(df[column])  # Keep non-mapped values intact



def load_data(testDataFileName = "test.csv", trainDataFileName = "train.csv", standart_path = "C:/Users/Win10 Pro x64/source/repos/Arbeits_Projekte/Projects_for_self_study/Linear Regression/data/"):
    test = pd.read_csv(standart_path + testDataFileName, keep_default_na=False)
    train = pd.read_csv(standart_path + trainDataFileName, keep_default_na=False)
    #print_data(test)
    #print_data(train)
    return test, train


def createMapping( debug=False ,name = "data_description.txt", standart_path = "C:/Users/Win10 Pro x64/source/repos/Arbeits_Projekte/Projects_for_self_study/Linear Regression/data/" ):
    with open( standart_path + name, 'r') as f:
        description_lines = f.readlines()

    #print(description_lines)

    # Initialize variables to store the mappings
    column_mappings = {}
    current_column = None   
   
    # Parse the data description file to create mappings
    for line in description_lines:
        stripped_line = line.strip()
        
        if not stripped_line:
            continue  # Skip empty lines

        # Check for new column headers (ends with ":")
        if ":" in stripped_line and not line.startswith(" "):
            current_column = stripped_line.split(":")[0]
            column_mappings[current_column] = {}
        elif current_column and line.startswith(" "):  # Indented lines are value mappings
            if(debug):
                print("\nstripped_line: ", stripped_line)
            parts = stripped_line.split("\t")
            if len(parts) >= 2:  # Expect at least code and description
                code = parts[0]
                index = len(column_mappings[current_column]) + 1  # Numerical index
                if(code == "WD "):
                    code = "WD"
                if(code == "NA "):
                    code = "NA"
                if(code == "nan"):
                    code = "NA"
                column_mappings[current_column][code] = index
                if(debug):
                    print(f"Mapping code: '{code}' to {index}")


    with open ( standart_path + "data_mapping.txt ", "w") as f:
        for column, mapping in column_mappings.items():
            f.write(column + ":\n")
            for key, value in mapping.items():
                f.write(f"{key}: {value}\n")
            

    return column_mappings


def save_data( csvData , name , standart_path = "C:/Users/Win10 Pro x64/source/repos/Arbeits_Projekte/Projects_for_self_study/Linear Regression/data/"):
    newName = 'modified_' + name
    csvData.to_csv( standart_path + newName + ".csv", index=False)
    return newName

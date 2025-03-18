import pandas as pd
from sys import argv
from sklearn import RandomForestClassifier, cross_val_score, roc_auc_score
#GSE37069 - Burn tissue
#GSE99039 - Parkinson's


#REMEMBER: Add myself to Abby's github as a contributor so I can push the changes!


# use sys.argv to accept arguments - name of input file, column name w/ class labels, column names to drop (comma-separated list), description of comparison (sex, autosomal, all) name of output file to create
# script is generic - not specific to just 1 dataset
# output file - for each cv fold, what is the roc_auc score? - 5 rows, one for each fold
# one column w/ name of input file - then we can merge all of the output files together at the end
# might need to write another script to reformat the expression data so that we can use it w the ml stuff


def main():
    try:
        if len(argv) != 5:
            raise ValueError("Incorrect number of arguments. Please provide exactly 5 arguments.")
       
        input_filename = argv[1]
        sex_col = argv[2]
        drop_cols = argv[3]
        output_filename = argv[4]


        with open(input_filename, 'r') as readFile, open(output_filename, 'w') as writeFile:
            df_in = pd.read_csv(readFile)

            #drop columns
            df_in = df_in.drop(columns = drop_cols)

            #only keep rows with existing sex data
            values_to_keep = ['male', 'female']
            df = df[df[sex_col].isin(values_to_keep)]

            #metadata_sex column
            X = df.iloc[:,0:5]
            #expression_sex column
            y = df.iloc[:,6]

            #hyper parameters
            rf = RandomForestClassifier(n_estimators = 1000,
                                        criterion = 'entropy',
                                        min_samples_split = 10,
                                        max_depth = 14,
                                        random_state = 42
            )

            cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
            print("Cross-validation scores for each fold:", cv_scores)

            #Calculate ROC AUC
            roc_auc = roc_auc_score(sex_col, y)
            print(roc_auc)

            #create output file


    except ValueError as ve:
        print(f"Error: {ve}")
       
if __name__ == "__main__":
    main()
import pandas as pd
from sys import argv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

#REMEMBER: Add myself to Abby's github as a contributor so I can push the changes!

# use sys.argv to accept arguments - name of input file, column name w/ class labels, column names to drop (comma-separated list), description of comparison (sex, autosomal, all) name of output file to create
# script is generic - not specific to just 1 dataset
# output file - for each cv fold, what is the roc_auc score? - 5 rows, one for each fold
# one column w/ name of input file - then we can merge all of the output files together at the end
# might need to write another script to reformat the expression data so that we can use it w the ml stuff


def main():
    try:
        if len(argv) != 5:
            raise ValueError("Incorrect number of arguments. Please provide exactly 4 arguments.")
        #refinebio_sex 
        expression_filename = argv[1]
        metadata_filename = argv[2]
        comparison_type = argv[3]
        output_filename = argv[4]


        with open(expression_filename, 'r') as expressionFile, open(metadata_filename, 'r') as metadataFile, open(output_filename, 'w') as writeFile:
            expression_df = pd.read_csv(expressionFile, sep='\t')
            metadata_df = pd.read_csv(metadataFile, sep='\t')
            # print("metadata_df: ", metadata_df.head())

            expression_df = expression_df.set_index('Gene')
            expression_df.index.name = 'refinebio_accession_code'
            print("before transposing:\n", expression_df.head())
            # print("\n\n\n\n\n\n\n")

            expression_df = expression_df.transpose()
            #expression_df = expression_df.rename(columns={'Gene':'refinebio_accession_code'}, inplace=True)
            #expression_df.index.name = 'refinebio_accession_code'
            print("after transposing:\n", expression_df.head())
            # print("#rows: ", len(expression_df))

            metadata_df = metadata_df[['refinebio_accession_code', "refinebio_sex"]]
            # print("should only be 2 columns: ", metadata_df)

            merged_df = pd.merge(metadata_df, expression_df, on='refinebio_accession_code')
            print("after merging:", merged_df.head())

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
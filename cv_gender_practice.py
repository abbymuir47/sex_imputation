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

# command line command to run the program: python3 cv_gender_practice.py GSE10358/GSE10358.tsv GSE10358/metadata_GSE10358.tsv xy myoutput.tsv    


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

            column_names = expression_df.columns.tolist()

            expression_df = expression_df.transpose()
            expression_df.columns = expression_df.iloc[0] 
            expression_df = expression_df[1:]
            expression_df.insert(0, 'refinebio_accession_code', column_names[1:])

            metadata_df = metadata_df[['refinebio_accession_code', "refinebio_sex"]]

            merged_df = pd.merge(metadata_df, expression_df, on='refinebio_accession_code')
            print("after merging:\n", merged_df.head())

            #only keep rows with existing sex data
            values_to_keep = ['male', 'female']
            merged_df = merged_df[merged_df['refinebio_sex'].isin(values_to_keep)]

            #metadata_sex column
            X = merged_df.iloc[2:]
            #expression_sex column
            y = merged_df.iloc[1]

            #hyper parameters
            print("about to make randomforest")
            rf = RandomForestClassifier(n_estimators = 1000,
                                        criterion = 'entropy',
                                        min_samples_split = 10,
                                        max_depth = 14,
                                        random_state = 42
            )

            print("about to do cross validation")
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
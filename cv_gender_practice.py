import pandas as pd
import numpy as np
from sys import argv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
            exp_df = pd.read_csv(expressionFile, sep='\t')
            metadata_df = pd.read_csv(metadataFile, sep='\t')

            column_names = exp_df.columns.tolist()

            exp_df = exp_df.transpose()
            exp_df.columns = exp_df.iloc[0] 
            exp_df = exp_df[1:]
            exp_df.insert(0, 'refinebio_accession_code', column_names[1:])

            metadata_df = metadata_df[['refinebio_accession_code', "refinebio_sex"]]

            merged_df = pd.merge(metadata_df, exp_df, on='refinebio_accession_code')

            #only keep rows with existing sex data
            values_to_keep = ['male', 'female']
            merged_df = merged_df[merged_df['refinebio_sex'].isin(values_to_keep)]

            #target vector, sex
            sex_col = merged_df['refinebio_sex']
            y = sex_col

            #feature matrix, gene expression data
            X = merged_df.drop(columns=["refinebio_accession_code", "refinebio_sex"])

            #hyper parameters
            rf_model = RandomForestClassifier(n_estimators = 1000,
                                        criterion = 'entropy',
                                        min_samples_split = 10,
                                        max_depth = 14
            )

            # Create binary for y_true (turn male/female into zeroes and ones)
            binary_sex_col = sex_col.map({'male': 1, 'female': 0})

            #Calculate ROC AUC
            print("Calculating ROC AUC... ")
            roc_auc_scores = cross_val_score(rf_model, X, binary_sex_col, cv=6, scoring='roc_auc')
            print(f'ROC AUC scores for each fold: {roc_auc_scores}')
            print(f'Mean ROC AUC score: {roc_auc_scores.mean()}')

            #create output file
            cv_folds = np.arange(1, len(roc_auc_scores) + 1)
            output_df = pd.DataFrame({
                'input_filename' : [expression_filename] * len(roc_auc_scores),
                'cv_fold' : cv_folds,
                'roc_score' : roc_auc_scores
            })

            output_df.to_csv(writeFile, sep='\t', index=False)
            print(f"ROC AUC has been written to {writeFile.name}.")

    except ValueError as ve:
        print(f"Error: {ve}")
       
if __name__ == "__main__":
    main()
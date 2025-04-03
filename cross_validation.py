import pandas as pd
import numpy as np
from sys import argv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# command line command to run the program: python3 cross_validation.py GSE10358/GSE10358.tsv GSE10358/metadata_GSE10358.tsv sex myoutput.tsv

def main():
    try:
        if len(argv) != 5:
            raise ValueError("Incorrect number of arguments. Please provide exactly 4 arguments.")
        
        #assign arguments to variables
        expression_filename = argv[1]
        metadata_filename = argv[2]
        comparison_type = argv[3]
        output_filename = argv[4]

        #create and filter a dataframe of gene expression data, calculate roc auc scores of sex imputation predictions for the data, and write the output to a tsv file
        expression_df = create_expression_dataframe(expression_filename, metadata_filename)
        expression_df = filter_by_comparison_type(expression_df, comparison_type)
        roc_auc_scores = calculate_roc_auc(expression_df)
        write_to_csv(expression_filename, roc_auc_scores, output_filename)

    except ValueError as ve:
        print(f"Error: {ve}")
    

def create_expression_dataframe(expression_filename, metadata_filename):
    with open(expression_filename, 'r') as expressionFile, open(metadata_filename, 'r') as metadataFile:
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
        return merged_df

#returns a list of ensembl ID's to delete, based on the user's comparison type input
#if a user inputs 'sex', this function will return a list of gene ID's for genes 1-23
#if a user inputs 'autosomal', this function will return a list of gene ID's for XY genes

def get_drop_columns(comparison_type):

    with open("ensembl_data.tsv", "r") as readFile:
        ensembl_df = pd.read_csv(readFile, sep='\t')

        if(comparison_type == "sex"):
            filtered_df = ensembl_df[ensembl_df['chromosome_name'].str.contains(r'\d{1,2}')]
        elif(comparison_type == "autosomal"):
            filtered_df = ensembl_df[ensembl_df['chromosome_name'].isin(['X', 'Y'])]
            print(f"there are {len(filtered_df)} xy genes")
        elif(comparison_type == "whole_genome"):
            filtered_df = ensembl_df
        else:
            return("comparison type not recognized; please enter 'sex', 'autosomal', or 'whole genome' as your argument")

        #dropping unneccessary columns to ensure that they aren't interfering with the cross validation results 
        filtered_df.drop(columns=['gene_biotype','external_gene_name'], axis=1, inplace=True)
        #print(filtered_df.head())
        
        return filtered_df['ensembl_gene_id'].tolist()

def filter_by_comparison_type(expression_df, comparison_type):
    expression_columns = expression_df.columns.tolist()

    #get ensembl gene ids for the indicated comparison type, then get the set of those that are included in the expression_df
    ensembl_ids_to_drop = get_drop_columns(comparison_type)
    intersection_set = set(expression_columns) & set(ensembl_ids_to_drop)
    intersection_list = list(intersection_set)

    #drop columns that are for unwanted gene IDs
    expression_df.drop(columns=intersection_list, axis=1, inplace=True)
    return expression_df

def calculate_roc_auc(expression_df):
    #target vector, sex
    sex_col = expression_df['refinebio_sex']
    y = sex_col

    #feature matrix, gene expression data
    X = expression_df.drop(columns=["refinebio_accession_code", "refinebio_sex"])

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
    roc_auc_scores = cross_val_score(rf_model, X, binary_sex_col, cv=5, scoring='roc_auc')
    print(f'ROC AUC scores for each fold: {roc_auc_scores}')
    print(f'Mean ROC AUC score: {roc_auc_scores.mean()}')
    return roc_auc_scores
    

def write_to_csv(expression_filename, roc_auc_scores, output_filename):
    #create output df
    cv_folds = np.arange(1, len(roc_auc_scores) + 1)
    output_df = pd.DataFrame({
        'input_filename' : [expression_filename] * len(roc_auc_scores),
        'cv_fold' : cv_folds,
        'roc_score' : roc_auc_scores
    })

    #write output df to tsv file
    with open(output_filename, 'w') as writeFile:
        output_df.to_csv(writeFile, sep='\t', index=False)
        print(f"ROC AUC has been written to {output_filename}.")


if __name__ == "__main__":
    main()
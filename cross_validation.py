import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress all ConvergenceWarnings from scikit-learn
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

from sys import argv
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif

# command line commands to run the program with different model types: 
# python3 cross_validation.py GSE10358/GSE10358.tsv GSE10358/metadata_GSE10358.tsv sex random_forest myoutput.tsv
# python3 cross_validation.py GSE10358/GSE10358.tsv GSE10358/metadata_GSE10358.tsv sex logistic_regression myoutput.tsv
# python3 cross_validation.py GSE10358/GSE10358.tsv GSE10358/metadata_GSE10358.tsv sex decision_trees myoutput.tsv

def main():
    try:
        if len(argv) != 6:
            raise ValueError("Incorrect number of arguments. Please provide exactly 4 arguments.")
        
        #assign arguments to variables
        expression_filename = argv[1]
        metadata_filename = argv[2]
        comparison_type = argv[3]
        model_type = argv[4]
        output_filename = argv[5]

        #create and filter a dataframe of gene expression data, calculate roc auc scores of sex imputation predictions for the data, and write the output to a tsv file
        columns, expression_df = create_expression_dataframe(expression_filename, metadata_filename)
        expression_df = filter_by_comparison_type(expression_df, comparison_type)
        roc_auc_scores = calculate_roc_auc(expression_df, model_type, columns)
        write_to_tsv(expression_filename, roc_auc_scores, output_filename)

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
        columns = merged_df.columns.tolist()
        
        return columns, merged_df

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
        filtered_df = filtered_df.drop(columns=['gene_biotype','external_gene_name'], axis=1)
        return filtered_df['ensembl_gene_id'].tolist()

#modifies dataframe to only include desired genes based on user's selected comparison type
def filter_by_comparison_type(expression_df, comparison_type):
    expression_columns = expression_df.columns.tolist()

    #get ensembl gene ids for the indicated comparison type, then get the set of those that are included in the expression_df
    ensembl_ids_to_drop = get_drop_columns(comparison_type)
    intersection_set = set(expression_columns) & set(ensembl_ids_to_drop)
    intersection_list = list(intersection_set)

    #drop columns that are for unwanted gene IDs
    expression_df = expression_df.drop(columns=intersection_list, axis=1)
    return expression_df

#calculates roc auc scores for the data
def calculate_roc_auc(expression_df, model_type, gene_names):
    #target vector, sex
    sex_col = expression_df['refinebio_sex']
    y = sex_col

    #feature matrix, gene expression data
    X = expression_df.drop(columns=["refinebio_accession_code", "refinebio_sex"])

    select_features(X,y, gene_names)

    #create model with correct criteria 
    model = make_model(model_type)
    
    # Create binary for y_true (turn male/female into zeroes and ones)
    binary_sex_col = sex_col.map({'male': 1, 'female': 0})

    #Calculate ROC AUC
    print("Calculating ROC AUC... ")

    random.seed(42)
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    roc_auc_scores = cross_val_score(model, X, binary_sex_col, cv=cv, scoring='roc_auc')
    print(f'ROC AUC scores for each fold, using {model_type} model: {roc_auc_scores}')
    print(f'Mean ROC AUC score, using {model_type} model: {roc_auc_scores.mean()}')
    return roc_auc_scores

#makes the classification model with the best criteria given a user's choice     
def make_model(model_type):
    try:
        if(model_type == "random_forest"):
            # Best Parameters: {'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 20, 'n_estimators': 500}, Best Score:  0.9625
            model = RandomForestClassifier(bootstrap=False, 
                                            max_depth=None, 
                                            min_samples_leaf=1, 
                                            min_samples_split=20, 
                                            n_estimators=500, 
                                            random_state = 42)
        elif(model_type == "decision_trees"):
            # Best Parameters: {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2}, Best Score:  0.9374113475177305
            model = tree.DecisionTreeClassifier(criterion='entropy',
                                                max_depth = None,
                                                min_samples_leaf=2, 
                                                min_samples_split=2, 
                                                random_state = 42)
        elif(model_type == "logistic_regression"):
            # Best Parameters: {'C': 0.01, 'l1_ratio': 0.5, 'penalty': 'elasticnet', 'solver': 'saga'}, Best Score:  0.9416666666666668
            model = LogisticRegression(penalty='elasticnet', 
                                        solver='saga', 
                                        C=0.01, 
                                        l1_ratio=0.5, 
                                        max_iter=1000)
    except ValueError as ve:
        print(f"Error: please enter model type as random_forest, decision_trees, or logistic_regression")

#writes the given roc auc scores to a tsv file 
def write_to_tsv(expression_filename, roc_auc_scores, output_filename):
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

def select_features(X,y, gene_names):
    print("Performing Univariate Feature Selection\n")
    print("X.shape: ", X.shape)
    selector = SelectKBest(f_classif, k=10)
    
    X_new = selector.fit_transform(X, y)
    print("New X.shape: ",X_new.shape)

    selected_mask = selector.get_support()
    selected_features = [gene_names[i] for i in range(len(gene_names)) if selected_mask[i]]

    print("Selected features: ", selected_features)




if __name__ == "__main__":
    main()
import os, torch
import pandas as pd
import numpy as np
from BertSequenceCLF import train_for_sequence_classification, predict_sequence_classification
from sklearn.model_selection import StratifiedKFold


def main(input_file, output_dir, textid_column, text_column, label_column, 
         model_name='emanjavacas/MacBERTh', n_folds=10, batch_size=32, epochs=5, 
         seed=42, loss_type='cross_entropy_standard', lr=2e-5): 
    
    # Load data
    df = pd.read_csv(input_file, sep=";", encoding='unicode_escape')
    df['text'] = df[text_column]
    df['label'] = df[label_column]
    df = df[[textid_column, 'text', 'label']].dropna().reset_index(drop=True)
    labels = sorted(df['label'].unique())
    label2id = {label: i for i, label in enumerate(labels)}

    # Store predictions
    experiment_name = model_name.split('/')[-1] + '-' + '-'.join(
        [str(batch_size), str(epochs), str(loss_type), str(lr), str(seed)])
    all_predictions = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Create folds
    X = df.drop('label', axis=1)
    y = df['label']
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))

    for test_fold_idx in range(n_folds):
        print(f"\n===== Fold {test_fold_idx + 1}/{n_folds} =====")
        
        test_idx = folds[test_fold_idx][1]
        remaining_folds = [i for i in range(n_folds) if i != test_fold_idx]
        dev_idx = folds[remaining_folds[0]][1]
        train_idx = []
        for fold_i in remaining_folds[1:]:
            train_idx.extend(folds[fold_i][1])
        
        # Prepare data splits
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        train_df = X_train.copy()
        train_df['label'] = y_train.values
        dev_df = X_dev.copy()
        dev_df['label'] = y_dev.values
        test_df = X_test.copy()
        test_df['label'] = y_test.values
        
        print('train-val-test sized: ', len(train_df), len(dev_df), len(test_df))
        
        # Model output dir
        fold_output_dir = output_dir+experiment_name+'/fold'+str(test_fold_idx)
        os.makedirs(fold_output_dir, exist_ok=True)

        # Train and test
        train_for_sequence_classification(model_name, fold_output_dir, train_df, dev_df, 
                                          label2id, batch_size, epochs, seed, loss_type, lr) 
        preds = predict_sequence_classification(fold_output_dir, model_name, test_df, label2id, batch_size)
        test_ids = test_df[textid_column].tolist()
        fold_predictions = pd.DataFrame({textid_column: test_ids, f'pred_{experiment_name}': preds, "test_fold": test_fold_idx})
        all_predictions.append(fold_predictions)

    
    # Save all predictions
    original_df = pd.read_csv(input_file, encoding='unicode_escape', sep=';')
    allpreds_df = pd.concat(all_predictions, ignore_index=True)
    final_df = pd.merge(original_df, allpreds_df, on=textid_column, how='left')
    output_file = os.path.join(output_dir, f"{experiment_name}_preds.csv")
    final_df.to_csv(output_file, index=False)
    
if __name__ == '__main__':

    csv_path = "...csv"
    textid_column = 'sentenceID'
    text_column = 'originalsentence'
    label_column = 'invective'
    output_dir = '../output/'
    model_name = 'FacebookAI/xlm-roberta-large'

    main(csv_path, output_dir, textid_column, text_column, label_column,
        model_name=model_name)

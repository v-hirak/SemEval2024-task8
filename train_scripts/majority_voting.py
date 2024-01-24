import json
import pandas as pd
import argparse
from statistics import mode

def get_mode(row):
   return mode(row)
  
def majority_voting(preds1, preds2, preds3):
    labels1 = pd.read_json(preds1, lines=True)[['id', 'label']]
    labels2 = pd.read_json(preds2, lines=True)[['id', 'label']]
    labels3 = pd.read_json(preds3, lines=True)[['id', 'label']]
    merged_labels12 = labels1.merge(labels2, on='id')
    merged_labels123 = merged_labels12.merge(labels3, on='id')


    merged_labels123['final_label'] = merged_labels123[['label_x', 'label_y', 'label']].values.tolist()
    merged_labels123['label'] = merged_labels123['final_label'].apply(get_mode)

    return merged_labels123[['id', 'label']]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--preds1", '-p1', type=str, required=True, help="Predictions of model X1")
  parser.add_argument("--preds2", '-p2', type=str, required=True, help="Predictions of model X2")
  parser.add_argument("--preds3", '-p3', type=str, required=True, help="Predictions of model X3")

  parser.add_argument("--final_path", '-p', type=str, required=True, help="Path to save predictions")
  args = parser.parse_args()

  predictions1 = args.preds1
  predictions2 = args.preds2
  predictions3 = args.preds3
  final_path = args.final_path

  preds = majority_voting(predictions1, predictions2, predictions3)

  preds.to_json(final_path, \
                lines=True, orient='records')



import json
with open('cynthia_churn.ipynb','r',encoding='utf-8') as f:
    nb=json.load(f)
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source',[]))
    if 'for model_name, model in models.items' in src:
        print('TRAIN_LOOP_CELL', i)
    if 'results.append(result)' in src and 'evaluate_model' in src and 'Confusion Matrix' in src:
        print('TRAIN_CELL_ALT', i)
        print(src[:2000])

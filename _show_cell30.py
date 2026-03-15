import json
with open('cynthia_churn.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    src = c.get('source', [])
    full = ''.join(src) if isinstance(src, list) else src
    if 'Cross-validation score' in full and 'evaluate_model' in full:
        print('Cell index:', i)
        print('Snippet:', repr(full[full.find('Cross-validation'):full.find('Cross-validation')+200]))
        break

# Patch cynthia_churn.ipynb: update evaluate_model cell (cell 30) for CV with ROC-AUC
import json
path = 'cynthia_churn.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][28]
src = cell['source']
if isinstance(src, list):
    full = ''.join(src)
else:
    full = src

old_cv = """    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')"""

new_cv = """    # Stratified K-Fold Cross-Validation (Accuracy + ROC-AUC)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(model, X_train, y_train, cv=skf, scoring=['accuracy', 'roc_auc', 'f1'], n_jobs=-1)
    cv_scores = cv_res['test_accuracy']
    cv_roc_auc_scores = cv_res['test_roc_auc']"""

old_store = """        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Confusion Matrix': cm,"""

new_store = """        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'CV ROC-AUC Mean': cv_roc_auc_scores.mean(),
        'CV ROC-AUC Std': cv_roc_auc_scores.std(),
        'Confusion Matrix': cm,"""

if old_cv in full and old_store in full:
    full = full.replace(old_cv, new_cv).replace(old_store, new_store)
    lines = full.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1] + '\n'] if lines[-1] else [])
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print('Patched cell 28 OK')
else:
    print('Pattern not found')
    print('Has old_cv:', old_cv in full)
    print('Has old_store:', old_store in full)

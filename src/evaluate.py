
import sys
import io
import time
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection   import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition     import PCA
from sklearn.preprocessing     import label_binarize, RobustScaler
from sklearn.metrics           import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec  as gridspec

from model      import NetworkTransformer
from load_data  import load_data, NUMERIC_FEATURES, ATTACK_TYPES, NUM_CLASSES
from preprocess import NetworkPreprocessor, prepare_data, compute_class_weights, LOG_COLS


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model over a full DataLoader, return (y_true, y_pred, y_prob)."""
    model.eval()
    ys, logits_list = [], []
    for bx, by in loader:
        logits_list.append(model(bx.to(device)).cpu())
        ys.append(by)
    logits = torch.cat(logits_list, 0)
    y_true = torch.cat(ys, 0).numpy()
    y_prob = torch.softmax(logits, dim=1).numpy()
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_pred, y_prob




def plot_roc(y_true, y_prob, save_path='./checkpoints/roc_curves.png'):

    y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))  # (N, 10)

    fpr, tpr, auc_val = {}, {}, {}

    for i in range(NUM_CLASSES):
        if y_bin[:, i].sum() == 0:
            continue                                            # no test samples → skip
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_val[i]        = auc(fpr[i], tpr[i])

    # micro-average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    auc_val['micro'] = auc(fpr['micro'], tpr['micro'])
    auc_val['macro'] = np.mean([auc_val[i] for i in range(NUM_CLASSES) if i in auc_val])

    # ── plot ──
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    for i in range(NUM_CLASSES):
        if i not in auc_val:
            continue
        ax.plot(fpr[i], tpr[i], color=colors[i], lw=1.8,
                label=f'{ATTACK_TYPES[i]:12s}  AUC = {auc_val[i]:.3f}')

    ax.plot(fpr['micro'], tpr['micro'], 'k--', lw=2.2,
            label=f'{"micro-avg":12s}  AUC = {auc_val["micro"]:.3f}')
    ax.plot([0, 1], [0, 1], 'gray', lw=0.8, ls=':')            # random baseline

    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title=f'ROC  (one-vs-rest)  —  macro AUC = {auc_val["macro"]:.3f}')
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {save_path}")
    plt.close()
    return auc_val



def measure_response_time(model, loader, device, n_warmup=3):
    """
    Per-sample detection latency in milliseconds.

    Why warmup matters on CUDA:
        The first few forward passes trigger kernel compilation and memory
        allocation.  Their timings are anomalously high.  We run n_warmup
        batches first and discard their times.

    Why synchronize():
        CUDA kernels run asynchronously.  Without synchronize(), perf_counter()
        returns before the GPU has finished — giving you CPU overhead only,
        not actual inference time.  This is the single most common GPU
        timing mistake.

    Reports mean latency (ms/sample) and throughput (samples/sec) — the
    throughput number is what you'd compare against other IDS algorithms.
    """
    model.eval()

    # warmup
    warmed = 0
    with torch.no_grad():
        for bx, _ in loader:
            model(bx.to(device))
            if device != 'cpu':
                torch.cuda.synchronize()
            warmed += 1
            if warmed >= n_warmup:
                break

    # timed inference
    records = []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            if device != 'cpu':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _  = model(bx)

            if device != 'cpu':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            records.append((t1 - t0, bx.size(0)))

    tot_time    = sum(t for t, _ in records)
    tot_samples = sum(n for _, n in records)
    mean_ms     = (tot_time / tot_samples) * 1000
    throughput  = tot_samples / tot_time                      

    per_batch_ms = np.array([(t / n) * 1000 for t, n in records])

    print(f"\n  Response Time  (device={device}, batch_size={records[0][1]})")
    print(f"    Samples            : {tot_samples:,}")
    print(f"    Mean latency       : {mean_ms:.3f} ms / sample")
    print(f"    Std                : {per_batch_ms.std():.3f} ms")
    print(f"    Throughput         : {throughput:,.0f} samples / sec")
    print(f"    Total inference    : {tot_time * 1000:.1f} ms")

    return {'mean_latency_ms': mean_ms, 'std_latency_ms': float(per_batch_ms.std()),
            'throughput_sps': throughput, 'total_samples': tot_samples}




def feature_selection_mi(X, y, save_path='./checkpoints/feature_importance_mi.png'):
    """
    Mutual Information between each feature and the 10-class target.

    MI answers: "How many bits of information about the attack type does
    knowing this single feature give me?"

    Computed on raw flows (not sequences) because MI is a per-sample metric.
    Log-transform is applied first so skewed byte/packet counts don't
    dominate the k-NN MI estimator.
    """
    Xm = X.copy()
    for col in Xm.columns:
        if col in LOG_COLS:
            Xm[col] = np.log1p(np.maximum(Xm[col].values, 0.0))

    print(f"  Computing MI on {len(X):,} flows x {len(X.columns)} features …",
          flush=True, end=' ')
    mi = mutual_info_classif(Xm, y, random_state=42, n_neighbors=5)
    print("done")

    df = (pd.DataFrame({'feature': Xm.columns, 'MI': mi})
            .sort_values('MI', ascending=False).reset_index(drop=True))

    # ── console table ──
    mx = df['MI'].max() or 1.0
    print(f"\n  {'Feature':<34} {'MI':>8}")
    print(f"  {'─' * 44}")
    for _, r in df.iterrows():
        bar = '█' * int(r['MI'] / mx * 24)
        print(f"  {r['feature']:<34} {r['MI']:>8.4f}  {bar}")

    # ── plot ──
    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))
    ax.barh(df['feature'][::-1], df['MI'][::-1], color=colors)
    ax.set(xlabel='Mutual Information Score',
           title='Feature Importance  (MI vs 10-class target)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {save_path}")
    plt.close()
    return df



def feature_reduction_pca(X, save_path='./checkpoints/pca_variance.png'):
    """
    PCA on numeric features (after log + RobustScale).

    Shows the effective dimensionality of TON_IOT traffic.
    Key numbers for the paper:
        - Components needed for 95 % and 99 % cumulative variance
        - This justifies the choice of d_model = 128
          (if 95 % variance is captured by, say, 8 components, then 128 is
          generous and the model has capacity to spare — which is fine for
          a transformer that also models temporal relationships).
    """
    num_cols = [c for c in X.columns if c in NUMERIC_FEATURES]
    Xn = X[num_cols].copy()
    for col in Xn.columns:
        if col in LOG_COLS:
            Xn[col] = np.log1p(np.maximum(Xn[col].values, 0.0))

    Xs     = RobustScaler().fit_transform(Xn)
    pca    = PCA(random_state=42).fit(Xs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    n_95 = int(np.argmax(cumvar >= 0.95) + 1)
    n_99 = int(np.argmax(cumvar >= 0.99) + 1)

    print(f"\n  PCA Feature Reduction:")
    print(f"    Original dimensions        : {Xs.shape[1]}")
    print(f"    Components for 95 % var    : {n_95}")
    print(f"    Components for 99 % var    : {n_99}")

    # ── plot ──
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = range(1, len(cumvar) + 1)

    a1.bar(xs, pca.explained_variance_ratio_, color='steelblue', alpha=0.75)
    a1.set(xlabel='Component', ylabel='Explained Variance Ratio',
           title='Per-Component Variance')
    a1.grid(axis='y', alpha=0.3)

    a2.plot(xs, cumvar, 'o-', color='darkorange', lw=2, ms=4)
    a2.axhline(0.95, color='red',   ls='--', lw=1.2, label=f'95 % @ {n_95} comp.')
    a2.axhline(0.99, color='green', ls='--', lw=1.2, label=f'99 % @ {n_99} comp.')
    a2.set(xlabel='Components', ylabel='Cumulative Variance',
           title='Cumulative Explained Variance', ylim=(0, 1.05))
    a2.legend(fontsize=9)
    a2.grid(alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {save_path}")
    plt.close()
    return pca, n_95, n_99




def classifier_fairness(y_true, y_pred,
                        save_path='./checkpoints/fairness.png'):
    """
    Equalized-odds analysis across the 10 attack types.

    In network IDS, "fairness" means the model doesn't systematically
    fail on certain attack types.  The standard formalization is
    Equalized Odds (Hardt et al. 2016):

        Equal Opportunity  : TPR should be the same across all classes
        Equalized Odds     : Both TPR AND FPR should be the same

    We report per-class values and the max-min gap for each.
    A gap of 0 = perfectly fair.  A large gap = the model is biased
    toward/against certain attack types.

    Definitions (per class i):
        TPR_i  =  P(predict i  |  true = i)     — recall
        FPR_i  =  P(predict i  |  true != i)    — false-alarm rate for class i
    """
    tpr = np.zeros(NUM_CLASSES)
    fpr = np.zeros(NUM_CLASSES)
    sup = np.zeros(NUM_CLASSES, dtype=int)

    for i in range(NUM_CLASSES):
        pos    = (y_true == i)
        neg    = ~pos
        sup[i] = pos.sum()

        if pos.sum() == 0:
            tpr[i] = np.nan
            continue

        tpr[i] = ((y_pred == i) & pos).sum() / pos.sum()

        if neg.sum() > 0:
            fpr[i] = ((y_pred == i) & neg).sum() / neg.sum()

    valid_tpr = tpr[~np.isnan(tpr)]
    gap_tpr   = float(valid_tpr.max() - valid_tpr.min())       # Equal Opportunity Gap
    gap_fpr   = float(fpr.max() - fpr.min())                   # Equalized Odds Gap

    # ── console table ──
    print(f"\n  Classifier Fairness  (Equalized Odds):")
    print(f"  {'Class':<13} {'TPR':>7} {'FPR':>10} {'Support':>9}")
    print(f"  {'─' * 43}")
    for i in range(NUM_CLASSES):
        print(f"  {ATTACK_TYPES[i]:<13} {tpr[i]:>7.4f} {fpr[i]:>10.6f} {sup[i]:>9,}")
    print(f"\n  Equal Opportunity Gap (TPR) : {gap_tpr:.4f}")
    print(f"  Equalized Odds Gap   (FPR) : {gap_fpr:.6f}")
    print(f"  (0.0 = perfectly fair across all classes)")

    # ── plot ──
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(NUM_CLASSES)

    # TPR bars — color-coded green/orange/red by severity
    c_tpr = ['#2ecc71' if t > 0.95 else '#f39c12' if t > 0.8 else '#e74c3c'
             for t in tpr]
    a1.bar(x, tpr, color=c_tpr, edgecolor='black', linewidth=0.5)
    a1.axhline(valid_tpr.min(), color='red', ls=':', lw=1.5,
               label=f'min TPR = {valid_tpr.min():.3f}')
    a1.set(xticks=x, xticklabels=ATTACK_TYPES, ylabel='TPR (Recall)',
           title=f'Equal Opportunity\nGap = {gap_tpr:.4f}', ylim=(0, 1.15))
    a1.tick_params(axis='x', rotation=30)
    a1.legend(fontsize=9)
    a1.grid(axis='y', alpha=0.3)

    # FPR bars
    a2.bar(x, fpr, color='indianred', edgecolor='black', linewidth=0.5)
    a2.axhline(fpr.max(), color='red', ls=':', lw=1.5,
               label=f'max FPR = {fpr.max():.6f}')
    a2.set(xticks=x, xticklabels=ATTACK_TYPES, ylabel='FPR',
           title=f'Equalized Odds (FPR)\nGap = {gap_fpr:.6f}')
    a2.tick_params(axis='x', rotation=30)
    a2.legend(fontsize=9)
    a2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {save_path}")
    plt.close()
    return {'tpr': tpr, 'fpr': fpr, 'eo_gap_tpr': gap_tpr, 'eo_gap_fpr': gap_fpr}



def _train_one_fold(X_tr, y_tr, X_va, y_va, X_te, y_te,
                    cat_indices, cat_vocab_sizes, num_numeric, device,
                    seq_len=10, stride=5, batch_size=128,
                    max_epochs=40, patience=8, lr=1e-3):
    """
    Train one model on a single k-fold partition.
    Returns (confusion_matrix, best_val_accuracy).

    Preprocessing prints are suppressed (contextlib.redirect_stdout) so
    the outer loop can print a clean per-fold summary line instead of
    per-class sequence counts for each fold.
    """
    # ── preprocess (suppressed output) ──
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        prep           = NetworkPreprocessor(seq_len=seq_len, stride=stride)
        X_tr_s, y_tr_s = prep.fit_transform(X_tr, y_tr)
        X_va_s, y_va_s = prep.transform(X_va, y_va)
        X_te_s, y_te_s = prep.transform(X_te, y_te)
        w              = compute_class_weights(y_tr_s)

    def _dl(Xs, ys, shuf):
        return DataLoader(
            TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)),
            batch_size=batch_size, shuffle=shuf, pin_memory=True
        )

    tr_dl = _dl(X_tr_s, y_tr_s, True)
    va_dl = _dl(X_va_s, y_va_s, False)
    te_dl = _dl(X_te_s, y_te_s, False)

    # ── model + optimiser ──
    model = NetworkTransformer(
        num_numeric=num_numeric, cat_indices=cat_indices,
        cat_vocab_sizes=cat_vocab_sizes, num_classes=NUM_CLASSES,
        d_model=128, nhead=8, num_layers=3,
    ).to(device)

    crit  = nn.CrossEntropyLoss(weight=w.to(device))
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(tr_dl),
        epochs=max_epochs, pct_start=0.1, anneal_strategy='cos'
    )

    # ── training loop ──
    best_acc, best_state, no_imp = 0.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        for bx, by in tr_dl:
            bx, by = bx.to(device), by.to(device)
            loss   = crit(model(bx), by)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        # validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in va_dl:
                correct += (model(bx.to(device)).argmax(1).cpu() == by).sum().item()
                total   += by.size(0)
        acc = correct / total

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    # ── evaluate on held-out test with best weights ──
    model.load_state_dict(best_state)
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for bx, by in te_dl:
            preds.append(model(bx.to(device)).argmax(1).cpu().numpy())
            labels.append(by.numpy())

    cm = confusion_matrix(np.concatenate(labels), np.concatenate(preds),
                          labels=range(NUM_CLASSES))
    return cm, best_acc


def kfold_confusion_matrices(X, y, n_folds=5, device='cpu',
                             save_path='./checkpoints/kfold_confusion_matrices.png',
                             **kw):
    """
    Stratified k-fold CV with full retraining per fold.

    For each fold:
        1. Hold out 1/k flows as the test set
        2. Split the remaining flows 85 / 15 into train / val
        3. Fit a fresh preprocessor on train only  (no leakage)
        4. Train a fresh model → early stopping on val accuracy
        5. Evaluate on held-out test → confusion matrix

    Produces k individual CMs plus one averaged CM.
    The average is computed on the raw (unnormalized) matrices, then
    row-normalized for display — this gives each sample equal weight.
    """
    cat_col_names   = ['proto_enc', 'service_enc', 'conn_state_enc']
    col_list        = X.columns.tolist()
    cat_indices     = [col_list.index(c) for c in cat_col_names]
    cat_vocab_sizes = [int(X[c].max()) + 1 for c in cat_col_names]
    num_numeric     = len(col_list) - len(cat_indices)

    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cms, accs = [], []

    print(f"\n{'=' * 58}")
    print(f"  {n_folds}-Fold CV  (trains {n_folds} models from scratch)")
    print(f"  Estimated: 2-4 min / fold on GPU, longer on CPU")
    print(f"{'=' * 58}")

    for fold, (tv_idx, te_idx) in enumerate(skf.split(X, y), 1):
        t0 = time.perf_counter()

        X_tv, X_te = X.iloc[tv_idx], X.iloc[te_idx]
        y_tv, y_te = y.iloc[tv_idx], y.iloc[te_idx]

        # 85 / 15 train-val split from the non-test portion
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tv, y_tv, test_size=0.15, random_state=42, stratify=y_tv
        )

        print(f"\n  Fold {fold}/{n_folds}  "
              f"(train {len(X_tr):,}  val {len(X_va):,}  test {len(X_te):,}) … ",
              end='', flush=True)

        cm, acc = _train_one_fold(
            X_tr, y_tr, X_va, y_va, X_te, y_te,
            cat_indices, cat_vocab_sizes, num_numeric, device, **kw
        )
        cms.append(cm)
        accs.append(acc)
        print(f"acc={acc:.4f}  ({time.perf_counter() - t0:.0f}s)")

    cms    = np.array(cms)                                      # (k, 10, 10)
    avg_cm = cms.mean(axis=0)

    print(f"\n  Fold accs : {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean ± Std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # ── plot: individual fold CMs (rows of 3) + averaged CM (full-width bottom) ──
    n_cols      = 3
    n_fold_rows = (n_folds + n_cols - 1) // n_cols             # ceil div
    n_rows      = n_fold_rows + 1                              # +1 for average row

    fig = plt.figure(figsize=(n_cols * 5, n_rows * 4.8))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    def _draw(ax, cm_mat, title):
        """Draw one row-normalised heatmap on the given axes."""
        rs   = cm_mat.sum(axis=1, keepdims=True)
        rs   = np.where(rs == 0, 1, rs)                        # avoid /0 for empty classes
        norm = cm_mat.astype(float) / rs

        im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(ATTACK_TYPES, rotation=38, ha='right', fontsize=6.5)
        ax.set_yticklabels(ATTACK_TYPES, fontsize=6.5)
        ax.set(xlabel='Predicted', ylabel='True', title=title)
        ax.title.set_fontsize(8.5)

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                c = 'white' if norm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{norm[i, j]:.2f}',
                        ha='center', va='center', color=c, fontsize=5.5)

    # individual folds
    for idx in range(n_folds):
        r, c = divmod(idx, n_cols)
        _draw(fig.add_subplot(gs[r, c]), cms[idx],
              f'Fold {idx + 1}  (acc={accs[idx]:.4f})')

    # average CM — spans full width of the last row
    _draw(fig.add_subplot(gs[n_fold_rows, :]), avg_cm,
          f'Average ({n_folds}-Fold)  —  {np.mean(accs):.4f} ± {np.std(accs):.4f}')

    plt.suptitle('K-Fold CV Confusion Matrices  (row-normalised)',
                 fontsize=12, y=1.01)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {save_path}")
    plt.close()
    return cms, avg_cm, accs


# ═══════════════════════════════════════════════════════════════
# MAIN  —  run all 6 evaluation sections
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── simple arg parsing ──
    pos_args   = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags      = set(sys.argv[1:]) - set(pos_args)
    csv_path   = pos_args[0] if pos_args else "train_test_network.csv"
    skip_kfold = '--no-kfold' in flags
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── load data ──
    X, y = load_data(csv_path)

    # ── categorical metadata (same derivation as train.py) ──
    cat_col_names   = ['proto_enc', 'service_enc', 'conn_state_enc']
    col_list        = X.columns.tolist()
    cat_indices     = [col_list.index(c) for c in cat_col_names]
    cat_vocab_sizes = [int(X[c].max()) + 1 for c in cat_col_names]
    num_numeric     = len(col_list) - len(cat_indices)

    # ── DataLoaders (for model-dependent evaluations) ──
    loaders, _, _ = prepare_data(X, y, seq_len=10, stride=5, batch_size=128)

    # ── load the best trained model ──
    ckpt = './checkpoints/best_model.pt'
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"No model at {ckpt}. Run train.py first.")

    model = NetworkTransformer(
        num_numeric=num_numeric, cat_indices=cat_indices,
        cat_vocab_sizes=cat_vocab_sizes, num_classes=NUM_CLASSES,
        d_model=128, nhead=8, num_layers=3,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"  Loaded best model → {ckpt}")

    # ── collect test-set predictions once ──
    y_true, y_pred, y_prob = collect_predictions(model, loaders['test'], device)

    print(f"\n{'=' * 58}")
    print(f"  Comprehensive Evaluation  |  test = {len(y_true):,} sequences")
    print(f"{'=' * 58}")

    # ── 1. ROC ──────────────────────────────────────────────
    print("\n── 1/6  ROC Curves ──")
    roc_auc = plot_roc(y_true, y_prob)

    # ── 2. Response Time ────────────────────────────────────
    print("\n── 2/6  Response Time ──")
    rt = measure_response_time(model, loaders['test'], device)

    # ── 3. Feature Selection (MI) ───────────────────────────
    print("\n── 3/6  Feature Selection (MI) ──")
    mi_df = feature_selection_mi(X, y)

    # ── 4. Feature Reduction (PCA) ──────────────────────────
    print("\n── 4/6  Feature Reduction (PCA) ──")
    pca, n95, n99 = feature_reduction_pca(X)

    # ── 5. Classifier Fairness ──────────────────────────────
    print("\n── 5/6  Classifier Fairness ──")
    fair = classifier_fairness(y_true, y_pred)

    # ── 6. K-Fold CV ────────────────────────────────────────
    if skip_kfold:
        print("\n── 6/6  K-Fold  (SKIPPED — remove --no-kfold to run) ──")
    else:
        print("\n── 6/6  K-Fold Confusion Matrices ──")
        cms, avg_cm, fold_accs = kfold_confusion_matrices(
            X, y, n_folds=5, device=device, max_epochs=40, patience=8
        )

    print(f"\n{'=' * 58}")
    print(f"  All evaluations complete.  Plots → ./checkpoints/")
    print(f"{'=' * 58}")
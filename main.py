"""
Usage: see README.md

Working:
1. Load .h5 files from datasets/Intra/train -> preprocess -> windows
2. Train model on those windows using training data, tuning data and test data
3. Test on datasets/Intra/test -> accuracy + confusion matrix
4. Repeat for Cross experiment
5. Save all plots
6. Save results table
"""

import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils            import *
from src.config.config    import *
from src.data.loader      import *
from src.models           import get_model
from src.training.trainer import *
from src.evaluation.plots import *

# Reproducability (NOTE: still stochastic behaviour)
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# Run the models on Intra
def run_intra(model_name: str, eval_only: bool = False) -> dict[str, float]:
    """
    Train and evaluate on the Intra-subject dataset.
    Intra means training on same subject.
    Returns dict with accuracy and F1 score.
    """
    print(f"\n{'═'*50}")
    print(f"Running Intra with model: {model_name.upper()}")
    print(f"{'═'*50}")

    # 1. Load the data in
    (train_loader, val_loader, test_loader) = build_loaders(
        INTRA_TRAIN, INTRA_TEST, verbose = True)

    # 2. Load the model
    model = get_model(model_name)

    # 3. Train or load results
    ckpt_dir = CHECKPOINTS_DIR / f"{model_name}_intra_best.pt"
    if eval_only:
        print(f"\n-- Loading checkpoint: {ckpt_dir} --")
        model.load_state_dict(torch.load(ckpt_dir, map_location = DEVICE))
        history, training_time = None, 0.0
    else:
        start_time = time.time()
        history = train(model, train_loader, val_loader,
            model_name = model_name, experiment = "intra",
            epochs = EPOCHS, lr = LEARNING_RATE,
            weight_decay = WEIGHT_DECAY, patience = PATIENCE,
            mixup_alpha = MIXUP_ALPHA_INTRA)
        training_time = time.time() - start_time
        if history:
            plot_training_curves(history, model_name.upper(), "Intra")

    # 4. Evaluate on the test set
    model = model.to(DEVICE)
    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    plot_confusion_matrix(y_true, y_pred, model_name.upper(), "Intra")
    metrics = print_report(y_true, y_pred, model_name.upper(), "Intra")

    # 5. Store extra metrics
    metrics["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics["training_time"] = training_time
    if history is not None:
        metrics["last-epoch"] = history.get("last-epoch", [-1])[0]

    return metrics

# Run the models on Cross
def run_cross(model_name: str, eval_only: bool = False) -> dict[str, float]:
    """
    Train and evaluate on the Cross-subject dataset.
    Cross-subject means to train on subjects A & B, test on different subjects C, D, E.
    Uses chunked loading (hint) to handle large number of training files.
    Returns dict with accuracy and F1 score.
    """
    print(f"\n{'═'*50}")
    print(f"Running cross with model: {model_name.upper()}")
    print(f"{'═'*50}")

    # 1. Build file list
    test_folders = [CROSS_TEST1, CROSS_TEST2, CROSS_TEST3]
    file_chunks, test_files, subject_stats = build_loaders_chunked(
        CROSS_TRAIN, test_folders,
        files_per_chunk = FILES_PER_CHUNK, verbose = True)

    # 2. Building the test loader
    print("\n-- Preprocessing test files…")
    X_test, y_test = process_files(test_files, subject_stats = subject_stats, verbose = True)
    test_ds     = MEGDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE,
        shuffle = False, num_workers = NUM_WORKERS, pin_memory = True)

    # 3. Build validation loader
    print("\n-- Building validation set from all training chunks…")
    (val_loader, X_train_all, y_train_all) = build_val_loader_from_chunks(
        file_chunks, subject_stats, val_fraction = VAL_SPLIT)

    # 4. Build the model
    model = get_model(model_name)
    ckpt_dir = CHECKPOINTS_DIR / f"{model_name}_cross_best.pt"

    # 5. Train or load results
    if eval_only:
        print(f"\n-- Loading checkpoint: {ckpt_dir}")
        model.load_state_dict(torch.load(ckpt_dir, map_location = DEVICE))
        history, training_time = None, 0.0
    else:
        start_time = time.time()
        history = train_chunked(
            model, file_chunks, val_loader, subject_stats,
            model_name = model_name, epochs = EPOCHS, lr = LEARNING_RATE,
            weight_decay = WEIGHT_DECAY, patience = PATIENCE,
            mixup_alpha = MIXUP_ALPHA_CROSS)
        training_time = time.time() - start_time
        if history:
            plot_training_curves(history, model_name.upper(), "Cross")

    # 6. Evaluate on the test set
    model = model.to(DEVICE)
    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    plot_confusion_matrix(y_true, y_pred, model_name.upper(), "Cross")
    metrics = print_report(y_true, y_pred, model_name.upper(), "Cross")
    
    # 7. Store extra metrics
    metrics["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics["training_time"] = training_time
    if history is not None:
        metrics["last-epoch"] = history.get("last-epoch", [-1])[0]
    return metrics

# Run all models and plot comparison
def run_all(experiment: str, eval_only: bool = False
        ) -> dict[str, dict[str, dict[str, float]]]:
    """Train all models and produce comparison plots"""
    all_results: dict[str, dict[str, dict[str, float]]] = {}
 
    for m in ALL_MODELS:
        try:
            if experiment in ("intra", "both"):
                all_results.setdefault("Intra", {})[MODEL_DISPLAY[m]] = run_intra(m, eval_only)
            if experiment in ("cross", "both"):
                all_results.setdefault("Cross", {})[MODEL_DISPLAY[m]] = run_cross(m, eval_only)
        except Exception as exc:
            print(f"\n[SKIP] {m} raised: {exc}")
 
    if "Intra" in all_results:
        plot_comparison_bar(all_results["Intra"], "Intra")
    if "Cross" in all_results:
        plot_comparison_bar(all_results["Cross"], "Cross")
    if "Intra" in all_results and "Cross" in all_results:
        plot_intra_vs_cross(
            {m: v["accuracy"] for m, v in all_results["Intra"].items()},
            {m: v["accuracy"] for m, v in all_results["Cross"].items()})
 
    save_results_csv(all_results)
    return all_results

def main():
    args = parse_args()
    set_seed()

    print(f"Model      : {args.model}")
    print(f"Experiment : {args.experiment}")
    print(f"Eval only  : {args.eval_only}")
    print(f"Using cuda : {torch.cuda.is_available()}")

    # Always initialise so the CSV save below never hits UnboundLocalError
    intra_metrics: dict[str, float] = {}
    cross_metrics: dict[str, float] = {}
 
    if args.model == "all":
        run_all(args.experiment, args.eval_only)
    else:
        if args.experiment in ("intra", "both"):
            intra_metrics = run_intra(args.model, args.eval_only)
        if args.experiment in ("cross", "both"):
            cross_metrics = run_cross(args.model, args.eval_only)
 
        if args.experiment == "both":
            display = MODEL_DISPLAY.get(args.model, args.model)
            if display is None:
                display = args.model
            save_results_csv({
                "Intra": {display: intra_metrics},
                "Cross": {display: cross_metrics}})

if __name__ == "__main__":
    main()

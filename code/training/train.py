from ultralytics import YOLO
import wandb
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Configuration
DATA_YAML = "datasets/cricket_ball/data.yaml"
if len(sys.argv) > 1:
    DATA_YAML = sys.argv[1]

PROJECT_NAME = "cricket-ball-detection"
BASE_MODEL = "yolo11m.pt"
IMGSZ = 832 
EPOCHS = 100
BATCH = 8

RUN_NAME = f"{BASE_MODEL[:-3]}-imgsz{IMGSZ}-epochs{EPOCHS}"

def log_validation(model, epoch, results=None):
    """
    Runs validation and logs results to wandb.
    """
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        device="0",
    )

    metrics = {
        "val/box_map50": val_results.box.map50,
        "val/box_map50_95": val_results.box.map,
        "val/precision": val_results.box.mp,
        "val/recall": val_results.box.mr,
        "epoch": epoch,
        "train/box_map50": results['metrics/mAP50(B)'] if results else None,
        "train/box_map50_95": results['metrics/mAP50-95(B)'] if results else None,
        "train/precision": results['metrics/precision(B)'] if results else None,  
        "train/recall": results['metrics/recall(B)'] if results else None,
    }

    try:
        wandb.log(metrics)
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")
        print(f"Metrics: {metrics}")


def main():
    try:
        run = wandb.init(
            project=PROJECT_NAME,
            name=RUN_NAME,
            config={
                "model": BASE_MODEL,
                "imgsz": IMGSZ,
                "epochs": EPOCHS,
                "batch": BATCH,
                "optimizer": "AdamW",
                "dataset": DATA_YAML,
            },
        )
        print("WandB initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        print("Continuing without wandb logging...")

    # Load model
    model = YOLO(BASE_MODEL)
    
    print(f"Epoch 0/{EPOCHS} - Initial validation")
    log_validation(model, epoch=0)

    # Train one epoch at a time
    checkpoint_path = None
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Training Epoch {epoch}/{EPOCHS}")
        print(f"{'='*50}")
        
        if checkpoint_path is not None:
            print(f"Loading weights from: {checkpoint_path}")
            model = YOLO(checkpoint_path)
        
        results = model.train(
            data=DATA_YAML,
            imgsz=IMGSZ,
            epochs=1,
            batch=BATCH,
            project="runs/detect",
            name=f"{RUN_NAME}_epoch_{epoch}",
            device=[0,1],
        )

        checkpoint_path = model.trainer.last
        
        # Run validation and log manually
        print(f"\nEpoch {epoch}/{EPOCHS} - Running validation")
        log_validation(model, epoch=epoch, results=results.results_dict if results else None)
        
        print(f"Checkpoint saved: {checkpoint_path}")

    # 6. Finish the run
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"{'='*50}")
    
    try:
        wandb.finish()
    except Exception as e:
        print(f"Warning: Failed to finish wandb cleanly: {e}")

if __name__ == "__main__":
    main()
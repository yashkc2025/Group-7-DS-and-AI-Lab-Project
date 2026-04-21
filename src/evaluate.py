import os
import pandas as pd
from ultralytics import YOLO
import shutil

def run_evaluation(weights_path='outputs/weights/v11l_640_stage1/weights/best.pt'):
    # 1. Load the trained model
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights not found at {weights_path}")
        return

    model = YOLO(weights_path)
    
    # 2. Run Validation
    print(f"📊 Running comprehensive evaluation on: {weights_path}")
    metrics = model.val(data='data/data.yaml', split='val')
    
    # 3. Extract and Save Per-Class Metrics to CSV
    class_names = model.names
    results_data = []
    
    for i, name in class_names.items():
        results_data.append({
            'Class_ID': i,
            'Class_Name': name,
            'Precision': metrics.results_dict.get('metrics/precision(B)', 0),
            'Recall': metrics.results_dict.get('metrics/recall(B)', 0),
            'mAP50': metrics.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': metrics.results_dict.get('metrics/mAP50-95(B)', 0)
        })
    
    df = pd.DataFrame(results_data)
    csv_out = 'outputs/csv/final_evaluation_metrics.csv'
    df.to_csv(csv_out, index=False)
    print(f"✅ CSV metrics saved to: {csv_out}")

    # 4. Move Visual Plots to /outputs/plots
    # YOLO saves plots in the run directory; we move them to our structured folder
    run_dir = metrics.save_dir
    visuals = ['confusion_matrix.png', 'F1_curve.png', 'PR_curve.png', 'results.png']
    
    for vis in visuals:
        src = os.path.join(run_dir, vis)
        if os.path.exists(src):
            shutil.copy(src, f'outputs/plots/{vis}')
    
    print("✅ Visual plots (Confusion Matrix, PR Curves) moved to 'outputs/plots/'.")

if __name__ == "__main__":
    run_evaluation()

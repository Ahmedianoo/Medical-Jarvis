import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
from datetime import datetime

from preprocess import preprocess_img
from segment import wbc
from wbc_features import WBCClassifier


class Logger:
    """
    Logger class that writes output to both console and file simultaneously.
    """
    def __init__(self, log_file_path: str):
        """
        Initialize logger with file path.
        
        Args:
            log_file_path: Path where log file will be saved
        """
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        """Write message to both terminal and file."""
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        """Close the log file."""
        self.log_file.close()


class WBCEvaluator:
    """
    Evaluator for WBC classification performance.
    
    This class compares classifier predictions against ground truth labels,
    computes accuracy metrics, generates confusion matrices, and analyzes
    feature distributions across different cell types.
    """
    
    def __init__(self, labels_csv_path: str, images_dir: str):
        """
        Initialize the evaluator with ground truth labels.
        
        Args:
            labels_csv_path: Path to CSV file containing ground truth labels
            images_dir: Directory containing blood smear images
        """
        self.labels_df = pd.read_csv(labels_csv_path)
        self.images_dir = Path(images_dir)
        self.classifier = WBCClassifier()
        
        self.labels_df.columns = self.labels_df.columns.str.strip()
        
        self.results = []
        self.confusion_data = []
        self.feature_analysis = defaultdict(list)
        
        print(f"Loaded {len(self.labels_df)} labels from CSV")
        print(f"Columns: {list(self.labels_df.columns)}")
        print(f"Sample data:\n{self.labels_df.head()}")
    
    def get_image_path(self, image_number: int) -> Path:
        """Construct the file path for a given image number."""
        return self.images_dir / f"BloodImage_{image_number:05d}.jpg"
    
    def evaluate_single_image(self, image_number: int, ground_truth: str) -> Dict:
        """
        Evaluate classifier performance on a single image.
        
        This method loads an image, runs the classifier, and compares
        the predicted label against the ground truth.
        
        Args:
            image_number: Numeric identifier for the image
            ground_truth: Expected cell type label
            
        Returns:
            Dictionary containing evaluation results including:
                - image_number: Image identifier
                - ground_truth: Expected label
                - predicted: Classifier's prediction
                - status: 'correct', 'incorrect', or error message
                - num_wbcs_detected: Number of cells found
        """
        img_path = self.get_image_path(image_number)
        
        if not img_path.exists():
            return {
                'image_number': image_number,
                'ground_truth': ground_truth,
                'predicted': None,
                'status': 'image_not_found',
                'num_wbcs_detected': 0
            }
        
        if pd.isna(ground_truth):
            return {
                'image_number': image_number,
                'ground_truth': None,
                'predicted': None,
                'status': 'no_ground_truth',
                'num_wbcs_detected': 0
            }
        
        ground_truth = str(ground_truth).strip().upper()
        
        # Handle images with multiple cells by selecting the first label
        if ',' in ground_truth:
            ground_truth_list = [gt.strip() for gt in ground_truth.split(',')]
            ground_truth = ground_truth_list[0]
            has_multiple_gt = True
        else:
            has_multiple_gt = False
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return {
                    'image_number': image_number,
                    'ground_truth': ground_truth,
                    'predicted': None,
                    'status': 'image_load_failed',
                    'num_wbcs_detected': 0
                }
            
            pre = preprocess_img(img)
            wbc_mask = wbc(pre)
            classifications = self.classifier.classify_all_wbcs(pre, wbc_mask)
            
            num_detected = len(classifications)
            
            if num_detected == 0:
                return {
                    'image_number': image_number,
                    'ground_truth': ground_truth,
                    'predicted': None,
                    'status': 'no_wbc_detected',
                    'num_wbcs_detected': 0
                }
            
            # Use the first detected cell for evaluation
            predicted_type = classifications[0]['type'].upper()
            
            nf = classifications[0]['nucleus_features']
            cf = classifications[0]['cytoplasm_features']
            
            # Store features for later analysis
            if nf and cf:
                self.feature_analysis[ground_truth].append({
                    'circularity': nf['circularity'],
                    'solidity': nf['solidity'],
                    'num_lobes': nf['num_lobes'],
                    'cn_ratio': cf['cn_ratio'],
                    'mean_hue': cf['mean_hue'],
                    'texture_variance': cf['texture_variance'],
                    'predicted': predicted_type
                })
            
            is_correct = (predicted_type == ground_truth)
            
            return {
                'image_number': image_number,
                'ground_truth': ground_truth,
                'predicted': predicted_type,
                'status': 'correct' if is_correct else 'incorrect',
                'num_wbcs_detected': num_detected,
                'has_multiple_gt': has_multiple_gt,
                'nucleus_features': nf,
                'cytoplasm_features': cf
            }
            
        except Exception as e:
            print(f"Error processing image {image_number}: {str(e)}")
            return {
                'image_number': image_number,
                'ground_truth': ground_truth,
                'predicted': None,
                'status': f'error: {str(e)}',
                'num_wbcs_detected': 0
            }
    
    def evaluate_all(self):
        """
        Run evaluation on all images in the dataset.
        
        Iterates through the label CSV, classifies each image,
        and stores results for subsequent analysis.
        """
        print("\nEvaluating all images...")
        print("="*60)
        
        total = len(self.labels_df)
        
        for idx, row in self.labels_df.iterrows():
            image_num = int(row['Image'])
            category = row['Category']
            
            result = self.evaluate_single_image(image_num, category)
            self.results.append(result)
            
            if result['status'] in ['correct', 'incorrect']:
                self.confusion_data.append({
                    'true': result['ground_truth'],
                    'pred': result['predicted']
                })
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{total} images...")
        
        print(f"\nCompleted processing {total} images.")
    
    def calculate_metrics(self) -> Dict:
        """
        Compute classification accuracy metrics.
        
        Returns:
            Dictionary containing:
                - overall_accuracy: Overall percentage correct
                - class_accuracy: Per-class accuracy breakdown
                - status_counts: Count of each evaluation status
        """
        valid_results = [r for r in self.results if r['status'] in ['correct', 'incorrect']]
        
        if not valid_results:
            print("No valid predictions to evaluate!")
            return {}
        
        total_valid = len(valid_results)
        correct = len([r for r in valid_results if r['status'] == 'correct'])
        
        accuracy = (correct / total_valid) * 100 if total_valid > 0 else 0
        
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for r in valid_results:
            gt = r['ground_truth']
            class_total[gt] += 1
            if r['status'] == 'correct':
                class_correct[gt] += 1
        
        class_accuracy = {
            cls: (class_correct[cls] / class_total[cls] * 100) 
            for cls in class_total.keys()
        }
        
        status_counts = Counter([r['status'] for r in self.results])
        
        metrics = {
            'overall_accuracy': accuracy,
            'total_images': len(self.results),
            'valid_predictions': total_valid,
            'correct_predictions': correct,
            'class_accuracy': class_accuracy,
            'status_counts': dict(status_counts)
        }
        
        return metrics
    
    def print_report(self):
        """
        Display a formatted evaluation report to the console.
        
        Prints overall accuracy, per-class accuracy, and processing statistics.
        """
        metrics = self.calculate_metrics()
        
        if not metrics:
            return
        
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.2f}%")
        print(f"Correct: {metrics['correct_predictions']}/{metrics['valid_predictions']}")
        
        print("\n" + "-"*60)
        print("Per-Class Accuracy:")
        print("-"*60)
        for cls, acc in sorted(metrics['class_accuracy'].items()):
            print(f"  {cls:20s}: {acc:6.2f}%")
        
        print("\n" + "-"*60)
        print("Processing Status:")
        print("-"*60)
        for status, count in sorted(metrics['status_counts'].items()):
            print(f"  {status:25s}: {count:4d}")
        
        print("\n" + "="*60)
    
    def analyze_misclassifications(self) -> pd.DataFrame:
        """
        Identify and summarize misclassification patterns.
        
        Groups errors by (ground_truth → predicted) pairs to reveal
        which cell types are most commonly confused.
        
        Returns:
            DataFrame containing all misclassified samples
        """
        misclassified = [r for r in self.results if r['status'] == 'incorrect']
        
        if not misclassified:
            print("\nNo misclassifications found!")
            return pd.DataFrame()
        
        print("\n" + "="*60)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*60)
        
        error_patterns = defaultdict(list)
        for r in misclassified:
            key = f"{r['ground_truth']} → {r['predicted']}"
            error_patterns[key].append(r['image_number'])
        
        print("\nCommon Misclassification Patterns:")
        print("-"*60)
        for pattern, images in sorted(error_patterns.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {pattern:30s}: {len(images):3d} cases")
            if len(images) <= 5:
                print(f"    Images: {images}")
        
        return pd.DataFrame(misclassified)
    
    def suggest_threshold_adjustments(self):
        """
        Analyze feature distributions to suggest threshold improvements.
        
        For each cell type, computes mean, standard deviation, and range
        of all features. Identifies which cell types are frequently confused
        to guide threshold refinement.
        """
        print("\n" + "="*60)
        print("THRESHOLD ADJUSTMENT SUGGESTIONS")
        print("="*60)
        
        for cell_type, features_list in self.feature_analysis.items():
            if not features_list:
                continue
            
            print(f"\n{cell_type}:")
            print("-" * 40)
            
            df = pd.DataFrame(features_list)
            
            print(f"  Sample count: {len(df)}")
            print(f"\n  Feature Statistics:")
            print(f"    Circularity:      {df['circularity'].mean():.3f} ± {df['circularity'].std():.3f} (range: {df['circularity'].min():.3f} - {df['circularity'].max():.3f})")
            print(f"    Solidity:         {df['solidity'].mean():.3f} ± {df['solidity'].std():.3f} (range: {df['solidity'].min():.3f} - {df['solidity'].max():.3f})")
            print(f"    Num Lobes:        {df['num_lobes'].mean():.1f} ± {df['num_lobes'].std():.1f} (range: {df['num_lobes'].min():.0f} - {df['num_lobes'].max():.0f})")
            print(f"    CN Ratio:         {df['cn_ratio'].mean():.3f} ± {df['cn_ratio'].std():.3f} (range: {df['cn_ratio'].min():.3f} - {df['cn_ratio'].max():.3f})")
            print(f"    Mean Hue:         {df['mean_hue'].mean():.1f} ± {df['mean_hue'].std():.1f} (range: {df['mean_hue'].min():.1f} - {df['mean_hue'].max():.1f})")
            print(f"    Texture Variance: {df['texture_variance'].mean():.1f} ± {df['texture_variance'].std():.1f}")
            
            correct_count = len(df[df['predicted'] == cell_type])
            accuracy = (correct_count / len(df)) * 100
            print(f"\n  Classification accuracy for this type: {accuracy:.1f}% ({correct_count}/{len(df)})")
            
            if accuracy < 100:
                misclassified_as = df[df['predicted'] != cell_type]['predicted'].value_counts()
                print(f"  Often misclassified as: {dict(misclassified_as)}")
    
    def plot_confusion_matrix(self, save_path="../results/wbc_features/evaluation_results/confusion_matrix.png"):
        """
        Generate and save a confusion matrix visualization.
        
        The confusion matrix shows how often each true label was
        predicted as each possible label, helping identify systematic errors.
        
        Args:
            save_path: Output file path for the confusion matrix image
        """
        if not self.confusion_data:
            print("No confusion matrix data available.")
            return
        
        all_classes = sorted(set([d['true'] for d in self.confusion_data] + 
                                [d['pred'] for d in self.confusion_data]))
        
        n_classes = len(all_classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        
        for d in self.confusion_data:
            true_idx = class_to_idx[d['true']]
            pred_idx = class_to_idx[d['pred']]
            cm[true_idx, pred_idx] += 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(all_classes)
        ax.set_yticklabels(all_classes)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax.text(j, i, str(cm[i, j]),
                             ha="center", va="center", 
                             color="black" if cm[i, j] < cm.max()/2 else "white",
                             fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix - WBC Classification', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
        plt.close()
    
    def plot_feature_distributions(self, save_path="../results/wbc_features/evaluation_results/feature_distributions.png"):
        """
        Visualize feature distributions across cell types.
        
        Creates histograms showing how each feature varies across
        different cell types. Overlapping distributions indicate
        features that may need threshold adjustment.
        
        Args:
            save_path: Output file path for the distribution plots
        """
        if not self.feature_analysis:
            print("No feature analysis data available.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        features = ['circularity', 'solidity', 'num_lobes', 'cn_ratio', 'mean_hue', 'texture_variance']
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            for cell_type, features_list in self.feature_analysis.items():
                if features_list:
                    values = [f[feature] for f in features_list]
                    ax.hist(values, alpha=0.5, label=cell_type, bins=20)
            
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.set_title(f'{feature.replace("_", " ").title()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature distributions saved to {save_path}")
        plt.close()
    
    def export_results(self, save_path="../results/wbc_features/evaluation_results/wbc_table.csv"):
        """
        Export detailed results to a CSV file for further analysis.
        
        Args:
            save_path: Output file path for the results CSV
        """
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(save_path, index=False)
        print(f"\nDetailed results exported to {save_path}")


def main():
    """
    Main entry point for the evaluation pipeline.
    
    Loads ground truth labels, runs the classifier on all images,
    computes metrics, generates visualizations, and exports results.
    """
    # Setup paths
    LABELS_CSV = "../results/wbc_features/labels.csv"
    IMAGES_DIR = "../data/input/JPEGImages"
    LOG_FILE = f"../results/wbc_features/evaluation_results/evaluation_results_log.txt"
    
    # Create output directory if it doesn't exist
    Path("../results/wbc_features/evaluation_results").mkdir(parents=True, exist_ok=True)
    
    # Setup logger to write to both console and file
    logger = Logger(LOG_FILE)
    sys.stdout = logger
    
    try:
        print(f"WBC Classification Evaluation")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        if not Path(LABELS_CSV).exists():
            print(f"Error: CSV file not found at {LABELS_CSV}")
            print("Please update the LABELS_CSV path in the script.")
            return
        
        if not Path(IMAGES_DIR).exists():
            print(f"Error: Images directory not found at {IMAGES_DIR}")
            print("Please update the IMAGES_DIR path in the script.")
            return
        
        evaluator = WBCEvaluator(LABELS_CSV, IMAGES_DIR)
        
        evaluator.evaluate_all()
        evaluator.print_report()
        evaluator.analyze_misclassifications()
        evaluator.suggest_threshold_adjustments()
        evaluator.plot_confusion_matrix("../results/wbc_features/evaluation_results/confusion_matrix.png")
        evaluator.plot_feature_distributions("../results/wbc_features/evaluation_results/feature_distributions.png")
        evaluator.export_results("../results/wbc_features/evaluation_results/wbc_table.csv")
        
        print("\n" + "="*60)
        print("Evaluation complete!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file saved to: {LOG_FILE}")
        print("="*60)
        
    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()
        print(f"\nAll output has been saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
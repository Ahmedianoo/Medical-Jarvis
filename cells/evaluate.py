import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
from datetime import datetime

from cells.preprocess import preprocess_img
from cells.segment import wbc
from wbc_features import WBCClassifier


class Logger:
    """Logger class that writes output to both console and file simultaneously."""
    def __init__(self, log_file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


class WBCEvaluator:
    """
    Evaluator for WBC classification performance.
    
    ENHANCED: Now calculates Precision, Recall, F1-Score, and Support for each class.
    """
    
    def __init__(self, labels_csv_path: str, images_dir: str):
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
        """Evaluate classifier performance on a single image."""
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
        """Run evaluation on all images in the dataset."""
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
    
    def calculate_per_class_metrics(self) -> Dict:
        """
        Calculate Precision, Recall, F1-Score, and Support for each class.
        
        Returns:
            Dictionary with per-class metrics
        """
        valid_results = [r for r in self.results if r['status'] in ['correct', 'incorrect']]
        
        if not valid_results:
            return {}
        
        # Get all unique classes
        all_classes = sorted(set([r['ground_truth'] for r in valid_results] + 
                                [r['predicted'] for r in valid_results]))
        
        class_metrics = {}
        
        for cls in all_classes:
            # True Positives: Correctly predicted as this class
            tp = len([r for r in valid_results 
                     if r['ground_truth'] == cls and r['predicted'] == cls])
            
            # False Positives: Incorrectly predicted as this class
            fp = len([r for r in valid_results 
                     if r['ground_truth'] != cls and r['predicted'] == cls])
            
            # False Negatives: This class but predicted as something else
            fn = len([r for r in valid_results 
                     if r['ground_truth'] == cls and r['predicted'] != cls])
            
            # True Negatives: Neither ground truth nor predicted as this class
            tn = len([r for r in valid_results 
                     if r['ground_truth'] != cls and r['predicted'] != cls])
            
            # Support: Total actual instances of this class
            support = tp + fn
            
            # Precision: Of all predicted as this class, how many were correct?
            precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
            
            # Recall (Sensitivity): Of all actual instances, how many did we find?
            recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
            
            # F1-Score: Harmonic mean of Precision and Recall
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            class_metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
        
        return class_metrics
    
    def calculate_metrics(self) -> Dict:
        """
        Compute all classification metrics including per-class and overall.
        
        Returns:
            Dictionary containing comprehensive metrics
        """
        valid_results = [r for r in self.results if r['status'] in ['correct', 'incorrect']]
        
        if not valid_results:
            print("No valid predictions to evaluate!")
            return {}
        
        total_valid = len(valid_results)
        correct = len([r for r in valid_results if r['status'] == 'correct'])
        
        overall_accuracy = (correct / total_valid) * 100 if total_valid > 0 else 0
        
        # Calculate per-class metrics
        class_metrics = self.calculate_per_class_metrics()
        
        # Calculate macro-averaged metrics (average across classes)
        if class_metrics:
            macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
            macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
            macro_f1 = np.mean([m['f1_score'] for m in class_metrics.values()])
        else:
            macro_precision = macro_recall = macro_f1 = 0.0
        
        # Calculate weighted-averaged metrics (weighted by support)
        total_support = sum([m['support'] for m in class_metrics.values()])
        if total_support > 0:
            weighted_precision = sum([m['precision'] * m['support'] for m in class_metrics.values()]) / total_support
            weighted_recall = sum([m['recall'] * m['support'] for m in class_metrics.values()]) / total_support
            weighted_f1 = sum([m['f1_score'] * m['support'] for m in class_metrics.values()]) / total_support
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0
        
        status_counts = Counter([r['status'] for r in self.results])
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'total_images': len(self.results),
            'valid_predictions': total_valid,
            'correct_predictions': correct,
            'class_metrics': class_metrics,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'status_counts': dict(status_counts)
        }
        
        return metrics
    
    def print_report(self):
        """Display a comprehensive evaluation report with all metrics."""
        metrics = self.calculate_metrics()
        
        if not metrics:
            return
        
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL METRICS")
        print("-"*80)
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        print(f"Correct: {metrics['correct_predictions']}/{metrics['valid_predictions']}")
        
        print(f"\n📈 MACRO-AVERAGED METRICS (Unweighted Average Across Classes)")
        print("-"*80)
        print(f"Macro Precision:  {metrics['macro_precision']:.2f}%")
        print(f"Macro Recall:     {metrics['macro_recall']:.2f}%")
        print(f"Macro F1-Score:   {metrics['macro_f1']:.2f}%")
        
        print(f"\n⚖️  WEIGHTED-AVERAGED METRICS (Weighted by Support)")
        print("-"*80)
        print(f"Weighted Precision:  {metrics['weighted_precision']:.2f}%")
        print(f"Weighted Recall:     {metrics['weighted_recall']:.2f}%")
        print(f"Weighted F1-Score:   {metrics['weighted_f1']:.2f}%")
        
        print(f"\n🎯 PER-CLASS DETAILED METRICS")
        print("-"*80)
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-"*80)
        
        for cls in sorted(metrics['class_metrics'].keys()):
            m = metrics['class_metrics'][cls]
            print(f"{cls:<15} {m['precision']:>9.2f}% {m['recall']:>9.2f}% "
                  f"{m['f1_score']:>9.2f}% {m['support']:>10d}")
        
        print("\n" + "="*80)
        print("METRIC DEFINITIONS:")
        print("-"*80)
        print("• Precision:  Of all predicted as this class, % that were correct")
        print("              (Answers: 'When we say X, are we usually right?')")
        print("• Recall:     Of all actual instances of this class, % we found")
        print("              (Answers: 'Do we catch most X cases?')")
        print("• F1-Score:   Harmonic mean of Precision and Recall")
        print("              (Balanced measure of both)")
        print("• Support:    Total number of actual instances in dataset")
        print("="*80)
        
        print("\n📋 Processing Status:")
        print("-"*80)
        for status, count in sorted(metrics['status_counts'].items()):
            print(f"  {status:25s}: {count:4d}")
        
        print("\n" + "="*80)
    
    def export_detailed_metrics(self, save_path="../results/wbc_features/evaluation_results/detailed_metrics.csv"):
        """
        Export detailed per-class metrics to CSV.
        
        Args:
            save_path: Output file path for the metrics CSV
        """
        metrics = self.calculate_metrics()
        
        if not metrics or not metrics['class_metrics']:
            print("No metrics to export.")
            return
        
        # Create DataFrame from class metrics
        metrics_data = []
        for cls, m in sorted(metrics['class_metrics'].items()):
            metrics_data.append({
                'Class': cls,
                'Precision (%)': round(m['precision'], 2),
                'Recall (%)': round(m['recall'], 2),
                'F1-Score (%)': round(m['f1_score'], 2),
                'Support': m['support'],
                'True Positives': m['tp'],
                'False Positives': m['fp'],
                'False Negatives': m['fn'],
                'True Negatives': m['tn']
            })
        
        # Add overall metrics row
        metrics_data.append({
            'Class': 'MACRO AVERAGE',
            'Precision (%)': round(metrics['macro_precision'], 2),
            'Recall (%)': round(metrics['macro_recall'], 2),
            'F1-Score (%)': round(metrics['macro_f1'], 2),
            'Support': sum([m['support'] for m in metrics['class_metrics'].values()]),
            'True Positives': '',
            'False Positives': '',
            'False Negatives': '',
            'True Negatives': ''
        })
        
        metrics_data.append({
            'Class': 'WEIGHTED AVERAGE',
            'Precision (%)': round(metrics['weighted_precision'], 2),
            'Recall (%)': round(metrics['weighted_recall'], 2),
            'F1-Score (%)': round(metrics['weighted_f1'], 2),
            'Support': sum([m['support'] for m in metrics['class_metrics'].values()]),
            'True Positives': '',
            'False Positives': '',
            'False Negatives': '',
            'True Negatives': ''
        })
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(save_path, index=False)
        print(f"\nDetailed metrics exported to {save_path}")
    
    def analyze_misclassifications(self) -> pd.DataFrame:
        """Identify and summarize misclassification patterns."""
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
        """Analyze feature distributions to suggest threshold improvements."""
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
        """Generate and save a confusion matrix visualization."""
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
        """Visualize feature distributions across cell types."""
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
        """Export detailed results to a CSV file for further analysis."""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(save_path, index=False)
        print(f"\nDetailed results exported to {save_path}")


def main():
    """Main entry point for the evaluation pipeline."""
    LABELS_CSV = "../results/wbc_features/labels.csv"
    IMAGES_DIR = "../data/input/JPEGImages"
    LOG_FILE = f"../results/wbc_features/evaluation_results/evaluation_results_log.txt"
    
    Path("../results/wbc_features/evaluation_results").mkdir(parents=True, exist_ok=True)
    
    logger = Logger(LOG_FILE)
    sys.stdout = logger
    
    try:
        print(f"WBC Classification Evaluation (with Precision, Recall, F1)")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        if not Path(LABELS_CSV).exists():
            print(f"Error: CSV file not found at {LABELS_CSV}")
            return
        
        if not Path(IMAGES_DIR).exists():
            print(f"Error: Images directory not found at {IMAGES_DIR}")
            return
        
        evaluator = WBCEvaluator(LABELS_CSV, IMAGES_DIR)
        
        evaluator.evaluate_all()
        evaluator.print_report()
        evaluator.analyze_misclassifications()
        evaluator.suggest_threshold_adjustments()
        evaluator.plot_confusion_matrix("../results/wbc_features/evaluation_results/confusion_matrix.png")
        evaluator.plot_feature_distributions("../results/wbc_features/evaluation_results/feature_distributions.png")
        evaluator.export_results("../results/wbc_features/evaluation_results/wbc_table.csv")
        evaluator.export_detailed_metrics("../results/wbc_features/evaluation_results/detailed_metrics.csv")
        
        print("\n" + "="*60)
        print("Evaluation complete!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file saved to: {LOG_FILE}")
        print("="*60)
        
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\nAll output has been saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
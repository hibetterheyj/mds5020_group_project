import json
import pandas as pd
from typing import Dict, List, Any


def generate_partial_summary():
    """Generate a partial summary using completed results"""
    # List of completed result files
    completed_configs = [
        {'name': 'linear_svc', 'display_name': 'LinearSVC'},
        {'name': 'svc_rbf', 'display_name': 'SVC with RBF Kernel'},
        {'name': 'svc_sigmoid', 'display_name': 'SVC with Sigmoid Kernel'}
    ]

    # Load results for completed classifiers
    all_results = {}
    for config in completed_configs:
        try:
            with open(f"{config['name']}_tuning_results.json", 'r') as f:
                results = json.load(f)
            all_results[config['name']] = results
            print(f"Loaded results for {config['display_name']}")
        except FileNotFoundError:
            print(f"Results file not found for {config['display_name']}")

    if not all_results:
        print("No results files found!")
        return

    # Generate markdown content
    markdown_content = "# SVC Hyperparameter Tuning - Partial Summary\n\n"
    markdown_content += "**Note:** Poly kernel tuning is still in progress.\n\n"
    
    # Experiment setup
    markdown_content += "## Experiment Setup\n\n"
    markdown_content += "- **Baseline**: Logistic Regression with TF-IDF features (0.8024 F1 score)\n"
    markdown_content += "- **Tuning Method**: RandomizedSearchCV with 200 iterations per configuration\n"
    markdown_content += "- **Cross-Validation**: 5-fold StratifiedKFold\n"
    markdown_content += "- **Evaluation Metric**: Weighted F1 Score\n\n"
    
    # Key parameters tuned
    markdown_content += "## Key Parameters Tuned\n\n"
    markdown_content += "1. **TF-IDF Parameters**:\n"
    markdown_content += "   - `max_features`: Number of top features to keep\n"
    markdown_content += "   - `ngram_range`: Range of n-grams to consider\n\n"
    markdown_content += "2. **SVC Parameters**:\n"
    markdown_content += "   - `C`: Regularization parameter\n"
    markdown_content += "   - `class_weight`: Class weighting strategy\n"
    markdown_content += "   - `kernel`: Kernel type (for non-linear SVC)\n"
    markdown_content += "   - `gamma`: Kernel coefficient\n"
    markdown_content += "   - `coef0`: Independent term in kernel function\n\n"
    markdown_content += "3. **Feature Configuration**:\n"
    markdown_content += "   - `use_handcrafted`: Whether to include handcrafted features\n\n"
    
    # Best results by classifier
    markdown_content += "## Best Results by Classifier\n\n"
    
    best_overall = None
    best_score = 0.0
    best_config = ""
    
    for config_name, results in all_results.items():
        best_result = max(results, key=lambda x: x['metrics']['mean_score'])
        
        display_name = next(c['display_name'] for c in completed_configs if c['name'] == config_name)
        
        markdown_content += f"### {display_name}\n"
        markdown_content += f"- **Best F1 Score**: {best_result['metrics']['mean_score']:.4f}\n"
        markdown_content += f"- **Standard Deviation**: {best_result['metrics']['std_score']:.4f}\n"
        markdown_content += f"- **Use Handcrafted Features**: {best_result['parameters']['use_handcrafted']}\n"
        markdown_content += f"- **Best Parameters**:\n"
        
        for param, value in best_result['parameters'].items():
            if param not in ['svc_type', 'kernel']:
                # Format parameter name for readability
                param_name = param.replace('classifier__', '').replace('features__text__', '').replace('text__', '')
                markdown_content += f"  - `{param_name}`: {value}\n"
        
        markdown_content += "\n"
        
        # Track overall best
        if best_result['metrics']['mean_score'] > best_score:
            best_score = best_result['metrics']['mean_score']
            best_overall = best_result
            best_config = display_name
    
    # Comparison across classifiers
    markdown_content += "## Cross-Classifier Comparison\n\n"
    
    for config_name, results in all_results.items():
        best_result = max(results, key=lambda x: x['metrics']['mean_score'])
        score = best_result['metrics']['mean_score']
        display_name = next(c['display_name'] for c in completed_configs if c['name'] == config_name)
        
        markdown_content += f"- **{display_name}**: {score:.4f}\n"
    
    markdown_content += f"\n### Best Overall Model (So Far)\n"
    markdown_content += f"- **Classifier**: {best_config}\n"
    markdown_content += f"- **F1 Score**: {best_score:.4f}\n"
    markdown_content += f"- **Parameters**:\n"
    
    for param, value in best_overall['parameters'].items():
        if param not in ['svc_type', 'kernel']:
            param_name = param.replace('classifier__', '').replace('features__text__', '').replace('text__', '')
            markdown_content += f"  - `{param_name}`: {value}\n"
    
    # Impact of handcrafted features
    markdown_content += "\n## Impact of Handcrafted Features\n\n"
    
    for config_name, results in all_results.items():
        handcrafted_results = [r for r in results if r['parameters']['use_handcrafted']]
        non_handcrafted_results = [r for r in results if not r['parameters']['use_handcrafted']]
        
        if handcrafted_results and non_handcrafted_results:
            best_handcrafted = max(handcrafted_results, key=lambda x: x['metrics']['mean_score'])
            best_non_handcrafted = max(non_handcrafted_results, key=lambda x: x['metrics']['mean_score'])
            
            diff = best_handcrafted['metrics']['mean_score'] - best_non_handcrafted['metrics']['mean_score']
            display_name = next(c['display_name'] for c in completed_configs if c['name'] == config_name)
            
            markdown_content += f"- **{display_name}**:\n"
            markdown_content += f"  - With handcrafted features: {best_handcrafted['metrics']['mean_score']:.4f}\n"
            markdown_content += f"  - Without handcrafted features: {best_non_handcrafted['metrics']['mean_score']:.4f}\n"
            markdown_content += f"  - Difference: {diff:+.4f}\n"
    
    # Save markdown summary
    with open('svc_tuning_partial_summary.md', 'w') as f:
        f.write(markdown_content)
    
    print("\nPartial summary generated: svc_tuning_partial_summary.md")
    print(f"Best overall score so far: {best_score:.4f} ({best_config})")


if __name__ == "__main__":
    generate_partial_summary()
"""
MLflow utilities for tracking 1000 LoRA models
Includes dashboard generation and model registry management
"""

import mlflow
import pandas as pd
from pathlib import Path
import json

class MLFlowManager:
    """Manage MLflow experiments, models, and registry"""
    
    def __init__(self, tracking_uri="file:./mlruns", experiment_name="LoRA-1000-Datasets"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def get_experiment_stats(self):
        """Get statistics from all runs"""
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        
        if not experiment:
            return None
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        stats = {
            'total_runs': len(runs),
            'completed_runs': sum(1 for r in runs if r.status == 'FINISHED'),
            'failed_runs': sum(1 for r in runs if r.status == 'FAILED'),
            'avg_perplexity': None,
            'avg_training_time': None,
            'best_model': None
        }
        
        if runs:
            metrics = [r.data.metrics for r in runs if r.status == 'FINISHED']
            if metrics:
                perplexities = [m.get('eval_perplexity', 0) for m in metrics]
                times = [m.get('training_time_minutes', 0) for m in metrics]
                
                stats['avg_perplexity'] = sum(perplexities) / len(perplexities) if perplexities else 0
                stats['avg_training_time'] = sum(times) / len(times) if times else 0
                
                # Find best model (lowest perplexity)
                best_idx = perplexities.index(min(perplexities))
                stats['best_model'] = runs[best_idx].data.params.get('dataset_name', 'Unknown')
        
        return stats
    
    def export_results_to_csv(self, output_path='mlflow_results.csv'):
        """Export all run results to CSV"""
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        
        if not experiment:
            print("No experiment found")
            return
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        results = []
        for run in runs:
            result = {
                'run_id': run.info.run_id,
                'dataset_name': run.data.params.get('dataset_name', 'Unknown'),
                'domain': run.data.params.get('domain', 'Unknown'),
                'style': run.data.params.get('style', 'Unknown'),
                'status': run.info.status,
                'perplexity': run.data.metrics.get('eval_perplexity', 0),
                'training_time_minutes': run.data.metrics.get('training_time_minutes', 0),
                'loss': run.data.metrics.get('eval_loss', 0),
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"✅ Exported {len(df)} results to {output_path}")
        return df
    
    def generate_report(self):
        """Generate human-readable report"""
        stats = self.get_experiment_stats()
        
        if not stats:
            print("No experiment data found")
            return
        
        report = f"""
{'='*60}
LoRA TRAINING REPORT
{'='*60}
Total Runs:           {stats['total_runs']}
Completed:            {stats['completed_runs']}
Failed:               {stats['failed_runs']}
Success Rate:         {stats['completed_runs']/max(stats['total_runs'], 1)*100:.1f}%

Performance Metrics:
├─ Avg Perplexity:    {stats['avg_perplexity']:.2f}
├─ Avg Training Time:  {stats['avg_training_time']:.2f} minutes
└─ Best Model:        {stats['best_model']}

MLflow UI:            {self.tracking_uri}
{'='*60}
        """
        print(report)
        return report
    
    def get_top_models(self, n=10, metric='eval_perplexity'):
        """Get top N models by metric"""
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        
        if not experiment:
            return []
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        # Sort by metric
        sorted_runs = sorted(
            runs,
            key=lambda r: r.data.metrics.get(metric, float('inf'))
        )
        
        top_models = []
        for run in sorted_runs[:n]:
            top_models.append({
                'dataset_name': run.data.params.get('dataset_name'),
                'perplexity': run.data.metrics.get('eval_perplexity'),
                'training_time': run.data.metrics.get('training_time_minutes'),
                'run_id': run.info.run_id
            })
        
        return top_models
    
    def register_best_models(self, top_n=10):
        """Register top N models to MLflow Model Registry"""
        top_models = self.get_top_models(top_n)
        
        for i, model in enumerate(top_models, 1):
            try:
                # Register model
                model_uri = f"runs:/{model['run_id']}/lora_adapter"
                mlflow.register_model(
                    model_uri,
                    f"lora_top_{i}_{model['dataset_name']}"
                )
                print(f"✅ Registered: {model['dataset_name']}")
            except Exception as e:
                print(f"⚠️ Failed to register {model['dataset_name']}: {e}")

def main():
    """Example usage"""
    manager = MLFlowManager()
    
    # Generate report
    manager.generate_report()
    
    # Export results
    manager.export_results_to_csv()
    
    # Get top models
    top_models = manager.get_top_models(n=5)
    print("\n🏆 Top 5 Models by Perplexity:")
    for i, model in enumerate(top_models, 1):
        print(f"{i}. {model['dataset_name']}: {model['perplexity']:.2f}")

if __name__ == '__main__':
    main()

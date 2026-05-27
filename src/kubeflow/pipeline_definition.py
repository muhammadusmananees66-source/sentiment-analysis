# src/kubeflow/pipeline_definition.py (COMPLETE WORKING VERSION)
import kfp
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, Artifact
from kfp.dsl import component
from typing import NamedTuple

# ============================================
# COMPONENT 1: Load Data from Hugging Face
# ============================================
@component(
    base_image="python:3.11",
    packages_to_install=[
        "datasets>=3.0.0",
        "transformers>=4.46.0",
        "torch>=2.5.0",
        "pandas>=2.2.3"
    ]
)
def load_data(
    dataset_name: str,
    split: str,
    output_dataset: Output[Dataset]
) -> str:
    """Load dataset from Hugging Face Hub"""
    import json
    from datasets import load_dataset
    import pandas as pd
    
    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)
    
    df = dataset.to_pandas()
    
    csv_path = f"{output_dataset.path}/data.csv"
    df.to_csv(csv_path, index=False)
    
    metadata = {
        "dataset_name": dataset_name,
        "split": split,
        "num_samples": len(df),
        "columns": list(df.columns)
    }
    with open(f"{output_dataset.path}/metadata.json", "w") as f:
        json.dump(metadata, f)
    
    print(f"✅ Data loaded: {len(df)} samples")
    return csv_path


# ============================================
# COMPONENT 2: Preprocess Data
# ============================================
@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas>=2.2.3",
        "numpy>=2.0.0",
        "transformers>=4.46.0",
        "torch>=2.5.0"
    ]
)
def preprocess_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    model_name: str = "distilbert-base-uncased"
) -> NamedTuple('Outputs', [
    ('num_samples', int),
    ('num_classes', int)
]):
    """Preprocess text data for sentiment analysis"""
    import pandas as pd
    from transformers import AutoTokenizer
    from collections import namedtuple
    
    df = pd.read_csv(f"{input_data.path}/data.csv")
    print(f"Processing {len(df)} samples")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text_column = 'text' if 'text' in df.columns else 'sentence'
    label_column = 'label' if 'label' in df.columns else 'labels'
    
    encodings = tokenizer(
        df[text_column].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None
    )
    
    df['input_ids'] = encodings['input_ids']
    df['attention_mask'] = encodings['attention_mask']
    df.to_csv(f"{output_data.path}/preprocessed.csv", index=False)
    
    tokenizer.save_pretrained(output_data.path)
    
    outputs = namedtuple('Outputs', ['num_samples', 'num_classes'])
    return outputs(len(df), df[label_column].nunique())


# ============================================
# COMPONENT 3: Train Model
# ============================================
@component(
    base_image="python:3.11",
    packages_to_install=[
        "torch>=2.5.0",
        "transformers>=4.46.0",
        "datasets>=3.0.0",
        "scikit-learn>=1.6.0",
        "pandas>=2.2.3"
    ]
)
def train_model(
    preprocessed_data: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics],
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
) -> float:
    """Train a sentiment analysis model"""
    import pandas as pd
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments
    )
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import json
    
    df = pd.read_csv(f"{preprocessed_data.path}/preprocessed.csv")
    
    class SentimentDataset(Dataset):
        def __init__(self, df):
            self.input_ids = torch.tensor(df['input_ids'].tolist())
            self.attention_mask = torch.tensor(df['attention_mask'].tolist())
            self.labels = torch.tensor(df['label'].tolist())
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]
            }
    
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = SentimentDataset(train_df)
    eval_dataset = SentimentDataset(eval_df)
    
    num_labels = df['label'].nunique()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda p: {
            'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1))
        }
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    accuracy = eval_results.get('eval_accuracy', 0.0)
    
    model.save_pretrained(model_output.path)
    tokenizer.save_pretrained(model_output.path)
    
    metrics_output.log_metric("accuracy", accuracy)
    metrics_output.log_metric("eval_loss", eval_results.get('eval_loss', 0.0))
    
    with open(f"{metrics_output.path}/metrics.json", "w") as f:
        json.dump(eval_results, f)
    
    print(f"✅ Training complete! Accuracy: {accuracy:.4f}")
    return accuracy


# ============================================
# COMPONENT 4: Deploy Model
# ============================================
@component(
    base_image="python:3.11",
    packages_to_install=["boto3", "fastapi", "uvicorn"]
)
def deploy_model(
    model: Input[Model],
    model_name: str = "sentiment-model"
) -> str:
    """Deploy model to SageMaker endpoint (simplified)"""
    import json
    
    model_path = model.path
    print(f"Model ready for deployment at: {model_path}")
    
    deployment_info = {
        "model_name": model_name,
        "model_path": model_path,
        "status": "ready",
        "endpoint": f"s3://mlops-models/{model_name}"
    }
    
    with open("/tmp/deployment.json", "w") as f:
        json.dump(deployment_info, f)
    
    return model_path


# ============================================
# MAIN PIPELINE DEFINITION - FIXED VERSION
# ============================================
@dsl.pipeline(
    name="sentiment-analysis-pipeline",
    description="End-to-end ML pipeline for sentiment analysis"
)
def sentiment_analysis_pipeline(
    dataset_name: str = "imdb",
    split: str = "train[:100]",
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """Define the pipeline workflow"""
    
    # Step 1: Load data
    load_task = load_data(
        dataset_name=dataset_name,
        split=split
    )
    
    # Step 2: Preprocess data
    preprocess_task = preprocess_data(
        input_data=load_task.outputs['output_dataset'],
        model_name=model_name
    )
    
    # Step 3: Train model
    train_task = train_model(
        preprocessed_data=preprocess_task.outputs['output_data'],
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Step 4: Deploy model
    deploy_task = deploy_model(
        model=train_task.outputs['model_output'],
        model_name="sentiment-model"
    )
    
    print("🎯 Pipeline defined successfully!")


# ============================================
# COMPILE THE PIPELINE
# ============================================
if __name__ == "__main__":
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=sentiment_analysis_pipeline,
        package_path="sentiment_pipeline.yaml"
    )
    
    print("✅ Pipeline compiled to sentiment_pipeline.yaml")
    print("\n📋 To run this pipeline:")
    print("   1. Upload sentiment_pipeline.yaml to Kubeflow Pipelines UI")
    print("   2. Or run with: kfp run create --experiment sentiment --run sentiment_pipeline.yaml")
#!/usr/bin/env python3
"""Deploy sentiment analysis model to Amazon SageMaker"""

import boto3
import time
import argparse
import sys

def deploy_to_sagemaker(image_uri, model_name, endpoint_name, instance_type, region):
    """Deploy model to SageMaker endpoint"""
    
    print(f"🚀 Deploying to SageMaker...")
    print(f"   Image: {image_uri}")
    print(f"   Model: {model_name}")
    print(f"   Endpoint: {endpoint_name}")
    print(f"   Instance: {instance_type}")
    print(f"   Region: {region}")
    
    sagemaker = boto3.client('sagemaker', region_name=region)
    account_id = boto3.client('sts').get_caller_identity()['Account']
    role_arn = f"arn:aws:iam::{account_id}:role/sagemaker-mlops-role"
    
    try:
        # Create model
        print("\n📦 Creating SageMaker model...")
        try:
            sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={'Image': image_uri},
                ExecutionRoleArn=role_arn
            )
            print(f"✅ Model created: {model_name}")
        except Exception as e:
            if 'AlreadyExists' in str(e):
                print(f"⚠️ Model {model_name} already exists")
        
        # Create endpoint config
        print("\n⚙️ Creating endpoint configuration...")
        endpoint_config_name = f"{model_name}-config"
        try:
            sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'prod',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': 1
                }]
            )
            print(f"✅ Endpoint config created: {endpoint_config_name}")
        except Exception as e:
            if 'AlreadyExists' in str(e):
                print(f"⚠️ Endpoint config already exists")
        
        # Deploy endpoint
        print(f"\n📡 Deploying endpoint (5-10 minutes)...")
        try:
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
        except Exception as e:
            if 'AlreadyExists' in str(e):
                print(f"⚠️ Endpoint exists, updating...")
                sagemaker.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name
                )
        
        # Wait for deployment
        while True:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            print(f"   Status: {status}")
            
            if status == 'InService':
                print(f"\n🎉 SUCCESS! Endpoint is live!")
                print(f"   Endpoint Name: {endpoint_name}")
                break
            elif status == 'Failed':
                print(f"\n❌ Failed: {response.get('FailureReason')}")
                sys.exit(1)
            time.sleep(30)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--endpoint-name', required=True)
    parser.add_argument('--instance-type', default='ml.t2.medium')
    parser.add_argument('--region', default='us-east-1')
    
    args = parser.parse_args()
    
    deploy_to_sagemaker(
        image_uri=args.image,
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        region=args.region
    )

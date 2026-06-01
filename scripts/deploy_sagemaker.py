#!/usr/bin/env python3
"""
Deploy sentiment analysis model to Amazon SageMaker
Uses existing ECR image (built by GitHub Actions)
"""

import boto3
import time
import json
import sys

def deploy_to_sagemaker():
    """Deploy model using existing ECR image"""
    
    print("🚀 Starting SageMaker Deployment...")
    
    # Get AWS account info
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    region = 'us-east-1'
    
    # Configuration
    ECR_IMAGE = f"{account_id}.dkr.ecr.{region}.amazonaws.com/sentiment-mlops:latest"
    ROLE_ARN = f"arn:aws:iam::{account_id}:role/sagemaker-mlops-role"
    MODEL_NAME = "sentiment-model"
    ENDPOINT_CONFIG_NAME = "sentiment-endpoint-config"
    ENDPOINT_NAME = "sentiment-endpoint"
    
    print(f"📋 Configuration:")
    print(f"   Account: {account_id}")
    print(f"   Region: {region}")
    print(f"   Image: {ECR_IMAGE}")
    print(f"   Role: {ROLE_ARN}")
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    # Step 1: Check if image exists in ECR
    print("\n🔍 Checking if ECR image exists...")
    ecr = boto3.client('ecr', region_name=region)
    try:
        ecr.describe_images(repositoryName='sentiment-mlops', imageIds=[{'imageTag': 'latest'}])
        print("✅ ECR image found")
    except Exception as e:
        print(f"❌ ECR image not found: {e}")
        print("\n📌 Please run GitHub Actions workflow first to build and push image to ECR")
        print("   Push to main branch to trigger the build")
        sys.exit(1)
    
    # Step 2: Create or update model
    print("\n📦 Creating/Updating SageMaker model...")
    try:
        sagemaker.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={
                'Image': ECR_IMAGE,
                'Environment': {
                    'LOG_LEVEL': 'INFO',
                    'MODEL_SERVER_WORKERS': '1'
                }
            },
            ExecutionRoleArn=ROLE_ARN
        )
        print("✅ Model created")
    except Exception as e:
        if 'AlreadyExists' in str(e):
            print("⚠️ Model already exists, skipping creation")
        else:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    # Step 3: Create endpoint configuration
    print("\n⚙️ Creating endpoint configuration...")
    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
            ProductionVariants=[
                {
                    'VariantName': 'prod-variant',
                    'ModelName': MODEL_NAME,
                    'InstanceType': 'ml.t2.medium',
                    'InitialInstanceCount': 1,
                    'InitialVariantWeight': 1.0,
                }
            ]
        )
        print("✅ Endpoint configuration created")
    except Exception as e:
        if 'AlreadyExists' in str(e):
            print("⚠️ Endpoint config already exists")
        else:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    # Step 4: Deploy endpoint
    print("\n📡 Deploying endpoint (this takes 5-10 minutes)...")
    print("   Please wait...")
    
    try:
        # Check if endpoint already exists
        try:
            existing_endpoint = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
            print(f"   Endpoint already exists with status: {existing_endpoint['EndpointStatus']}")
            print("   Updating endpoint...")
            sagemaker.update_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=ENDPOINT_CONFIG_NAME
            )
        except:
            # Create new endpoint
            sagemaker.create_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=ENDPOINT_CONFIG_NAME
            )
        
        # Wait for deployment
        while True:
            response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
            status = response['EndpointStatus']
            print(f"   Status: {status}")
            
            if status == 'InService':
                print("\n🎉 SUCCESS! Endpoint is live!")
                print(f"   Endpoint Name: {ENDPOINT_NAME}")
                break
            elif status == 'Failed':
                print(f"\n❌ Deployment failed: {response.get('FailureReason')}")
                sys.exit(1)
            elif status == 'RollingBack':
                print("   Deployment rolling back...")
                sys.exit(1)
            
            time.sleep(30)
            
    except Exception as e:
        print(f"❌ Error deploying endpoint: {e}")
        sys.exit(1)
    
    # Step 5: Test the endpoint
    print("\n🧪 Testing the endpoint...")
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    test_texts = [
        "This movie is absolutely amazing! I loved it!",
        "Terrible product, complete waste of money.",
        "It was okay, nothing special."
    ]
    
    for text in test_texts:
        payload = json.dumps({'text': text})
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )
        result = json.loads(response['Body'].read().decode())
        print(f"\n   Input: {text[:50]}...")
        print(f"   Response: {result}")
    
    print("\n" + "="*50)
    print("✅ DEPLOYMENT COMPLETE!")
    print("="*50)
    print(f"Endpoint Name: {ENDPOINT_NAME}")
    print(f"AWS Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{ENDPOINT_NAME}")
    print("\n⚠️ IMPORTANT: Delete endpoint after demo to stop charges:")
    print(f"   aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME}")

if __name__ == "__main__":
    deploy_to_sagemaker()

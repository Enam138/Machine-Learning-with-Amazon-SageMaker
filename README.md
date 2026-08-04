# ML Model Deployment with Amazon SageMaker

> An end-to-end machine learning deployment project that trains an XGBoost regression model using Amazon SageMaker and exposes real-time predictions through a Flask API running on Amazon EC2 and Amazon API Gateway.

## Project Overview

This project demonstrates the complete lifecycle of deploying a machine learning model on AWS, from preparing training data and training an XGBoost model to exposing the trained model through a production-style REST API for real-time inference.

The solution uses the **California Housing dataset** to train a regression model for predicting housing values based on numerical housing and demographic features.

Rather than stopping at model development in a notebook, the project focuses on the engineering required to turn a trained machine learning model into an accessible cloud service.

The architecture combines:

- Amazon SageMaker for model training and real-time inference
- Amazon S3 for training data and model artifact storage
- Amazon EC2 for hosting the Flask application
- Amazon API Gateway for exposing the prediction REST API
- AWS IAM for service-to-service authorization
- Amazon CloudWatch for logging and monitoring
- Boto3 for communication between the Flask application and SageMaker
- Gunicorn and Nginx for serving the Flask application

The result is an end-to-end AWS machine learning deployment architecture capable of accepting housing features through an HTTPS API and returning predictions from a deployed SageMaker endpoint.


## Project Objectives

The primary objectives of this project were to:

- Build an end-to-end machine learning deployment workflow on AWS
- Train an XGBoost regression model using Amazon SageMaker
- Store training data and model artifacts securely in Amazon S3
- Evaluate model performance before deployment
- Deploy the trained model to a real-time SageMaker inference endpoint
- Build a Flask API that communicates with SageMaker using Boto3
- Host the Flask application on Amazon EC2
- Expose the application through Amazon API Gateway
- Secure communication between AWS services using IAM roles
- Monitor application and model operations using Amazon CloudWatch
- Demonstrate practical MLOps and cloud engineering concepts


## Architecture

<img width="1280" height="853" alt="akwannya3" src="https://github.com/user-attachments/assets/40da88e0-4b25-4edb-b68c-9214cf78c9ca" />


The architecture contains two major workflows:

1. **Offline Model Training Pipeline**
2. **Online Real-Time Inference Pipeline**

This separation allows model development and model serving to operate as distinct stages of the machine learning lifecycle.


# Architecture Overview

## 1. Offline Model Training Pipeline

The model training workflow begins with the California Housing dataset.

```text
California Housing Dataset
          │
          ▼
    Data Preprocessing
          │
          ▼
   Amazon S3 Bucket
    (Training Data)
          │
          ▼
Amazon SageMaker Training Job
          │
          ▼
      XGBoost Model
          │
          ▼
     Model Evaluation
   RMSE | MAE | R² Score
          │
          ▼
      model.tar.gz
          │
          ▼
      Amazon S3
    (Model Artifact)
          │
          ▼
Amazon SageMaker Endpoint
```

The dataset is cleaned and prepared before being uploaded to Amazon S3.

Amazon SageMaker then performs the model training process using the XGBoost algorithm.

After training, the model is evaluated using regression metrics before the resulting model artifact is stored in Amazon S3 and deployed to a SageMaker real-time inference endpoint.

## 2. Real-Time Inference Pipeline

Once the model has been deployed, predictions are served through the following architecture:

```text
End User
   │
   │ HTTPS / JSON
   ▼
Amazon API Gateway
   │
   │ Forward Request
   ▼
Amazon EC2
Flask REST API
   │
   │ Boto3 InvokeEndpoint
   ▼
Amazon SageMaker Endpoint
   │
   │ XGBoost Inference
   ▼
Prediction
   │
   ▼
Flask API
   │
   │ JSON Response
   ▼
Amazon API Gateway
   │
   ▼
End User
```

The end user does not communicate directly with SageMaker.

Instead, the Flask application acts as the application layer between the public API and the machine learning inference endpoint.


# End-to-End Data Flow

A prediction request follows six primary stages.

### Step 1 — Client Request

A user submits housing features from a client such as:

- Postman
- Web application
- Mobile application
- REST API client

The request is formatted as JSON and sent over HTTPS.


### Step 2 — Amazon API Gateway

Amazon API Gateway receives the request through the prediction REST endpoint.

Example:

```text
POST /predict
```

API Gateway forwards the request to the Flask application running on Amazon EC2.


### Step 3 — Flask API

The Flask application:

1. Receives the JSON request
2. Parses the input
3. Validates the required features
4. Converts the input into the format expected by the model
5. Uses the AWS SDK for Python (Boto3) to invoke the SageMaker endpoint


### Step 4 — SageMaker Inference

The Flask application invokes the deployed SageMaker endpoint using:

```text
InvokeEndpoint
```

The SageMaker endpoint hosts the trained XGBoost model and performs real-time inference using the supplied housing features.


### Step 5 — Prediction Response

SageMaker returns the prediction to the Flask application.

The Flask application converts the prediction into a structured JSON response.


### Step 6 — Client Response

The response travels back through Amazon API Gateway and is returned to the client over HTTPS.

```text
Client
   │
   ▼
API Gateway
   │
   ▼
EC2 / Flask
   │
   ▼
SageMaker Endpoint
   │
   ▼
Prediction
   │
   ▼
EC2 / Flask
   │
   ▼
API Gateway
   │
   ▼
Client
```


# AWS Services Used

| AWS Service | Purpose |
|-------------|---------|
| Amazon SageMaker | Model training, hosting, and real-time inference |
| Amazon S3 | Stores training data, model artifacts, and dependencies |
| Amazon EC2 | Hosts the Flask backend application |
| Amazon API Gateway | Exposes the prediction REST API |
| AWS IAM | Controls permissions between AWS services |
| Amazon CloudWatch | Provides logging, monitoring, and operational visibility |


# Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.10 | Application and machine learning development |
| XGBoost | Regression model |
| Flask | Backend REST API |
| Boto3 | AWS SDK used to invoke SageMaker |
| Gunicorn | Python WSGI application server |
| Nginx | Reverse proxy for the Flask application |
| JSON | API request and response format |
| REST | API communication architecture |


# Machine Learning Workflow

The machine learning workflow consists of seven major stages.

## 1. Dataset

The project uses the **California Housing dataset** for the regression task.

The dataset contains housing and demographic attributes used to predict housing values.


## 2. Data Preprocessing

Before model training, the dataset is prepared through several preprocessing operations.

These include:

- Data cleaning
- Feature preparation
- Train/test splitting
- Feature scaling where required
- Formatting data for model training

The objective is to ensure that the training data is consistent and suitable for the XGBoost algorithm.


## 3. Upload Training Data to Amazon S3

The prepared training dataset is uploaded to Amazon S3.

Amazon S3 acts as the persistent storage layer used by SageMaker during training.

```text
Prepared Dataset
       │
       ▼
   Amazon S3
       │
       ▼
SageMaker Training
```

## 4. SageMaker Training Job

Amazon SageMaker is used to train the XGBoost regression model.

The training job:

- Retrieves training data from Amazon S3
- Initializes the XGBoost training environment
- Trains the regression model
- Produces the trained model artifact

Using SageMaker separates model training from the local development environment and demonstrates managed cloud-based ML training.


## 5. Model Evaluation

After training, the model is evaluated using regression metrics.

The evaluation process includes:

### Root Mean Squared Error — RMSE

Measures the average magnitude of prediction errors while penalizing larger errors.

### Mean Absolute Error — MAE

Measures the average absolute difference between predicted and actual values.

### R² Score

Measures how much variance in the target variable is explained by the model.

These metrics help determine whether the model performs sufficiently well before deployment.


## 6. Model Artifact

After successful training, SageMaker produces the trained model artifact.

```text
model.tar.gz
```

The artifact is stored in Amazon S3.

This artifact contains the trained model required for inference.


## 7. Real-Time Endpoint Deployment

The trained model is deployed to an Amazon SageMaker real-time inference endpoint.

The endpoint:

- Hosts the XGBoost model
- Accepts inference requests
- Performs predictions
- Returns prediction results

The Flask application communicates with this endpoint through Boto3.


# API Layer

The Flask application provides the application logic between API Gateway and SageMaker.

Its responsibilities include:

- Receiving requests
- Parsing JSON
- Validating input
- Preparing inference payloads
- Invoking SageMaker
- Processing prediction responses
- Returning structured JSON

This prevents clients from communicating directly with the SageMaker endpoint.


# Example Prediction Request

A client can submit a request to the prediction API.

Example structure:

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

> The exact payload structure must match the feature order and preprocessing expected by the deployed model.


# Example Prediction Response

A successful response may resemble:

```json
{
  "prediction": 4.52
}
```

The prediction is generated by the XGBoost model hosted on the SageMaker endpoint.


# Security Architecture

Security controls are applied across the architecture.

```text
Internet
   │
 HTTPS
   ▼
API Gateway
   │
   ▼
EC2 / Flask
   │
IAM Role
   ▼
SageMaker Endpoint
   │
   ▼
Private Model Artifacts
   │
   ▼
Amazon S3
```

## HTTPS Communication

Client requests are transmitted securely using HTTPS.

This protects data while it travels between clients and the public API.


## IAM Roles and Policies

IAM controls communication between AWS services.

The EC2 instance receives only the permissions required to invoke the SageMaker endpoint.

SageMaker receives the permissions required to access training data and model artifacts stored in Amazon S3.

This follows the **Principle of Least Privilege**.


## Amazon S3 Security

Amazon S3 stores:

- Training data
- Model artifacts
- Supporting ML files

The bucket is not intended for direct public access.

Access is controlled through AWS IAM.


## EC2 Security

The EC2 instance hosts the Flask API.

Security controls include:

- Security Group rules
- IAM instance role
- Restricted inbound access
- Controlled outbound communication

Nginx acts as the reverse proxy while Gunicorn serves the Flask application.


## No Hardcoded AWS Credentials

The application uses IAM roles and temporary AWS credentials rather than embedding long-term AWS access keys in application code.

This reduces the risk of credential exposure.


# IAM Permission Flow

The architecture follows a service-to-service permission model.

```text
EC2
 │
 │ IAM Instance Role
 ▼
SageMaker InvokeEndpoint

SageMaker
 │
 │ Execution Role
 ▼
Amazon S3

Lambda-style embedded credentials
        ✗

IAM Roles
        ✓
```

This approach improves security and simplifies credential management.


# Monitoring and Observability

Amazon CloudWatch provides operational visibility across the architecture.

Monitoring can include:

- SageMaker endpoint metrics
- Model invocation activity
- Endpoint latency
- Endpoint errors
- Application logs
- EC2 operational metrics

CloudWatch helps identify application failures and performance problems.


# Scalability

The architecture separates the public API, application logic, model hosting, and storage layers.

```text
API Layer
     │
Application Layer
     │
ML Inference Layer
     │
Storage Layer
```

This separation allows individual components to be managed and scaled independently as requirements evolve.

Amazon SageMaker manages the model-serving infrastructure, while Amazon S3 provides highly scalable object storage.


# Cost Considerations

Several components of this architecture can generate ongoing AWS charges.

These include:

- Amazon EC2 runtime
- SageMaker real-time endpoint runtime
- SageMaker training jobs
- Amazon S3 storage
- API Gateway requests
- CloudWatch logs and metrics

Unlike purely event-driven serverless workloads, EC2 instances and SageMaker real-time endpoints may continue generating charges while provisioned.

For development environments, unused resources should be stopped or deleted when they are no longer required.


# Key Features

The project demonstrates:

- End-to-end ML model deployment
- Cloud-based model training
- Real-time inference
- REST API integration
- Managed model hosting
- Secure AWS service communication
- Model artifact management
- Cloud monitoring
- Production-style Flask serving
- Separation of training and inference workflows


# Key Design Decisions

## SageMaker for Model Training and Hosting

Amazon SageMaker was selected because it provides managed infrastructure for both model training and real-time inference.

This reduces the operational complexity of manually configuring ML infrastructure.


## Amazon S3 for ML Artifact Storage

Amazon S3 provides durable object storage for:

- Training datasets
- Model artifacts
- Supporting files

Its native integration with SageMaker makes it suitable for the ML workflow.


## EC2 for the Flask Application

Amazon EC2 provides control over the application runtime and allows Flask, Gunicorn, Nginx, and Boto3 to operate as the API application layer.


## API Gateway for Public API Exposure

API Gateway provides a managed entry point for client requests and separates external clients from the underlying EC2 application.


## Boto3 for SageMaker Invocation

The Flask application uses Boto3 to communicate with AWS services.

This enables the application to invoke the SageMaker endpoint using the EC2 instance's IAM role instead of hardcoded credentials.


# Project Structure

```text
ml-model-deployment/
│
├── README.md
│
├── LICENSE
├── .gitignore
│
├── architecture/
│   ├── architecture-overview.md
│   ├── aws-services.md
│   ├── request-flow.md
│   └── ml-pipeline.md
│
├── deployment/
│   ├── prerequisites.md
│   ├── deployment-guide.md
│   └── validation.md
│
├── security/
│   ├── security-controls.md
│   ├── iam-security.md
│   ├── api-security.md
│   ├── endpoint-security.md
│   ├── threat-model.md
│   └── best-practices.md
│
├── mlops/
│   ├── model-training.md
│   ├── model-serving.md
│   ├── inference.md
│   └── model-lifecycle.md
│
├── monitoring/
│   └── monitoring.md
│
├── troubleshooting/
│   └── common-issues.md
│
├── diagrams/
│   └── architecture.png
│
├── screenshots/
│
└── lessons-learned.md
```


# MLOps Concepts Demonstrated

Although this project does not attempt to implement a complete enterprise MLOps platform, it demonstrates several important MLOps concepts.

These include:

### Model Training

Training workloads are executed using managed SageMaker infrastructure.

### Artifact Management

Trained model artifacts are persisted in Amazon S3.

### Model Evaluation

Regression metrics are used to assess model performance before deployment.

### Model Deployment

The trained model is deployed as a SageMaker real-time endpoint.

### Model Serving

Applications consume the model through a REST-based application architecture.

### Monitoring

CloudWatch provides operational visibility into deployed AWS resources.


# AWS Well-Architected Considerations

The project incorporates principles from the AWS Well-Architected Framework.

## Security

- IAM-based access
- Least-privilege permissions
- HTTPS communication
- Private model artifacts
- No hardcoded AWS credentials

## Reliability

- Managed SageMaker model hosting
- Durable Amazon S3 storage
- Separation of application components

## Performance Efficiency

- Managed ML training
- Real-time inference endpoint
- Dedicated application layer

## Cost Optimization

- Pay-per-use API requests
- Managed object storage
- Ability to remove development endpoints and EC2 resources when unused

## Operational Excellence

- CloudWatch monitoring
- Service separation
- Documented architecture
- Structured deployment workflow


# Challenges Addressed

This project addresses several practical ML deployment challenges:

- Moving a model beyond local development
- Managing model artifacts
- Hosting a model for real-time inference
- Connecting application code to a managed ML endpoint
- Exposing predictions through a REST API
- Managing AWS permissions securely
- Monitoring deployed ML infrastructure
- Separating model training from application serving


# Future Improvements

The architecture provides a foundation that can be extended with additional MLOps capabilities.

Potential enhancements include:

- Automated model retraining
- CI/CD for application and model deployment
- SageMaker Model Registry
- SageMaker Pipelines
- Model versioning
- Data drift monitoring
- Model quality monitoring
- API authentication and authorization
- AWS WAF
- Auto Scaling
- Private networking using Amazon VPC
- Containerized Flask deployment
- Blue/green model deployment
- Canary inference deployment

These improvements would move the solution closer to a full production MLOps platform.


# What I Learned

Building this project strengthened my practical understanding of:

- Amazon SageMaker
- Machine learning deployment
- AWS API integration
- Amazon EC2
- Amazon S3
- API Gateway
- AWS IAM
- Amazon CloudWatch
- Flask REST APIs
- Boto3
- XGBoost
- Real-time inference
- Cloud security
- MLOps architecture

Most importantly, the project demonstrated that building a machine learning model is only one part of the ML lifecycle. Delivering predictions to real users requires additional engineering around infrastructure, APIs, security, deployment, monitoring, and model lifecycle management.


# Documentation

Detailed technical documentation is available throughout this repository.

| Section | Description |
|---------|-------------|
| `architecture/` | AWS architecture and system design |
| `deployment/` | Environment setup and deployment procedures |
| `security/` | IAM, API, endpoint, and infrastructure security |
| `mlops/` | Model training, serving, inference, and lifecycle |
| `monitoring/` | CloudWatch monitoring and observability |
| `troubleshooting/` | Common issues and resolutions |
| `lessons-learned.md` | Key technical and architectural lessons |


# Conclusion

The **ML Model Deployment with Amazon SageMaker** project demonstrates an end-to-end approach to taking a machine learning model from data preparation and training to a cloud-hosted real-time prediction service.

By combining Amazon SageMaker, Amazon S3, Amazon EC2, Amazon API Gateway, AWS IAM, Amazon CloudWatch, Flask, Boto3, and XGBoost, the project demonstrates how machine learning can be integrated with cloud infrastructure and application engineering to create an accessible inference service.

The project goes beyond model development by addressing the broader engineering requirements of machine learning systems, including model hosting, API integration, security, monitoring, infrastructure management, and real-time inference.

It therefore serves as a practical demonstration of **AWS cloud engineering, machine learning deployment, API development, and foundational MLOps principles**.

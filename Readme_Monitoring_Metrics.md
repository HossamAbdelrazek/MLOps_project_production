# Hand Gesture Recognition API - Monitoring & Deployment

## Overview
This repository contains a FastAPI-based hand gesture recognition service with comprehensive monitoring, testing, and deployment automation.

## Monitoring Metrics

We have implemented three types of monitoring metrics to ensure robust system observability:

### 1. Model-Related Metrics

#### `gesture_prediction_confidence`
- **Type**: Histogram
- **Description**: Distribution of prediction confidence scores from the ML model
- **Reasoning**: 
  - Critical for monitoring model performance degradation over time
  - Low confidence scores may indicate model drift or poor input quality
  - Helps identify when the model needs retraining
  - Enables alerting when confidence drops below acceptable thresholds
- **Buckets**: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#### `gesture_predictions_total`
- **Type**: Counter
- **Description**: Total number of predictions by gesture type and direction
- **Reasoning**:
  - Monitors prediction distribution to detect bias
  - Helps identify if certain gestures are over/under-represented
  - Useful for understanding user behavior patterns

### 2. Data-Related Metrics

#### `invalid_input_requests_total`
- **Type**: Counter with labels for error types
- **Description**: Tracks requests with invalid input data
- **Error Types Monitored**:
  - `empty_data`: Empty input arrays
  - `invalid_length`: Incorrect number of landmarks (not 63)
  - `invalid_numeric_values`: NaN or infinite values
  - `data_conversion_error`: Type conversion failures
  - `model_not_loaded`: Requests when model unavailable
  - `unknown_gesture`: Predictions for unrecognized gestures
- **Reasoning**:
  - Essential for monitoring data quality issues
  - Helps identify client-side problems or API misuse
  - Enables proactive debugging and user support
  - Critical for maintaining service reliability

### 3. Server-Related Metrics

#### `model_health_status`
- **Type**: Gauge
- **Description**: Binary indicator of model and encoder loading status (1=healthy, 0=unhealthy)
- **Reasoning**:
  - Provides immediate visibility into critical service dependencies
  - Enables automated alerting for model loading failures
  - Essential for service availability monitoring
  - Helps with rapid incident response

#### `prediction_processing_time_seconds`
- **Type**: Histogram
- **Description**: Processing time for prediction requests
- **Reasoning**:
  - Critical for monitoring API performance and SLA compliance
  - Helps identify performance bottlenecks
  - Enables capacity planning and scaling decisions
  - Important for user experience optimization

## Additional Monitoring Features

- **HTTP Request Metrics**: Automatically tracked by Prometheus FastAPI Instrumentator
- **Health Check Endpoints**: `/` and `/health` for service monitoring
- **Metrics Info Endpoint**: `/metrics-info` for documentation

## Testing Strategy

### Unit Tests (`test_gesture_api.py`)

Our comprehensive test suite covers:

- **API Endpoints**: Health checks and prediction endpoints
- **Input Validation**: Various invalid input scenarios
- **Error Handling**: Model loading failures and prediction errors
- **Business Logic**: Gesture mapping validation
- **Edge Cases**: Empty inputs, wrong data types, CORS headers

### Test Categories

1. **Happy Path Tests**: Valid inputs and expected outputs
2. **Validation Tests**: Input format and type validation
3. **Error Handling Tests**: Exception scenarios and fallbacks
4. **Integration Tests**: End-to-end API functionality

## Monitoring Setup

### Local Development
```bash
# Start all services
docker-compose up -d

# Access services
# API: http://localhost:8080
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)
```

### Grafana Dashboard

Our dashboard includes:
- **API Health Status**: Real-time model health indicator
- **Request Rate**: API traffic monitoring
- **Prediction Confidence**: Model performance visualization
- **Invalid Requests**: Data quality monitoring
- **Gesture Distribution**: Usage pattern analysis
- **Response Time**: Performance metrics

## Deployment

### AWS ECS Deployment

The GitHub Actions workflow automates:
1. **Testing**: Unit test execution
2. **Building**: Docker image creation
3. **Pushing**: ECR repository upload
4. **Deployment**: ECS service update
5. **Verification**: Health check validation

### Required AWS Resources

- **ECS Cluster**: Container orchestration
- **ECR Repository**: Docker image storage
- **Application Load Balancer**: Traffic distribution
- **CloudWatch**: Logging and monitoring
- **IAM Roles**: Service permissions

### Environment Variables

Set these secrets in GitHub:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Alerting Recommendations

Based on our metrics, set up alerts for:

1. **Model Health**: `model_health_status == 0`
2. **High Error Rate**: `rate(invalid_input_requests_total[5m]) > 0.1`
3. **Low Confidence**: `histogram_quantile(0.5, gesture_prediction_confidence_bucket) < 0.7`
4. **High Latency**: `histogram_quantile(0.95, prediction_processing_time_seconds_bucket) > 1.0`
5. **API Downtime**: `up{job="gesture-api"} == 0`

## Performance Optimization

Monitor these metrics to optimize:
- **Batch Processing**: Group predictions for efficiency
- **Model Optimization**: Reduce inference time
- **Caching**: Cache frequent predictions
- **Scaling**: Auto-scale based on request rate

## Contributing

1. Run tests: `pytest test_gesture_api.py -v`
2. Check metrics: `curl http://localhost:8080/metrics`
3. Monitor dashboard: Grafana at http://localhost:3000
4. Follow deployment workflow for production changes
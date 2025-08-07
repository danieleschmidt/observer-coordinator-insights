# API Specifications & Contracts

This document provides comprehensive technical specifications for the Observer Coordinator Insights REST API, including data models, endpoint contracts, error handling, and integration patterns for developers building on top of the platform.

## Table of Contents

1. [API Architecture](#api-architecture)
2. [Authentication & Authorization](#authentication--authorization)
3. [Data Models & Schemas](#data-models--schemas)
4. [Endpoint Specifications](#endpoint-specifications)
5. [Error Handling](#error-handling)
6. [Rate Limiting & Throttling](#rate-limiting--throttling)
7. [Webhooks & Events](#webhooks--events)
8. [API Versioning](#api-versioning)
9. [OpenAPI Specification](#openapi-specification)
10. [Integration Patterns](#integration-patterns)

## API Architecture

### REST API Design Principles

Observer Coordinator Insights follows RESTful design principles with the following characteristics:

- **Resource-based URLs**: Endpoints represent resources, not actions
- **HTTP Verbs**: Standard HTTP methods (GET, POST, PUT, DELETE, PATCH)
- **Stateless**: Each request contains all necessary information
- **JSON Format**: Request and response bodies use JSON
- **HATEOAS**: Hypermedia as the Engine of Application State
- **Consistent Error Format**: Standardized error responses

### Base URL Structure

```
Production:  https://api.insights.company.com/v1/
Staging:     https://staging-api.insights.company.com/v1/
Development: http://localhost:8000/api/
```

### Request/Response Format

#### Request Headers
```http
Content-Type: application/json
Authorization: Bearer <token>
Accept: application/json
X-Request-ID: <unique-request-id>
X-API-Version: v1
```

#### Response Headers
```http
Content-Type: application/json
X-Request-ID: <unique-request-id>
X-Rate-Limit-Remaining: 950
X-Rate-Limit-Reset: 1640995200
```

## Authentication & Authorization

### JWT Authentication

The API uses JWT (JSON Web Tokens) for authentication:

```typescript
interface AuthToken {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  refresh_token?: string;
  scope?: string[];
}
```

#### Authentication Flow

```typescript
// 1. Obtain token
POST /auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=password&username=user@company.com&password=secret

// Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "def502004b8c9a...",
  "scope": ["analytics:read", "analytics:create"]
}

// 2. Use token in requests
GET /analytics/jobs
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### OAuth 2.0 Integration

For third-party integrations:

```typescript
// OAuth 2.0 Authorization Code Flow
interface OAuthConfig {
  client_id: string;
  response_type: "code";
  redirect_uri: string;
  scope: string;
  state: string;
}

// Authorization URL
GET /auth/authorize?client_id=<id>&response_type=code&redirect_uri=<uri>&scope=<scope>&state=<state>

// Token exchange
POST /auth/token
{
  "grant_type": "authorization_code",
  "client_id": "<client_id>",
  "client_secret": "<client_secret>",
  "code": "<authorization_code>",
  "redirect_uri": "<redirect_uri>"
}
```

### Role-Based Access Control (RBAC)

```typescript
interface UserRole {
  role_id: string;
  name: string;
  permissions: Permission[];
}

interface Permission {
  resource: string;  // e.g., "analytics", "teams", "admin"
  actions: string[]; // e.g., ["read", "create", "update", "delete"]
}

// Example roles
const ROLES = {
  admin: {
    permissions: ["*:*"]  // Full access
  },
  analyst: {
    permissions: [
      "analytics:read",
      "analytics:create", 
      "teams:read"
    ]
  },
  viewer: {
    permissions: [
      "analytics:read",
      "teams:read"
    ]
  }
};
```

## Data Models & Schemas

### Core Data Models

#### Employee Data Model

```typescript
interface EmployeeData {
  employee_id: string;
  red_energy: number;     // 0-100
  blue_energy: number;    // 0-100
  green_energy: number;   // 0-100
  yellow_energy: number;  // 0-100
  department?: string;
  role_level?: string;
  tenure_years?: number;
  created_at?: string;    // ISO 8601 datetime
  updated_at?: string;    // ISO 8601 datetime
}

// JSON Schema
const EmployeeDataSchema = {
  type: "object",
  required: ["employee_id", "red_energy", "blue_energy", "green_energy", "yellow_energy"],
  properties: {
    employee_id: { type: "string", minLength: 1, maxLength: 50 },
    red_energy: { type: "number", minimum: 0, maximum: 100 },
    blue_energy: { type: "number", minimum: 0, maximum: 100 },
    green_energy: { type: "number", minimum: 0, maximum: 100 },
    yellow_energy: { type: "number", minimum: 0, maximum: 100 },
    department: { type: "string", maxLength: 100 },
    role_level: { type: "string", enum: ["junior", "mid", "senior", "lead", "manager"] },
    tenure_years: { type: "number", minimum: 0, maximum: 50 }
  },
  additionalProperties: false
};
```

#### Clustering Job Model

```typescript
interface ClusteringJob {
  job_id: string;
  user_id: string;
  status: JobStatus;
  method: ClusteringMethod;
  parameters: ClusteringParameters;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress: number;        // 0.0 to 1.0
  employee_count: number;
  results?: ClusteringResults;
  error_message?: string;
  estimated_completion?: string;
}

enum JobStatus {
  QUEUED = "queued",
  PROCESSING = "processing", 
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled"
}

enum ClusteringMethod {
  ESN = "esn",
  SNN = "snn", 
  LSM = "lsm",
  HYBRID = "hybrid_reservoir"
}

interface ClusteringParameters {
  n_clusters: number;
  method: ClusteringMethod;
  optimize_clusters?: boolean;
  secure_mode?: boolean;
  normalization?: string;
  language?: string;
  custom_params?: Record<string, any>;
}
```

#### Clustering Results Model

```typescript
interface ClusteringResults {
  cluster_count: number;
  employee_count: number;
  processing_time_seconds: number;
  method_used: ClusteringMethod;
  clusters: Cluster[];
  metrics: ClusteringMetrics;
  quality_assessment: QualityAssessment;
}

interface Cluster {
  cluster_id: number;
  name: string;
  employee_count: number;
  centroid: EnergyProfile;
  characteristics: ClusterCharacteristics;
  employees: string[];  // List of employee IDs
  confidence_score: number;
}

interface EnergyProfile {
  red_energy: number;
  blue_energy: number;
  green_energy: number;
  yellow_energy: number;
}

interface ClusterCharacteristics {
  dominant_energy: string;
  energy_distribution: EnergyProfile;
  personality_traits: string[];
  ideal_roles: string[];
  team_contribution: string;
  communication_style: string;
  decision_making: string;
  stress_response: string;
}

interface ClusteringMetrics {
  silhouette_score: number;
  calinski_harabasz_score: number;
  davies_bouldin_score: number;
  stability_score: number;
  interpretability_score: number;
}

interface QualityAssessment {
  overall_score: number;        // 0.0 to 1.0
  cluster_separation: number;   // 0.0 to 1.0
  within_cluster_cohesion: number; // 0.0 to 1.0
  stability_rating: string;     // "low", "medium", "high"
  recommendations: string[];
}
```

#### Team Formation Model

```typescript
interface TeamComposition {
  team_id: string;
  name?: string;
  members: TeamMember[];
  balance_metrics: TeamBalanceMetrics;
  predicted_performance: PerformancePrediction;
  formation_strategy: string;
  constraints_applied: TeamConstraint[];
  created_at: string;
}

interface TeamMember {
  employee_id: string;
  cluster_id: number;
  energy_profile: EnergyProfile;
  predicted_role: string;
  contribution_score: number;
  compatibility_scores: Record<string, number>;
}

interface TeamBalanceMetrics {
  energy_balance_score: number;     // 0.0 to 1.0
  diversity_score: number;          // 0.0 to 1.0
  complementarity_score: number;    // 0.0 to 1.0
  conflict_risk: number;            // 0.0 to 1.0
  communication_efficiency: number; // 0.0 to 1.0
}

interface PerformancePrediction {
  overall_performance: number;      // 0.0 to 1.0
  innovation_potential: number;     // 0.0 to 1.0
  execution_capability: number;     // 0.0 to 1.0
  collaboration_quality: number;    // 0.0 to 1.0
  adaptability: number;             // 0.0 to 1.0
  confidence_interval: [number, number];
}

interface TeamConstraint {
  type: string;  // "department", "experience", "role", "location"
  parameters: Record<string, any>;
  weight: number; // 0.0 to 1.0
}
```

### Validation Schemas

```typescript
// Request validation middleware
interface ValidationSchema {
  body?: object;
  query?: object;
  params?: object;
  headers?: object;
}

// Example endpoint validation
const UploadAnalyticsSchema: ValidationSchema = {
  body: {
    type: "object",
    properties: {
      file: { type: "string", format: "binary" },
      n_clusters: { type: "integer", minimum: 2, maximum: 20 },
      method: { type: "string", enum: ["esn", "snn", "lsm", "hybrid_reservoir"] },
      secure_mode: { type: "boolean" },
      language: { type: "string", enum: ["en", "de", "es", "fr", "ja", "zh"] }
    },
    required: ["file"]
  }
};
```

## Endpoint Specifications

### Analytics Endpoints

#### Upload and Analyze Data

```http
POST /analytics/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

**Request Parameters:**
```typescript
interface UploadRequest {
  file: File;              // CSV file with employee data
  n_clusters?: number;     // Default: 4, Range: 2-20
  method?: ClusteringMethod; // Default: "hybrid_reservoir"
  secure_mode?: boolean;   // Default: false
  optimize_clusters?: boolean; // Default: false
  language?: string;       // Default: "en"
  custom_params?: Record<string, any>;
}
```

**Response:**
```typescript
interface UploadResponse {
  job_id: string;
  status: "processing";
  message: string;
  employee_count: number;
  estimated_completion: string; // ISO 8601 datetime
  progress_url: string;
  webhook_url?: string;
}

// Example
{
  "job_id": "job_abc123def456",
  "status": "processing",
  "message": "Analysis started successfully",
  "employee_count": 150,
  "estimated_completion": "2025-01-15T14:30:00Z",
  "progress_url": "/analytics/status/job_abc123def456",
  "webhook_url": "https://client.com/webhooks/clustering-complete"
}
```

#### Get Analysis Results

```http
GET /analytics/results/{job_id}
Authorization: Bearer <token>
```

**Response:**
```typescript
interface AnalysisResults {
  job_id: string;
  status: "completed";
  completion_time: string;
  processing_time_seconds: number;
  results: ClusteringResults;
  download_urls: {
    json: string;
    csv: string;
    pdf_report: string;
    visualization: string;
  };
  _links: {
    self: string;
    teams: string;
    visualization: string;
  };
}
```

#### List Analysis Jobs

```http
GET /analytics/jobs?limit=20&offset=0&status=completed&method=esn
Authorization: Bearer <token>
```

**Query Parameters:**
```typescript
interface JobListQuery {
  limit?: number;     // Default: 20, Max: 100
  offset?: number;    // Default: 0
  status?: JobStatus[];
  method?: ClusteringMethod[];
  created_after?: string;  // ISO 8601 datetime
  created_before?: string; // ISO 8601 datetime
  sort?: string;      // "created_at", "-created_at", "completion_time"
}
```

**Response:**
```typescript
interface JobListResponse {
  jobs: ClusteringJob[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  _links: {
    self: string;
    next?: string;
    prev?: string;
  };
}
```

### Team Formation Endpoints

#### Generate Team Compositions

```http
POST /teams/generate
Content-Type: application/json
Authorization: Bearer <token>
```

**Request:**
```typescript
interface TeamGenerationRequest {
  clustering_job_id: string;
  num_teams: number;           // 2-10
  strategy?: string;           // "balanced", "specialized", "diverse"
  constraints?: TeamConstraint[];
  optimization_criteria?: {
    innovation_weight?: number;    // 0.0-1.0
    execution_weight?: number;     // 0.0-1.0
    harmony_weight?: number;       // 0.0-1.0
  };
  min_team_size?: number;      // Default: 3
  max_team_size?: number;      // Default: 8
}

// Example request
{
  "clustering_job_id": "job_abc123def456",
  "num_teams": 3,
  "strategy": "balanced",
  "constraints": [
    {
      "type": "department",
      "parameters": {
        "engineering": {"min": 1, "max": 3},
        "design": {"min": 1, "max": 2}
      },
      "weight": 0.8
    }
  ],
  "optimization_criteria": {
    "innovation_weight": 0.4,
    "execution_weight": 0.4,
    "harmony_weight": 0.2
  },
  "min_team_size": 4,
  "max_team_size": 6
}
```

**Response:**
```typescript
interface TeamGenerationResponse {
  generation_id: string;
  status: "completed";
  teams: TeamComposition[];
  overall_metrics: {
    average_balance_score: number;
    total_coverage_score: number;
    optimization_score: number;
  };
  alternatives?: TeamComposition[][]; // Alternative team configurations
  _links: {
    self: string;
    validate: string;
    export: string;
  };
}
```

#### Validate Team Composition

```http
POST /teams/validate
Content-Type: application/json
Authorization: Bearer <token>
```

**Request:**
```typescript
interface TeamValidationRequest {
  team_members: string[];      // Array of employee IDs
  clustering_job_id: string;
  validation_criteria: string[]; // ["balance", "performance", "compatibility"]
}
```

**Response:**
```typescript
interface TeamValidationResponse {
  validation_id: string;
  overall_score: number;       // 0.0-1.0
  detailed_scores: {
    balance_score: number;
    performance_prediction: number;
    compatibility_score: number;
    diversity_score: number;
  };
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  risk_factors: {
    level: "low" | "medium" | "high";
    factors: string[];
  };
}
```

### Health & Admin Endpoints

#### System Health

```http
GET /health
```

**Response:**
```typescript
interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string;
  version: string;
  environment: string;
  components: {
    database: ComponentHealth;
    redis: ComponentHealth;
    clustering_engine: ComponentHealth;
    file_storage: ComponentHealth;
  };
  metrics: {
    uptime_seconds: number;
    active_jobs: number;
    total_jobs_today: number;
    average_response_time_ms: number;
  };
}

interface ComponentHealth {
  status: "healthy" | "degraded" | "unhealthy";
  response_time_ms?: number;
  last_check: string;
  details?: Record<string, any>;
}
```

#### System Metrics (Admin Only)

```http
GET /admin/metrics
Authorization: Bearer <admin_token>
```

**Response:**
```typescript
interface SystemMetrics {
  timestamp: string;
  performance: {
    cpu_usage: number;           // 0.0-1.0
    memory_usage: number;        // 0.0-1.0  
    disk_usage: number;          // 0.0-1.0
    active_connections: number;
  };
  processing: {
    jobs_queued: number;
    jobs_processing: number;
    jobs_completed_today: number;
    average_processing_time_seconds: number;
    throughput_jobs_per_hour: number;
  };
  errors: {
    error_rate_24h: number;      // 0.0-1.0
    last_error: string;
    top_errors: Array<{
      type: string;
      count: number;
      last_occurrence: string;
    }>;
  };
}
```

## Error Handling

### Standard Error Response

```typescript
interface APIError {
  error: string;                 // Error code
  message: string;              // Human-readable description
  details?: Record<string, any>; // Additional error details
  timestamp: string;            // ISO 8601 datetime
  request_id: string;           // Unique request identifier
  documentation_url?: string;   // Link to relevant documentation
}

// Example error response
{
  "error": "validation_error",
  "message": "Invalid energy values: red_energy must be between 0 and 100",
  "details": {
    "field": "red_energy",
    "value": 150,
    "constraint": "range",
    "valid_range": [0, 100]
  },
  "timestamp": "2025-01-15T14:30:00Z",
  "request_id": "req_abc123def456",
  "documentation_url": "https://docs.insights.com/api/errors#validation_error"
}
```

### HTTP Status Codes

| Code | Status | Description | When to Use |
|------|--------|-------------|-------------|
| 200 | OK | Request successful | Successful GET, PUT, PATCH |
| 201 | Created | Resource created | Successful POST |
| 202 | Accepted | Request accepted for processing | Async operations |
| 204 | No Content | Successful request with no response body | DELETE operations |
| 400 | Bad Request | Invalid request format or data | Validation errors |
| 401 | Unauthorized | Missing or invalid authentication | Auth failures |
| 403 | Forbidden | Insufficient permissions | Authorization failures |
| 404 | Not Found | Resource doesn't exist | Missing resources |
| 409 | Conflict | Resource conflict | Duplicate resources |
| 422 | Unprocessable Entity | Request valid but semantically incorrect | Business logic errors |
| 429 | Too Many Requests | Rate limit exceeded | Rate limiting |
| 500 | Internal Server Error | Unexpected server error | Server-side errors |
| 502 | Bad Gateway | Upstream service error | Service dependencies |
| 503 | Service Unavailable | Service temporarily unavailable | Maintenance mode |

### Error Categories

#### Validation Errors
```typescript
interface ValidationError extends APIError {
  error: "validation_error";
  details: {
    field: string;
    value: any;
    constraint: string;
    message: string;
  };
}
```

#### Authentication Errors
```typescript
interface AuthError extends APIError {
  error: "authentication_error" | "authorization_error" | "token_expired";
  details?: {
    required_scope?: string[];
    token_type?: string;
    expires_at?: string;
  };
}
```

#### Resource Errors
```typescript
interface ResourceError extends APIError {
  error: "resource_not_found" | "resource_conflict" | "resource_gone";
  details: {
    resource_type: string;
    resource_id: string;
    available_actions?: string[];
  };
}
```

#### Rate Limiting Errors
```typescript
interface RateLimitError extends APIError {
  error: "rate_limit_exceeded";
  details: {
    limit: number;
    window_seconds: number;
    retry_after_seconds: number;
    current_usage: number;
  };
}
```

## Rate Limiting & Throttling

### Rate Limit Headers

```http
X-Rate-Limit-Limit: 1000        // Requests per hour
X-Rate-Limit-Remaining: 999     // Remaining requests
X-Rate-Limit-Reset: 1640995200  // Unix timestamp of reset
X-Rate-Limit-Window: 3600       // Window size in seconds
```

### Rate Limit Tiers

```typescript
interface RateLimitTier {
  tier: string;
  requests_per_hour: number;
  burst_limit: number;
  concurrent_requests: number;
}

const RATE_LIMIT_TIERS = {
  free: {
    tier: "free",
    requests_per_hour: 100,
    burst_limit: 10,
    concurrent_requests: 2
  },
  premium: {
    tier: "premium", 
    requests_per_hour: 1000,
    burst_limit: 50,
    concurrent_requests: 10
  },
  enterprise: {
    tier: "enterprise",
    requests_per_hour: 10000,
    burst_limit: 200,
    concurrent_requests: 50
  }
};
```

### Handling Rate Limits

```typescript
// Client-side rate limit handling
class APIClient {
  async makeRequest(url: string, options: RequestInit): Promise<Response> {
    const response = await fetch(url, options);
    
    if (response.status === 429) {
      const retryAfter = parseInt(response.headers.get('Retry-After') || '60');
      
      console.log(`Rate limit exceeded. Retrying after ${retryAfter} seconds.`);
      await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
      
      return this.makeRequest(url, options);
    }
    
    return response;
  }
}
```

## Webhooks & Events

### Webhook Configuration

```typescript
interface WebhookConfig {
  webhook_id: string;
  url: string;
  events: string[];
  secret?: string;          // For signature verification
  active: boolean;
  created_at: string;
  last_delivery?: WebhookDelivery;
}

interface WebhookDelivery {
  delivery_id: string;
  event_type: string;
  status: "success" | "failed";
  response_code?: number;
  delivered_at: string;
  next_retry?: string;
}
```

### Supported Events

```typescript
enum WebhookEvent {
  CLUSTERING_STARTED = "clustering.started",
  CLUSTERING_PROGRESS = "clustering.progress", 
  CLUSTERING_COMPLETED = "clustering.completed",
  CLUSTERING_FAILED = "clustering.failed",
  TEAM_GENERATED = "team.generated",
  ANALYSIS_EXPORTED = "analysis.exported",
  SYSTEM_MAINTENANCE = "system.maintenance"
}
```

### Webhook Payload Format

```typescript
interface WebhookPayload {
  event: WebhookEvent;
  timestamp: string;
  data: Record<string, any>;
  webhook_id: string;
  delivery_id: string;
}

// Example webhook payload
{
  "event": "clustering.completed",
  "timestamp": "2025-01-15T14:30:00Z",
  "data": {
    "job_id": "job_abc123def456",
    "status": "completed",
    "employee_count": 150,
    "cluster_count": 4,
    "processing_time_seconds": 127,
    "results_url": "https://api.insights.com/analytics/results/job_abc123def456"
  },
  "webhook_id": "wh_123456789",
  "delivery_id": "del_987654321"
}
```

### Webhook Security

```typescript
// Signature verification
function verifyWebhookSignature(
  payload: string, 
  signature: string, 
  secret: string
): boolean {
  const expectedSignature = crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
  
  return signature === `sha256=${expectedSignature}`;
}

// Example webhook handler
app.post('/webhooks/insights', (req, res) => {
  const signature = req.headers['x-insights-signature'];
  const payload = JSON.stringify(req.body);
  
  if (!verifyWebhookSignature(payload, signature, WEBHOOK_SECRET)) {
    return res.status(401).json({ error: 'Invalid signature' });
  }
  
  // Process webhook event
  handleWebhookEvent(req.body);
  res.status(200).json({ received: true });
});
```

## API Versioning

### URL-based Versioning

```http
GET /v1/analytics/jobs     // Version 1
GET /v2/analytics/jobs     // Version 2
```

### Header-based Versioning

```http
GET /analytics/jobs
Accept: application/vnd.insights.v1+json
```

### Version Compatibility Matrix

| Version | Status | Supported Until | Breaking Changes | Migration Guide |
|---------|--------|-----------------|------------------|----------------|
| v1 | Current | 2025-12-31 | None | N/A |
| v2 | Beta | TBD | New data models | [v1→v2 Migration](migrations/v1-to-v2.md) |

## OpenAPI Specification

### Complete OpenAPI 3.0 Specification

```yaml
openapi: 3.0.3
info:
  title: Observer Coordinator Insights API
  version: 1.0.0
  description: |
    Enterprise-grade neuromorphic clustering system for organizational analytics.
    
    ## Authentication
    This API uses OAuth 2.0 with JWT tokens for authentication.
    
    ## Rate Limiting
    API requests are rate limited based on your subscription tier.
    
    ## Support
    For API support, contact developers@terragon-labs.com
  contact:
    name: API Support
    email: developers@terragon-labs.com
    url: https://docs.insights.com
  license:
    name: Apache 2.0
    url: https://www.apache.org/licenses/LICENSE-2.0.html

servers:
  - url: https://api.insights.company.com/v1
    description: Production server
  - url: https://staging-api.insights.company.com/v1
    description: Staging server
  - url: http://localhost:8000/api
    description: Development server

security:
  - bearerAuth: []
  - oAuth2: ['analytics:read', 'analytics:create']

paths:
  /analytics/upload:
    post:
      summary: Upload and analyze employee data
      description: |
        Upload a CSV file containing employee personality data and start
        neuromorphic clustering analysis.
      operationId: uploadAnalytics
      tags:
        - Analytics
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              $ref: '#/components/schemas/UploadRequest'
      responses:
        '202':
          description: Analysis started successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UploadResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '413':
          $ref: '#/components/responses/PayloadTooLarge'
        '429':
          $ref: '#/components/responses/RateLimitExceeded'

  /analytics/results/{job_id}:
    get:
      summary: Get analysis results
      description: Retrieve clustering analysis results for a completed job
      operationId: getAnalysisResults
      tags:
        - Analytics
      parameters:
        - $ref: '#/components/parameters/JobId'
      responses:
        '200':
          description: Analysis results retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AnalysisResults'
        '404':
          $ref: '#/components/responses/NotFound'
        '202':
          description: Analysis still in progress
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobStatus'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    oAuth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: /auth/authorize
          tokenUrl: /auth/token
          scopes:
            analytics:read: Read analytics data
            analytics:create: Create analytics jobs
            teams:read: Read team data
            teams:create: Create team compositions
            admin:read: Read system metrics
            admin:write: Modify system configuration

  parameters:
    JobId:
      name: job_id
      in: path
      required: true
      schema:
        type: string
        pattern: '^job_[a-zA-Z0-9]{12}$'
      description: Unique job identifier
      example: job_abc123def456

  responses:
    BadRequest:
      description: Invalid request format or validation error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/APIError'
          example:
            error: validation_error
            message: "Invalid energy values: red_energy must be between 0 and 100"
            details:
              field: red_energy
              value: 150
    
    Unauthorized:
      description: Authentication failed or missing
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/APIError'
    
    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/APIError'
    
    RateLimitExceeded:
      description: Rate limit exceeded
      headers:
        X-Rate-Limit-Limit:
          schema:
            type: integer
          description: Request limit per hour
        X-Rate-Limit-Remaining:
          schema:
            type: integer
          description: Remaining requests in current window
        X-Rate-Limit-Reset:
          schema:
            type: integer
          description: Unix timestamp when rate limit resets
        Retry-After:
          schema:
            type: integer
          description: Seconds to wait before retrying
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/APIError'

  schemas:
    UploadRequest:
      type: object
      required:
        - file
      properties:
        file:
          type: string
          format: binary
          description: CSV file containing employee personality data
        n_clusters:
          type: integer
          minimum: 2
          maximum: 20
          default: 4
          description: Number of clusters to generate
        method:
          $ref: '#/components/schemas/ClusteringMethod'
        secure_mode:
          type: boolean
          default: false
          description: Enable data anonymization and enhanced security
        language:
          type: string
          enum: [en, de, es, fr, ja, zh]
          default: en
          description: Language for result interpretation
    
    UploadResponse:
      type: object
      properties:
        job_id:
          type: string
          pattern: '^job_[a-zA-Z0-9]{12}$'
          example: job_abc123def456
        status:
          type: string
          enum: [processing]
        message:
          type: string
          example: "Analysis started successfully"
        employee_count:
          type: integer
          minimum: 1
        estimated_completion:
          type: string
          format: date-time
        progress_url:
          type: string
          format: uri
    
    ClusteringMethod:
      type: string
      enum:
        - esn
        - snn
        - lsm
        - hybrid_reservoir
      description: |
        Neuromorphic clustering method:
        * `esn` - Echo State Network (fastest)
        * `snn` - Spiking Neural Network (noise resilient)
        * `lsm` - Liquid State Machine (complex patterns)
        * `hybrid_reservoir` - Combined approach (highest accuracy)
    
    APIError:
      type: object
      required:
        - error
        - message
        - timestamp
        - request_id
      properties:
        error:
          type: string
          description: Error code identifier
        message:
          type: string
          description: Human-readable error description
        details:
          type: object
          description: Additional error context
        timestamp:
          type: string
          format: date-time
        request_id:
          type: string
          description: Unique request identifier for debugging
        documentation_url:
          type: string
          format: uri
          description: Link to relevant documentation
```

This comprehensive API specification provides developers with everything needed to integrate with Observer Coordinator Insights, ensuring consistent, reliable, and secure interactions with the neuromorphic clustering platform.
#!/bin/bash
# Production Deployment Script for Observer Coordinator Insights
# Supports multiple deployment targets: Docker Compose, Kubernetes, and Cloud Providers

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_TARGET=${DEPLOYMENT_TARGET:-"docker-compose"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
VERSION=${VERSION:-"latest"}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Help function
show_help() {
    cat << EOF
Observer Coordinator Insights - Production Deployment Script

USAGE:
    ./deploy.sh [OPTIONS] [COMMAND]

COMMANDS:
    deploy          Deploy the application (default)
    build           Build application images
    test            Run deployment tests
    rollback        Rollback to previous version
    status          Check deployment status
    logs            View application logs
    scale           Scale services up/down
    backup          Create database backup
    restore         Restore from backup
    cleanup         Clean up unused resources

OPTIONS:
    -t, --target TARGET     Deployment target (docker-compose, kubernetes, aws, azure, gcp)
    -e, --env ENV          Environment (production, staging, development)
    -v, --version VERSION  Application version to deploy
    -c, --config FILE      Configuration file path
    -n, --namespace NS     Kubernetes namespace
    -r, --region REGION    Cloud provider region
    -d, --dry-run          Show what would be deployed without executing
    -h, --help             Show this help message

EXAMPLES:
    # Deploy with Docker Compose (default)
    ./deploy.sh

    # Deploy to Kubernetes
    ./deploy.sh -t kubernetes -n production

    # Deploy to AWS with specific version
    ./deploy.sh -t aws -v v4.0.0 -r us-west-2

    # Scale services
    ./deploy.sh scale --api=5 --worker=3

    # Check deployment status
    ./deploy.sh status

ENVIRONMENT VARIABLES:
    DEPLOYMENT_TARGET      Deployment target (docker-compose, kubernetes, aws, azure, gcp)
    ENVIRONMENT           Environment name (production, staging, development)
    VERSION               Application version
    DATABASE_PASSWORD     Database password
    REDIS_PASSWORD        Redis password
    SECRET_KEY            Application secret key
    ENCRYPTION_KEY        Data encryption key

EOF
}

# Prerequisites check
check_prerequisites() {
    local target="$1"
    
    log_info "Checking prerequisites for $target deployment..."
    
    # Common prerequisites
    command -v docker >/dev/null 2>&1 || log_error "Docker is not installed"
    command -v python3 >/dev/null 2>&1 || log_error "Python 3 is not installed"
    
    case "$target" in
        "docker-compose")
            command -v docker-compose >/dev/null 2>&1 || command -v docker >/dev/null 2>&1 || log_error "Docker Compose is not available"
            ;;
        "kubernetes")
            command -v kubectl >/dev/null 2>&1 || log_error "kubectl is not installed"
            command -v helm >/dev/null 2>&1 || log_warning "Helm is recommended for Kubernetes deployments"
            ;;
        "aws")
            command -v aws >/dev/null 2>&1 || log_error "AWS CLI is not installed"
            command -v kubectl >/dev/null 2>&1 || log_error "kubectl is not installed"
            ;;
        "azure")
            command -v az >/dev/null 2>&1 || log_error "Azure CLI is not installed"
            command -v kubectl >/dev/null 2>&1 || log_error "kubectl is not installed"
            ;;
        "gcp")
            command -v gcloud >/dev/null 2>&1 || log_error "Google Cloud SDK is not installed"
            command -v kubectl >/dev/null 2>&1 || log_error "kubectl is not installed"
            ;;
    esac
    
    log_success "Prerequisites check completed"
}

# Generate environment file
generate_env_file() {
    local env_file="$1"
    
    log_info "Generating environment file: $env_file"
    
    cat > "$env_file" << EOF
# Observer Coordinator Insights - Environment Configuration
ENVIRONMENT=${ENVIRONMENT}
VERSION=${VERSION}

# Database Configuration
DATABASE_USER=observer_user
DATABASE_PASSWORD=${DATABASE_PASSWORD:-$(openssl rand -base64 32)}
DATABASE_HOST=postgres
DATABASE_PORT=5432
DATABASE_NAME=observer_coordinator

# Redis Configuration
REDIS_PASSWORD=${REDIS_PASSWORD:-$(openssl rand -base64 32)}
REDIS_HOST=redis
REDIS_PORT=6379

# Application Security
SECRET_KEY=${SECRET_KEY:-$(openssl rand -base64 64)}
ENCRYPTION_KEY=${ENCRYPTION_KEY:-$(openssl rand -base64 32)}

# Monitoring
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-$(openssl rand -base64 16)}
VAULT_ROOT_TOKEN=${VAULT_ROOT_TOKEN:-$(openssl rand -hex 16)}

# Performance Tuning
API_WORKERS=4
WORKER_PROCESSES=2
CACHE_SIZE=10000
MAX_CONNECTIONS=100

# Feature Flags
ENABLE_METRICS=true
ENABLE_TRACING=true
ENABLE_AUDIT_LOGGING=true
ENABLE_AUTO_SCALING=true
EOF
    
    log_success "Environment file generated"
}

# Build application images
build_images() {
    log_info "Building application images..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    docker build \
        -f Dockerfile.production \
        -t "terragon/observer-coordinator-insights:${VERSION}" \
        -t "terragon/observer-coordinator-insights:latest" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="${VERSION}" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        .
    
    log_success "Images built successfully"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    local compose_file="$PROJECT_ROOT/deployment/docker-compose.production.yml"
    local env_file="$PROJECT_ROOT/.env.production"
    
    # Generate environment file if it doesn't exist
    if [[ ! -f "$env_file" ]]; then
        generate_env_file "$env_file"
    fi
    
    # Create required directories
    mkdir -p "$PROJECT_ROOT/volumes"/{postgres,redis,logs,prometheus,grafana,loki,vault,backups}
    
    # Deploy services
    cd "$PROJECT_ROOT"
    docker-compose -f "$compose_file" --env-file "$env_file" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    docker-compose -f "$compose_file" ps
    
    log_success "Docker Compose deployment completed"
    log_info "Application available at: http://localhost"
    log_info "Monitoring available at: http://localhost:3000 (admin/\$GRAFANA_ADMIN_PASSWORD)"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    local namespace="${KUBERNETES_NAMESPACE:-observer-coordinator-insights}"
    
    log_info "Deploying to Kubernetes namespace: $namespace"
    
    cd "$PROJECT_ROOT"
    
    # Apply namespace
    kubectl apply -f k8s/production/namespace.yaml
    
    # Create secrets
    create_kubernetes_secrets "$namespace"
    
    # Apply configurations
    kubectl apply -f k8s/production/configmap.yaml -n "$namespace"
    kubectl apply -f k8s/production/service.yaml -n "$namespace"
    kubectl apply -f k8s/production/deployment.yaml -n "$namespace"
    kubectl apply -f k8s/production/hpa.yaml -n "$namespace"
    
    # Wait for rollout
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/observer-coordinator-api -n "$namespace" --timeout=600s
    kubectl rollout status deployment/observer-coordinator-worker -n "$namespace" --timeout=600s
    kubectl rollout status deployment/observer-coordinator-scheduler -n "$namespace" --timeout=600s
    
    log_success "Kubernetes deployment completed"
    
    # Get service endpoints
    kubectl get services -n "$namespace"
}

# Create Kubernetes secrets
create_kubernetes_secrets() {
    local namespace="$1"
    
    log_info "Creating Kubernetes secrets..."
    
    # Check if secret already exists
    if kubectl get secret observer-coordinator-secrets -n "$namespace" >/dev/null 2>&1; then
        log_warning "Secrets already exist, skipping creation"
        return
    fi
    
    # Generate secrets
    local database_password="${DATABASE_PASSWORD:-$(openssl rand -base64 32)}"
    local redis_password="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
    local secret_key="${SECRET_KEY:-$(openssl rand -base64 64)}"
    local encryption_key="${ENCRYPTION_KEY:-$(openssl rand -base64 32)}"
    
    # Create secret
    kubectl create secret generic observer-coordinator-secrets \
        --from-literal=database-url="postgresql://observer_user:${database_password}@postgres:5432/observer_coordinator" \
        --from-literal=redis-url="redis://:${redis_password}@redis:6379/0" \
        --from-literal=secret-key="$secret_key" \
        --from-literal=encryption-key="$encryption_key" \
        -n "$namespace"
    
    log_success "Kubernetes secrets created"
}

# Deploy to AWS
deploy_aws() {
    local region="${AWS_REGION:-us-west-2}"
    
    log_info "Deploying to AWS region: $region"
    
    # Check AWS credentials
    aws sts get-caller-identity >/dev/null || log_error "AWS credentials not configured"
    
    # Deploy infrastructure with Terraform (if available)
    if [[ -d "$PROJECT_ROOT/infrastructure/aws" ]]; then
        log_info "Deploying AWS infrastructure..."
        cd "$PROJECT_ROOT/infrastructure/aws"
        terraform init
        terraform plan -var="region=$region" -var="environment=$ENVIRONMENT"
        terraform apply -var="region=$region" -var="environment=$ENVIRONMENT" -auto-approve
    fi
    
    # Deploy to EKS
    local cluster_name="observer-coordinator-${ENVIRONMENT}"
    aws eks update-kubeconfig --region "$region" --name "$cluster_name"
    
    # Use Kubernetes deployment
    KUBERNETES_NAMESPACE="observer-coordinator-${ENVIRONMENT}"
    deploy_kubernetes
    
    log_success "AWS deployment completed"
}

# Scale services
scale_services() {
    log_info "Scaling services..."
    
    case "$DEPLOYMENT_TARGET" in
        "docker-compose")
            # Docker Compose scaling
            local compose_file="$PROJECT_ROOT/deployment/docker-compose.production.yml"
            docker-compose -f "$compose_file" up -d --scale api="${API_REPLICAS:-3}" --scale worker="${WORKER_REPLICAS:-2}"
            ;;
        "kubernetes")
            local namespace="${KUBERNETES_NAMESPACE:-observer-coordinator-insights}"
            kubectl scale deployment observer-coordinator-api --replicas="${API_REPLICAS:-3}" -n "$namespace"
            kubectl scale deployment observer-coordinator-worker --replicas="${WORKER_REPLICAS:-2}" -n "$namespace"
            ;;
        *)
            log_error "Scaling not supported for deployment target: $DEPLOYMENT_TARGET"
            ;;
    esac
    
    log_success "Services scaled successfully"
}

# Check deployment status
check_status() {
    log_info "Checking deployment status..."
    
    case "$DEPLOYMENT_TARGET" in
        "docker-compose")
            local compose_file="$PROJECT_ROOT/deployment/docker-compose.production.yml"
            docker-compose -f "$compose_file" ps
            ;;
        "kubernetes")
            local namespace="${KUBERNETES_NAMESPACE:-observer-coordinator-insights}"
            kubectl get pods -n "$namespace"
            kubectl get services -n "$namespace"
            kubectl get hpa -n "$namespace"
            ;;
        *)
            log_error "Status check not supported for deployment target: $DEPLOYMENT_TARGET"
            ;;
    esac
}

# View application logs
view_logs() {
    log_info "Viewing application logs..."
    
    case "$DEPLOYMENT_TARGET" in
        "docker-compose")
            local compose_file="$PROJECT_ROOT/deployment/docker-compose.production.yml"
            docker-compose -f "$compose_file" logs -f --tail=100 api worker scheduler
            ;;
        "kubernetes")
            local namespace="${KUBERNETES_NAMESPACE:-observer-coordinator-insights}"
            kubectl logs -f -l app.kubernetes.io/name=observer-coordinator-insights -n "$namespace" --tail=100
            ;;
        *)
            log_error "Log viewing not supported for deployment target: $DEPLOYMENT_TARGET"
            ;;
    esac
}

# Create backup
create_backup() {
    log_info "Creating database backup..."
    
    local backup_name="observer_coordinator_backup_$(date +%Y%m%d_%H%M%S)"
    
    case "$DEPLOYMENT_TARGET" in
        "docker-compose")
            docker-compose -f "$PROJECT_ROOT/deployment/docker-compose.production.yml" exec postgres \
                pg_dump -U observer_user observer_coordinator > "./volumes/backups/${backup_name}.sql"
            ;;
        "kubernetes")
            local namespace="${KUBERNETES_NAMESPACE:-observer-coordinator-insights}"
            kubectl exec -it deployment/postgres -n "$namespace" -- \
                pg_dump -U observer_user observer_coordinator > "/tmp/${backup_name}.sql"
            kubectl cp "${namespace}/postgres-pod:/tmp/${backup_name}.sql" "./${backup_name}.sql"
            ;;
    esac
    
    log_success "Backup created: ${backup_name}.sql"
}

# Cleanup resources
cleanup() {
    log_info "Cleaning up unused resources..."
    
    # Docker cleanup
    docker system prune -f
    docker volume prune -f
    
    case "$DEPLOYMENT_TARGET" in
        "docker-compose")
            local compose_file="$PROJECT_ROOT/deployment/docker-compose.production.yml"
            docker-compose -f "$compose_file" down --remove-orphans
            ;;
        "kubernetes")
            local namespace="${KUBERNETES_NAMESPACE:-observer-coordinator-insights}"
            kubectl delete all --all -n "$namespace"
            ;;
    esac
    
    log_success "Cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment..."
    log_info "Target: $DEPLOYMENT_TARGET"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    # Check prerequisites
    check_prerequisites "$DEPLOYMENT_TARGET"
    
    # Build images
    if [[ "$DEPLOYMENT_TARGET" != "kubernetes" ]] || [[ "${BUILD_IMAGES:-true}" == "true" ]]; then
        build_images
    fi
    
    # Deploy based on target
    case "$DEPLOYMENT_TARGET" in
        "docker-compose")
            deploy_docker_compose
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
        "aws")
            deploy_aws
            ;;
        "azure")
            log_error "Azure deployment not yet implemented"
            ;;
        "gcp")
            log_error "GCP deployment not yet implemented"
            ;;
        *)
            log_error "Unknown deployment target: $DEPLOYMENT_TARGET"
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            DEPLOYMENT_TARGET="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -n|--namespace)
            KUBERNETES_NAMESPACE="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        deploy)
            COMMAND="deploy"
            shift
            ;;
        build)
            COMMAND="build"
            shift
            ;;
        test)
            COMMAND="test"
            shift
            ;;
        status)
            COMMAND="status"
            shift
            ;;
        logs)
            COMMAND="logs"
            shift
            ;;
        scale)
            COMMAND="scale"
            shift
            ;;
        backup)
            COMMAND="backup"
            shift
            ;;
        cleanup)
            COMMAND="cleanup"
            shift
            ;;
        --api=*)
            API_REPLICAS="${1#*=}"
            shift
            ;;
        --worker=*)
            WORKER_REPLICAS="${1#*=}"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            ;;
    esac
done

# Execute command
COMMAND=${COMMAND:-deploy}

case "$COMMAND" in
    deploy)
        deploy
        ;;
    build)
        check_prerequisites "$DEPLOYMENT_TARGET"
        build_images
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
    scale)
        scale_services
        ;;
    backup)
        create_backup
        ;;
    cleanup)
        cleanup
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        ;;
esac
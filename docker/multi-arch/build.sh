#!/bin/bash
# Multi-architecture Docker build script for global deployment

set -euo pipefail

# Configuration
REGISTRY="${DOCKER_REGISTRY:-your-registry.com/neuromorphic-clustering}"
VERSION="${VERSION:-4.0-global}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
}

# Check requirements
check_requirements() {
    log_info "Checking build requirements..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker buildx version &> /dev/null; then
        log_error "Docker Buildx is not available"
        exit 1
    fi
    
    # Check if buildx builder exists
    if ! docker buildx ls | grep -q "multiarch"; then
        log_info "Creating multiarch builder..."
        docker buildx create --name multiarch --driver docker-container --use
        docker buildx bootstrap
    else
        log_info "Using existing multiarch builder"
        docker buildx use multiarch
    fi
    
    log_success "Build requirements satisfied"
}

# Build single architecture image
build_single_arch() {
    local platform=$1
    local arch_suffix=$2
    
    log_info "Building for platform: $platform"
    
    docker buildx build \
        --platform "$platform" \
        --build-arg BUILDPLATFORM="linux/amd64" \
        --build-arg TARGETPLATFORM="$platform" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        --tag "$REGISTRY:$VERSION-$arch_suffix" \
        --tag "$REGISTRY:latest-$arch_suffix" \
        --file docker/multi-arch/Dockerfile \
        --push \
        ../../
    
    log_success "Built and pushed $platform image"
}

# Build multi-architecture image
build_multiarch() {
    log_info "Building multi-architecture image..."
    
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        --tag "$REGISTRY:$VERSION" \
        --tag "$REGISTRY:latest" \
        --file docker/multi-arch/Dockerfile \
        --push \
        ../../
    
    log_success "Built and pushed multi-architecture image"
}

# Build cloud-specific images
build_cloud_images() {
    log_info "Building cloud provider specific images..."
    
    # AWS optimized
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        --tag "$REGISTRY:$VERSION-aws" \
        --tag "$REGISTRY:aws" \
        --label "cloud.provider=aws" \
        --label "deployment.type=ecs-fargate" \
        --file docker/multi-arch/Dockerfile \
        --push \
        ../../
    
    # Azure optimized  
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        --tag "$REGISTRY:$VERSION-azure" \
        --tag "$REGISTRY:azure" \
        --label "cloud.provider=azure" \
        --label "deployment.type=container-instances" \
        --file docker/multi-arch/Dockerfile \
        --push \
        ../../
    
    # GCP optimized
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        --tag "$REGISTRY:$VERSION-gcp" \
        --tag "$REGISTRY:gcp" \
        --label "cloud.provider=gcp" \
        --label "deployment.type=cloud-run" \
        --file docker/multi-arch/Dockerfile \
        --push \
        ../../
    
    log_success "Built and pushed cloud-specific images"
}

# Scan images for vulnerabilities
scan_images() {
    log_info "Scanning images for vulnerabilities..."
    
    if command -v trivy &> /dev/null; then
        trivy image --exit-code 1 --severity HIGH,CRITICAL "$REGISTRY:$VERSION" || {
            log_warning "Vulnerability scan found issues"
            return 1
        }
        log_success "Vulnerability scan passed"
    else
        log_warning "Trivy not found, skipping vulnerability scan"
    fi
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    log_info "Generating SBOM..."
    
    if command -v syft &> /dev/null; then
        syft "$REGISTRY:$VERSION" -o spdx-json > "sbom-$VERSION.json"
        log_success "SBOM generated: sbom-$VERSION.json"
    else
        log_warning "Syft not found, skipping SBOM generation"
    fi
}

# Main build function
main() {
    local build_type=${1:-multiarch}
    
    log_info "Starting multi-architecture build process..."
    log_info "Registry: $REGISTRY"
    log_info "Version: $VERSION"
    log_info "Build Date: $BUILD_DATE"
    log_info "Git Commit: $GIT_COMMIT"
    
    check_requirements
    
    case $build_type in
        "amd64")
            build_single_arch "linux/amd64" "amd64"
            ;;
        "arm64")
            build_single_arch "linux/arm64" "arm64"
            ;;
        "multiarch")
            build_multiarch
            ;;
        "cloud")
            build_cloud_images
            ;;
        "all")
            build_multiarch
            build_cloud_images
            scan_images
            generate_sbom
            ;;
        *)
            log_error "Unknown build type: $build_type"
            echo "Usage: $0 [amd64|arm64|multiarch|cloud|all]"
            exit 1
            ;;
    esac
    
    log_success "Build process completed successfully!"
    
    # Display image information
    log_info "Built images:"
    docker buildx imagetools inspect "$REGISTRY:$VERSION" 2>/dev/null || log_warning "Could not inspect image"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
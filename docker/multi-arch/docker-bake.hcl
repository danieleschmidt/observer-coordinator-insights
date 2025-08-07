# Docker Bake configuration for multi-architecture builds
# Usage: docker buildx bake -f docker-bake.hcl

variable "REGISTRY" {
  default = "your-registry.com/neuromorphic-clustering"
}

variable "VERSION" {
  default = "4.0-global"
}

variable "BUILD_DATE" {
  default = ""
}

group "default" {
  targets = ["app"]
}

group "all-platforms" {
  targets = ["app-multiarch"]
}

target "app" {
  context = "../../"
  dockerfile = "docker/multi-arch/Dockerfile"
  tags = [
    "${REGISTRY}:${VERSION}",
    "${REGISTRY}:latest"
  ]
  labels = {
    "org.opencontainers.image.created" = "${BUILD_DATE}"
    "org.opencontainers.image.revision" = ""
    "org.opencontainers.image.version" = "${VERSION}"
  }
}

target "app-multiarch" {
  inherits = ["app"]
  platforms = [
    "linux/amd64",
    "linux/arm64"
  ]
  tags = [
    "${REGISTRY}:${VERSION}-multiarch",
    "${REGISTRY}:latest-multiarch"
  ]
}

target "app-dev" {
  inherits = ["app"]
  target = "base"
  tags = [
    "${REGISTRY}:${VERSION}-dev",
    "${REGISTRY}:dev"
  ]
  args = {
    BUILDKIT_INLINE_CACHE = 1
  }
}

# Cloud provider specific builds
target "aws" {
  inherits = ["app-multiarch"]
  tags = [
    "${REGISTRY}:${VERSION}-aws",
    "${REGISTRY}:aws"
  ]
  labels = {
    "cloud.provider" = "aws"
    "deployment.type" = "ecs-fargate"
  }
}

target "azure" {
  inherits = ["app-multiarch"]
  tags = [
    "${REGISTRY}:${VERSION}-azure",
    "${REGISTRY}:azure"
  ]
  labels = {
    "cloud.provider" = "azure"
    "deployment.type" = "container-instances"
  }
}

target "gcp" {
  inherits = ["app-multiarch"]
  tags = [
    "${REGISTRY}:${VERSION}-gcp", 
    "${REGISTRY}:gcp"
  ]
  labels = {
    "cloud.provider" = "gcp"
    "deployment.type" = "cloud-run"
  }
}
# Outputs for AWS Infrastructure

#----------------------------------
# Network Outputs
#----------------------------------

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnet_ids
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.vpc.database_subnet_ids
}

#----------------------------------
# Load Balancer Outputs
#----------------------------------

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.main.zone_id
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.main.arn
}

output "target_group_arn" {
  description = "ARN of the Target Group"
  value       = aws_lb_target_group.main.arn
}

#----------------------------------
# DNS and SSL Outputs
#----------------------------------

output "domain_name" {
  description = "Domain name of the application"
  value       = var.domain_name
}

output "application_url" {
  description = "URL of the application"
  value       = "https://${var.domain_name}"
}

output "certificate_arn" {
  description = "ARN of the SSL certificate"
  value       = aws_acm_certificate.main.arn
}

#----------------------------------
# ECS Outputs
#----------------------------------

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.main.name
}

output "ecs_task_definition_arn" {
  description = "ARN of the ECS task definition"
  value       = aws_ecs_task_definition.main.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task.arn
}

output "ecs_execution_role_arn" {
  description = "ARN of the ECS execution role"
  value       = aws_iam_role.ecs_task_execution.arn
}

#----------------------------------
# Database Outputs
#----------------------------------

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = false
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.main.port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = aws_db_instance.main.db_name
}

output "rds_username" {
  description = "RDS master username"
  value       = aws_db_instance.main.username
  sensitive   = true
}

output "rds_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.main.id
}

output "rds_replica_endpoint" {
  description = "RDS read replica endpoint"
  value       = var.create_read_replica ? aws_db_instance.read_replica[0].endpoint : null
}

#----------------------------------
# Redis Outputs
#----------------------------------

output "redis_endpoint" {
  description = "Redis cache endpoint"
  value       = aws_elasticache_cluster.main.cache_nodes[0].address
}

output "redis_port" {
  description = "Redis cache port"
  value       = aws_elasticache_cluster.main.cache_nodes[0].port
}

output "redis_cluster_endpoint" {
  description = "Redis cluster configuration endpoint"
  value       = var.create_redis_cluster ? aws_elasticache_replication_group.main[0].configuration_endpoint_address : null
}

output "redis_auth_token" {
  description = "Redis authentication token"
  value       = var.create_redis_cluster ? random_password.redis_auth_token[0].result : null
  sensitive   = true
}

#----------------------------------
# Storage Outputs
#----------------------------------

output "s3_alb_logs_bucket" {
  description = "S3 bucket for ALB access logs"
  value       = aws_s3_bucket.alb_logs.id
}

output "s3_app_storage_bucket" {
  description = "S3 bucket for application file storage"
  value       = aws_s3_bucket.app_storage.id
}

output "s3_alb_logs_bucket_arn" {
  description = "ARN of S3 bucket for ALB access logs"
  value       = aws_s3_bucket.alb_logs.arn
}

output "s3_app_storage_bucket_arn" {
  description = "ARN of S3 bucket for application storage"
  value       = aws_s3_bucket.app_storage.arn
}

#----------------------------------
# Security Outputs
#----------------------------------

output "kms_key_id" {
  description = "KMS key ID for encryption"
  value       = aws_kms_key.main.key_id
}

output "kms_key_arn" {
  description = "KMS key ARN for encryption"
  value       = aws_kms_key.main.arn
}

output "rds_kms_key_id" {
  description = "KMS key ID for RDS encryption"
  value       = aws_kms_key.rds.key_id
}

output "s3_kms_key_id" {
  description = "KMS key ID for S3 encryption"
  value       = aws_kms_key.s3.key_id
}

output "ecs_kms_key_id" {
  description = "KMS key ID for ECS encryption"
  value       = aws_kms_key.ecs.key_id
}

#----------------------------------
# Secrets Manager Outputs
#----------------------------------

output "app_secrets_arn" {
  description = "ARN of application secrets in Secrets Manager"
  value       = aws_secretsmanager_secret.app_secrets.arn
}

output "db_password_secret_arn" {
  description = "ARN of database password secret"
  value       = aws_secretsmanager_secret.db_password.arn
}

#----------------------------------
# Monitoring Outputs
#----------------------------------

output "cloudwatch_log_group_name" {
  description = "Name of the main CloudWatch log group"
  value       = aws_cloudwatch_log_group.main.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the main CloudWatch log group"
  value       = aws_cloudwatch_log_group.main.arn
}

#----------------------------------
# Service Discovery Outputs
#----------------------------------

output "service_discovery_namespace_id" {
  description = "ID of the service discovery namespace"
  value       = aws_service_discovery_private_dns_namespace.main.id
}

output "service_discovery_service_id" {
  description = "ID of the service discovery service"
  value       = aws_service_discovery_service.main.id
}

#----------------------------------
# Connection Information
#----------------------------------

output "database_url" {
  description = "Database connection URL"
  value       = "postgresql://${aws_db_instance.main.username}:${random_password.db_password.result}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
  sensitive   = true
}

output "redis_url" {
  description = "Redis connection URL"
  value       = var.create_redis_cluster ? "rediss://${aws_elasticache_replication_group.main[0].configuration_endpoint_address}:6379" : "redis://${aws_elasticache_cluster.main.cache_nodes[0].address}:${aws_elasticache_cluster.main.cache_nodes[0].port}"
  sensitive   = false
}

#----------------------------------
# Resource Information
#----------------------------------

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "project_name" {
  description = "Project name"
  value       = var.project_name
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "availability_zones" {
  description = "Availability zones used"
  value       = local.azs
}

#----------------------------------
# Cost and Capacity Information
#----------------------------------

output "ecs_service_desired_count" {
  description = "Desired count of ECS tasks"
  value       = var.ecs_service_desired_count
}

output "ecs_autoscale_min_capacity" {
  description = "Minimum ECS autoscale capacity"
  value       = var.ecs_autoscale_min_capacity
}

output "ecs_autoscale_max_capacity" {
  description = "Maximum ECS autoscale capacity"
  value       = var.ecs_autoscale_max_capacity
}

output "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  value       = var.rds_allocated_storage
}

output "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  value       = var.rds_max_allocated_storage
}

#----------------------------------
# Deployment Information
#----------------------------------

output "deployment_summary" {
  description = "Summary of deployed resources"
  value = {
    environment           = var.environment
    project_name         = var.project_name
    aws_region           = var.aws_region
    application_url      = "https://${var.domain_name}"
    ecs_cluster          = aws_ecs_cluster.main.name
    ecs_service          = aws_ecs_service.main.name
    rds_instance         = aws_db_instance.main.id
    redis_cluster        = var.create_redis_cluster ? aws_elasticache_replication_group.main[0].id : aws_elasticache_cluster.main.cluster_id
    s3_app_bucket        = aws_s3_bucket.app_storage.id
    load_balancer        = aws_lb.main.dns_name
    ssl_certificate      = aws_acm_certificate.main.arn
    vpc_id              = module.vpc.vpc_id
  }
}

#----------------------------------
# Security and Compliance Information
#----------------------------------

output "security_summary" {
  description = "Summary of security configurations"
  value = {
    encryption_at_rest_enabled    = true
    encryption_in_transit_enabled = true
    kms_keys_created             = [
      aws_kms_key.main.key_id,
      aws_kms_key.rds.key_id,
      aws_kms_key.s3.key_id,
      aws_kms_key.ecs.key_id
    ]
    deletion_protection_enabled = var.enable_deletion_protection
    backup_retention_days       = var.rds_backup_retention_period
    ssl_certificate_arn         = aws_acm_certificate.main.arn
    vpc_private_subnets        = length(module.vpc.private_subnet_ids)
    security_groups_created    = [
      aws_security_group.alb.id,
      aws_security_group.ecs_tasks.id,
      aws_security_group.rds.id,
      aws_security_group.redis.id
    ]
  }
}
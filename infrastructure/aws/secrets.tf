# Secrets Manager resources for AWS Infrastructure

#----------------------------------
# Application Secrets
#----------------------------------

resource "aws_secretsmanager_secret" "app_secrets" {
  name        = "${local.name_prefix}-app-secrets"
  description = "Application secrets for Observer Coordinator Insights"
  
  kms_key_id              = aws_kms_key.main.arn
  recovery_window_in_days = var.enable_deletion_protection ? 30 : 0

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    jwt_secret_key           = random_password.jwt_secret.result
    api_key                 = random_password.api_key.result
    encryption_key          = random_password.encryption_key.result
    redis_auth_token        = var.create_redis_cluster ? random_password.redis_auth_token[0].result : ""
    openai_api_key          = var.openai_api_key != "" ? var.openai_api_key : ""
    smtp_password           = var.smtp_password != "" ? var.smtp_password : ""
    webhook_secret          = random_password.webhook_secret.result
    session_secret          = random_password.session_secret.result
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# Database password secret
resource "aws_secretsmanager_secret" "db_password" {
  name        = "${local.name_prefix}-db-password"
  description = "Database password for Observer Coordinator Insights"
  
  kms_key_id              = aws_kms_key.main.arn
  recovery_window_in_days = var.enable_deletion_protection ? 30 : 0

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id = aws_secretsmanager_secret.db_password.id
  secret_string = jsonencode({
    username = aws_db_instance.main.username
    password = random_password.db_password.result
    engine   = "postgres"
    host     = aws_db_instance.main.endpoint
    port     = aws_db_instance.main.port
    dbname   = aws_db_instance.main.db_name
  })
}

#----------------------------------
# Random Passwords and Keys
#----------------------------------

resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

resource "random_password" "api_key" {
  length  = 32
  special = false
  upper   = true
  lower   = true
  numeric = true
}

resource "random_password" "encryption_key" {
  length  = 32
  special = false
  upper   = true
  lower   = true
  numeric = true
}

resource "random_password" "webhook_secret" {
  length  = 32
  special = true
}

resource "random_password" "session_secret" {
  length  = 48
  special = true
}

#----------------------------------
# Secrets Rotation Configuration
#----------------------------------

resource "aws_secretsmanager_secret_rotation" "app_secrets" {
  count = var.enable_secrets_rotation ? 1 : 0
  
  secret_id           = aws_secretsmanager_secret.app_secrets.id
  rotation_lambda_arn = aws_lambda_function.secrets_rotation[0].arn

  rotation_rules {
    automatically_after_days = 90
  }
}

resource "aws_secretsmanager_secret_rotation" "db_password" {
  count = var.enable_secrets_rotation ? 1 : 0
  
  secret_id           = aws_secretsmanager_secret.db_password.id
  rotation_lambda_arn = aws_lambda_function.db_rotation[0].arn

  rotation_rules {
    automatically_after_days = 60
  }
}

#----------------------------------
# Lambda Functions for Rotation
#----------------------------------

resource "aws_lambda_function" "secrets_rotation" {
  count = var.enable_secrets_rotation ? 1 : 0
  
  filename      = "secrets_rotation.zip"
  function_name = "${local.name_prefix}-secrets-rotation"
  role          = aws_iam_role.secrets_rotation_role[0].arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.11"
  timeout       = 30

  environment {
    variables = {
      SECRETS_MANAGER_ENDPOINT = "https://secretsmanager.${var.aws_region}.amazonaws.com"
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_function" "db_rotation" {
  count = var.enable_secrets_rotation ? 1 : 0
  
  filename      = "db_rotation.zip"
  function_name = "${local.name_prefix}-db-rotation"
  role          = aws_iam_role.secrets_rotation_role[0].arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.11"
  timeout       = 30

  environment {
    variables = {
      SECRETS_MANAGER_ENDPOINT = "https://secretsmanager.${var.aws_region}.amazonaws.com"
      RDS_INSTANCE_IDENTIFIER  = aws_db_instance.main.id
    }
  }

  vpc_config {
    subnet_ids         = module.vpc.private_subnet_ids
    security_group_ids = [aws_security_group.lambda_rotation[0].id]
  }

  tags = local.common_tags
}

resource "aws_security_group" "lambda_rotation" {
  count = var.enable_secrets_rotation ? 1 : 0
  
  name_prefix = "${local.name_prefix}-lambda-rotation"
  description = "Security group for Lambda rotation functions"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.rds.id]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-lambda-rotation-sg"
  })
}

#----------------------------------
# Lambda Permissions for Secrets Manager
#----------------------------------

resource "aws_lambda_permission" "allow_secrets_manager_app" {
  count = var.enable_secrets_rotation ? 1 : 0
  
  statement_id  = "AllowExecutionFromSecretsManager"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.secrets_rotation[0].function_name
  principal     = "secretsmanager.amazonaws.com"
}

resource "aws_lambda_permission" "allow_secrets_manager_db" {
  count = var.enable_secrets_rotation ? 1 : 0
  
  statement_id  = "AllowExecutionFromSecretsManager"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.db_rotation[0].function_name
  principal     = "secretsmanager.amazonaws.com"
}

#----------------------------------
# Parameter Store for Non-Secret Configuration
#----------------------------------

resource "aws_ssm_parameter" "app_config" {
  for_each = {
    "/${local.name_prefix}/config/environment"     = var.environment
    "/${local.name_prefix}/config/log_level"       = var.log_level
    "/${local.name_prefix}/config/debug_mode"      = tostring(var.debug_mode)
    "/${local.name_prefix}/config/max_workers"     = tostring(var.max_workers)
    "/${local.name_prefix}/config/timeout"         = tostring(var.request_timeout)
    "/${local.name_prefix}/config/domain_name"     = var.domain_name
    "/${local.name_prefix}/config/s3_bucket"       = aws_s3_bucket.app_storage.id
    "/${local.name_prefix}/config/redis_endpoint"  = var.create_redis_cluster ? aws_elasticache_replication_group.main[0].configuration_endpoint_address : aws_elasticache_cluster.main.cache_nodes[0].address
  }

  name  = each.key
  type  = "String"
  value = each.value

  tags = local.common_tags
}

resource "aws_ssm_parameter" "secure_config" {
  for_each = {
    "/${local.name_prefix}/config/database_url" = "postgresql://${aws_db_instance.main.username}:${random_password.db_password.result}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
  }

  name  = each.key
  type  = "SecureString"
  value = each.value
  key_id = aws_kms_key.main.arn

  tags = local.common_tags
}
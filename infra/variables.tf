variable "name" {
  description = "The name of the stack"
  default     = "web"
}

variable "environment" {
  description = "the name of your environment, e.g. \"prod\""
  default     = "dev"
}

variable "region" {
  description = "the AWS region in which resources are created, you must set the availability_zones variable as well if you define this value to something other than the default"
  default     = "eu-west-2"
}

variable "availability_zones" {
  description = "a comma-separated list of availability zones, defaults to all AZ of the region, if set to something other than the defaults, both private_subnets and public_subnets have to be defined as well"
  default     = ["eu-west-2a", "eu-west-2b"]
}

variable "cidr" {
  description = "The CIDR block for the VPC."
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "a list of CIDRs for private subnets in your VPC, must be set if the cidr variable is defined, needs to have as many elements as there are availability zones"
  default     = ["10.0.0.0/20", "10.0.32.0/20"]
}

variable "public_subnets" {
  description = "a list of CIDRs for public subnets in your VPC, must be set if the cidr variable is defined, needs to have as many elements as there are availability zones"
  default     = ["10.0.16.0/20", "10.0.48.0/20"]
}


variable "key_path" {
  description = "path to pulic key for jumping between ec2s in public and private network"
  default     = "~/.ssh/aws-dev-key.pub"
}


variable "domain_name" {

  description = "Domain name to be used"
  default     = "cbay.io"
}

variable "db_password" {
  description = "Database root password"
  default     = "rootrootroot"
}
variable "db_user" {
  description = "Database root user"
  default     = "root"
}
variable "db_name" {
  description = "Database name"
  default     = "django"
}

variable "target_group_port" {
  description = "Port to use for load balance to balance load across"
  default     = 80

}
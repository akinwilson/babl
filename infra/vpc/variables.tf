variable "name" {
  description = "Name of service"
}

variable "environment" {
  description = "the name of your environment, e.g. \"dev\", \" staging \" or \"prod\""
}

variable "cidr" {
  description = "The CIDR block for the VPC."
}

variable "public_subnets" {
  description = "List of public subnets"
}

variable "region" {
  description = "region for s3 bucket"
}

variable "availability_zones" {
  description = "List of availability zones"
}

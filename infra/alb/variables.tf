variable "name" {
  description = "name of of stack module"
}


variable "environment" {
  description = "the name of your environment, e.g. \"dev\", \"staging\" and \"prod\""
}

variable "public_subnets" {
  description = "Comma separated list of subnet IDs"
}

variable "vpc_id" {
  description = "VPC ID"
}

variable "sg" {
  description = "Comma separated list of security groups"
}

variable "webserver" {
  description = "ec2 instance of webserver in private subnets"
}

variable "target_group_port" {
  description = "Port for load balance across of targets"
}

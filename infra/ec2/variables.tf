variable "name" {
  description = "Name of stack module"
}

variable "environment" {
  description = "the name of your environment, e.g. 'prod' or 'dev'"
}

variable "vpc_id" {
  description = "The vpc to launch the EC2 instance into"
}

variable "availability_zones" {
  description = "a comma-separated list of availability zones, defaults to all AZ of the region, if set to something other than the defaults, both private_subnets and public_subnets have to be defined as well"
  default     = ["eu-west-2a", "eu-west-2b"]
}

variable "pub_subnet" {
  description = "The subnet (public) to launch the EC2 instance into"
}


variable "key_path" {
  description = "Local path to private shh key for ssh-forwarding into ec2 in private network"
}

variable "sg_jumper" {
  description = "jumper ec2 security group"
}
variable "sg_web" {
  description = "web ec2 security group"
}

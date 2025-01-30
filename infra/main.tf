terraform {
  backend "s3" {
    bucket = var.tf_bucket
    key    = "terraform-svc"
    region = "eu-west-2"
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "3.73.0"
    }
  }
  required_version = ">=1.1.0"
}

provider "aws" {
  region = var.region
}

module "vpc" {
  source             = "./vpc"
  name               = var.name
  cidr               = var.cidr
  region             = var.region
  public_subnets     = var.public_subnets
  availability_zones = var.availability_zones
  environment        = var.environment
}

module "sg" {
  source      = "./sg"
  name        = var.name
  environment = var.environment
  vpc_id      = module.vpc.id
}


module "alb" {
  source            = "./alb"
  name              = var.name
  environment       = var.environment
  target_group_port = var.target_group_port
  public_subnets    = module.vpc.public
  webserver         = module.ec2.webserver
  vpc_id            = module.vpc.id
  sg                = [module.sg.alb]
}



module "ec2" {
  source      = "./ec2"
  name        = var.name
  environment = var.environment
  key_path    = var.key_path
  instance_type = var.instance_type
  vpc_id      = module.vpc.id
  pub_subnet  = module.vpc.public
  sg_jumper   = module.sg.ec2_jumper
  sg_web      = module.sg.ec2_web
}


module "ecr" {
  source      = "./ecr"
  name        = var.name
  environment = var.environment

}


data "aws_iam_policy_document" "iam_policy_document" {
  statement {
    sid     = "AllowSpecificS3FullAccess"
    actions = ["s3:*"]
    effect  = "Allow"
    resources = [
      "arn:aws:s3:::*/*",
      "arn:aws:s3:::*",
      "arn:aws:s3:::${var.tf_bucket}",
    ]
  }

  statement {
    sid = "AllowSecurityGroups"
    actions = [
      "ec2:DescribeSecurityGroups",
      "ec2:DescribeSecurityGroupsRules",
      "ec2:DescribeTags",
      "ec2:CreateTags",
      "ec2:AuthorizeSecurityGroupIngress",
      "ec2:RevokeSecurityGroupIngress",
      "ec2:AuthorizeSecurityGroupEgress",
      "ec2:ModifySecurityGroupEgress",
      "ec2:ModifySecurityGroupRuleDescriptionIngress",
      "ec2:ModifySecurityGroupRuleDescriptionEgress",
      "ec2:ModifySecurityGroupRules",
      "ec2:CreateSecurityGroup"
    ]
    effect    = "Allow"
    resources = ["*"]
  }
  statement {
    sid = "AllowEC2"
    actions = [
      "ec2:*"
    ]
    effect    = "Allow"
    resources = ["*"]
  }
  statement {
    sid = "AllowIAM"
    actions = [
      "iam:*"
    ]
    effect    = "Allow"
    resources = ["*"]
  }
}

resource "aws_iam_policy" "iam_policy" {
  name   = "terraform-iam-policy"
  path   = "/"
  policy = data.aws_iam_policy_document.iam_policy_document.json
}

resource "aws_iam_user" "terraform_agent_user" {
  name = "terraform_agent_user"
}

resource "aws_iam_user_policy_attachment" "tf_attach" {
  user       = aws_iam_user.terraform_agent_user.name
  policy_arn = aws_iam_policy.iam_policy.arn
}

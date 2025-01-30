
resource "aws_security_group" "alb" {
  name   = "${var.name}-sg-alb-${var.environment}"
  vpc_id = var.vpc_id
  ingress {
    protocol         = "tcp"
    from_port        = 80
    to_port          = 80
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  ingress {
    protocol         = "tcp"
    from_port        = 443
    to_port          = 443
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  egress {
    protocol         = "-1"
    from_port        = 0
    to_port          = 0
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  tags = {
    Name        = "${var.name}-sg-alb-${var.environment}"
    Environment = var.environment
  }
}


resource "aws_security_group" "ec2_web" {
  name   = "${var.name}-sg-ec2-web-${var.environment}"
  vpc_id = var.vpc_id
  ingress {
    protocol        = "-1"
    from_port       = 0
    to_port         = 0
    security_groups = ["${aws_security_group.alb.id}"]
  }
  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    // Do not use this in production, should be limited to your own IP
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    protocol         = "-1"
    from_port        = 0
    to_port          = 0
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  tags = {
    Name        = "${var.name}-sg-ec2-web-${var.environment}"
    Environment = var.environment
  }
}


resource "aws_security_group" "ec2_jumper" {
  name   = "${var.name}-sg-ec2-jumper-${var.environment}"
  vpc_id = var.vpc_id
  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    // Do not use this in production, should be limited to your own IP
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    protocol         = "-1"
    from_port        = 0
    to_port          = 0
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  tags = {
    Name        = "${var.name}-sg-ec2-jumper-${var.environment}"
    Environment = var.environment
  }
}
output "alb" {
  value = aws_security_group.alb.id
}


output "ec2_jumper" {
  value = aws_security_group.ec2_jumper.id
}


output "ec2_web" {
  value = aws_security_group.ec2_web.id
}

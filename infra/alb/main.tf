## application load balance to acccess babl frontend through a url 

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/lb_target_group
resource "aws_lb_target_group" "front" {
  name     = "application-front"
  port     = 80
  protocol = "HTTP"
  vpc_id   = var.vpc_id
  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval            = 10
    matcher             = 200
    path                = "/"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 3
    unhealthy_threshold = 2
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/lb_target_group_attachment
resource "aws_lb_target_group_attachment" "main" {
  count            = length(var.webserver)
  target_group_arn = aws_lb_target_group.front.arn
  target_id        = element(var.webserver.*.id, count.index)
  port             = var.target_group_port # 1337
}


resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.front.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.front.arn
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/lb
resource "aws_lb" "front" {
  name               = "${var.name}-front-lb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = var.sg
  subnets            = var.public_subnets.*.id

  enable_deletion_protection = false

  tags = {
    Name        = "${var.name}-alb-${var.environment}"
    Environment = var.environment
  }
}
output "alb" {
  value = aws_lb.front
}

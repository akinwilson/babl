resource "aws_key_pair" "main" {
  key_name   = "aws-dev-key"
  public_key = file(var.key_path)
}

data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"] # Canonical
}

# https://medium.com/adessoturkey/how-to-connect-to-private-ec2-instance-database-via-bastion-host-5b05a256f9f7
resource "aws_instance" "web" {
  ami                    = data.aws_ami.ubuntu.id
  # count                  = length(var.availability_zones)
  count                  = 1 
  instance_type          = "t2.micro"
  subnet_id              = var.priv_subnet[count.index].id
  vpc_security_group_ids = [var.sg_web]
  key_name               = aws_key_pair.main.key_name
  user_data              = file("${path.root}/${path.module}/${var.environment}_startup.sh")


  tags = {
    Name        = "${var.name}-priv-ec2-${var.environment}-${count.index}"
    Environment = var.environment
  }
}


resource "aws_instance" "jumper" {

  ami = data.aws_ami.ubuntu.id
  # only add jumper for dev envrionment
  # count                  = var.environment == "dev" ? length(var.availability_zones) : 0
  count                  = var.environment == "dev" ? 1 : 0
  instance_type          = "t2.micro"
  subnet_id              = var.pub_subnet[count.index].id
  vpc_security_group_ids = [var.sg_jumper]
  key_name               = aws_key_pair.main.key_name
  user_data              = file("${path.root}/${path.module}/${var.environment}_startup.sh")
  tags = {
    Name        = "${var.name}-pub-ec2-${var.environment}-${count.index}"
    Environment = var.environment
  }
}
output "webserver" {
  value = aws_instance.web

}

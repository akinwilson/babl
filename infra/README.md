## Infrastructure for remote fitting jobs 

## Overview 

This repository contains the infrastructure as code to provision an [EC2](https://aws.amazon.com/pm/ec2/?gclid=CjwKCAiAneK8BhAVEiwAoy2HYRJ80Nf5rCxwkctj5RfldsFhHREjb7a2I3d1mB2irtQkLJnRmlUmwBoCMXoQAvD_BwE&trk=c49c9dff-8619-4ee1-8d47-8585eb10c61e&sc_channel=ps&ef_id=CjwKCAiAneK8BhAVEiwAoy2HYRJ80Nf5rCxwkctj5RfldsFhHREjb7a2I3d1mB2irtQkLJnRmlUmwBoCMXoQAvD_BwE:G:s&s_kwcid=AL!4422!3!638364387973!e!!g!!ec2!19090032168!140900569821)  on [AWS](https://aws.amazon.com/free/?gclid=CjwKCAiAneK8BhAVEiwAoy2HYarfGp5s2zmOntzYKVk4GZzDUoqVBT_OTMyRcaph4714VAcTKDzgvBoCGH0QAvD_BwE&trk=d5254134-67ca-4a35-91cc-77868c97eedd&sc_channel=ps&ef_id=CjwKCAiAneK8BhAVEiwAoy2HYarfGp5s2zmOntzYKVk4GZzDUoqVBT_OTMyRcaph4714VAcTKDzgvBoCGH0QAvD_BwE:G:s&s_kwcid=AL!4422!3!433803620858!e!!g!!aws!1680401428!67152600164) to allow for the fitting job of deep learning models to occur remotely rather than locally.

Deep learning models may be train via two execution providers; the [CPU](https://en.wikipedia.org/wiki/Central_processing_unit) or [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit). Both have their advantages and drawbacks. 



With using the CPU, as long as your available amount of [RAM](https://en.wikipedia.org/wiki/Random-access_memory) memory  permits loading the model into memory with enough leeway to account for additional gradient parameters to be stored during fitting, you are able to train your deep learning models using it as an execution provider, that being at a considerable slower rate when compared to using the GPU as an execution provider. 

With using the GPU(s), you may see a substantial speed up in fitting time of your models, and your RAM memory requirements may shrink but this is at  the cost of requiring more [VRAM](https://en.wikipedia.org/wiki/Video_random-access_memory), which on a per unit comparison (RAM versus VRAM), is far more costly.


This repository is for those with locally resource-contraint environments, whether it be RAM, VRAM or CPU and/or GPU [clock rate](https://en.wikipedia.org/wiki/Clock_rate), and wish to execute their fitting routines in an environemnt with access to more desirable resources without a concern for expenditure associated to renting node(s).  



## Usage 

To start, run the followng script to ensure you have all the required CLI tools: 
```
./check_cli_tools.sh
```
Next, review the `aws_env_vars` file and tailor it to your AWS account. 

Once, happy with the tailored file, to export its content as environment variables run 
```
export $(cat aws_env_args | xargs)
```
**Note** you are free to call your [terraform state version control](https://developer.hashicorp.com/terraform/language/state) bucket anything you would like with in the constraints AWS places on bucket names, but obviously a semantically logical name is perferable for a seperation of concerns. 

We want to create a `.tfvars` file in order to make use of these inside of our terraform files. From the command line, run:

```
cat <<EOF > $AWS_ACCOUNT.tfvars
name="${AWS_ACCOUNT}"
region="${AWS_REGION}"
environment="${STAGE}"
EOF
```
This will create the `AWS_ACCONT.tfvars` file which `terraform` will as an input. 


Next, provision the state-store bucket with:
```
./utils/create-s3-tf-backend-bucket.sh 
```










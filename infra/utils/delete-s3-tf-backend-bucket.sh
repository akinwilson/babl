#! /bin/bash
set -e

AWS_PROFILE=$AWS_PROFILE
AWS_ACCOUNT=$AWS_ACCOUNT
AWS_REGION=$AWS_REGION
AWS_TF_BUCKET=$AWS_TF_BUCKET
echo ""
echo "Deleting state store bucket of terraform..."
echo ""
echo "aws account: ${AWS_ACCOUNT}"
echo "aws profile: ${AWS_PROFILE}"
echo "aws region: ${AWS_REGION}"
echo "aws s3 bucket name: ${AWS_BUCKET}"

# Delete objects inside state-store bucket 
# NOTE: using conditional operator || first try removing object of bucket, if param validation fails due to no 
# objects being present inside bucket, then force remove bucket
aws s3api delete-objects --bucket $AWS_BUCKET --delete "$(aws s3api list-object-versions --bucket iac-svc --query='{Objects: Versions[].{Key:Key,VersionId:VersionId}}')" > /dev/null && aws s3 rb s3://$AWS_BUCKET --force || aws s3 rb s3://$AWS_BUCKET --force
echo "" 
echo "Finished cleaning up backend of terrform state store"
echo ""
exit 1



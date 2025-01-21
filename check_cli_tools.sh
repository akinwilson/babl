#!/bin/sh

GREEN='\033[0;32m'
LIGHTPURPLE='\033[1;35m'
LIGHTCYAN='\033[1;36m'
YELLOW='\033[1;33m'
# reset 
RESET='\033[0m'


project_title=$(cat <<"EOF"

 _           _     _ 
| |__   __ _| |__ | |
| '_ \ / _` | '_ \| |
| |_) | (_| | |_) | |
|_.__/ \__,_|_.__/|_|
EOF
)

echo "${YELLOW}${project_title}${RESET}\n"


# checking if verion returned is null with -z 
if [ -z "$(docker -v)" ]; then 
  echo "Unable to find docker"
  echo -e "To install docker, please follow this guide: ${GREEN}https://docs.docker.com/get-docker${RESET}"
  exit 1 
fi
# checking if docker-compose is available 
if [ -x "$(command -v docker-compose)" ]; then
  echo "Unable to find docker-compose"
  echo "To install docker-compose, please follow this guide: ${GREEN}https://docs.docker.com/compose/install/linux/${RESET}"
  exit 1
fi 



echo "\nYou have all the required CLI tools for  ${LIGHTPURPLE}local deployment${RESET}.\n\nReady to begin.\n\n"



#!/bin/bash
 
# Usage:
# ./create_user.sh username password
 
USERNAME=$1
PASSWORD=$2
 
# Check if username provided
if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "Usage: $0 <username> <password>"
    exit 1
fi
 
# Check if user already exists
if id "$USERNAME" &>/dev/null; then
    echo "User '$USERNAME' already exists."
    exit 1
fi
 
# Create user with home directory
sudo useradd -m "$USERNAME"
 
# Set password
echo "${USERNAME}:${PASSWORD}" | sudo chpasswd
 
echo "User '$USERNAME' created successfully."

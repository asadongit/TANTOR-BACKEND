#!/bin/bash

#!/bin/bash

# Usage:
# ./delete_user.sh username

USERNAME=$1

if [ -z "$USERNAME" ]; then
    echo "Usage: $0 <username>"

USERNAME=$1

if [ -z "$USERNAME" ]; then
    echo "Usage: $0 <username>"
    exit 1
fi


# Check if user exists
if ! id "$USERNAME" &>/dev/null; then
    echo "User '$USERNAME' does not exist."
    exit 1
fi


# Delete user with home directory
sudo userdel -r "$USERNAME"

echo "User '$USERNAME' deleted successfully."
sudo userdel -r "$USERNAME"

echo "User '$USERNAME' deleted successfully."
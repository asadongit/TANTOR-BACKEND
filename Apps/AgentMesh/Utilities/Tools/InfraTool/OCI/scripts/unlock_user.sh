#!/bin/bash

# Usage:
# ./unlock_user.sh username

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

# Unlock user account
sudo passwd -u "$USERNAME"

echo "User '$USERNAME' unlocked successfully."
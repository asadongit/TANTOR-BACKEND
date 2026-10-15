#!/bin/bash
USER_OCID=$1
TENANCY_OCID=$2
FINGERPRINT=$3
REGION=$4
B64_PRIVATE_KEY=$5
NSG_OCID=$6

if [ -z "$NSG_OCID" ]; then
    echo "Error: Missing required arguments."
    exit 1
fi

echo "========================================"
echo "OCI CLI NSG VM Detachment over SSH"
echo "========================================"

# 1. Install OCI CLI if not present
if ! command -v oci &> /dev/null; then
    echo "OCI CLI not found. Installing OCI CLI..."
    if command -v yum &> /dev/null; then
        sudo yum install -y python3-oci-cli
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3-oci-cli
    else
        echo "Error: Neither yum nor apt-get found. Cannot install OCI CLI automatically."
        exit 1
    fi
fi

if ! command -v oci &> /dev/null; then
    echo "Error: OCI CLI installation failed."
    exit 1
fi

# 2. Get local VNIC ID using metadata service
echo "Querying OCI local metadata service for VNIC ID..."
VNIC_JSON=$(curl -s -H "Authorization: Bearer Oracle" http://169.254.169.254/opc/v2/vnics/)
VNIC_ID=$(echo "$VNIC_JSON" | grep -oE '"vnicId"\s*:\s*"[^"]+"' | head -n 1 | cut -d'"' -f4)

if [ -z "$VNIC_ID" ] || [[ "$VNIC_ID" != ocid1.vnic.* ]]; then
    echo "Error: Could not retrieve valid VNIC OCID from metadata service. Metadata returned: $VNIC_JSON"
    exit 1
fi
echo "Resolved VNIC OCID: $VNIC_ID"

# 3. Create temporary OCI configuration directory and files
TEMP_DIR=$(mktemp -d -t oci-cli-XXXXXX)
CONFIG_PATH="$TEMP_DIR/config"
KEY_PATH="$TEMP_DIR/key.pem"

# Write private key
echo "$B64_PRIVATE_KEY" | base64 -d > "$KEY_PATH"
chmod 600 "$KEY_PATH"

# Write config file
cat <<EOF > "$CONFIG_PATH"
[DEFAULT]
user=$USER_OCID
fingerprint=$FINGERPRINT
key_file=$KEY_PATH
tenancy=$TENANCY_OCID
region=$REGION
EOF

# 4. Fetch current NSG associations
echo "Fetching current NSGs associated with the VNIC..."
CURRENT_NSGS_JSON=$(oci --config-file "$CONFIG_PATH" network vnic get --vnic-id "$VNIC_ID" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Error: Failed to fetch VNIC details using OCI CLI."
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Parse nsgIds
CURRENT_NSGS_LIST=$(echo "$CURRENT_NSGS_JSON" | grep -oE '"nsg-ids"\s*:\s*\[[^]]*\]' | tr -d ' \n\r' | cut -d'[' -f2 | cut -d']' -f1 | tr -d '"')

# Filter out target NSG
IFS=',' read -ra ADDR <<< "$CURRENT_NSGS_LIST"
UPDATED_NSGS_LIST=""
FOUND_NSG=false

for nsg in "${ADDR[@]}"; do
    if [ "$nsg" = "$NSG_OCID" ]; then
        FOUND_NSG=true
    else
        if [ -n "$nsg" ]; then
            if [ -z "$UPDATED_NSGS_LIST" ]; then
                UPDATED_NSGS_LIST="\"$nsg\""
            else
                UPDATED_NSGS_LIST="$UPDATED_NSGS_LIST,\"$nsg\""
            fi
        fi
    fi
done

if [ "$FOUND_NSG" = false ]; then
    echo "NSG '$NSG_OCID' is not associated with this VNIC."
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Update VNIC NSGs
echo "Updating VNIC NSGs (removing target) to: [$UPDATED_NSGS_LIST]"
UPDATE_RES=$(oci --config-file "$CONFIG_PATH" network vnic update --vnic-id "$VNIC_ID" --nsg-ids "[$UPDATED_NSGS_LIST]" 2>&1)
UPDATE_STATUS=$?

# Clean up
rm -rf "$TEMP_DIR"

if [ $UPDATE_STATUS -eq 0 ]; then
    echo "Successfully detached NSG '$NSG_OCID' from VNIC '$VNIC_ID'."
else
    echo "Error: Failed to update VNIC NSGs. OCI CLI returned: $UPDATE_RES"
    exit 1
fi

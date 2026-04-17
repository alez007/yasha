#!/bin/bash
set -e

# Use the environment variables set during build, or default to 1000
TARGET_UID=${MSHIP_UID:-1000}
TARGET_GID=${MSHIP_GID:-1000}

# Fix permissions for the cache directory
# If it was bind-mounted, it might be owned by root
if [ -d "/.cache" ]; then
    chown -R $TARGET_UID:$TARGET_GID /.cache
fi

# Also ensure the workspace has the right permissions
chown $TARGET_UID:$TARGET_GID /modelship

# Drop privileges and execute the main command
# gosu takes user name or UID:GID
exec gosu $TARGET_UID:$TARGET_GID "$@"

#!/usr/bin/env bash
# daily_backup.sh - Automated daily backup for crypto-ai-trader state
# Safely backs up SQLite databases, position review histories, and portfolio state.
# Prunes backups older than 30 days.

set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-$HOME/crypto-backups}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
DEST="$BACKUP_DIR/$TIMESTAMP"

mkdir -p "$DEST"
echo "=== Starting Crypto AI Trader Daily Backup: $TIMESTAMP ==="
echo "Target backup directory: $DEST"

# 1. Safely backup fundamental_eval.db using SQLite online backup API
if [ -f "$PROJECT_ROOT/fundamental_eval.db" ]; then
    echo "Backing up fundamental_eval.db..."
    sqlite3 "$PROJECT_ROOT/fundamental_eval.db" ".backup '$DEST/fundamental_eval.db'"
elif [ -f "fundamental_eval.db" ]; then
    echo "Backing up fundamental_eval.db from current dir..."
    sqlite3 "fundamental_eval.db" ".backup '$DEST/fundamental_eval.db'"
else
    echo "Notice: fundamental_eval.db not found, skipping."
fi

# 2. Backup position_review_history.json
if [ -f "$PROJECT_ROOT/position_review_history.json" ]; then
    echo "Backing up position_review_history.json..."
    cp "$PROJECT_ROOT/position_review_history.json" "$DEST/position_review_history.json"
elif [ -f "position_review_history.json" ]; then
    echo "Backing up position_review_history.json from current dir..."
    cp "position_review_history.json" "$DEST/position_review_history.json"
else
    echo "Notice: position_review_history.json not found, skipping."
fi

# 3. Backup any active paper portfolio state or sqlite databases in project root
for db_file in "$PROJECT_ROOT"/*.db; do
    if [ -f "$db_file" ] && [ "$(basename "$db_file")" != "fundamental_eval.db" ]; then
        base_name=$(basename "$db_file")
        echo "Backing up database: $base_name..."
        sqlite3 "$db_file" ".backup '$DEST/$base_name'"
    fi
done

for json_file in "$PROJECT_ROOT"/*history*.json "$PROJECT_ROOT"/*portfolio*.json; do
    if [ -f "$json_file" ] && [ "$(basename "$json_file")" != "position_review_history.json" ]; then
        base_name=$(basename "$json_file")
        echo "Backing up state file: $base_name..."
        cp "$json_file" "$DEST/$base_name"
    fi
done

echo "Backup complete: $DEST"

# 4. Prune backups older than 30 days
echo "Pruning backups older than 30 days from $BACKUP_DIR..."
find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true

echo "=== Backup routine successfully finished ==="

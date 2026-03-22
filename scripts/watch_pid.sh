#!/bin/bash
# Usage: ./watch_pid.sh <PID> <command...>
# Watches a process by PID and runs a command when it exits.

if [ $# -lt 2 ]; then
    echo "Usage: $0 <PID> <command...>"
    exit 1
fi

PID=$1
shift
CMD="$@"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "Error: PID $PID does not exist"
    exit 1
fi

echo "Watching PID $PID..."
while kill -0 "$PID" 2>/dev/null; do
    sleep 5
done

echo "PID $PID exited, starting: $CMD"
exec $CMD

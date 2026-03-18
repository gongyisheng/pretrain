#!/bin/bash
# Tests for scripts/watch_pid.sh

SCRIPT="$(dirname "$0")/../scripts/watch_pid.sh"
PASS=0
FAIL=0

pass() { echo "PASS: $1"; ((PASS++)); }
fail() { echo "FAIL: $1"; ((FAIL++)); }

# Test 1: no arguments prints usage and exits non-zero
out=$("$SCRIPT" 2>&1); code=$?
if [[ $code -ne 0 && "$out" == *"Usage"* ]]; then
    pass "no args prints usage"
else
    fail "no args prints usage (exit=$code)"
fi

# Test 2: only one argument (missing command) exits non-zero
out=$("$SCRIPT" 999999 2>&1); code=$?
if [[ $code -ne 0 && "$out" == *"Usage"* ]]; then
    pass "one arg prints usage"
else
    fail "one arg prints usage (exit=$code)"
fi

# Test 3: invalid PID exits non-zero with error message
out=$("$SCRIPT" 999999 echo done 2>&1); code=$?
if [[ $code -ne 0 && "$out" == *"does not exist"* ]]; then
    pass "invalid PID error"
else
    fail "invalid PID error (exit=$code, out=$out)"
fi

# Test 4: watch a real process and run command after it exits
MARKER=$(mktemp)
sleep 2 &
WATCHED_PID=$!
"$SCRIPT" "$WATCHED_PID" touch "$MARKER" &
WATCHER_PID=$!

wait "$WATCHED_PID"           # let sleep finish
wait "$WATCHER_PID"           # let watcher finish

if [[ -f "$MARKER" ]]; then
    pass "runs command after process exits"
else
    fail "runs command after process exits"
fi
rm -f "$MARKER"

# Test 5: command output is correct
MARKER=$(mktemp)
sleep 1 &
WATCHED_PID=$!
out=$("$SCRIPT" "$WATCHED_PID" touch "$MARKER" 2>&1)

if [[ "$out" == *"Watching PID $WATCHED_PID"* && "$out" == *"exited"* ]]; then
    pass "correct output messages"
else
    fail "correct output messages (out=$out)"
fi
rm -f "$MARKER"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]

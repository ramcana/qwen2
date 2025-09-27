#!/bin/bash
# Frontend test runner script for WSL
# Usage: ./run-frontend-tests.sh [test-pattern]

cd frontend
npm test -- --watchAll=false ${1:+--testPathPattern=$1}
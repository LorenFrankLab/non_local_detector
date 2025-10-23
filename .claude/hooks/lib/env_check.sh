#!/bin/bash
# Environment validation utilities for non_local_detector project

# Check if conda environment is activated
check_conda_env() {
    local required_env="non_local_detector"

    # Check CONDA_DEFAULT_ENV
    if [ "$CONDA_DEFAULT_ENV" = "$required_env" ]; then
        return 0
    fi

    # Check python path
    local python_path=$(which python 2>/dev/null)
    if [[ "$python_path" == *"/envs/$required_env/"* ]]; then
        return 0
    fi

    return 1
}

# Get activation command
get_activation_cmd() {
    echo "conda activate non_local_detector"
}

# Auto-prepend conda activation to command
prepend_activation() {
    local cmd="$1"
    echo "conda activate non_local_detector && $cmd"
}

# Check if command needs conda environment
needs_conda() {
    local cmd="$1"

    # Check for Python-related commands
    if [[ "$cmd" =~ ^(python|pytest|pip|black|ruff|mypy) ]]; then
        return 0
    fi

    # Check for full paths to conda binaries
    if [[ "$cmd" =~ /envs/[^/]+/bin/ ]]; then
        return 0
    fi

    return 1
}

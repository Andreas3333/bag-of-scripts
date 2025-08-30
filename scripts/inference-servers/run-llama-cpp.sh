#!/usr/bin/env bash


script_name=$(basename "$(realpath "$0")")

DATE=$(date +"%F:%H:%M:%S")
LOG_DIRECTORY="${HOME}/.agent-scripts/llama-cpp/logs"
PID_FILE="${HOME}/.agent-scripts/llama-cpp/llama-cpp.pid"

LLAMA_LOCAL_MODEL="${HOME}/Library/Caches/llama.cpp/ggml-org_Qwen2.5-Coder-1.5B-Q8_0-GGUF_qwen2.5-coder-1.5b-q8_0.gguf"
LLAMA_MODEL_SHORT_NAME="Qwen2.5-Coder-1.5B-Q8_0-GGUF"
LLAMA_SERVER_PORT=8100gti
LOG_FILE="${LOG_DIRECTORY}/llama.cpp.${DATE}.log"


function usage() {

    cat << EOF
Usage: ${script_name%.*} [OPTIONS]

Start a local Ollama server instance. This script does not download the specified model.
So you must must pre-download the model before running.

Options:
    --clear-logs  Clear all old log files
    --list-logs   List all log files
EOF
    exit 0
}

if [[ $1 =~ ^(-h|--help|help)$ ]]; then
    usage
fi


while getopts ":-:" opt
do
    case "${opt}" in
        -)
            case "${OPTARG}" in
                "clear-logs") CLEAR_LOGS="true";;
                "list-logs") LIST_LOGS="true";;
                *) echo "Unknown long option: --${OPTARG}" >&2; exit 1;;
            esac
        ;;
        \?) echo "Invalid option for ${script_name%.*} script: -${OPTARG}"; exit 1;;
        :) echo "Option -${OPTARG} requires an argument." >&2; exit 1;;
    esac
done


if [ "$LIST_LOGS" ]; then
    files=( "$(ls "${LOG_DIRECTORY}")" )
    for file in "${files[@]}"; do
        echo "- ${LOG_DIRECTORY}/${file}"
    done
    exit $?
fi

if [ "$CLEAR_LOGS" ]; then
    files=( "$(ls "${LOG_DIRECTORY}")" )
    for file in "${files[@]}"; do
        log_file="${LOG_DIRECTORY}/${file}"
        echo "${log_file}"
        rm "${log_file}"
    done
    echo "Cleared old log files";
    exit 0
fi


echo "llama is running at 127.0.0.0:${LLAMA_SERVER_PORT} serving ${LLAMA_MODEL_SHORT_NAME}..."
nohup llama-server \
    --log-file "${LOG_FILE}" \
    --log-colors --log-timestamps \
    -m "${LLAMA_LOCAL_MODEL}" \
    --port "${LLAMA_SERVER_PORT}" \
    -ngl 99 -fa -ub 1024 -b 1024 \
    --ctx-size 0 --cache-reuse 256 > /dev/null 2>&1 & echo $! > "$PID_FILE"

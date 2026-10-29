#!/bin/sh
set -eu

MODEL_VERSION="${ACTIVE_MODEL_VERSION:-v0.1.0-baseline}"
MODEL_ROOT="${MODEL_ARTIFACTS_DIR:-/var/data/models}"
PERSISTED_MODEL="${MODEL_ROOT}/${MODEL_VERSION}"
BUNDLED_MODEL="/app/artifacts/models/${MODEL_VERSION}"

mkdir -p "${MODEL_ROOT}" /var/data/feedback

model_is_complete() {
    test -f "$1/config.json" && 
    test -f "$1/model.safetensors" && 
    test -f "$1/tokenizer.json" && 
    test -f "$1/label_mapping.json"
}

if ! model_is_complete "${PERSISTED_MODEL}"; then
    mkdir -p "${PERSISTED_MODEL}"

    if model_is_complete "${BUNDLED_MODEL}"; then
        cp -a "${BUNDLED_MODEL}/." "${PERSISTED_MODEL}/"
    else
        if [ -n "${DAGSHUB_TOKEN:-}" ]; then
            : "${DAGSHUB_USERNAME:?DAGSHUB_USERNAME is required when DAGSHUB_TOKEN is set}"
            dvc remote modify --local dagshub auth basic
            dvc remote modify --local dagshub user "${DAGSHUB_USERNAME}"
            dvc remote modify --local dagshub password "${DAGSHUB_TOKEN}"
        fi

        dvc pull "artifacts/models/${MODEL_VERSION}"

        if ! model_is_complete "${BUNDLED_MODEL}"; then
            echo "ERROR: Model files are missing after DVC pull: ${BUNDLED_MODEL}" >&2
            exit 1
        fi

        cp -a "${BUNDLED_MODEL}/." "${PERSISTED_MODEL}/"
    fi
fi

if ! model_is_complete "${PERSISTED_MODEL}"; then
    echo "ERROR: Model is incomplete: ${PERSISTED_MODEL}" >&2
    exit 1
fi

export MODEL_ARTIFACTS_DIR="${MODEL_ROOT}"

exec python -m uvicorn ADKBot.api:app --host 0.0.0.0 --port "${PORT:-6655}"

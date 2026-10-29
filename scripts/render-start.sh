#!/bin/sh
# Materialize the pinned baseline once on Render's persistent disk, then start
# the API on the port assigned by the platform.  Credentials are runtime
# environment variables and are never written into the image or repository.
set -eu

MODEL_VERSION="${ACTIVE_MODEL_VERSION:-v0.1.0-baseline}"
MODEL_ROOT="${MODEL_ARTIFACTS_DIR:-/app/artifacts/models}"
PERSISTED_MODEL="${MODEL_ROOT}/${MODEL_VERSION}"
BUNDLED_MODEL="/app/artifacts/models/${MODEL_VERSION}"

mkdir -p "${MODEL_ROOT}" /var/data/feedback

model_is_complete() {
    test -f "$1/config.json" && \
    test -f "$1/model.safetensors" && \
    test -f "$1/tokenizer.json" && \
    test -f "$1/label_mapping.json"
}

if ! model_is_complete "${PERSISTED_MODEL}"; then
    mkdir -p "${PERSISTED_MODEL}"
    if [ -f "${BUNDLED_MODEL}/config.json" ]; then
        cp -a "${BUNDLED_MODEL}/." "${PERSISTED_MODEL}/"
    else
        if [ -n "${DAGSHUB_TOKEN:-}" ]; then
            : "${DAGSHUB_USERNAME:?DAGSHUB_USERNAME is required when DAGSHUB_TOKEN is set}"
            dvc remote modify --local dagshub auth basic
            dvc remote modify --local dagshub user "${DAGSHUB_USERNAME}"
            dvc remote modify --local dagshub password "${DAGSHUB_TOKEN}"
        fi
        dvc pull "artifacts/models/${MODEL_VERSION}"
        model_is_complete "${BUNDLED_MODEL}"
        cp -a "${BUNDLED_MODEL}/." "${PERSISTED_MODEL}/"
    fi
fi

model_is_complete "${PERSISTED_MODEL}"

exec python -m uvicorn ADKBot.api:app --host 0.0.0.0 --port "${PORT:-6655}"

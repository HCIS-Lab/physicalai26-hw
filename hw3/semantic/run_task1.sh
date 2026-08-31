#!/usr/bin/env bash
# =============================================================================
# TAICA HW3 — Task 1 one-command runner (REUSE grounding + SHACL validation)
#
#   STEP 1  Check the toolchain (java / python)
#   STEP 2  Prepare Apache Jena (its `shacl` CLI does the validation)
#   STEP 3  REUSE grounding: ground_execution.py
#           -> output/data.ttl        (your FK/IK execution process as RDF)
#   STEP 4  SHACL validation:
#           - YOUR shapes.ttl vs output/data.ttl        -> output/validation.ttl
#           - TA ta-shapes-full.ttl vs data.ttl         -> output/ta-validation.ttl
#           - YOUR shapes.ttl vs ta-faulty-execution.ttl-> output/probe-validation.ttl
#   STEP 5  Scoring (score_semantic.py parses the validation reports)
#
# Usage (from the hw3 root or from semantic/):
#   bash semantic/run_task1.sh --student-id <your id>
#   bash semantic/run_task1.sh --student-id ... --own   # after Tasks 2-3: use YOUR FK/IK
#
# Environment variables:
#   PYTHON     python interpreter (default: python; must be the taica-hw3 env)
#   JENA_HOME  an already-extracted apache-jena directory
#              (default: auto-download into semantic/.cache)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

JENA_VERSION=4.10.0
PYTHON="${PYTHON:-python}"
GROUND_ARGS=("$@")

echo "== STEP 1/5 | Checking toolchain =="
command -v java >/dev/null 2>&1 || { echo "ERROR: 'java' not found (JDK 11+ required)" >&2; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "ERROR: python '$PYTHON' not found" >&2; exit 1; }
java -version 2>&1 | head -1
"$PYTHON" -c "import numpy, pybullet; print('python + numpy + pybullet OK')"

echo
echo "== STEP 2/5 | Preparing Apache Jena ${JENA_VERSION} (shacl CLI) =="
if [ -z "${JENA_HOME:-}" ]; then
    JENA_HOME="$PWD/.cache/apache-jena-$JENA_VERSION"
fi
if [ ! -d "$JENA_HOME/lib" ]; then
    echo "Jena not found at $JENA_HOME — downloading (~30 MB) ..."
    mkdir -p .cache
    TARBALL=".cache/apache-jena-$JENA_VERSION.tar.gz"
    MIRRORS=(
        "https://repo1.maven.org/maven2/org/apache/jena/apache-jena/$JENA_VERSION/apache-jena-$JENA_VERSION.tar.gz"
        "https://archive.apache.org/dist/jena/binaries/apache-jena-$JENA_VERSION.tar.gz"
    )
    ok=0
    for url in "${MIRRORS[@]}"; do
        echo "Fetching $url"
        if curl -fL -C - --retry 5 --retry-delay 2 -o "$TARBALL" "$url"; then ok=1; break; fi
        echo "WARN: mirror failed, trying next ..."
    done
    [ "$ok" -eq 1 ] || { echo "ERROR: could not download Apache Jena" >&2; exit 1; }
    tar -xzf "$TARBALL" -C .cache
fi
export JENA_HOME
echo "JENA_HOME = $JENA_HOME"

echo
echo "== STEP 3/5 | REUSE grounding: FK/IK execution process -> output/data.ttl =="
if [ -n "${GROUND_SCRIPT:-}" ]; then          # overridable for TA testing
    "$PYTHON" "$GROUND_SCRIPT" ${GROUND_ARGS[@]+"${GROUND_ARGS[@]}"}
else
    "$PYTHON" ground_execution.py ${GROUND_ARGS[@]+"${GROUND_ARGS[@]}"}
fi

echo
echo "== STEP 4/5 | SHACL validation (Jena shacl CLI) =="
SHACL="$JENA_HOME/bin/shacl"
run_shacl () {  # $1 shapes, $2 data, $3 out
    echo "---- shacl validate --shapes $1 --data $2 -> $3"
    "$SHACL" validate --shapes "$1" --data "$2" > "$3" || true
    grep -q 'sh:conforms  *true' "$3" \
        && echo "     conforms: true" \
        || echo "     conforms: false ($(grep -c 'sh:ValidationResult' "$3" || true) result(s))"
}
run_shacl shapes.ttl output/data.ttl output/validation.ttl
run_shacl ta-shapes-full.ttl output/data.ttl output/ta-validation.ttl
run_shacl shapes.ttl ta-faulty-execution.ttl output/probe-validation.ttl

echo
echo "== STEP 5/5 | Scoring Task 1 =="
"$PYTHON" score_semantic.py

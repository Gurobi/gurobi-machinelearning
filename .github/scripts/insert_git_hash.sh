if [[ $# -eq 0 ]] ; then
    echo "Missing '_version.py' input argument"
    exit 1
fi

VERSION_PY="$1"
GIT_HASH=$(git rev-parse --verify HEAD)

SEARCH_STRING='$Format:%H$'
SEARCH_ESCAPED=$(sed 's/[^^]/[&]/g; s/\^/\\^/g' <<<"$SEARCH_STRING") # escape it.

sed -i "s/$SEARCH_ESCAPED/$GIT_HASH/" "$VERSION_PY"

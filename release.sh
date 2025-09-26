#!/usr/bin/env bash
set -euo pipefail

LEVEL=${1:-}
[[ -z "$LEVEL" ]] && { echo "Usage: $0 [patch|minor|major|x.y.z]"; exit 1; }

PKG=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].name')
OLD_V=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version')

# dry-run preview of the new version
if [[ "$LEVEL" =~ ^[0-9]+\.[0-9]+\.[0-9+](-.+)?$ ]]; then
    NEW_V="$LEVEL" # user gave explicit version
else
    NEW_V=$(cargo release version "$LEVEL" --quiet 2>&1 | awk '/Upgrading.*from/{print $NF}' || echo "")
    [[ -z "$NEW_V" ]] && NEW_V=$(cargo release version "$LEVEL" --quiet 2>&1 | grep -oP '\d+\.\d+\.\d+[^ ]*' | head -1)
fi
[[ -z "$NEW_V" ]] && { echo "Could not determine new version"; exit 1; }

BRANCH="release-${NEW_V}"
REMOTE=${REMOTE:-origin}

echo "=== ${PKG}  ${OLD_V}  ->  ${NEW_V}  (branch ${BRANCH})"

git fetch --tags "$REMOTE"
git switch -c "$BRANCH"

## bump + changelog + commit  (no publish/push)
cargo release --no-publish --no-push --no-confirm --execute "$LEVEL"

## create tag on this commit
git tag -a "v${NEW_V}" -m "v${NEW_V}"

## push both branch and tag
git push -u "$REMOTE" "$BRANCH" "v${NEW_V}"

REPO_URL=$(git remote get-url "$REMOTE" | sed 's/\.git$//' | sed 's/git@/https:\/\//' | sed 's/:/\//')
echo
echo "✅  branch ${BRANCH}  and  tag v${NEW_V}  pushed"
echo "   open PR:  ${REPO_URL}/compare/${BRANCH}"

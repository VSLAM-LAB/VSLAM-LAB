#!/bin/bash
# Usage: ./git-clone.sh <url> <dest_dir> [<branch>]
# Clones <url> (GitHub, Hugging Face, ...) recursively into <dest_dir>, relative to the VSLAM-LAB root
# (Baselines/<Name>, Capabilities/sources/<name>), and does nothing if <dest_dir> already exists, so every
# `fetch-source` task can be re-run safely. <branch> is optional; when given, that branch is checked out (and its
# submodules are initialised) instead of the repo's default branch.

url=$1
dest_dir=$2
git_branch=$3

if [ -z "$url" ] || [ -z "$dest_dir" ]; then
  echo "usage: $0 <url> <dest_dir> [<branch>]" >&2
  exit 2
fi

if [ -d "$dest_dir" ]; then
  exit 0
fi

branch_flag=()
if [ -n "$git_branch" ]; then
  branch_flag=(--branch "$git_branch")
fi

git clone --recursive "${branch_flag[@]}" "$url" "$dest_dir"

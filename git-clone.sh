#!/bin/bash
# Usage: ./git-clone.sh <github_profile> <repo> [<folder_name>] [<branch>]
# Clones into Baselines/<folder_name> (default: <repo>) and skips if it already exists.
# <branch> is optional; when given, that branch is checked out (and its submodules
# are initialised) instead of the repo's default branch.

get_git_url() {
  local git_profile=$1
  local git_repo=$2
  echo "https://github.com/$git_profile/$git_repo.git"
  #echo "git@github.com:$git_profile/$git_repo.git"
}

git_profile=$1
git_repo=$2
folder_name=${3:-$git_repo}
git_branch=$4
vslamlab_baselines_folder="Baselines/${folder_name}"

if [ -d "$vslamlab_baselines_folder" ]; then
  exit 0
fi

branch_flag=()
if [ -n "$git_branch" ]; then
  branch_flag=(--branch "$git_branch")
fi

git_url=$(get_git_url "${git_profile}" "${git_repo}")
git clone --recursive "${branch_flag[@]}" "${git_url}" "${vslamlab_baselines_folder}"

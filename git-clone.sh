#!/bin/bash

get_git_url() {
  local git_profile=$1
  local git_repo=$2
  echo "https://github.com/$git_profile/$git_repo.git"
  #echo "git@github.com:$git_profile/$git_repo.git"
}

git_profile=$1
git_repo=$2
folder_name=${3:-$git_repo}
vslamlab_baselines_folder="Baselines/${folder_name}"

if [ -d "$vslamlab_baselines_folder" ]; then
  exit 0
fi

git_url=$(get_git_url "${git_profile}" "${git_repo}")
git clone --recursive "${git_url}" "${vslamlab_baselines_folder}"


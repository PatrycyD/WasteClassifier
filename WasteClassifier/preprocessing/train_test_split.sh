#!/bin/bash

source /home/peprycy/WasteClassifier/WasteClassifier/preprocessing/config.cfg

TEST_PATH="$SPLIT_IMAGES_PATH/test"
TRAIN_PATH="$SPLIT_IMAGES_PATH/train"

function main(){

  merge_paths

#  echo $all_paths | tr ' ' '\n'

  if [[ -d "$SPLIT_IMAGES_PATH" ]]
  then
    rm -rf "$SPLIT_IMAGES_PATH"
  fi
  mkdir -p "$TRAIN_PATH"
  mkdir -p "$TEST_PATH"

  for category in $all_paths
  do
    category_name=$(echo "$category" | sed 's|.*\/||g')

    category_test_path="${TEST_PATH}/$category_name"
    category_train_path="${TRAIN_PATH}/$category_name"

    [[ ! -d "${category_test_path}" ]] && mkdir "${category_test_path}" && echo "Created $category_name folder in test directory"
    [[ ! -d "${category_train_path}" ]] && mkdir "${category_train_path}" && echo "Created $category_name folder in train directory"

    images_num=$(find "$category" -type f | wc -l)
    test_indexes=$(python -c "import random; print(random.sample(range(1, $images_num), int($TEST_PERCENT*$images_num)))")
    test_indexes="${test_indexes//,/}"
    i=1

    train_index=$(find "$category_train_path/" -type f | wc -l)
    test_index=$(find "$category_train_path/" -type f | wc -l)

    for img in "$category"/*
    do
      copy_images "$img"
      i=$((i+1))
    done
    echo -e "\nCopied photos to $category_name test and train directories\n"
  done
}

function merge_paths(){

  all_paths=""
  for path in "$TRASHNET_RESIZED_PATH"/*
  do
    all_paths="${all_paths} ${path}"
  done
  all_paths=$(echo "$all_paths")

  all_paths="${all_paths} $ORGANIC_PATH"

  for path in "$TRASHBOX_PATH"/*
  do
    all_paths="${all_paths} ${path}"
  done
}

function copy_images(){
  local image=$1

#  file_name=$(tr -d ' ' <<< "${image##*/}")  # last path part with space removed
  if [[ $test_indexes = *" $i "* ]]  # if photo index was marked as test index
    then
      cp "$image" "${category_test_path}/${category_name}${test_index}.jpg"
      test_index=$((test_index + 1))
    else
      cp "$image" "${category_train_path}/${category_name}${train_index}.jpg"
      train_index=$((train_index + 1))
  fi

}

main
ROOT_PATH=$(python -c 'import preprocessing.config as config; print(config.ROOT_PATH)' | sed 's/\\/\//g' | sed 's/://')
ROOT_PATH="/${ROOT_PATH}"

RESIZED_PATH=$(python -c 'import preprocessing.config as config; print(config.RESIZED_PATH)' | sed 's/\\/\//g' | sed 's/://')
RESIZED_PATH="/${RESIZED_PATH}"

SPLITTED_IMAGES_PATH=$(python -c 'import preprocessing.config as config; print(config.SPLITTED_IMAGES_PATH)' | sed 's/\\/\//g' | sed 's/://')
SPLITTED_IMAGES_PATH="/${SPLITTED_IMAGES_PATH}"

TEST_PATH="$SPLITTED_IMAGES_PATH/test"
TRAIN_PATH="$SPLITTED_IMAGES_PATH/train"

TEST_PERCENT=$(python -c 'import preprocessing.config as config; print(config.TEST_PERCENT)')

function main(){
  if [[ -d "$SPLITTED_IMAGES_PATH" ]]
  then
    rm -rf "$SPLITTED_IMAGES_PATH"
  fi
  mkdir -p "$TRAIN_PATH"
  mkdir "$TEST_PATH"

  for category in "$RESIZED_PATH"/*
  do
    category_name=$(echo $category | sed 's|.*\/||g')
    category_test_path="${TEST_PATH}/$category_name"
    category_train_path="${TRAIN_PATH}/$category_name"
    mkdir "${category_test_path}"
    mkdir "${category_train_path}"

    images_num=$(find "$category" -type f | wc -l)
    test_indexes=$(python -c "import random; print(random.sample(range(1, $images_num), int($TEST_PERCENT*$images_num)))")
    test_indexes="${test_indexes//,/}"
    i=1

    for img in "$category"/*
    do
      if [[ $test_indexes = *" $i "* ]]
      then
        cp "$img" "$category_test_path"
      else
        cp "$img" "$category_train_path"
      fi
      i=$((i+1))
    done
  done
}

main
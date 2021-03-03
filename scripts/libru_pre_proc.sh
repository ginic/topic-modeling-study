# !bin/bash

# Script for turning lib.ru text files into same format as the Joanna rus data, Author/title.txt
# which are named like Author_Title.txt

LIB_RU_IN=$1
OUT_DIR=$2

mkdir -p $OUT_DIR
for d in $LIB_RU_IN/* ; do
    if [ -d "$d" ]; then
        author=$(basename $d)
        echo "Author $author"
        for f in $d/*.txt ; do
            title=$(basename $f)
            target="${author}_${title}"
            echo "Copy $f to $target"
            cp $f $OUT_DIR/$target
        done
    fi
done

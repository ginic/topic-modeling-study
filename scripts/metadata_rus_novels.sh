# !/bin/bash
# Quick script for outputting metadata for JoannaRus data to tsv format
# [doc_id]\t[author_label]\t[title]
# Doc ids in this case are just 'author_title'
# Usage: ./metadata_rus_novels.sh RussianNovels/corpus metadata.tsv

RUS_CORPUS=$1
TSV_OUT=$2

ls $RUS_CORPUS | awk -F"[_.]" '{print $1"_"$2"\t"$1"\t"$2}' > $TSV_OUT
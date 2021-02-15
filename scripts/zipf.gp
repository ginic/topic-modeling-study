#!/user/bin/env gnuplot

# Quick gnuplot for Zipf curves of document frequency. Probably will switch to better python option
# once I figure out what all we need, but for now this is fine.
# Run as `gnuplot ../scripts/zipf.p`
#

countsfile = 'russian_novels_counts.tsv'

set terminal png noenhanced size 1200,800
set output 'russian_novels_zipf.png'
set xlabel "Term Rank"
set ylabel "Document Frequency"
set title sprintf("Zipf curve %s", countsfile)

plot for [i=0:*] countsfile using i:3 title 'Document frequency'
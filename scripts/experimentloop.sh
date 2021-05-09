# Loop to build many experiments
for n in 50 100 250 500
do
    for i in {0..10}
    do
        make experiment EXP_COUNT=$i NUM_TOPICS=$n
        make stemmed_post_proc_experiment EXP_COUNT=$i STEM_METHOD=pymystem3 NUM_TOPICS=$n
        for s in "pymystem3 stanza snowball truncate"
        do
            make stemmed_corpus_experiment EXP_COUNT=$i STEM_METHOD=$s NUM_TOPICS=$n
        done
    done
done
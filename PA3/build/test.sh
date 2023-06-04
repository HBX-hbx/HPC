#!/bin/bash

# am
# amazon_cogdl
# arxiv
# citation
# collab
# ddi
# ppa
# products
# protein
# reddit.dgl
# wikikg2
# yelp
# youtube

export GTEST_FILTER="SpMMTest*"
datadir=~/PA3/data/
dataset=wikikg2
len=256

srun -p gpu ./test/unit_tests --dataset $dataset --len $len --datadir $datadir
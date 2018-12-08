#!/bin/sh

python main.py -epochs 10 -gpu 1,2,3 -saved_model models/best.h5 -batch_size 64 -embedding_dim 500 -hidden_size 500 -drop 0.1

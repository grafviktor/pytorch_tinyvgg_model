#!/bin/bash

file_name=./data/pizza_steak_sushi/test/steak/2117351.jpg
python predict.py --file_name "$file_name"
open "$file_name"
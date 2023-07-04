#!/bin/bash
tensorman run -p 8888:8888 --gpu --jupyter -- python3.8 -m notebook --ip=0.0.0.0 --port=8888 --allow-root

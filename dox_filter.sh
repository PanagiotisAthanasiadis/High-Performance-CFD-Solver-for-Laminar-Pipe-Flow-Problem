#!/bin/bash
# Replaces kernel<<<...>>> with just kernel
sed 's/<<<[^>]*>>>//g' "$1"

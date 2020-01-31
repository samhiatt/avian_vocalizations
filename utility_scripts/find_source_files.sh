#!/bin/sh

find . -type f \! -path "./.*" ! -path "./data/*" ! -path "./report/*" ! -path "./html/*" ! -path "./output/*" ! -name "*.pyc" 

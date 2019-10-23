#!/bin/sh

pdoc3 . --config sort_identifiers=False --html -o new_html; rm -rf html/; mv new_html html

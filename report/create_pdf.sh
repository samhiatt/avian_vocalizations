#!/bin/sh

if [ -z "$(which pdflatex)$(which pandoc)" ]; then 
  INSTALL_CMD="HOME=$HOME sudo ./linux_install_pandoc_pdflatex.sh"
  echo "pdflatex and/or pandoc missing. Enter admin password to install with \n'$INSTALL_CMD'";
  $INSTALL_CMD; 
fi

if [ -z "$1" ]; then
  echo "Must supply source filename." ;
else
  pandoc -F pandoc-crossref -F pandoc-citeproc \
    --variable papersize=a4paper \
    `#--variable classoption=twocolumn` \
    -s $1 -o "${1%.*}".pdf \
    && echo "Created ${1%.*}.pdf" 
fi

#!/bin/sh
# Daily: gzip market recordings from previous days (the research tools read .jsonl.gz directly).
# Today's file is still being written, so it is left alone.
find /opt/hltrader/app/data -maxdepth 1 -name 'raw-*.jsonl' -mmin +1500 -exec gzip -q {} \;

#!/bin/sh

ssh -o ProxyCommand='nc -X connect -x localhost:48455 %h %p' root@0.tcp.ngrok.io -p $1

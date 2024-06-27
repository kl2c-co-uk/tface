#!/bin/bash
sling-tface


cd /home/piter/tface.git && git fetch --all && git push  git@github.com:kl2c-co-uk/tface.git --all





git branch -r | grep -v '\->' | while read remote; do git checkout --track $remote ; git pull; done


git fetch --all && git push --all github






open container

$ docker run -it --name sling-tface --entrypoint /bin/sh alpine/git

create basic container layout

; git init ./sling
; cd ./sling
; git remote add big5pi0 http://192.168.0.9:3000/peter/tface.git
; git remote add github git@github.com:g-pechorin/tface.git
; ssh-keygen -b 4096
; cat /root/.ssh/id_ed25519.pub 
; ssh git@github.com
exit

add key to github

> ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAIk4N5gIxNBoaWacGntvgSKxbvZm1ZOVYU2PW9IZWop root@61b3f28fe792



run something

docker run -it --name sling-tface alpine/git /bin/sh 






docker exec -it --name sling-tface /bin/sh 



git init ./sling


docker create --name sling-tface alpine/git 

docker run -it --rm --entrypoint /bin/sh alpine/git



docker run -it --name sling-tface --entrypoint /bin/sh alpine/git





docker create --name sling-tface alpine/git  --entrypoint /bin/sh -C "cd /git/sling && git fetch --all && git push --all github"



git clone http://192.168.0.9:3000/peter/tface.git
cd tface

git fetch --all http://192.168.0.9:3000/peter/tface.git

git clone http://big5pi0:3000/peter/tface.git
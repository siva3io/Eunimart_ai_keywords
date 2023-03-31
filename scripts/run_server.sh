#!/bin/bash

cd /home/azureuser/vdezi_ai_keywords    #replace service_repowith  your repo name 

git stash

git pull https://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@github.com/eunimart/vdezi_ai_keywords.git

source /home/azureuser/vdezi_ai_keywords/venv/bin/activate

pip3 install requests==2.25.1

pip3 install -r /home/azureuser/vdezi_ai_keywords/requirements.txt

python3 serve.py




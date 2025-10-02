#!/bin/bash
export COMPOSE_BAKE=true

docker compose up -d

docker exec -it padif_app /bin/bash -c "/mnt/ccdc/CSDS2020/CSD_2020/bin/ccdc_activator -a -k YOUR_CCDC_LICENSE_KEY_HERE"
docker exec -it padif_app /bin/bash

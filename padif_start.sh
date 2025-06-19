#!/bin/bash
export COMPOSE_BAKE=true

docker compose up -d

docker exec -it padif_app /bin/bash -c "/mnt/ccdc/CSDS2020/CSD_2020/bin/ccdc_activator -a -k 96DAD1-E0657B-44F7B2-E17D11-EB9203-EEBCCE"
docker exec -it padif_app /bin/bash

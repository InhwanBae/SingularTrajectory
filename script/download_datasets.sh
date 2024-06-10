#!/bin/bash

echo "Download ETH-UCY datasets"

wget -O datasets.zip 'https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=0'
unzip -q datasets.zip
rm -rf datasets.zip

wget -O datasets-domainextension.zip https://github.com/InhwanBae/SingularTrajectory/releases/download/v0.1/datasets-domainextension.zip
unzip -q datasets-domainextension.zip
rm -rf datasets-domainextension.zip

wget -O datasets-imageextension.zip https://github.com/InhwanBae/SingularTrajectory/releases/download/v0.1/datasets-imageextension.zip
unzip -q datasets-imageextension.zip
rm -rf datasets-imageextension.zip

echo "Done."

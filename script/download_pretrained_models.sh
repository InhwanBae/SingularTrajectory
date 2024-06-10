#!/bin/bash

task_array=("stochastic" "deterministic" "momentary" "domain" "fewshot" "domain-stochastic" "allinone")

for (( i=0; i<${#task_array[@]}; i++ ))
do
  echo "Download pre-trained model for ${task_array[$i]} task."
  wget -O ${task_array[$i]}.zip https://github.com/InhwanBae/SingularTrajectory/releases/download/v1.0/SingularTrajectory-${task_array[$i]}-pretrained.zip
  unzip -q ${task_array[$i]}.zip
  rm -rf ${task_array[$i]}.zip
done

echo "Done."

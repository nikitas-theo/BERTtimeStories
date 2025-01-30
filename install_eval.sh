# clone the official evaluation pipeline repo
git clone https://github.com/babylm/evaluation-pipeline-2024

# move all files in the evaluation pipeline
cp evaluation_files/* evaluation-pipeline-2024/

# move the task.py file to the correct location 
# This fixes an evaluation issue with minimal pairs evaluations (EWoK, BLiMP), 
# where when both options have the same probability the first one was selected by default. 
# After the change a random option is selected, removing any unwanted bias.
mv evaluation-pipeline-2024/task.py evaluation-pipeline-2024/lm_eval/api/task.py
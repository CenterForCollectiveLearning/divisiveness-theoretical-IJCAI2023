declare -a StringArray=("UM50" "UM10" "IC")
array=($(seq 4 11))
for i in "${array[@]}"; do
    for val in "${StringArray[@]}"; do
        python run_manipulation.py --type "${val}" --alternatives "${i}" --method "borda" --iterations=100 --step=6;
    done
done
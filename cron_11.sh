declare -a StringArray=("UM50" "UM10" "IC")
declare -a voting=("copeland" "borda")
declare -a steps=("6" "10")

for step in "${steps[@]}"; do
    for vot in "${voting[@]}"; do
        for val in "${StringArray[@]}"; do
            python run_manipulation.py --type "${val}" --alternatives 11 --method "${vot}" --iterations=100 --step="${step}";
        done
    done
done
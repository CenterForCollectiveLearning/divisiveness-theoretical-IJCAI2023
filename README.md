# Divisiveness Theoretical

In the text we refer to:

**Experiment 1** as the results shown in Section 3.2 and Appendix A.1

**Experiment 2** as the results shown in Section 4.1 and Appendix A.2

**Experiment 3** as the results shown in Section 4.2 and Appendix A.3

### File names

For each Experiment X $\in$ {1,2,3} and method Y $\in$ {IC, UM10, UM50} and number of agents N we include:

Experiment X Toy Example.ipynb to show how the data was collected

### Data collection

The data we collected for building our figures in Experiments 2 and 3 can be found in
experiment_X_method_Y_profile_N.csv

In case of Experiment 3, we split the files by voting rule R $\in$ {copeland, borda}. Due to computational time to calculate divisiveness, we run the divisiveness every S new agents (for this manuscript, S $\in$ {6, 10})
experiment_3_method_Y_profile_N_step_S_R.csv

For each figure in the paper and appendix we include a file showing how the figures were created:
Experiment X Figure #.ipynb

### Additional considerations

The files `helpers.py` and `rules.py` contain functions used throughout the notebooks.

The file `run_manipulation.py` sets the heuristic implemented in **Experiment 3**, along with generating the data. In addition, we include the script files `cron_borda.sh` and `cron_copeland.sh` to initialize the file `run_manipulation.py`.

The folder `data/` also includes files generated in `run_manipulation.py` for Experiment 3. Those files were concatenated by using `File_Creation.ipynb` in `deprecated/` folder.

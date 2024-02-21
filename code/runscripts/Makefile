clean:
	@rm -f ../slurm_out/*.out

dual_run:
	@sbatch dual_latent.sh

latent_run:
	@sbatch dual_edges.sh

# If we dont want to wait for the queue and we don't need any major ressources
_run:
	python run.py

# If we dont want to wait for the queue and we don't need any major ressources

queue:
	@squeue -u phk851

conda:
	@conda activate master
cancel:
	@scancel -u phk851

cleanrun:
# @ hides the command when we use it in the terminal
# $$ is necessary when we want to use multiple commands and concat them.
# In a Makefile each line of a recipe is executed in a separated shell instance. so you need to escape the '$' sign to prevent make from interpreting it as a makefile varialbe. Use '$$' to rpresent a single '$'
out:
	@cat ../slurm_out/$$(ls -rt ../slurm_out/ | tail -n1)
all_out:
	@cat ../slurm_out/*
clean:
	rm -f ../slurm_out/*.out

vgae:
	@sbatch VGAEsbatch.sh

# If we dont want to wait for the queue and we don't need any major ressources
_vgae:
	python VGAErun.py

directions:
	@sbatch DirectionsSbatch.sh

# If we dont want to wait for the queue and we don't need any major ressources
_directions:
	python DirectionsRun.py

# @ hides the command when we use it in the terminal
# $$ is necessary when we want to use multiple commands and concat them.
# In a Makefile each line of a recipe is executed in a separated shell instance. so you need to escape the '$' sign to prevent make from interpreting it as a makefile varialbe. Use '$$' to rpresent a single '$'
out:
	@cat ../slurm_out/$$(ls -rt ../slurm_out/ | tail -n1)

cancel:
	scancel -u hlp179
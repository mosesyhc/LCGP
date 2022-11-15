#!/bin/bash
while IFS=$'\t' read -r n method seed noiseconst ipfrac fname
do
    JOB=`sbatch <<EOJ
#!/bin/bash
#SBATCH --account=p30845
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --mail-user=mosesyhc@u.northwestern.edu
#SBATCH -J lcgp
#SBATCH --output=error_output/R-%x.%j.out
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem=8G

# unload modules that may have been loaded when job was submitted
module purge all

# load the version of python you want to use
module load python/anaconda3.6
source activate /projects/p30845/pythonenvs/vigpenv

# By default all file paths are relative to the directory where you submitted the job.
python slurm_testgenfunc.py --n=${n} --method=${method} --seed=${seed} --noiseconst=${noiseconst} --ipfrac=${ipfrac} --fname=${fname}
EOJ
`

done < params/params.txt
exit


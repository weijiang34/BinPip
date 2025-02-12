import pandas as pd 
import os 
import argparse

CONDA_DIR = '/g/data/oo46/wj6768/miniconda3'
MAMBA_DIR = ''
METAWRAP_SCRIPTS_DIR = '/g/data/oo46/wj6768/tools/metaWRAP/bin/metawrap-scripts'

# ['pbs', 'gadi']
JOB_MANAGER = 'gadi'
# For all general PBS users:
MAX_THREADS= 32
MAIL_ADDRESS = '379004663@qq.com'
# For gadi users, specifically:
GADI_PROJECT = 'mp96'
GADI_STORAGE = 'gdata/oo46+gdata/mp96'

# Workflow
# 1. grouping
# 2. mapping:
#       s1: concat & indexing
#       s2: mapping
# 3. binning:
#       s3: generating sequence files
#       s4: self-training
#       s5: binning
# 4. 

# Check
# 1. grouping
#       # of groups detected (not compulsory)
# 2. maping:
#       s1: concat & indexing
#           check:  concat: $group_dir/semibin_output/combined_output/concatenated.fa
#                   indexing:  $group_dir/semibin_output/combined_output/concatenated.fa.*.bt2l
#       s2: mapping
#           check: $group_dir/bamsam_output/$fileHeader.bam $fileHeader.mapped.bam $fileHeader.mapped.sorted.bam $fileHeader.sam
# 3. binning:
#       s3: generating sequence files
#           check: 
#       s4: self-training
#           check: model.h5

class BashHeader():
    def __init__(self, job_name, ncpus) -> None:
        self.job_name = job_name
        self.ncpus = ncpus

    def get_header(self):
        header = []
        header.append(f"#!/bin/bash")

        # header = [x+"\n" for x in header]
        return header

class PBSHeader(BashHeader):
    def __init__(self, job_name, ncpus, ngpus, mem, walltime, mail_addr, log_o, log_e) -> None:
        super().__init__(job_name, ncpus)
        self.ngpus = ngpus
        self.mem = mem
        self.walltime = walltime
        self.mail_addr = mail_addr
        self.log_o = log_o
        self.log_e = log_e
    
    # return a list of header lines
    def get_header(self):
        header = []
        header.append(f"#!/bin/bash")
        if self.job_name!='':
            header.append(f"#PBS -N {self.job_name}")
        if self.log_o!='':
            header.append(f"#PBS -o {self.log_o}")
        if self.log_e!='':
            header.append(f"#PBS -e {self.log_e}")
        header.append(f"#PBS -j oe")
        header.append(f"#PBS -m abe")
        header.append(f"#PBS -M {self.mail_addr}")
        header.append(f"#PBS -l ncpus={self.ncpus}")
        if self.ngpus>0:
            header.append(f"#PBS -l ngpus={self.ngpus}")
        header.append(f"#PBS -l mem={self.mem}")
        header.append(f"#PBS -l walltime={self.walltime}")

        # header = [x+"\n" for x in header]
        return header
        
class GadiHeader(PBSHeader):
    def __init__(self, job_name, ncpus, ngpus, mem, walltime, mail_addr, log_o, log_e, project, storage, node_type, jobfs) -> None:
        super().__init__(job_name, ncpus, ngpus, mem, walltime, mail_addr, log_o, log_e)
        self.project = project
        self.storage = storage
        self.node_type = node_type
        self.jobfs = jobfs

    # return a list of header lines
    def get_header(self):
        header = []
        header.append(f"#!/bin/bash")
        if self.job_name!='':
            header.append(f"#PBS -N {self.job_name}")
        if self.log_o!='':
            header.append(f"#PBS -o {self.log_o}")
        if self.log_e!='':
            header.append(f"#PBS -e {self.log_e}")
        header.append(f"#PBS -j oe")
        header.append(f"#PBS -m abe")
        if self.mail_addr!='':
            header.append(f"#PBS -M {self.mail_addr}")
        header.append(f"#PBS -l ncpus={self.ncpus}")
        if self.ngpus>0:
            header.append(f"#PBS -l ngpus={self.ngpus}")
        header.append(f"#PBS -l mem={self.mem}")
        header.append(f"#PBS -l walltime={self.walltime}")

        header.append(f"#PBS -P {self.project}")
        header.append(f"#PBS -l storage={self.storage}")
        if self.ngpus>0:
            header.append(f"#PBS -q gpuvolta")
        else:
            header.append(f"#PBS -q {self.node_type}")
        header.append(f"#PBS -l jobfs={self.jobfs}")

        # header = [x+"\n" for x in header]
        return header

job_managers = {"pbs":PBSHeader,"gadi":GadiHeader}

def group_by_age(prj_dir, manifest, age_gender, ngroups):
    manifest = pd.read_csv(manifest, header=0, names=["fileHeader","fq1","fq2","fa"])
    age_gender = pd.read_csv(age_gender, header=0, names=["sample_id","age","sexbirth"])

    manifest["sample_id"] = manifest["fileHeader"].str.split('_').apply(lambda x: [sep for sep in x if "HOAM" in sep][0])
    age_gender_dict = {row["sample_id"]:[row["age"],row["sexbirth"]] for _, row in age_gender.iterrows()}
    df = manifest.copy()
    df["age"] = manifest.apply(lambda x: age_gender_dict[x["sample_id"]][0] if x["sample_id"] in age_gender_dict.keys() else None, axis=1)
    df["sexbirth"] = manifest.apply(lambda x: age_gender_dict[x["sample_id"]][1] if x["sample_id"] in age_gender_dict.keys() else None, axis=1)
    df_na = df[df['age'].isna()]
    df_na['group'] = "unknown"
    df_non_na = df[~(df['age'].isna())]#.astype({"age":int})
    df_non_na['group'] = pd.qcut(df_non_na['age'], ngroups).tolist()
    # df_non_na['group'] = df_non_na['group'].fillna("unknown")
    df_non_na['group'] = df_non_na['group'].apply(lambda x: str(x).translate(str.maketrans({"(":"", "]":"", ",":"_", " ":""})))
    df = pd.concat([df_non_na, df_na], axis=0)
    df = df[['sample_id','fileHeader','fq1','fq2','fa','age','sexbirth','group']]
    df.to_csv(os.path.join(prj_dir,"sample_summary.csv"), index=None)
    for group in df['group'].unique():
        os.makedirs(os.path.join(prj_dir,group), exist_ok=True)
    #     group_manifest = df[df['group']==group].loc[:,["sample_id","fa"]]
    #     group_manifest.to_csv(os.path.join(prj_dir, f"{group}.csv"))
    print(df["group"].unique())
    return 

def mapping(prj_dir):
    threads = MAX_THREADS
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)
    # manifest = pd.read_csv(manifest_path, header=None, names=["fileHeader","sample_id","fa"])
    groups = sample_summary["group"].unique().tolist()
    for group in groups:
        group_df = sample_summary[sample_summary['group']==group]
        print(f"[INFO] Group {group}:\tn={group_df.shape[0]}")

        fasta_input_list = group_df['fa'].tolist()
        out_dir = os.path.join(prj_dir, group, "semibin_output", "combined_output")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(prj_dir, group,'bamsam_output'), exist_ok=True)
        # Commands of concatenating, and building index (Mapping s1)
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='24:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='24:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} bar",
            f"SemiBin concatenate_fasta -o {out_dir} -i {' '.join(fasta_input_list)}",
            f"bowtie2-build --threads {threads} -f {out_dir}/concatenated.fa {out_dir}/concatenated.fa",
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s1_concat_indexing.pbs"), 'w') as f:
            f.writelines(bash_commands)
        # Commands of bowtie2 mapping (Mapping s2)
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} bar",
        ] + \
        [   f"echo \"{row['fileHeader']} s2_mapping started.\" ;" + \
            f"bowtie2 -q --fr --threads {threads} " + \
            f"-x \"{os.path.join(prj_dir,group,'semibin_output','combined_output','concatenated.fa')}\" " + \
            f"-1 \"{row['fq1']}\" " + \
            f"-2 \"{row['fq2']}\" " + \
            f"-S \"{os.path.join(prj_dir,group,'bamsam_output',row['fileHeader']+'.sam')}\"; " + \
            f"samtools view -@ {threads-1} -h -b -S \"{os.path.join(prj_dir,group,'bamsam_output',row['fileHeader']+'.sam')}\" -o \"{os.path.join(prj_dir,group,'bamsam_output',row['fileHeader']+'.bam')}\"; " + \
            f"samtools view -@ {threads-1} -b -F 4 \"{os.path.join(prj_dir,group,'bamsam_output',row['fileHeader']+'.bam')}\" -o \"{os.path.join(prj_dir,group,'bamsam_output',row['fileHeader']+'.mapped.bam')}\"; " + \
            f"samtools sort -@ {threads-1} -m 1000000000 \"{os.path.join(prj_dir,group,'bamsam_output',row['fileHeader']+'.mapped.bam')}\" -o \"{os.path.join(prj_dir,group,'bamsam_output',row['fileHeader']+'.mapped.sorted.bam')}\";" \
            f"rm {os.path.join(prj_dir,group,'bamsam_output',row['fileHeader']+'.sam')} ;" + \
            f"rm {os.path.join(prj_dir,group,'bamsam_output',row['fileHeader']+'.mapped.bam')} ;" + \
            f"echo \"{row['fileHeader']} s2_mapping finished.\" ;" \
            for _, row in group_df.iterrows() 
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s2_mapping.pbs"), 'w') as f:
            f.writelines(bash_commands)

def binning(prj_dir):
    threads = MAX_THREADS
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)
    groups = sample_summary["group"].unique().tolist()
    for group in groups:
        group_df = sample_summary[sample_summary['group']==group]
        print(f"[INFO] Group {group}:\tn={group_df.shape[0]}")
        # Commands of generating sequence files (Binning 1 s3)
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus='28',ngpus=0,mem=f'1000GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='hugemembw',jobfs='350GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus='28',ngpus=0,mem=f'1000GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} bar",
            f"mkdir {os.path.join(prj_dir,group,'semibin_output','multi_sample_output')}",
            f"SemiBin generate_sequence_features_multi -i {os.path.join(prj_dir,group,'semibin_output','combined_output','concatenated.fa')} " + \
            f"-b {os.path.join(prj_dir,group,'bamsam_output', '*.mapped.sorted.bam')} " + \
            f"-o {os.path.join(prj_dir,group,'semibin_output','multi_sample_output')}",
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s3_generating_sequence_files.pbs"), 'w') as f:
            f.writelines(bash_commands)
        # Commands of self-training (Binning 2 s4)
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} bar",
        ] + \
        [   f"echo \"{row['fileHeader']} s4_self-training start.\";" + \
            f"SemiBin2 train_self --data {os.path.join(prj_dir,group,'semibin_output','multi_sample_output','samples',row['fileHeader'],'data.csv')} " + \
            f"--data-split {os.path.join(prj_dir,group,'semibin_output','multi_sample_output','samples',row['fileHeader'],'data_split.csv')} " + \
            f"--output {os.path.join(prj_dir,group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'])} ;" + \
            f"echo \"{row['fileHeader']} s4_self-training finished.\";" \
            for _, row in group_df.iterrows()
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s4_self-training.pbs"), 'w') as f:
            f.writelines(bash_commands)
        # Commands of binning (Binning 3 s5)
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} bar",
        ] + \
        [   f"echo \"{row['fileHeader']} s5_binning start.\";" + \
            f"SemiBin2 bin_short " + \
            f"-i {row['fa']} " + \
            f"--model {os.path.join(prj_dir,group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'],'model.h5')} " + \
            f"--data {os.path.join(prj_dir,group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'],'data.csv')} " + \
            f"-o {os.path.join(prj_dir,group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'])} ;" + \
            f"echo \"{row['fileHeader']} s5_binning finished.\";"
            for _, row in group_df.iterrows()
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group,f"{group}_s5_binning.pbs"), 'w') as f:
            f.writelines(bash_commands)

    return

# COMPLETENESS and CONTAMINATION PREDICTION WITH CHECKM
def checkM(prj_dir):
    threads = MAX_THREADS
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)
    groups = sample_summary["group"].unique().tolist()
    for group in groups:
        group_df = sample_summary[sample_summary['group']==group]
        print(f"[INFO] Group {group}:\tn={group_df.shape[0]}")
        os.makedirs(os.path.join(prj_dir, group, "checkm_output"), exist_ok=True)
        # Commands of running checkM2 (CheckM2 1 s6)
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} checkm2",
        ] + \
        [   f"echo \"{row['fileHeader']} s6_checkm start.\";" + \
            f"checkm2 predict --threads {threads} -x .fa.gz --force " + \
            f"--input {os.path.join(prj_dir, group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'], 'output_bins')} " + \
            f"--output-directory {os.path.join(prj_dir, group, 'checkm_output', row['fileHeader'])} ;" + \
            f"echo \"{row['fileHeader']} s6_checkm finished.\";"
            for _, row in group_df.iterrows()
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s6_checkm.pbs"), 'w') as f:
            f.writelines(bash_commands)
    return

def filtering(prj_dir, drep_coverage=0.3, drep_cutoff=0.98):
    threads = MAX_THREADS
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)
    groups = sample_summary["group"].unique().tolist()
    for group in groups:
        # Commands of selecting genomes (filtering 1 s7)
        group_df = sample_summary[sample_summary['group']==group]
        print(f"[INFO] Group {group}:\tn={group_df.shape[0]}")
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} bar",
        ] + \
        [   
f'''
for folder in {os.path.join(prj_dir, group, "checkm_output")}/*; do
    if [ -d "$folder" ]; then
        # Extract folder name
        folder_name=$(basename "$folder")

        # Check if the quality_report.txt file exists in the folder
        if [ -f "$folder/quality_report.tsv" ]; then
            # Rename the file by prefixing the folder name
            mv "$folder/quality_report.tsv" "$folder/${{folder_name}}_quality_report.tsv"
            echo "File in $folder renamed to ${{folder_name}}_quality_report.tsv"
        else
            echo "No quality_report.tsv file found in $folder"
        fi
    fi
done
''',
f'''
samples_dir="{os.path.join(prj_dir,group,"semibin_output","multi_sample_output","samples")}"
draft_genomes_dir="{os.path.join(prj_dir,group,"semibin_output","multi_sample_output","samples","draft_genomes")}"

#modify path to directory
for folder in {os.path.join(prj_dir,group,"semibin_output","multi_sample_output","samples","*","output_bins")}; do
    echo "$folder"
    sampleID=$(basename $(dirname "$folder"))
    echo "$sampleID"
    quality_report="{os.path.join(prj_dir, group, "checkm_output")}/${{sampleID}}/${{sampleID}}_quality_report.tsv"
    echo "$quality_report"
    mkdir -p "$draft_genomes_dir/$sampleID"

    # first line (header of the quality report.tsv files) will have output: "Bin Name not high quality for sample $sampleID", it's fine and the program will continue
    while IFS=$'\\t' read -r Name Completeness Contamination _; do
        echo "Bin: $Name Completeness: $Completeness, Contamination: $Contamination"

        if (( $(echo "$Completeness > 60 && $Contamination < 5" | bc -l) )); then
            bin_path="${{folder}}/${{Name}}.gz"
            cp -r "$bin_path" "$draft_genomes_dir/$sampleID"
            mv "$draft_genomes_dir/\"$sampleID\"/${{Name}}.gz" "$draft_genomes_dir/\"$sampleID\"/${{sampleID}}_${{Name}}.gz"
            echo "Copied bin $Name for sample $sampleID"
        else
            echo "Bin $Name not high quality for sample $sampleID"
        fi
    done < "$quality_report"
done
            '''
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s7_draft-genomes.pbs"), 'w') as f:
            f.writelines(bash_commands)

        # Commands of dereplication (filtering 2 s8)
        drep_draft_genomes_dir = os.path.join(prj_dir, group, 'drep_draft_genomes','nc{}_sa{}'.format(drep_coverage,drep_cutoff))
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            "source ~/.bashrc",
            "micromamba activate drep",
        ] + \
        [   
            f"mkdir -p {drep_draft_genomes_dir}",
            f"dRep dereplicate {drep_draft_genomes_dir} -g {os.path.join(prj_dir, group, 'semibin_output', 'multi_sample_output', 'samples', 'draft_genomes', '*', '*.fa.gz')} --S_algorithm fastANI -nc {drep_coverage} -sa {drep_cutoff} --ignoreGenomeQuality",
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s8_derep_nc{drep_coverage}_sa{drep_cutoff}.pbs"), 'w') as f:
            f.writelines(bash_commands)
    return

def classify(prj_dir, drep_coverage=0.3, drep_cutoff=0.98):
    threads = MAX_THREADS
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)
    groups = sample_summary["group"].unique().tolist()
    for group in groups:
        group_df = sample_summary[sample_summary['group']==group]
        print(f"[INFO] Group {group}:\tn={group_df.shape[0]}")

        # Commands of classification with GTDBtk (classify 1 s9)
        drep_draft_genomes_dir = os.path.join(prj_dir, group, 'drep_draft_genomes',f'nc{drep_coverage}_sa{drep_cutoff}', 'dereplicated_genomes') #os.path.join(prj_dir, group, 'semibin_output', 'multi_sample_output', 'samples', 'drep_draft_genomes', 'dereplicated_genomes')
        out_dir = os.path.join(prj_dir, group,'gtdbtk_output', f'nc{drep_coverage}_sa{drep_cutoff}')
        mash_db = os.path.join(prj_dir, group,'gtdbtk_output', f'nc{drep_coverage}_sa{drep_cutoff}', 'mash_db')
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} gtdbtk",
        ] + \
        [   
            f"gtdbtk classify_wf --genome_dir {drep_draft_genomes_dir} --out_dir {out_dir} --mash_db {mash_db} -x gz --cpus {threads}",
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s9_classify_nc{drep_coverage}_sa{drep_cutoff}.pbs"), 'w') as f:
            f.writelines(bash_commands)

    return 

def quantify(prj_dir, drep_coverage=0.3, drep_cutoff=0.98):
    threads = MAX_THREADS
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)
    groups = sample_summary["group"].unique().tolist()
    for group in groups:
        group_df = sample_summary[sample_summary['group']==group]

        quant_output = os.path.join(prj_dir,group,'quant_output', f'nc{drep_coverage}_sa{drep_cutoff}')
        drep_draft_genomes_dir = os.path.join(prj_dir, group, 'drep_draft_genomes',f'nc{drep_coverage}_sa{drep_cutoff}','dereplicated_genomes')
        merged_fasta_dir = os.path.join(quant_output,'merged_drep_genomes.fa')
        salmon_index = os.path.join(quant_output,'salmon_index')
        abundance_output_dir = os.path.join(quant_output,'bin_abundance_table.tab')

        print(f"[INFO] Group {group}:\tn={group_df.shape[0]}")
        # Commands of quantification with salmon - renaming, merging, and indexing (quant 1 s10)
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} salmon",
        ] + \
        [   
            f"mkdir -p {salmon_index}",
            f"if [ -f {merged_fasta_dir} ]; then",
            f"    rm {merged_fasta_dir};",
            f"fi",
            f"for file in $(ls {drep_draft_genomes_dir}); do", 
            f"    if [ -f \"{drep_draft_genomes_dir}/$file\" ]; then", 
            f"        header=$(echo \"$file\" | cut -d'.' -f1)", 
            f"        gunzip {drep_draft_genomes_dir}/$file",
            f"        sed -i \"s/^>/>${{header}}_/\" {drep_draft_genomes_dir}/${{file%.gz}}", 
            f"        cat {drep_draft_genomes_dir}/${{file%.gz}} >> {merged_fasta_dir}",
            f"    fi", 
            f"done", 

        ] + \
        [   
            f"salmon index -p {threads} -t {merged_fasta_dir} -i {salmon_index}",
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s10_quant_index_nc{drep_coverage}_sa{drep_cutoff}.pbs"), 'w') as f:
            f.writelines(bash_commands)

        # Commands of quantification with salmon - quantification (quant 2 s11)
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} salmon",
        ] + \
        [   
            f"echo \"{row['fileHeader']} s11_quantification start.\"; " + \
            f"salmon quant -i {salmon_index} --libType IU " + \
            f"-1 {row['fq1']} -2 {row['fq2']} " + \
            f"-o {os.path.join(quant_output,'alignment_files',row['fileHeader']+'.quant')} --meta -p {threads}; " + \
            f"echo \"{row['fileHeader']} s11_quantification finished.\";"
            for _, row in group_df.iterrows()
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s11_quantify_nc{drep_coverage}_sa{drep_cutoff}.pbs"), 'w') as f:
            f.writelines(bash_commands)
        # Commands of post-quantification process - split salmon output into bins (quant 3 s12)
        metawrap_summarise_salmon = os.path.join(METAWRAP_SCRIPTS_DIR,'summarize_salmon_files.py')
        metawrap_split_salmon = os.path.join(METAWRAP_SCRIPTS_DIR,'split_salmon_out_into_bins.py')
        if JOB_MANAGER=='gadi':
            header = GadiHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o='',
                project='mp96',storage=GADI_STORAGE,node_type='normalsl',jobfs='20GB'
            ).get_header()
        if JOB_MANAGER=='pbs':
            header = PBSHeader(
                job_name='',ncpus=MAX_THREADS,ngpus=0,mem=f'{int(MAX_THREADS*4)}GB',
                walltime='48:00:00',mail_addr=MAIL_ADDRESS,log_e='',log_o=''
            )
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} salmon",
        ] + \
        [   
            f"cd {os.path.join(quant_output,'alignment_files')}", 
            f"{metawrap_summarise_salmon}",
            f"mkdir -p {os.path.join(quant_output,'quant_files')}; mv {os.path.join(quant_output, 'alignment_files', '*.quant.counts')} {os.path.join(quant_output,'quant_files')} ",
            f"{metawrap_split_salmon} {os.path.join(quant_output,'quant_files')} {drep_draft_genomes_dir} {merged_fasta_dir} > {abundance_output_dir}",
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, group, f"{group}_s12_post_quant_nc{drep_coverage}_sa{drep_cutoff}.pbs"), 'w') as f:
            f.writelines(bash_commands)
    return

def complete_check(prj_dir, drep_coverage=0.3, drep_cutoff=0.98):

    def check_s1(prj_dir, group, sample_summary):
        files = os.listdir(os.path.join(prj_dir, group, "semibin_output", "combined_output"))
        bt2_indices = [file for file in files if ".bt2" in file]
        bt2_revs = [file for file in files if ".rev" in file]
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "concat": False, #[False]*len(sample_summary[sample_summary["group"]==group]["fileHeader"]), 
            "indexing": False, #[False]*len(sample_summary[sample_summary["group"]==group]["fileHeader"]),
        })
        

        if os.path.isfile(os.path.join(prj_dir, group, "semibin_output", "combined_output", "concatenated.fa")): status["concat"] = True
        else: status["concat"] = False

        if len(bt2_indices)==6 and len(bt2_revs)==2: status["indexing"] = True
        else: status["indexing"] = False

        return status
    
    def check_s2(prj_dir, group, sample_summary):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "bam": False,
            # "mapped.bam": False,
            "mapped.sorted.bam": False,
            "mapping": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(prj_dir, group, "bamsam_output", f"{x.name}.bam")) else False, 
            # True if os.path.isfile(os.path.join(prj_dir, group, "bamsam_output", f"{x.name}.mapped.bam")) else False, 
            True if os.path.isfile(os.path.join(prj_dir, group, "bamsam_output", f"{x.name}.mapped.sorted.bam")) else False,
            False,
        ], index=["bam", "mapped.sorted.bam", "mapping"]), axis=1) # , "mapped.bam"
        status["mapping"] = status.apply(lambda x: True if x.iloc[:2].all() else False, axis=1)
        return status

    def check_s3(prj_dir, group, sample_summary):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "data": False,
            "data_cov": False,
            "data_split": False,
            "data_split_cov": False,
            "generating": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(prj_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "data.csv")) else False, 
            True if os.path.isfile(os.path.join(prj_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "data_cov.csv")) else False, 
            True if os.path.isfile(os.path.join(prj_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "data_split.csv")) else False,
            True if os.path.isfile(os.path.join(prj_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "data_split_cov.csv")) else False,
            False,
        ], index=["data", "data_cov", "data_split", "data_split_cov", "generating"]), axis=1)
        status["generating"] = status.apply(lambda x: True if x.iloc[:4].all() else False, axis=1)
        return status

    def check_s4(prj_dir, group, sample_summary):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "self-training": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(prj_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "model.h5")) else False, 
        ], index=["self-training"]), axis=1)      
        return status
    
    def check_s5(prj_dir, group, sample_summary):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "contig_bins": False,
            "recluster_bins_info": False,
            "output_bins": False,
            "binning": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(prj_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "contig_bins.tsv")) else False, 
            True if os.path.isfile(os.path.join(prj_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "recluster_bins_info.tsv")) else False,
            True if os.path.isdir(os.path.join(prj_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "output_bins")) else False, 
            False,
        ], index=["self-contig_bins", "recluster_bins_info", "output_bins", "binning"]), axis=1)
        status["binning"] = status.apply(lambda x: True if x.iloc[:3].all() else False, axis=1)  
        return status
    
    def check_s6(prj_dir, group, sample_summary):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "checkm": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if (os.path.isfile(os.path.join(prj_dir, group, "checkm_output", f"{x.name}", "quality_report.tsv")) or os.path.isfile(os.path.join(prj_dir, group, "checkm_output", f"{x.name}", f"{x.name}_quality_report.tsv"))) else False, 
        ], index=["checkm"]), axis=1)
        return status
    
    def check_s7(prj_dir, group, sample_summary):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "draft-genomes": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isdir(os.path.join(prj_dir, group, "semibin_output", "multi_sample_output", "samples", "draft_genomes", f"{x.name}")) else False, 
        ], index=["draft-genomes"]), axis=1)
        return status
    
    def check_s8(prj_dir, group, sample_summary, drep_coverage=drep_coverage, drep_cutoff=drep_cutoff):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "drep": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(prj_dir, group, "drep_draft_genomes", f'nc{drep_coverage}_sa{drep_cutoff}', 'data_tables', 'Widb.csv')) else False, 
        ], index=["drep"]), axis=1)
        return status

    def check_s9(prj_dir, group, sample_summary, drep_coverage=drep_coverage, drep_cutoff=drep_cutoff):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "classification": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(prj_dir, group, "gtdbtk_output", f'nc{drep_coverage}_sa{drep_cutoff}', "gtdbtk.bac120.summary.tsv",)) else False, 
        ], index=["classification"]), axis=1)
        return status
    
    def check_s10(prj_dir, group, sample_summary, drep_coverage=drep_coverage, drep_cutoff=drep_cutoff):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "renaming_and_drep_indexing": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(prj_dir, group, "quant_output", f'nc{drep_coverage}_sa{drep_cutoff}', "salmon_index", 'info.json',)) else False, 
        ], index=["renaming_and_drep_indexing"]), axis=1)
        return status
    
    def check_s11(prj_dir, group, sample_summary, drep_coverage=drep_coverage, drep_cutoff=drep_cutoff):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "quantification": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(prj_dir, group, "quant_output", f'nc{drep_coverage}_sa{drep_cutoff}', 'alignment_files', f'{x.name}.quant', 'quant.sf')) else False, 
        ], index=["quantification"]), axis=1)
        return status
    
    # abundance_output_dir = os.path.join(quant_output,'bin_abundance_table.tab')
    def check_s12(prj_dir, group, sample_summary, drep_coverage=drep_coverage, drep_cutoff=drep_cutoff):
        status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "quant_postproc": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(prj_dir, group, 'quant_output', f'nc{drep_coverage}_sa{drep_cutoff}', 'bin_abundance_table.tab')) else False, 
        ], index=["quant_postproc"]), axis=1)
        return status
    
    sample_summary = pd.read_csv(os.path.join(prj_dir, "sample_summary.csv"), sep=',')
    groups = sample_summary["group"].sort_values().unique()

    steps = {
        "s1": check_s1, 
        "s2": check_s2,
        "s3": check_s3,
        "s4": check_s4,
        "s5": check_s5,
        "s6": check_s6,
        "s7": check_s7,
        "s8": check_s8,
        "s9": check_s9,
        "s10": check_s10,
        "s11": check_s11,
        "s12": check_s12,
    }
    status_list = []
    for group in groups:
        group_status = pd.DataFrame({
            "fileHeader": sample_summary[sample_summary["group"]==group]["fileHeader"],
            "group": group,
        }).set_index("fileHeader")

        for key in steps.keys():
            s_status = steps[key](prj_dir, group, sample_summary)
            group_status = pd.merge(group_status, s_status, how='left', on='fileHeader')

        status_list.append(group_status)

        summary = pd.DataFrame({
            "fileHeader": group_status.index.tolist(),
            "group": group,
            "s1_concat_indexing": group_status[['concat', 'indexing']].apply(lambda x: x.all(), axis=1),
            "s2_mapping": group_status["mapping"],
            "s3_generating": group_status["generating"],
            "s4_self-training": group_status["self-training"],
            "s5_binning": group_status["binning"],
            "s6_checkm": group_status["checkm"],
            "s7_draft-genomes": group_status["draft-genomes"],
            "s8_drep": group_status["drep"],
            "s9_classification": group_status["classification"],
            "s10_quant_indexing": group_status["renaming_and_drep_indexing"],
            "s11_quantification": group_status["quantification"],
            "s12_quant_postproc": group_status["quant_postproc"],
        })
        summary = summary.iloc[:,2:].apply(lambda x: str(x[x==True].count())+f"/{summary.shape[0]}", axis=0)
        summary = pd.DataFrame(pd.concat([pd.Series([group], index=["group"]), summary], axis=0)).T
        print("-"*100)
        print(summary.to_string(index=False, justify='center', ))
        print("-"*100)
        
    status = pd.concat(status_list, axis=0)
    status.to_csv(os.path.join(prj_dir, "completeness.csv"))
    
    

    return 

def main():
    pwd = os.getcwd()

    parser = argparse.ArgumentParser(prog="Binning Pipeline", description="This is a commandline software for the pipeline.")
    parser.add_argument("-p","--prj_dir", default=pwd, help="Specify your project directory. (e.g. './test', defalut: \'./\')")

    subparsers  = parser.add_subparsers(
        title="Modules",
        dest="modules",
        description="Modules that proceed different functions.", 
        help="Please specify one and only one of these options.", 
        required=True
    )

    subparser_grouping = subparsers.add_parser("grouping", help="grouping by age")
    subparser_grouping.add_argument("-m","--manifest", required=True, help="a 4-column csv file: fileHeader,fq1,fq2,fa")
    subparser_grouping.add_argument("-a","--age_gender", required=True, help="a three-column csv file: sample_id,age,sexbirth")
    subparser_grouping.add_argument("-n","--ngroups", type=int, help="default: 4, groups to split")

    subparser_mapping = subparsers.add_parser("mapping", help="concatenating and building index, and mapping reads to contigs")
    # subparser_mapping.add_argument("-g","--group", required=True, help="please specify a group name")

    subparser_binning = subparsers.add_parser("binning", help="generating sequence files, self-training, and binning")
    # subparser_binning.add_argument("-g","--group", required=True, help="please specify a group name")

    subparser_binning = subparsers.add_parser("checkm", help="provide quality_report.tsv per sample with usefull information about quality of each bin")
    # subparser_binning.add_argument("-g","--group", required=True, help="please specify a group name")

    subparser_check = subparsers.add_parser("check", help="check completeness of each step.")
    subparser_check.add_argument("-nc","--drep_coverage", type=float, default=0.3, help="default: 0.3, nucleotide coverage for dreplication with salmon")
    subparser_check.add_argument("-sa","--drep_ani", type=float, default=0.98, help="default: 0.98, average nucleotide identity for dreplication with salmon")

    subparser_filter = subparsers.add_parser("filter", help="select genomes and dereplication.")
    subparser_filter.add_argument("-nc","--drep_coverage", type=float, default=0.3, help="default: 0.3, nucleotide coverage for dreplication with salmon")
    subparser_filter.add_argument("-sa","--drep_ani", type=float, default=0.98, help="default: 0.98, average nucleotide identity for dreplication with salmon")

    subparser_classify = subparsers.add_parser("classify", help="classify MAGs with GTDB-TK.")
    subparser_classify.add_argument("-nc","--drep_coverage", type=float, default=0.3, help="default: 0.3, nucleotide coverage for dreplication with salmon")
    subparser_classify.add_argument("-sa","--drep_ani", type=float, default=0.98, help="default: 0.98, average nucleotide identity for dreplication with salmon")

    subparser_quant = subparsers.add_parser("quant", help="quantify MAGs with salmon.")
    subparser_quant.add_argument("-nc","--drep_coverage", type=float, default=0.3, help="default: 0.3, nucleotide coverage for dreplication with salmon")
    subparser_quant.add_argument("-sa","--drep_ani", type=float, default=0.98, help="default: 0.98, average nucleotide identity for dreplication with salmon")

    args = parser.parse_args()

    if args.modules=="grouping":
        group_by_age(
            prj_dir=args.prj_dir, 
            manifest=args.manifest,
            age_gender=args.age_gender,
            ngroups=args.ngroups,
        )
    if args.modules=="mapping":
        mapping(
            prj_dir=args.prj_dir, 
        )
    if args.modules=="binning":
        binning(
            prj_dir=args.prj_dir, 
        )
    
    if args.modules=="checkm":
        checkM(
            prj_dir=args.prj_dir, 
        )

    if args.modules=="check":
        complete_check(
            prj_dir=args.prj_dir, drep_coverage=args.drep_coverage, drep_cutoff=args.drep_ani
        )

    if args.modules=="filter":
        filtering(
            prj_dir=args.prj_dir, drep_coverage=args.drep_coverage, drep_cutoff=args.drep_ani
        )

    if args.modules=="classify":
        classify(
            prj_dir=args.prj_dir, drep_coverage=args.drep_coverage, drep_cutoff=args.drep_ani
        )
    
    if args.modules=="quant":
        quantify(
            prj_dir=args.prj_dir, drep_coverage=args.drep_coverage, drep_cutoff=args.drep_ani
        )

if __name__=='__main__':
    main()

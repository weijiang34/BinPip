import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import os 
import argparse
import re
import glob

CONDA_DIR = '/g/data/oo46/wj6768/miniconda3'
MAMBA_DIR = ''
METAWRAP_SCRIPTS_DIR = '/g/data/mp96/wj6768/tools/metaWRAP/bin/metawrap-scripts'

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
    group_dir = os.path.join(prj_dir, "grouped")
    
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
        os.makedirs(os.path.join(group_dir,group), exist_ok=True)
    #     group_manifest = df[df['group']==group].loc[:,["sample_id","fa"]]
    #     group_manifest.to_csv(os.path.join(prj_dir, f"{group}.csv"))
    print(df["group"].unique())
    return 

def mapping(prj_dir):
    threads = MAX_THREADS
    group_dir = os.path.join(prj_dir, "grouped")
    
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)
    # manifest = pd.read_csv(manifest_path, header=None, names=["fileHeader","sample_id","fa"])
    groups = sample_summary["group"].unique().tolist()
    for group in groups:
        group_df = sample_summary[sample_summary['group']==group]
        print(f"[INFO] Group {group}:\tn={group_df.shape[0]}")

        fasta_input_list = group_df['fa'].tolist()
        out_dir = os.path.join(group_dir,group, "semibin_output", "combined_output")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(group_dir,group,'bamsam_output'), exist_ok=True)
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
        with open(os.path.join(group_dir,group, f"{group}_s1_concat_indexing.pbs"), 'w') as f:
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
            f"-x \"{os.path.join(group_dir,group,'semibin_output','combined_output','concatenated.fa')}\" " + \
            f"-1 \"{row['fq1']}\" " + \
            f"-2 \"{row['fq2']}\" " + \
            f"-S \"{os.path.join(group_dir,group,'bamsam_output',row['fileHeader']+'.sam')}\"; " + \
            f"samtools view -@ {threads-1} -h -b -S \"{os.path.join(group_dir,group,'bamsam_output',row['fileHeader']+'.sam')}\" -o \"{os.path.join(group_dir,group,'bamsam_output',row['fileHeader']+'.bam')}\"; " + \
            f"samtools view -@ {threads-1} -b -F 4 \"{os.path.join(group_dir,group,'bamsam_output',row['fileHeader']+'.bam')}\" -o \"{os.path.join(group_dir,group,'bamsam_output',row['fileHeader']+'.mapped.bam')}\"; " + \
            f"samtools sort -@ {threads-1} -m 1000000000 \"{os.path.join(group_dir,group,'bamsam_output',row['fileHeader']+'.mapped.bam')}\" -o \"{os.path.join(group_dir,group,'bamsam_output',row['fileHeader']+'.mapped.sorted.bam')}\";" \
            f"rm {os.path.join(group_dir,group,'bamsam_output',row['fileHeader']+'.sam')} ;" + \
            f"rm {os.path.join(group_dir,group,'bamsam_output',row['fileHeader']+'.mapped.bam')} ;" + \
            f"echo \"{row['fileHeader']} s2_mapping finished.\" ;" \
            for _, row in group_df.iterrows() 
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(group_dir,group, f"{group}_s2_mapping.pbs"), 'w') as f:
            f.writelines(bash_commands)

def binning(prj_dir):
    threads = MAX_THREADS
    group_dir = os.path.join(prj_dir, "grouped")
    
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
            
            f"mkdir {os.path.join(group_dir,group,'semibin_output','multi_sample_output')}",
            f"SemiBin generate_sequence_features_multi -i {os.path.join(group_dir,group,'semibin_output','combined_output','concatenated.fa')} " + \
            f"-b {os.path.join(group_dir,group,'bamsam_output', '*.mapped.sorted.bam')} " + \
            f"-o {os.path.join(group_dir,group,'semibin_output','multi_sample_output')}",
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(group_dir,group, f"{group}_s3_generating_sequence_files.pbs"), 'w') as f:
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
            f"SemiBin2 train_self --data {os.path.join(group_dir,group,'semibin_output','multi_sample_output','samples',row['fileHeader'],'data.csv')} " + \
            f"--data-split {os.path.join(group_dir,group,'semibin_output','multi_sample_output','samples',row['fileHeader'],'data_split.csv')} " + \
            f"--output {os.path.join(group_dir,group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'])} ;" + \
            f"echo \"{row['fileHeader']} s4_self-training finished.\";" \
            for _, row in group_df.iterrows()
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(group_dir,group, f"{group}_s4_self-training.pbs"), 'w') as f:
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
            f"--model {os.path.join(group_dir,group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'],'model.h5')} " + \
            f"--data {os.path.join(group_dir,group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'],'data.csv')} " + \
            f"-o {os.path.join(group_dir,group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'])} ;" + \
            f"echo \"{row['fileHeader']} s5_binning finished.\";"
            for _, row in group_df.iterrows()
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(group_dir,group,f"{group}_s5_binning.pbs"), 'w') as f:
            f.writelines(bash_commands)

    return

# COMPLETENESS and CONTAMINATION PREDICTION WITH CHECKM
def checkM(prj_dir):
    threads = MAX_THREADS
    group_dir = os.path.join(prj_dir, "grouped")
    
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)
    groups = sample_summary["group"].unique().tolist()
    for group in groups:
        group_df = sample_summary[sample_summary['group']==group]
        print(f"[INFO] Group {group}:\tn={group_df.shape[0]}")
        os.makedirs(os.path.join(group_dir,group, "checkm_output"), exist_ok=True)
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
            f"--input {os.path.join(group_dir,group,'semibin_output','multi_sample_output', 'samples', row['fileHeader'], 'output_bins')} " + \
            f"--output-directory {os.path.join(group_dir,group, 'checkm_output', row['fileHeader'])} ;" + \
            f"echo \"{row['fileHeader']} s6_checkm finished.\";"
            for _, row in group_df.iterrows()
        ]
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(group_dir,group, f"{group}_s6_checkm.pbs"), 'w') as f:
            f.writelines(bash_commands)
    return

def filtering(prj_dir, drep_coverage=0.3, drep_ani=0.99):
    '''
    rename, filter low-quality, merge in group, and derep across groups
    '''
    threads = MAX_THREADS
    group_dir = os.path.join(prj_dir, "grouped")
    
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
         # rename quality report files
f'''
for folder in {os.path.join(group_dir,group, "checkm_output")}/*; do
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
# move and rename genomes files
f'''
samples_dir="{os.path.join(group_dir,group,"semibin_output","multi_sample_output","samples")}"
draft_genomes_dir="{os.path.join(prj_dir,"draft_genomes")}"

#modify path to directory
for folder in {os.path.join(group_dir,group,"semibin_output","multi_sample_output","samples","*","output_bins")}; do
    echo "$folder"
    sampleID=$(basename $(dirname "$folder"))
    echo "$sampleID"
    quality_report="{os.path.join(group_dir,group, "checkm_output")}/${{sampleID}}/${{sampleID}}_quality_report.tsv"
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
        with open(os.path.join(group_dir, group, f"{group}_s7_draft-genomes.pbs"), 'w') as f:
            f.writelines(bash_commands)

    # Commands of dereplication (filtering 2 s8)
    drep_output_dir = os.path.abspath(os.path.join(prj_dir, 'drep_draft_genomes','nc{}_sa{}'.format(drep_coverage,drep_ani)))
    drep_draft_genomes_dir = os.path.abspath(os.path.join(drep_output_dir, 'dereplicated_genomes'))
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
        f"mkdir -p {drep_output_dir}",
        f"echo \"[INFO] Running dRep ...\"",
        f"dRep dereplicate {drep_output_dir} -g {os.path.join(prj_dir, 'draft_genomes', '*', '*.fa.gz')} --S_algorithm fastANI -nc {drep_coverage} -sa {drep_ani} -p {threads} --ignoreGenomeQuality",
        f"echo \"[INFO] Renaming dereplicated genomes ...\"",
        f"for file in $(ls {drep_draft_genomes_dir}/*.fa.gz); do", 
        f"    if [ -f \"$file\" ]; then", 
        f"        header=$(basename \"$file\" | cut -d'.' -f1)", 
        f"        gunzip $file",
        f"        sed -i \"s/^>/>${{header}}_/\" ${{file%.gz}}", 
        f"    fi", 
        f"done", 
        f"echo \"[INFO] Done.\"",
    ]
    bash_commands = [x+"\n" for x in bash_commands]
    with open(os.path.join(prj_dir, f"s8_derep_nc{drep_coverage}_sa{drep_ani}.pbs"), 'w') as f:
        f.writelines(bash_commands)
    return

def classify(prj_dir, drep_coverage=0.3, drep_ani=0.99):
    threads = MAX_THREADS
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)
    # Commands of classification with GTDBtk (classify 1 s9)
    drep_draft_genomes_dir = os.path.join(prj_dir, 'drep_draft_genomes',f'nc{drep_coverage}_sa{drep_ani}', 'dereplicated_genomes') #os.path.join(prj_dir, 'semibin_output', 'multi_sample_output', 'samples', 'drep_draft_genomes', 'dereplicated_genomes')
    out_dir = os.path.join(prj_dir,'gtdbtk_output', f'nc{drep_coverage}_sa{drep_ani}')
    mash_db = os.path.join(prj_dir,'gtdbtk_output', f'nc{drep_coverage}_sa{drep_ani}', 'mash_db')
    # check last step completeness
    # if complete:
    #   go on
    # else:
    #   let users use -f to force the process
    # now forcely go on
    
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
        f"gtdbtk classify_wf --genome_dir {drep_draft_genomes_dir} --out_dir {out_dir} --mash_db {mash_db} -x fa --cpus {threads}",
    ]
    bash_commands = [x+"\n" for x in bash_commands]
    with open(os.path.join(prj_dir, f"s9_classify_nc{drep_coverage}_sa{drep_ani}.pbs"), 'w') as f:
        f.writelines(bash_commands)

    return 

def quantify(prj_dir, drep_coverage=0.3, drep_ani=0.99, chunksize=10):
    threads = MAX_THREADS
    quant_output = os.path.join(prj_dir,'quant_output', f'nc{drep_coverage}_sa{drep_ani}')
    drep_draft_genomes_dir = os.path.join(prj_dir, 'drep_draft_genomes',f'nc{drep_coverage}_sa{drep_ani}','dereplicated_genomes')
    # drep_draft_genomes_dir = os.path.join(prj_dir, 'derep_genomes',f'nc{drep_coverage}_sa{drep_ani}','dereplicated_genomes')
    merged_fasta_path = os.path.join(quant_output,'merged_drep_genomes.fa')
    salmon_index = os.path.join(quant_output,'salmon_index')
    abundance_output_dir = os.path.join(quant_output,'bin_abundance_table.tab')
    
    sample_summary = pd.read_csv(os.path.join(prj_dir,"sample_summary.csv"), header=0)

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
        f"if [ -f {merged_fasta_path} ]; then",
        f"    rm {merged_fasta_path};",
        f"fi",
        f"cat {drep_draft_genomes_dir}/*.fa > {merged_fasta_path}",
        # rename dereplicated genomes with sampleID
        # f"for file in $(ls {drep_draft_genomes_dir}); do", 
        # f"    if [ -f \"{drep_draft_genomes_dir}/$file\" ]; then", 
        # f"        header=$(echo \"$file\" | cut -d'.' -f1)", 
        # f"        gunzip {drep_draft_genomes_dir}/$file",
        # f"        sed -i \"s/^>/>${{header}}_/\" {drep_draft_genomes_dir}/${{file%.gz}}", 
        # f"        cat {drep_draft_genomes_dir}/${{file%.gz}} >> {merged_fasta_path}",
        # f"    fi", 
        # f"done", 
    ] + \
    [   
        f"salmon index -p {threads} -t {merged_fasta_path} -i {salmon_index}",
    ]
    bash_commands = [x+"\n" for x in bash_commands]
    with open(os.path.join(prj_dir, f"s10_quant_index_nc{drep_coverage}_sa{drep_ani}.pbs"), 'w') as f:
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
    # make splitted pbs files and dir
    os.makedirs(os.path.join(prj_dir, f"s11_quantify_nc{drep_coverage}_sa{drep_ani}.pbs.split"), exist_ok=True)
    all_cmds = [   
        f"echo \"{row['fileHeader']} s11_quantification start.\"; " + \
        f"salmon quant -i {salmon_index} --libType IU " + \
        f"-1 {row['fq1']} -2 {row['fq2']} " + \
        f"-o {os.path.join(quant_output,'alignment_files',row['fileHeader']+'.quant')} --meta -p {threads}; " + \
        f"echo \"{row['fileHeader']} s11_quantification finished.\";"
        for _, row in sample_summary.iterrows()
    ]
    # write splitted commands
    chunked_cmds = [all_cmds[i:i + chunksize] for i in range(0, len(all_cmds), chunksize)]
    for idx, chunked_cmd in enumerate(chunked_cmds):
        bash_commands = header + \
        [
            f"source {os.path.join(CONDA_DIR,'bin','activate')} salmon",
        ] + chunked_cmd
        bash_commands = [x+"\n" for x in bash_commands]
        with open(os.path.join(prj_dir, f"s11_quantify_nc{drep_coverage}_sa{drep_ani}.pbs.split", f"s11_quantify_nc{drep_coverage}_sa{drep_ani}.pbs.{idx}"), 'w') as f:
            f.writelines(bash_commands)
    # write full commands
    bash_commands = header + \
    [
        f"source {os.path.join(CONDA_DIR,'bin','activate')} salmon",
    ] + all_cmds
    bash_commands = [x+"\n" for x in bash_commands]
    with open(os.path.join(prj_dir, f"s11_quantify_nc{drep_coverage}_sa{drep_ani}.pbs"), 'w') as f:
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
        f"{metawrap_split_salmon} {os.path.join(quant_output,'quant_files')} {drep_draft_genomes_dir} {merged_fasta_path} > {abundance_output_dir}",
    ]
    bash_commands = [x+"\n" for x in bash_commands]
    with open(os.path.join(prj_dir, f"s12_post_quant_nc{drep_coverage}_sa{drep_ani}.pbs"), 'w') as f:
        f.writelines(bash_commands)
    return

def mapping_rate(prj_dir, drep_coverage=0.3, drep_ani=0.99, single_log=None, *args, **kwargs):
    '''
    This functions requires the quantification step to be completed in one go, 
    otherwise it may fail when running. Please use with caution.
    '''
    # single job log
    quant_output_dir = os.path.join(prj_dir, 'quant_output', f'nc{drep_coverage}_sa{drep_ani}')
    quant_log_pattern = os.path.join(prj_dir, f's11_quantify_nc{drep_coverage}_sa{drep_ani}.pbs.o[0-9]*')
    quant_logs = glob.glob(quant_log_pattern)
    quant_log_newest = sorted(quant_logs, key=os.path.getmtime)[-1]
    log_single = [quant_log_newest]
    # split job logs
    quant_log_split_pattern = os.path.join(prj_dir, f's11_quantify_nc{drep_coverage}_sa{drep_ani}.pbs.split', f's11_quantify_nc{drep_coverage}_sa{drep_ani}.pbs.[0-9]*.o[0-9]*')
    quant_logs_split = glob.glob(quant_log_split_pattern)
    if len(quant_logs_split)==0:
        print(f"[INFO] No split quantification logs found, using single log: {quant_log_newest}")
        log_split = [0]
    else:
        df_quant_logs_split = pd.DataFrame(quant_logs_split, columns=['log'])
        df_quant_logs_split = pd.concat([df_quant_logs_split, df_quant_logs_split['log'].str.split('/', expand=True).iloc[:,-1].str.split('.pbs.', expand=True)[1].str.split('.o', expand=True).rename({0:'num', 1:'id_num'}, axis=1)], axis=1)
        df_quant_logs_split = df_quant_logs_split.groupby('num').apply(lambda x: x.sort_values('id_num').iloc[-1,:]).reset_index(drop=True)
        log_split = df_quant_logs_split['log'].unique().tolist()
    
    # make list of logs
    logs = log_single if max(list(map(lambda x: int(x.split('.o')[-1]), log_single))) > max(list(map(lambda x:int(x), log_split))) else log_split
    if single_log is not None:
        logs = [single_log]
    if len(logs)==0:
        print(f"[ERROR] No quantification log found.")
        return
    mapping_rates = []
    sample_names = []
    for log in logs:
        with open(log, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if '### [ output ] => ' in line:
                '''output names comes first in log'''
                sample_names.append(line.split('/')[-1].split('.')[0])
            if 'Mapping rate' in line:
                '''then comes the mapping rate'''
                mapping_rates.append(line.split()[-1])

    # start extracting mapping rates
    # print(len(mapping_rates), len(sample_names))
    # return 
    if len(mapping_rates)!=len(sample_names):
        print(f"[ERROR] Mapping rate are not extracted")
    else:
        mapping_rate_df = pd.DataFrame({
            "sample": sample_names,
            "mapping_rate": mapping_rates,
        })
        mapping_rate_df.to_csv(os.path.join(quant_output_dir, f'mapping_rate.csv'), index=False)
        mapping_rate_df['mapping_rate'] = mapping_rate_df['mapping_rate'].str.rstrip('%').astype('float')
        # plot distribution
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        sns.histplot(mapping_rate_df, x='mapping_rate', bins='auto', ax=ax, color='skyblue', shrink=0.85)
        ax = plt.gca()
        ax.set_xlim(0, 100)
        ax.set_ylabel('sample count')
        ax.set_xlabel('mapping rate (%)')
        plt.title(f'Mapping rate distribution ({prj_dir} nc={drep_coverage} sa={drep_ani})')
        plt.savefig(os.path.join(quant_output_dir, f'mapping_rate_distribution.png'))

    # for group in groups:
    #     quant_output_dir = os.path.join(group_dir, group, 'quant_output', f'nc{drep_coverage}_sa{drep_ani}')
    #     quant_log_pattern = os.path.join(group_dir, group, f'{group}_s11_quantify_nc{drep_coverage}_sa{drep_ani}.pbs.e*')
    #     quant_logs = glob.glob(quant_log_pattern)
    #     if len(quant_logs)==0:
    #         print(f"[ERROR] No quantification log found for group {group}")
    #         continue
    #     quant_log_newest = sorted(quant_logs, key=os.path.getmtime)[-1]
    #     with open(quant_log_newest, 'r') as f:
    #         lines = f.readlines()
    #     mapping_rates = []
    #     sample_names = []
    #     for line in lines:
    #         if 'Mapping rate' in line:
    #             mapping_rates.append(line.split()[-1])
    #         if '### [ output ] => ' in line:
    #             sample_names.append(line.split('/')[-1].split('.')[0])
    #     if len(mapping_rates)==len(sample_names):
    #         mapping_rate_df = pd.DataFrame({
    #             "sample": sample_names,
    #             "mapping_rate": mapping_rates,
    #         })
    #         mapping_rate_df.to_csv(os.path.join(quant_output_dir, f'mapping_rate.csv'), index=False)
    #         mapping_rate_dfs.append(mapping_rate_df)
    #     else:
    #         print(f"[ERROR] Mapping rate not extracted for group {group}")
    
    # mapping_rate_df = pd.concat(mapping_rate_dfs)
    # mapping_rate_df['mapping_rate'] = mapping_rate_df['mapping_rate'].str.rstrip('%').astype('float')
    # # plot distribution
    # fig = plt.figure(figsize=(10, 6))
    # gs = GridSpec(1, 1, figure=fig)
    # ax = fig.add_subplot(gs[0, 0])
    # sns.histplot(mapping_rate_df, x='mapping_rate', bins='auto', ax=ax, color='skyblue', shrink=0.85)
    # ax =plt.gca()
    # ax.set_ylabel('sample count')
    # ax.set_xlabel('mapping rate (%)')
    # plt.title(f'Mapping rate distribution ({prj_dir} nc={drep_coverage} sa={drep_ani})')
    # plt.savefig(os.path.join(quant_output_dir, f'mapping_rate_distribution.png'))
    
    return

class CompleteCheck:
    def __init__(self, prj_dir, drep_coverage=0.3, drep_ani=0.99):
        self.prj_dir = prj_dir
        self.drep_coverage = drep_coverage
        self.drep_ani = drep_ani
        self.group_dir = os.path.join(prj_dir, "grouped")
        self.sample_summary = pd.read_csv(os.path.join(prj_dir, "sample_summary.csv"), sep=',')
        self.groups = self.sample_summary["group"].sort_values().unique()
        self.drep_output_dir = os.path.join(prj_dir, 'drep_draft_genomes', f'nc{drep_coverage}_sa{drep_ani}')
        self.gtdbtk_output_dir = os.path.join(prj_dir, 'gtdbtk_output', f'nc{drep_coverage}_sa{drep_ani}')
        self.quant_output_dir = os.path.join(prj_dir, 'quant_output', f'nc{drep_coverage}_sa{drep_ani}')
        self.steps = {
            "s1": self.check_s1, 
            "s2": self.check_s2,
            "s3": self.check_s3,
            "s4": self.check_s4,
            "s5": self.check_s5,
            "s6": self.check_s6,
            "s7": self.check_s7,
            "s8": self.check_s8,
            "s9": self.check_s9,
            "s10": self.check_s10,
            "s11": self.check_s11,
            "s12": self.check_s12,
        }

    def check_s1(self, group, *args, **kwargs):
        files = os.listdir(os.path.join(self.group_dir, group, "semibin_output", "combined_output"))
        bt2_indices = [file for file in files if ".bt2" in file]
        bt2_revs = [file for file in files if ".rev" in file]
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "concat": False, #[False]*len(sample_summary[sample_summary["group"]==group]["fileHeader"]), 
            "indexing": False, #[False]*len(sample_summary[sample_summary["group"]==group]["fileHeader"]),
        })
        

        if os.path.isfile(os.path.join(self.group_dir, group, "semibin_output", "combined_output", "concatenated.fa")): status["concat"] = True
        else: status["concat"] = False

        if len(bt2_indices)==6 and len(bt2_revs)==2: status["indexing"] = True
        else: status["indexing"] = False

        return status
    
    def check_s2(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "bam": False,
            # "mapped.bam": False,
            "mapped.sorted.bam": False,
            "mapping": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(self.group_dir, group, "bamsam_output", f"{x.name}.bam")) else False, 
            # True if os.path.isfile(os.path.join(group_dir,group, "bamsam_output", f"{x.name}.mapped.bam")) else False, 
            True if os.path.isfile(os.path.join(self.group_dir, group, "bamsam_output", f"{x.name}.mapped.sorted.bam")) else False,
            False,
        ], index=["bam", "mapped.sorted.bam", "mapping"]), axis=1) # , "mapped.bam"
        status["mapping"] = status.apply(lambda x: True if x.iloc[:2].all() else False, axis=1)
        return status

    def check_s3(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "data": False,
            "data_cov": False,
            "data_split": False,
            "data_split_cov": False,
            "generating": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(self.group_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "data.csv")) else False, 
            True if os.path.isfile(os.path.join(self.group_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "data_cov.csv")) else False, 
            True if os.path.isfile(os.path.join(self.group_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "data_split.csv")) else False,
            True if os.path.isfile(os.path.join(self.group_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "data_split_cov.csv")) else False,
            False,
        ], index=["data", "data_cov", "data_split", "data_split_cov", "generating"]), axis=1)
        status["generating"] = status.apply(lambda x: True if x.iloc[:4].all() else False, axis=1)
        return status

    def check_s4(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "self-training": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(self.group_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "model.h5")) else False, 
        ], index=["self-training"]), axis=1)      
        return status
    
    def check_s5(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "contig_bins": False,
            "recluster_bins_info": False,
            "output_bins": False,
            "binning": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(self.group_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "contig_bins.tsv")) else False, 
            True if os.path.isfile(os.path.join(self.group_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "recluster_bins_info.tsv")) else False,
            True if os.path.isdir(os.path.join(self.group_dir, group, "semibin_output", "multi_sample_output", "samples", f"{x.name}", "output_bins")) else False, 
            False,
        ], index=["self-contig_bins", "recluster_bins_info", "output_bins", "binning"]), axis=1)
        status["binning"] = status.apply(lambda x: True if x.iloc[:3].all() else False, axis=1)  
        return status
    
    def check_s6(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "checkm": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if (os.path.isfile(os.path.join(self.group_dir, group, "checkm_output", f"{x.name}", "quality_report.tsv")) or os.path.isfile(os.path.join(self.group_dir, group, "checkm_output", f"{x.name}", f"{x.name}_quality_report.tsv"))) else False, 
        ], index=["checkm"]), axis=1)
        return status
    
    def check_s7(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "draft-genomes": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isdir(os.path.join(self.prj_dir, "draft_genomes", f"{x.name}")) else False, 
        ], index=["draft-genomes"]), axis=1)
        return status
    
    def check_s8(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "drep": False,
        }).set_index("fileHeader")
        status['drep'] = [True] * status['drep'].shape[0] if os.path.exists(os.path.join(self.drep_output_dir, 'data_tables', 'Widb.csv')) else [False] * status['drep'].shape[0]
        return status

    def check_s9(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "classification": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(self.gtdbtk_output_dir, "gtdbtk.bac120.summary.tsv",)) else False, 
        ], index=["classification"]), axis=1)
        return status
    
    def check_s10(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "renaming_and_drep_indexing": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(self.quant_output_dir, "salmon_index", 'info.json',)) else False, 
        ], index=["renaming_and_drep_indexing"]), axis=1)
        return status
    
    def check_s11(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary['group']==group]["fileHeader"],
            "quantification": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(self.quant_output_dir, 'alignment_files', f'{x.name}.quant', 'quant.sf')) else False, 
        ], index=["quantification"]), axis=1)
        return status
    
    # abundance_output_dir = os.path.join(quant_output,'bin_abundance_table.tab')
    def check_s12(self, group, *args, **kwargs):
        status = pd.DataFrame({
            "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
            "quant_postproc": False,
        }).set_index("fileHeader")
        status = status.apply(lambda x: pd.Series([
            True if os.path.isfile(os.path.join(self.quant_output_dir, 'bin_abundance_table.tab')) else False, 
        ], index=["quant_postproc"]), axis=1)
        return status
    
    def check_all(self, ):
        status_list = []
        for group in self.groups:
            group_status = pd.DataFrame({
                "fileHeader": self.sample_summary[self.sample_summary["group"]==group]["fileHeader"],
                "group": group,
            }).set_index("fileHeader")

            for key in self.steps.keys():
                s_status = self.steps[key](group, nc=self.drep_coverage, sa=self.drep_ani)
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
        status.to_csv(os.path.join(self.prj_dir, "completeness.csv"))
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
    subparser_check.add_argument("-sa","--drep_ani", type=float, default=0.99, help="default: 0.99, average nucleotide identity for dreplication with salmon")

    subparser_filter = subparsers.add_parser("filter", help="select genomes and dereplication.")
    subparser_filter.add_argument("-nc","--drep_coverage", type=float, default=0.3, help="default: 0.3, nucleotide coverage for dreplication with salmon")
    subparser_filter.add_argument("-sa","--drep_ani", type=float, default=0.99, help="default: 0.99, average nucleotide identity for dreplication with salmon")

    subparser_classify = subparsers.add_parser("classify", help="classify MAGs with GTDB-TK.")
    subparser_classify.add_argument("-nc","--drep_coverage", type=float, default=0.3, help="default: 0.3, nucleotide coverage for dreplication with salmon")
    subparser_classify.add_argument("-sa","--drep_ani", type=float, default=0.99, help="default: 0.99, average nucleotide identity for dreplication with salmon")

    subparser_quant = subparsers.add_parser("quant", help="quantify MAGs with salmon.")
    subparser_quant.add_argument("-nc","--drep_coverage", type=float, default=0.3, help="default: 0.3, nucleotide coverage for dreplication with salmon")
    subparser_quant.add_argument("-sa","--drep_ani", type=float, default=0.99, help="default: 0.99, average nucleotide identity for dreplication with salmon")
    
    subparser_mapping_rate = subparsers.add_parser("mapping_rate", help="extract mapping rate from salmon quantification log files.")
    subparser_mapping_rate.add_argument("-nc","--drep_coverage", type=float, default=0.3, help="default: 0.3, nucleotide coverage used in dreplication step")
    subparser_mapping_rate.add_argument("-sa","--drep_ani", type=float, default=0.99, help="default: 0.99, average nucleotide identity used in dreplication step")
    subparser_mapping_rate.add_argument("-l","--single_log", type=str, default=None, help="default: None, specify an absolute path to a single log file to extract mapping rate from it. If not specified, the script will try to find the latest log file from the quantification step.")

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
        complete_check = CompleteCheck(
            prj_dir=args.prj_dir, drep_coverage=args.drep_coverage, drep_ani=args.drep_ani
        )
        complete_check.check_all()

    if args.modules=="filter":
        filtering(
            prj_dir=args.prj_dir, drep_coverage=args.drep_coverage, drep_ani=args.drep_ani
        )

    if args.modules=="classify":
        classify(
            prj_dir=args.prj_dir, drep_coverage=args.drep_coverage, drep_ani=args.drep_ani
        )
    
    if args.modules=="quant":
        quantify(
            prj_dir=args.prj_dir, drep_coverage=args.drep_coverage, drep_ani=args.drep_ani
        )
    
    if args.modules=="mapping_rate":
        mapping_rate(
            prj_dir=args.prj_dir, drep_coverage=args.drep_coverage, drep_ani=args.drep_ani, single_log=args.single_log
        )

if __name__=='__main__':
    main()
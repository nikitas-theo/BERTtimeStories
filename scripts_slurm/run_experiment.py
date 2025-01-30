import argparse
import os


def replace_in_file(inputfile, outputfile, exp_name, conf_dir, base_script, num_gpus):
    # Read the input file
    with open(inputfile, "r") as file:
        filedata = file.read()

    # Replace the target strings
    filedata = filedata.replace("<JOB-NAME>", exp_name)
    filedata = filedata.replace("<CONF-DIR>", conf_dir)
    filedata = filedata.replace("<BASE-SCRIPT>", base_script)

    if num_gpus > 1:
        filedata = filedata.replace("<NUM-GPUS>", str(num_gpus))
        filedata = filedata.replace("<RAM>", str(75 + 15 * (num_gpus - 1)) + "G")

    # Write the modified content to the output file
    with open(outputfile, "w") as file:
        file.write(filedata)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate and submit a SLURM job script."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="The name of the experiment config (without .yaml).",
    )
    parser.add_argument(
        "--base_script",
        type=str,
        help="The base script to run.",
        choices=["_base_LTG-BERT", "_base_GPT-Neo", "_small_LTG-BERT"],
    )
    parser.add_argument("--conf_dir", type=str, help="The configuration directory.")
    parser.add_argument(
        "--num_gpus", type=int, help="The number of GPUs to use.", default=1
    )

    # Parse the arguments
    args = parser.parse_args()

    exp_name = args.exp_name
    conf_dir = args.conf_dir
    base_script = args.base_script + ".yaml"
    num_gpus = args.num_gpus

    # Print the provided job name and conf directory
    print(f"Job Name: {exp_name}")
    print(f"Conf Dir: {conf_dir}")

    # Define the input and output file paths
    if args.num_gpus == 1:
        inputfile = "./scripts_slurm/srun_script_template.slurm"
        outputfile = f"./scripts_slurm/{conf_dir}/srun_script_{exp_name}.slurm"
    else:
        inputfile = "./scripts_slurm/srun_script_template_multi_gpu.slurm"
        outputfile = (
            f"./scripts_slurm/{conf_dir}/srun_script_{exp_name}_GPUS_{num_gpus}.slurm"
        )

    # Create the necessary directories
    os.makedirs(f"./scripts_slurm/{conf_dir}/", exist_ok=True)

    # Check if the input file exists
    if os.path.isfile(inputfile):
        # Replace placeholders in the file and save it
        replace_in_file(
            inputfile, outputfile, exp_name, conf_dir, base_script, num_gpus
        )
        print(
            f"Replaced all occurrences of '<JOB-NAME>' with '{exp_name}' and '<CONF-DIR>' with '{conf_dir}' in '{outputfile}'."
        )

    else:
        print(f"Input file '{inputfile}' not found!")


if __name__ == "__main__":
    main()

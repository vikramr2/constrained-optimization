"""The main CLI logic, containing also the main algorithm"""
from __future__ import annotations

from typing import Optional

import typer
import os
import shutil

from compare_clusterings import compare_clustering

from common.utils import ClustererSpec
from common.clusterers.ikc_wrapper import IkcClusterer
from common.clusterers.leiden_wrapper import LeidenClusterer, Quality

    
def main(
    input: str = typer.Option(..., "--input", "-i", help="The input network."),
    existing_clustering: str = typer.Option(..., "--existing-clustering", "-e", help="The existing clustering of the input network to be reclustered."),
    quiet: Optional[bool] = typer.Option(False, "--quiet", "-q", help="Silence output messages."), # (VR) Change: Removed working directory parameter since no FileIO occurs during runtime anymore
    clusterer_spec: ClustererSpec = typer.Option(..., "--clusterer", "-c", help="Clustering algorithm used to obtain the existing clustering."),
    k: int = typer.Option(-1, "--k", "-k", help="(IKC Only) k parameter."),
    resolution: float = typer.Option(-1, "--resolution", "-g", help="(Leiden Only) Resolution parameter."),
    threshold: str = typer.Option("", "--threshold", "-t", help="Connectivity threshold which all clusters should be above."),
    output: str = typer.Option("", "--output", "-o", help="Output filename."),
    outdir: str = typer.Option("", "--outdir", "-od", help="Output directory."),
    cores: int = typer.Option(4, "--nprocs", "-n", help="Number of cores to run in parallel.")
):
    if not os.path.exists('working_dir'):
        os.mkdir('working_dir')

    # Extra args
    added = f"-n {cores} "

    if quiet:
        added = added + '-q '

    # (VR) Check -g and -k parameters for Leiden and IKC respectively
    if clusterer_spec == ClustererSpec.leiden:
        assert resolution != -1, "Leiden requires resolution"
        added = added + f"-g {resolution}"
    elif clusterer_spec == ClustererSpec.leiden_mod:
        assert resolution == -1, "Leiden with modularity does not support resolution"
    else:
        assert k != -1, "IKC requires k"
        added = added + f"-k {k}"

    iter = 1
    prev_file = existing_clustering
    graph = input

    while True:
        # print(f'python3 batch_mincut.py -i {graph} -e {prev_file} -c {clusterer_spec.name} -t {threshold} -o working_dir/{output}.mincut{iter} {added}')
        os.system(f'python3 batch_mincut.py -i {graph} -e {prev_file} -c {clusterer_spec.name} -t {threshold} -o working_dir/{output}.mincut{iter} {added}')
        new_file = f'working_dir/{output}.mincut{iter}.after.tsv'

        if compare_clustering(prev_file, new_file):
            shutil.move(new_file, outdir)
            break

        iter += 1
        prev_file = new_file
        # print(f'python3 recluster.py -i {graph} -e {new_file} -c {clusterer_spec.name} -o working_dir/{output}.reclustering{iter} {added}')
        os.system(f'python3 recluster.py -i {graph} -e {new_file} -c {clusterer_spec.name} -o working_dir/{output}.reclustering{iter} {added}')

        graph = f'working_dir/{output}.reclustering{iter}.newgraph.tsv'
        

def entry_point():
    typer.run(main)

if __name__ == "__main__":
    entry_point()

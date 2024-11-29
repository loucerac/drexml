#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import requests

THIS_VERSION = 1.0


def build_gtex_url(version, qcv="RNASeQCv2.4.2"):
    """Build gtex url from versions."""

    url_parts = [
        "https://storage.googleapis.com",
        "adult-gtex",
        "bulk-gex",
        f"v{version}",
        "rna-seq",
        f"GTEx_Analysis_v{version}_{qcv}_gene_reads.gct.gz",
    ]

    return "/".join(url_parts)


@click.group()
def main():
    """Data downloader for drexml."""

    print(f"Running data downloader {THIS_VERSION}")


@main.command()
@click.option("--version", default="10", help="GTeX version.")
@click.option("--output", default="gtex.gc.gz", help="The name of the output file.")
def download_gtex(version, output):
    """Downloads the GTeX rnaseq database."""

    url = build_gtex_url(version=version)
    click.echo(url)

    try:
        response = requests.get(url, stream=True, timeout=100)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        click.echo(f"GTeX rnaseq database downloaded successfully to {output}")

    except requests.exceptions.RequestException as e:
        click.echo(f"Error downloading GTeX rnaseq database: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drugbank parser.
Adapted from https://github.com/dhimmel/drugbank/blob/gh-pages/parse.ipynb
Adaptaed from https://github.com/babelomics/drexml-retinitis.git
"""

import collections
import configparser
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import click
import pandas as pd
import requests
from biothings_client import get_client

THIS_VERSION = 1.0


def read_genes_df(path):
    """Wrapper to fix NANs in entrex IDs (IEEE-int has no NA)."""
    return pd.read_csv(path, sep="\t", dtype={"entrez_id": str})


def collapse_list_values(row):
    """Collapse cells containing lists into strings."""
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = "|".join(value)
    return row


def convert_gene_ids(gene_ids, source="entrezgene", target="uniprot,symbol"):
    """Myege wrapper."""
    renamer = {
        "uniprot": "uniprot_id",
        "entrezgene": "entrez_id",
        "symbol": "symbol_id",
    }
    client = get_client("gene")
    genes_converted = client.querymany(
        gene_ids,
        scopes=source,
        fields=target,
        species="human",
        as_dataframe=True,
    )

    genes_converted = genes_converted.reset_index(names=[source])
    cols_query = genes_converted.columns.isin(["uniprot", "entrezgene", "symbol"])
    genes_converted = genes_converted.loc[:, cols_query]
    genes_converted = genes_converted.rename(columns=renamer)

    return genes_converted


def build_drug_dataset(root):
    """Parse Drugbank drugs from Drugbank XML root."""
    ns = "{http://www.drugbank.ca}"
    inchikey_template = (
        "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
    )
    inchi_template = (
        "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"
    )

    rows = list()
    for _, drug in enumerate(root):
        row = collections.OrderedDict()
        assert drug.tag == ns + "drug"
        row["type"] = drug.get("type")
        row["drugbank_id"] = drug.findtext(ns + "drugbank-id[@primary='true']")
        row["name"] = drug.findtext(ns + "name")
        row["description"] = drug.findtext(ns + "description")
        row["groups"] = [
            group.text for group in drug.findall("{ns}groups/{ns}group".format(ns=ns))
        ]
        row["atc_codes"] = [
            code.get("code")
            for code in drug.findall("{ns}atc-codes/{ns}atc-code".format(ns=ns))
        ]
        row["categories"] = [
            x.findtext(ns + "category")
            for x in drug.findall("{ns}categories/{ns}category".format(ns=ns))
        ]
        row["inchi"] = drug.findtext(inchi_template.format(ns=ns))
        row["inchikey"] = drug.findtext(inchikey_template.format(ns=ns))

        # Add drug aliases
        aliases = {
            elem.text
            for elem in drug.findall(
                f"{ns}international-brands/{ns}international-brand"
            )
            + drug.findall(f"{ns}synonyms/{ns}synonym[@language='English']")
            + drug.findall(f"{ns}international-brands/{ns}international-brand")
            + drug.findall(f"{ns}products/{ns}product/{ns}name")
        }
        aliases.add(row["name"])
        row["aliases"] = sorted(aliases)

        rows.append(row)

    rows = list(map(collapse_list_values, rows))

    columns = [
        "drugbank_id",
        "name",
        "type",
        "groups",
        "atc_codes",
        "categories",
        "inchikey",
        "inchi",
        "description",
    ]
    drugbank_df = pd.DataFrame.from_dict(rows)[columns]

    return drugbank_df


def build_protein_df(root, use_groups=False):
    """Parse Drugbank associated proteins from Drugbank XML root."""
    ns = "{http://www.drugbank.ca}"

    protein_rows = list()
    for _, drug in enumerate(root):
        drugbank_id = drug.findtext(ns + "drugbank-id[@primary='true']")
        for category in ["target", "enzyme", "carrier", "transporter"]:
            proteins = drug.findall("{ns}{cat}s/{ns}{cat}".format(ns=ns, cat=category))
            for protein in proteins:
                row = {"drugbank_id": drugbank_id, "category": category}
                row["organism"] = protein.findtext("{}organism".format(ns))
                row["known_action"] = protein.findtext("{}known-action".format(ns))
                actions = protein.findall("{ns}actions/{ns}action".format(ns=ns))
                row["actions"] = "|".join(action.text for action in actions)
                uniprot_ids = [
                    polypep.text
                    for polypep in protein.findall(
                        f"{ns}polypeptide/{ns}external-identifiers/{ns}external-identifier[{ns}resource='UniProtKB']/{ns}identifier"
                    )
                ]

                if len(uniprot_ids) != 1:
                    if use_groups:
                        row["is_protein_group_target"] = True
                        row["uniprot_id"] = "|".join(uniprot_ids)
                    else:
                        continue
                else:
                    row["is_protein_group_target"] = False
                    row["uniprot_id"] = uniprot_ids[0]

                ref_text = protein.findtext(
                    "{ns}references[@format='textile']".format(ns=ns)
                )
                pmids = re.findall(r"pubmed/([0-9]+)", str(ref_text))
                row["pubmed_ids"] = "|".join(pmids)
                protein_rows.append(row)

    protein_df = pd.DataFrame.from_dict(protein_rows)
    protein_df.uniprot_id = protein_df.uniprot_id.str.split("|")
    protein_df = protein_df.explode("uniprot_id")

    return protein_df


@click.group()
def main():
    """Drugbank parser for drexml."""

    print(f"Running drugbank parser {THIS_VERSION}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
@click.option("--kind", type=click.Choice(["drugbank", "gtex"], case_sensitive=False))
def translate(path, output, kind):
    """Gene translation tool using mygene."""
    print("Running mygene translation tool.")

    path = Path(path)
    output = Path(output)

    kind = kind.lower()
    if kind == "drugbank":
        data = pd.read_csv(path, sep="\t")
        ids = data["uniprot_id"].unique()
        this_source = "uniprot"
        this_target = "entrezgene"
    elif kind == "gtex":
        data = pd.read_feather(path)
        ids = data.columns[data.columns.str.startswith("X")].str.replace("X", "")
        this_source = "entrezgene"
        this_target = "symbol"

    genes_df = convert_gene_ids(ids, source=this_source, target=this_target)
    genes_df.to_csv(output, sep="\t", index=False)
    print(f"Wrote {output}")


@main.command()
@click.argument("drugbank-output", type=click.Path(exists=False), nargs=1)
@click.argument("genes-output", type=click.Path(exists=False), nargs=1)
@click.option("--drugbank-path", type=click.Path(exists=True))
@click.option("--drugbank-genes-path", type=click.Path(exists=True))
@click.option("--gtex-genes-path", type=click.Path(exists=True))
def filter_db(
    drugbank_output, genes_output, drugbank_path, drugbank_genes_path, gtex_genes_path
):
    """Filter the dataset using targets with a known action."""
    print("Running drugbank target filter.")

    data = pd.read_csv(drugbank_path, sep="\t")
    genes_drugbank = read_genes_df(drugbank_genes_path)
    genes_gtex = read_genes_df(gtex_genes_path)
    data = (
        data.merge(genes_drugbank, how="inner")
        .merge(genes_gtex, how="inner")
        .query(
            (
                'category=="target"'
                ' & groups.str.contains("^approved")'
                ' & (~groups.str.contains("withdrawn"))'
                ' & organism=="Humans"'
                ' & (known_action=="yes")'
            )
        )
        .sort_values("drugbank_id")
    )

    data.to_csv(drugbank_output, sep="\t", index=False)

    genes_gtex["drugbank_approved_targets"] = genes_gtex.entrez_id.isin(
        data.entrez_id.unique()
    )
    genes_gtex.to_csv(genes_output, sep="\t", index=False)


@main.command()
@click.argument("xml-path", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
@click.option("--use-groups", is_flag=True, default=False, help="number of greetings")
def parse(xml_path, output, use_groups):
    """Drugbank parse XML and save to TSV."""
    print("Running XML parser.")
    xml_path = Path(xml_path)
    with zipfile.ZipFile(xml_path) as this_zip_file:
        with this_zip_file.open("full database.xml") as xml_file:
            tree = ET.parse(xml_file)
    root = tree.getroot()

    drugbank_df = build_drug_dataset(root)

    protein_df = build_protein_df(root, use_groups=use_groups)

    db_df = drugbank_df.merge(protein_df, how="inner")
    db_df.to_csv(output, sep="\t", index=False)
    print(f"Wrote {output}")


@main.command()
@click.option("--version", default="5-1-12", help="DrugBank version.")
@click.option("--user", help="Your DrugBank username.")
@click.option(
    "--password",
    help="Your DrugBank password.",
)
@click.option("--filename", default="drugbank.zip", help="The name of the output file.")
def download_drugbank(version, user, password, filename):
    """Downloads the DrugBank full database."""

    if not user or not password:
        try:
            cfg = configparser.ConfigParser()
            cfg.read(Path.home().joinpath(".config", "drugbank.ini"))
        except Exception as e:
            click.echo(e)
            sys.exit(
                "Drugbank credentials not provided or .config/drugbank.ini not filled."
            )
        user = cfg["drugbank"]["username"]
        password = cfg["drugbank"]["password"]

    url = f"https://go.drugbank.com/releases/{version}/downloads/all-full-database"
    click.echo(url)

    try:
        response = requests.get(url, auth=(user, password), stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        click.echo(f"DrugBank database downloaded successfully to {filename}")

    except requests.exceptions.RequestException as e:
        click.echo(f"Error downloading DrugBank database: {e}")


if __name__ == "__main__":
    main()

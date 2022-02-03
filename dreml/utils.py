import pathlib
import mygene


def entrez2symbol(entrez_lst):
    """Converts a list of entrez genes to symbol."""
    myg = mygene.MyGeneInfo()
    frame = myg.querymany(
        entrez_lst,
        scopes="entrezgene",
        fields="symbol",
        returnall=True,
        as_dataframe=True,
    )

    return frame


def save_converter(frame, fpath):
    """Saves the output dataframe of entrez2symbol as a TSV."""
    frame.to_csv(
        pathlib.Path(fpath),
        sep="\t",
        index=True,
        index_label="entrez",
        columns=["symbol"],
    )

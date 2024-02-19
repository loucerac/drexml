#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
from pathlib import Path
import dotenv


# In[59]:


benchmark_folder = Path(dotenv.find_dotenv()).parent
experiments_folder = benchmark_folder.joinpath("experiments")
results_folder = benchmark_folder.joinpath("results")


# In[60]:


result_dfs = []
for map_size in [1, 25, 50, 75, 100]:
    map_size_str = f"{map_size:03d}"
    disease_name = f"disease_{map_size_str}"
    disease_folder = experiments_folder.joinpath(disease_name)
    csv_path = disease_folder.joinpath(f"{disease_name}.csv")
    df_i = pd.read_csv(csv_path, sep=",")
    df_i["map_size"] = map_size_str
    result_dfs.append(df_i)


# In[61]:


df = pd.concat(result_dfs, axis=0, ignore_index=True)
df["device"] = df["command"].str.contains("gpus").replace({True: "GPU", False: "CPU"})
fname = f"benchmark_results_{df['device'].iloc[0].lower()}.tsv"
df.to_csv(results_folder.joinpath(fname), sep="\t")


# In[ ]:





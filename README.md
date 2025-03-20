# Polars-MAS: Multiple Association testing

`polars-mas` is a python library and CLI tool meant to perform large scale multiple association tests, primarily seen in academic research. Currently this tool only supports Firth's logistic regression. Will run as a stand in replacement for PheWAS R package analysis, especially for Phecodes. `polars-mas` is built to leverage the speed and memory efficiency of the `polars` dataframe library and it's interoperability with the `sklearn` and `statsmodels` libraries. 

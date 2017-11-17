# To train the PG network
python train_pg.py -e 5

# To visualize the results
python plot.py data/SingleAgent_05-11-2017_19-26-40 --value AverageReturn

# To train Multiple agent problem
python MAA3C.py
# To visualize results
python MAplot.py data/MultiAgent_16-11-2017_17-06-18 --value AverageReturn1 AverageReturn2 AverageTotalReturn
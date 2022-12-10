## Codes explanation
Not sure if this is how it's done. hehe


- lr_sweep_harpodata: Parameter sweep to select best lr model.

- rf_sweep_harpodata: Parameter sweep to select best rf model.

- rf_harpodata.ipynb: It gets the best rf model metrics. The train//test split is done manually before feeding the pycaret model. Makes a hundred splits to show how much the metrics vary.

- rf_harpodata_assumingsplit.ipynb: It gets the best rf model metrics. It assumes that the pycaret module handles train//test split correctly not influencing it with either upsampling or downsampling. Makes a hundred splits to show how much the metrics vary.
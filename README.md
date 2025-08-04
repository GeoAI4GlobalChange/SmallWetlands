# Small wetland CH4 emissions    
The repository contains codes of identifying small wetlands and estimating wetland methane emissions at the global scale. Specifically, we used the GWL_FCS30 wetland dataset, and used ancillary datasets including HydroLAKES, SWOT PLD, Global River Width from Landsat, and Global Dam Watch Database, to exclude water bodies such as lakes, ponds, rivers, streams, and reservoirs.  
## Code introduction
1)	Identifying small wetlands    
In the ”Identifying_small_wetlands.py” file, it contains code to identify small wetlands and estimate wetland extent.    
2) Quantifying CH4 emission from small wetlands    
In the”Estimating_CH4_emissions.py” file, run the “global_wetland_CH4_emissions” function.  

## Data availability
The wetland dataset is from GWL_FCS30: https://zenodo.org/records/7340516  
For the input drivers, GPP was obtained from the GOSIF dataset, which is available at https://globalecology.unh.edu/data/GOSIF-GPP.html.  
Other variables (soil temperature, air temperature, air pressure, precipitation, wind speed, snow cover, and soil water content) were obtained from ERA5-land datasets,  
which are available at (https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics).  
HydroLAKES: https://www.hydrosheds.org/products/hydrolakes  
SWOT PLD:  https://hydroweb.next.theia-land.fr/  
Global River Width from Landsat: https://zenodo.org/records/1297434#.YfGIXf7MLcu  
Global Dam Watch Database: https://www.globaldamwatch.org/database  
Please read the corresponding paper for other datasets used for analysis.  

## Computing environment  
1) Packages required: rasterio, opencv-python, netCDF4, pandas, numpy, scipy, pytorch, scikit-learn, pickle, tqdm  
2) Anaconda is recommended to install all the packages required by runing: Conda install list_of_packages  
3) The code has been tested on: Windows 11 version 23H2 with Python 3.8.19 and Intel(R)Core(TM)i9-9900K CPU    

## References
Details of the research will be seen in our paper (under review now). If you have any questions about the code or dataset, please contact fali2@stanford.edu.

# Human Mobility Datasets Enriched With Contextual and Social Dimensions

This repository contains the code and documentation concerning the pipeline used to generate the two semantically enriched trajectory datasets presented in the resource paper *Human Mobility Datasets Enriched With Contextual and Social Dimensions* by Chiara Pugliese ([CNR-IIT](https://www.iit.cnr.it/en/)), Francesco Lettich ([CNR-ISTI](https://www.isti.cnr.it/en/)), Guido Rocchietti (CNR-ISTI), Chiara Renso (CNR-ISTI), and Fabio Pinelli ([IMT Lucca](https://www.imtlucca.it/), CNR-ISTI). 

The paper has been submitted to the **Resource Track** of the [ACM CIKM 2025](https://cikm2025.org/) international conference -- stay tuned for future updates!


## Overview of the pipeline's Jupyter Notebooks and Python scripts

In the following, we provide a brief description of the various Jupyter notebooks that implement our pipeline. We suggest to execute the notebooks have to be executed in the order they appear.
Finally, note that once all the notebooks have been executed, our pipeline continues with the application of the [MAT-Builder system](https://github.com/chiarap2/MAT_Builder).

**1 - OSM NYC GPX traces downloader.ipynb**: 
a multi-threaded downloader slices the given New York City's bounding box into manageable tiles and fetches GPX trackpoints via the OSM API, complete with retry logic to handle transient failures. The resulting GPX files are organized by tile for later processing.

**2.1.1 - Multiple GPX to GeoPandas trajectory processing (NY case).ipynb**:
this notebook reads GPX files from notebook 1, extracts both track metadata and track points, and converts them into GeoPandas dataframes. It applies data cleaning steps, fixing nonstandard tags, parsing timestamps, merging metadata, and removing duplicates. Finally, it concatenates the per-tile results into a unified Parquet dataset with consistent categorical user IDs.

**2.1.2 - Concatenate dataframes multiple bounding boxes (NY case).ipynb**:
once all per-tile Parquet files are generated, this notebook loads each into GeoPandas, concatenates them into a single “mega” dataframe, converts timestamps to the America/New_York timezone, and de-identifies users. Ultimately writes out a merged nyc_merged.parquet.

**2.2 - Single GPX to GeoPandas trajectory processing (Paris case).ipynb**:
this notebook preprocesses a single pre-existing GPX file containing trajectories moving around Paris. It is assumed that such GPX has been downloaded with the [JOSM tool](https://josm.openstreetmap.de/). This notebook reads track metadata and points from this GPX in streamed chunks, to handle large file sizes without exhausting memory. It filters and renames columns, merges metadata links into user IDs, and cleans timestamp formats before converting UTC times to Europe/Paris. The final GeoPandas dataframe is saved in the Parquet format.

**4 - OSM raw trajectory preprocessing.ipynb**:
starting from the preprocessed trajectories coming from the notebooks 2.1.2 or 2.2 (thus either Paris or NYC), this notebook computes per-user summaries—total observations, time spans, and sampling rates. Then, it filters out trajectories that fall below duration or frequency thresholds. The preprocessed trajectories are saved to a new Parquet file.

**5 - Ensure MAT-Builder compatibility.ipynb**:
to prepare trajectories for the MAT-Builder pipeline, this notebook reads the Parquet files outputted by notebook 4, and assigns a unique identifier for each user and trajectory, splits the geometry column into separate latitude and longitude fields, and drops the original geometry to match MAT-Builder’s expected schema. The notebook ultimately writes out a new Parquet file, ready for ingestion by MAT-Builder.

**6 - Generate dataset POI from OpenStreetMap.ipynb**:
leveraging OSMnx, this notebook defines a function to download Points of Interest by tag within a specified bounding box, then standardizes the resulting dataframe by renaming fields, selecting essential columns, and filtering out entries without names. It fetches multiple POI categories such as amenities, shops, tourism sites, historic landmarks, and leisure spots, and saves the compiled dataset as a Parquet file, ready for ingestion by MAT-Builder.

**7 - Meteostat daily data downloader.ipynb**:
this notebook automates the retrieval of historical daily weather records from Meteostat’s bulk CSV endpoints for a list of station IDs, filtering for data post-1990 and selecting key variables like average temperature and precipitation. It handles missing values through interpolation and forward-filling, then classifies each day’s weather based on precipitation thresholds. The cleaned, labeled weather dataset is finally exported to Parquet, ready for ingestion by MAT-Builder.

**8 - Generate Posts.py**: standalone Python scripts that uses the Meta’s meta-llama/Llama-3.3-70B-Instruct model to generate synthetic realistic social media posts from stop segments augmented with POIs. More specifically: executing this script first requires to have (1) the preprocessed trajectories of either Paris or NYC from notebook 5, and (2) the POIs of Paris or New York City. Then, give these two as input to the MAT-Builder system, which has then to be used to (1) compress the raw trajectoires, (2) segment the trajectories into stop and move segments, and finally (3) augment the stop segments with the POIs located within 50 meters from their centroids. One of the outputs of step (3) is a file named ```enriched_occasional.parquet```, which can then be given as input to this script to generate synthjetic but realistic posts based on the name and categories of the closest POI associated with each stop.  

**9 - Prepare social media dataset for MAT-Builder.ipynb**:
In this final notebook, the social media metadata produced by the Python script 8 are processed to generate the final synthetic tweets. The final output consists of a Parquet file , ready for ingestion by MAT-Builder.

## TO BE COMPLETED WITH MAT-Builder! 

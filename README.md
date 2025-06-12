# Human Mobility Datasets Enriched With Contextual and Social Dimensions

This repository contains the code and documentation concerning the pipeline used to generate the two semantically enriched trajectory datasets presented in the resource paper *Human Mobility Datasets Enriched With Contextual and Social Dimensions* by Chiara Pugliese ([CNR-IIT](https://www.iit.cnr.it/en/)), Francesco Lettich ([CNR-ISTI](https://www.isti.cnr.it/en/)), Guido Rocchietti (CNR-ISTI), Chiara Renso (CNR-ISTI), and Fabio Pinelli ([IMT Lucca](https://www.imtlucca.it/), CNR-ISTI). 

The paper has been submitted to the **Resource Track** of the [ACM CIKM 2025](https://cikm2025.org/) international conference.


## Overview of the pipeline

In the following, we provide a brief description of the various Jupyter notebooks that implement our pipeline. We suggest to execute the notebooks have to be executed in the order they appear.
Finally, note that once all the notebooks have been executed, our pipeline continues with the application of the [MAT-Builder system](https://github.com/chiarap2/MAT_Builder).

**1 - OSM NYC GPX traces downloader.ipynb**: 
a multi-threaded downloader slices the given New York City's bounding box into manageable tiles and fetches GPX trackpoints via the OSM API, complete with retry logic to handle transient failures. The resulting GPX files are organized by tile for later processing.

**2.1.1 - Multiple GPX to GeoPandas trajectory processing (NY case).ipynb**:
This notebook reads GPX files from notebook 1, extracts both track metadata and track points, and converts them into GeoPandas dataframes. It applies data cleaning steps, fixing nonstandard tags, parsing timestamps, merging metadata, and removing duplicates. Finally, it concatenates the per-tile results into a unified Parquet dataset with consistent categorical user IDs.

**2.1.2 - Concatenate dataframes multiple bounding boxes (NY case).ipynb**:
Once all per-tile Parquet files are generated, this notebook loads each into GeoPandas, concatenates them into a single “mega” dataframe, converts timestamps to the America/New_York timezone, and de-identifies users. Ultimately writes out a merged nyc_merged.parquet.

**2.2 - Single GPX to GeoPandas trajectory processing (Paris case).ipynb**:
This notebook preprocesses a single pre-existing GPX file containing trajectories moving around Paris. It is assumed that such GPX has been downloaded with the [JOSM tool](https://josm.openstreetmap.de/). This notebook reads track metadata and points from this GPX in streamed chunks, to handle large file sizes without exhausting memory. It filters and renames columns, merges metadata links into user IDs, and cleans timestamp formats before converting UTC times to Europe/Paris. The final GeoPandas dataframe is saved in the Parquet format.

**4 - OSM raw trajectory preprocessing.ipynb**
Starting from the preprocessed trajectories coming from the notebooks 2.1.2 or 2.2 (thus either Paris or NYC), this notebook computes per-user summaries—total observations, time spans, and sampling rates. Then, it filters out trajectories that fall below duration or frequency thresholds. The preprocessed trajectories are saved to a new Parquet file.

**5 - Ensure MAT-Builder compatibility.ipynb**
To prepare trajectories for the MAT-Builder pipeline, this notebook reads the Parquet files outputted by the notebook 4, and assigns a unique traj_id for each user, splits the geometry column into separate latitude and longitude fields, and drops the original geometry to match MAT-Builder’s expected schema. After resetting the index, it writes out a new Parquet file ready for ingestion by MAT-Builder.

**6 - Generate dataset POI from OpenStreetMap.ipynb**
Leveraging OSMnx, this notebook defines a function to download Points of Interest by tag within a specified bounding box, then standardizes the resulting dataframe by renaming fields, selecting essential columns, and filtering out entries without names. It demonstrates how to fetch multiple POI categories such as amenities, shops, tourism sites, historic landmarks, and leisure spots, and saves the compiled dataset as a Parquet file for spatial enrichment tasks.

**7 - Meteostat daily data downloader.ipynb**
This notebook automates the retrieval of historical daily weather records from Meteostat’s bulk CSV endpoints for a list of station IDs, filtering for data post-1990 and selecting key variables like average temperature and precipitation. It handles missing values through interpolation and forward-filling, then classifies each day’s weather based on precipitation thresholds. The cleaned, labeled weather dataset is finally exported to Parquet for integration with trajectory data.

**9 - Prepare social media dataset for MAT-Builder.ipynb**
In this final notebook, enriched stop-level social media data are loaded and pared down to user IDs, arrival and leave timestamps, and sentiment scores. Synthetic tweet timestamps are uniformly sampled between stop start and end times, and placeholder text strings are randomly assigned positive or negative sentiment labels. The resulting post-level dataset—complete with tweet_id, uid, tweet_created, and text fields—is written out as a Parquet file compatible with MAT-Builder.

## TO BE COMPLETED WITH MAT-Builder! 

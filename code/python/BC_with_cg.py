import cugraph
import cudf
import json


ctuName="capture20110815-2.binetflow.labels.gz"

# read data into a cuDF DataFrame using read_csv
gdf = cudf.read_csv('../../data/csv/'+ctuName, names=["SrcAddr", "DstAddr","Bytes"], dtype=["str", "str","float32"])

# We now have data as edge pairs

# create a Graph using the source (src) and destination (dst) vertex pairs
G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source='SrcAddr', destination='DstAddr', edge_attr='Bytes')
BC_cg=cugraph.betweenness_centrality(G)

with open("BC_cg.json", 'w') as f:
    json.dump(BC_cg, f, indent=2) 
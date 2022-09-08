from optparse import OptionParser
import graph_tool.all as gt
import sys,time
import pandas as pd
sys.path.append("./code/python/functions/")

def parsearg():
    parser = OptionParser(usage="usage: %prog [opt] ",
                          version="%prog 1.0")
    parser.add_option("-i", "--input",
                      action="store",
                      dest="input",
                      default=False,
                      help="set name of the input file")
    parser.add_option("-o", "--output",
                      action="store", # optional because action defaults to "store"
                      dest="output",
                      help="set name of the output file",)
    (opt, args) = parser.parse_args()
    return opt

if __name__ == '__main__':
   opt = parsearg()
   if opt.input == None or opt.output == None:
        print("[Py] Parameters missing. Please use --help for look at available parameters.")
        sys.exit()
   else:
       ncol_file  = pd.read_csv(opt.input, sep=' ', header=None)
       print("[Py]", len(ncol_file),"rows read") 
       ncol_file = ncol_file.drop(ncol_file[ncol_file[2] == 0].index)
       print("[Py]", len(ncol_file),"rows remains")   
       print("[Py] Calculating AC")
       #net = nx.from_pandas_edgelist(ncol_file, source = 0, target = 1, edge_attr = 2 )
       net = gt.load_graph_from_csv(opt.input, skip_first=False, directed=True,
       hashed=True, 
       eprop_names=["weight"], 
       eprop_types=['int'],
       csv_options={'delimiter':" "})
       net.list_properties()
       start = time.time()
       w= net.edge_properties["weight"]
       nodes=pd.DataFrame(net.vp.name)
       nodes_df=pd.DataFrame(nodes)
       ac =gt.katz(net,weight=None,norm=False,alpha=0.1,epsilon=1e-02,max_iter=10)
       print(f"[Py] Total time elapsed: {(time.time() - start):.4F} seconds")
       output_df = pd.DataFrame(ac)
       output_df = pd.concat([nodes_df,output_df],axis=1)
       output_df.to_csv(opt.output,index=False,header=False)
       
